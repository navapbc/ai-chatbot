# Add Jev (TypeSafe System One) judgments to the form-filling agent

## Summary

Added `lib/jev/`, a shared wrapper around TypeSafe's System One model (Jev), and wired it into
three tools in both agent trees: gap-analysis rows, form-summary rows, and a new
`resolveFieldValue` tool. Every judgment is annotate-only — it adds a `jev` field to something
the agent already produced and never drops or rewrites a row.

## Changes made

**Shared core** — [`lib/jev/`](../../lib/jev)

- `client.ts` (feature gate + a never-throwing `askJev`) and `questions.ts` (all three questions,
  written in code so they can be versioned; nothing composes one at runtime). `client.ts` is
  deliberately free of `server-only` so the Eve runtime can import it instead of forking a copy.
- `enrich.ts` annotates gap and summary rows — total functions, so a disabled flag, a missing
  record or an unreachable API all return the input unchanged. Concurrency capped at 6.
- `flatten.ts` generates candidates from the record; `participant.ts` recovers the record by
  brace-matching JSON out of the caseworker's opening message, scanning every user turn.

**Legacy route** — the three tools became factories bound to the record, wired in
[`app/(chat)/api/chat/route.ts`](<../../app/(chat)/api/chat/route.ts>):

- `gap-analysis.ts` / `form-summary.ts` — `createGapAnalysisTool(participant)` and
  `createFormSummaryTool(participant)` replace the previously static exports.
- `resolve-field-value.ts` — new tool. Candidates come from the record in code, so the answer is
  a verbatim copy of an existing value; `none` is always offered.

**Eve tree** — Eve tools take no constructor argument and `ToolContext` carries no message
history, so the record travels through session state instead:

- `agent/hooks/participant.ts` lifts it out of `message.received`; `agent/lib/participant-state.ts`
  holds the `defineState` slot; `agent/lib/jev.ts` reads it back and returns null outside an Eve
  context, so a subagent degrades to un-annotated rows.
- `agent/tools/{gap_analysis,form_summary,resolve_field_value}.ts` and
  `agent/subagents/form_review/tools/form_summary.ts` call the same `lib/jev/` code.

**Online scoring** — `evals/online/scorers/hallucination-jev.ts` is a Jev judge that runs beside
the Sonnet hallucination judge on the same traces rather than replacing it. Off unless
`JEV_ONLINE_SCORING=true`; `apply.ts` omits a disabled scorer from its rule entirely, because
Braintrust pauses rules rather than individual scorers.

**Config** — `TYPESAFE_API_KEY` (secret) and `JEV_FEATURES` in `terraform/cloud_run.tf`, the
`jev_features` variable, `.env.example`, and preview-only enablement in `deploy.yml`.

**Tests** — `tests/agent/jev-participant.test.ts` (record extraction, feature gate) and
`tests/agent/jev-participant-hook.test.ts` (the Eve hook, with `eve/context` mocked).

**Config fix, 2026-09-22** — `terraform/variables.tf` had `jev_features` defaulting to `"all"`.
`promote.yml` passes no `jev_features`, so it took that default: promoting any image turned every
Jev feature on in the target environment. Reset to `""` and documented why in both the variable
and the `deploy.yml` comment, which had described the old `""` default as if it still existed.
Also corrected a stale note in `cloud_run.tf` claiming the Eve sibling had no Jev integration.

## What it improves or fixes

- **Gap-analysis false positives are now visible.** Each row carries a verdict —
  `genuinely_missing` / `present_in_record` / `derivable_from_record` — with the full probability
  distribution, so asking a caseworker for something the record already answers is measurable.
- **Filled values are checked against the record**, along with the `source` label the agent
  attached to each one (`database` / `caseworker` / `inferred`).
- **`resolveFieldValue` narrows invention** on ambiguous fields by selecting among record-derived
  candidates rather than letting the model compose a value. It does not eliminate it: the
  resolved string still passes through the agent before `browser fill` types it.
- **Nothing blocks on TypeSafe being reachable.** Annotations are absent when Jev fails; the
  cards, the tools and the form run behave exactly as they did before.
- **Prod is off by default again** after the config fix. Before: a promotion enabled all three
  features. After: only preview opts in, through an explicit `-var`.

Caseworker-facing behavior is unchanged in this phase — both cards render from the tool *input*
(`components/message.tsx`), so the annotations reach the model and the Braintrust trace but not
the UI. That is deliberate: acting on a verdict needs a threshold chosen from observed data.

## How to run / verify

```bash
pnpm exec vitest run --config vitest.config.node.mjs tests/agent/jev-participant.test.ts tests/agent/jev-participant-hook.test.ts
```

18 tests, covering record extraction and the feature gate.

To exercise it against the live API, set `TYPESAFE_API_KEY` and `JEV_FEATURES=all` in
`.env.local`, run `pnpm dev`, start an application from the landing page (the participant record
has to be in the opening message), and look for `missingFields[].jev` or `fields[].jev` on the
`execute_tool gapAnalysis` / `execute_tool formSummary` span in Braintrust.

To preview the online scorer config without pushing it:

```bash
JEV_ONLINE_SCORING=true pnpm eval:online:apply dev --dry-run
```

## Not yet done

- No offline eval covers the three questions, though "questions in code can be regression-scored"
  is the stated reason they live in code.
- `askJev` failures are silent — a row with no `jev` key could mean the flag is off, the record
  is absent, or the API rate-limited. The verdict distributions need that denominator.
- `JEV_JUDGE_MODEL` in `evals/online/scorers/shared.ts` is unverified, and an org Owner still has
  to enable built-in Jev in Braintrust before the online rule can be activated.
