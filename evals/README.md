# Evals

Offline evaluation suite for the web-automation chat agent. Runs the agent against fixed scenarios with mocked tool surfaces, scores its behavior, and uploads results to [Braintrust](https://www.braintrust.dev) for tracking.

The goal: catch regressions when prompts, tools, or models change — before they reach production.

## Quick start

```bash
# Run all suites against the default model (gpt-5-mini), one at a time
pnpm eval

# Run a single suite
pnpm eval:tool-selection

# Run against a different model (any prefix from EVAL_MODEL table below)
EVAL_MODEL=claude-opus-4-7 pnpm eval

# Run in CI mode (skips the dotenv wrapper)
pnpm eval:ci
```

`pnpm eval` uses `dotenv-cli` to load `.env.local`. `pnpm eval:ci` expects env vars to already be in the shell — used by the GitHub Actions matrix.

Both `eval` and `eval:ci` run the suite files **one at a time** (a shell loop), not concurrently. Running all suites in parallel makes ~40 multi-step agents fire at once; with large cached system prompts each call requests ~10k tokens, which saturates a single model's tokens-per-minute (TPM) limit (e.g. OpenAI's 500k TPM for gpt-5.1) and fails cases with `429 rate_limit_exceeded`. Sequential execution keeps each leg well under the ceiling. The loop still exits non-zero if any suite fails.

## The 10 suites

| Suite | What it measures | Heuristic scorers | LLM-as-judge |
|-------|------------------|-------------------|--------------|
| **tool-selection** | Agent picks the right tool for each user request (one-step) | first-tool-correct, all-expected-tools-called, no-hallucinated-tools | — |
| **autonomous-progression** | Agent moves through a multi-page form without being nudged | used-database-data, filled-form-fields, progressed-autonomously, stopped-before-submit, did-not-modify-database, showed-review, not-overly-verbose | — |
| **navigation** | Agent handles modals, county pickers, income popups; doesn't open new tabs or hit Back | navigated-past-landing, avoided-external-links, handled-county-modal, handled-income-modal, did-not-use-back, did-not-open-new-tab, stayed-on-site, reached-review, stopped-at-review | — |
| **clicking-ui-interaction** | Agent handles date masks, phone formats, native + select2 dropdowns, collapsible sections | handled-date-field, handled-phone-field, handled-native-dropdown, handled-select2-dropdown, expanded-collapsible-section, verified-masked-fields, filled-collapsible-fields | — |
| **ask-questions** | Agent asks for genuinely missing fields, in plain English, without overstepping into sensitive territory | called-gap-analysis, missing-field-coverage, did-not-assume-sensitive-fields, did-not-ask-for-known-fields, filled-known-fields | **ask-questions-judge** |
| **deduction** | Agent infers age from DOB, mailing from physical, ethnicity mapping, language carry-over, household composition | inferred-age-from-dob, inferred-mailing-address, mapped-ethnicity, carried-language, correct-household-size, included-household-members, inferred-nearest-office, no-false-gaps | — |
| **hallucination** | Agent doesn't invent household members, fabricate SSNs/emails, or attribute invented data to "database" | did-not-invent-spouse, correct-household-size, correct-household-member, no-fabricated-names, accurate-data-values, did-not-fabricate-marital-status, form-summary-sources-accurate, did-not-fabricate-email | **hallucination-judge**, **summary-attribution-judge** |
| **verbosity** | Agent communicates concisely, no play-by-play, no technical jargon, no wall-of-text | responses-are-concise, does-not-narrate-every-action, no-technical-jargon, provides-updates, text-is-infrequent, no-play-by-play | **verbosity-judge** |
| **regression-scenarios** | Cross-walked from [cwilkes-npbc/AI-Evaluations](https://github.com/cwilkes-npbc/AI-Evaluations) (Rosa 339688 × WIC/IHSS, Carolina 339702 × WIC/IHSS). Form-specific gap behaviors not exercised by the per-category suites. Scorers return `null` for non-applicable scenarios. | checked-wic-auth, selected-applying-for-self, mapped-gender-to-sex, asked-mother-eligibility, selected-blind-from-flag | — |
| **session-carryover** | Multi-turn sessions — agent must persist the participant identity across sequential user messages (WIC → IHSS → BenefitsCal) and answer cross-context Q&A from existing context. | same-user-across-turns, did-not-reask-for-user, answered-age, answered-last-name | — |

## Scoring

Every scorer returns a value in `[0, 1]`.

- **Heuristic scorers** — typically `0` (fail) or `1` (pass). A few return fractions for set-intersection style checks (e.g., `all_expected_tools_called` returns hits/expected).
- **LLM-as-judge scorers** — choice scoring `A=1.0 / B=0.5 / C=0.0` with chain-of-thought. The judge picks one letter; the score is the mapped value. See each registered scorer file in `scorers/` for the rubric.

In the Braintrust dashboard, each experiment shows the mean across rows for every scorer. A regression is typically a multi-point drop on one or more scorers between two adjacent experiments.

## Cross-model matrix

`EVAL_MODEL` controls which model runs the agent under test. Recognised prefixes:

| Prefix | Provider | Required API key |
|--------|----------|------------------|
| `gpt-*` / `o1*` / `o3*` | `@ai-sdk/openai` | `OPENAI_API_KEY` |
| `claude-*` | `@ai-sdk/anthropic` (direct) | `ANTHROPIC_API_KEY` |
| `gemini-*` | `@ai-sdk/google` | `GOOGLE_GENERATIVE_AI_API_KEY` |

Default is `gpt-5-mini`. CI runs a 4-leg matrix in `.github/workflows/evals.yml` over `gpt-5.1` / `claude-opus-4-7` / `claude-opus-4-8` / `gemini-3-pro`. Each leg uploads to a distinct Braintrust experiment (the model id is suffixed via `evalExperimentName()` in `helpers.ts`).

Production uses `claude-opus-4-7` via Vertex AI (see `lib/ai/providers.ts:17`). The CI matrix uses **direct Anthropic API** instead of Vertex for simpler secret management. Model behavior is identical between routes — only auth and rate-limit ceilings differ.

## Token usage & cost

Each suite logs the task agent's token usage (aggregated across all agent steps via `result.totalUsage`) to its Braintrust task span using the canonical metric names `prompt_tokens` / `completion_tokens` / `prompt_cached_tokens` — so they land in Braintrust's native token columns and `total_tokens` is auto-derived. A custom `estimated_cost_usd` metric is logged in the same `span.log` call (so it rides alongside the token metrics), computed from `evals/pricing.ts` for the active `EVAL_MODEL`. As a custom metric it does not appear in the CLI summary table — find it per-row in the experiment in the Braintrust UI. Only the system-under-test's usage is captured — LLM-as-judge scorer calls are excluded.

The per-model rates in `evals/pricing.ts` are **estimates marked `TODO(verify)`** — confirm them against the provider pricing pages before trusting the dollar figures. For unpriced models the cost key is omitted (with `pricing_known: false` in metadata) so a missing price reads as "unknown", not "free". The helpers live in `helpers.ts` (`logResultUsage`, `logUsageAndCost`, `addUsage`).

## Registered scorers

LLM-as-judge scorers live in `scorers/` and are registered in the Braintrust dashboard so they appear under the project's Scorers tab and can be invoked by `slug`.

Currently registered:

| File | Slug | Used in |
|------|------|---------|
| `scorers/hallucination.ts` | `hallucination-judge` | `hallucination.eval.ts` |
| `scorers/summary-attribution.ts` | `summary-attribution-judge` | `hallucination.eval.ts` |
| `scorers/ask-questions.ts` | `ask-questions-judge` | `ask-questions.eval.ts` |
| `scorers/verbosity.ts` | `verbosity-judge` | `verbosity.eval.ts` |

All four use `gpt-4o` + chain-of-thought + `ifExists: "replace"` so re-pushing overwrites the rubric.

### Push a scorer to Braintrust

After editing a rubric, push it once so the Scorers tab and `initFunction()` calls see the update:

```bash
npx braintrust push evals/scorers/<name>.ts
```

The `Eval()` calls invoke registered scorers via `initFunction({ projectName: "labs-asp", slug: "<slug>" })`. The eval file serializes the agent's `RunState` into a single string and passes it as `{ output: serialized }` — the registered prompt template fills `{{output}}` at evaluation time.

## Datasets

| File | Contents |
|------|----------|
| `datasets/participants.json` | Synthetic participant database records — names, addresses, household composition, income, ethnicity, language. Each suite picks a participant whose data shape matches the scenario (sparse for ask-questions, household for hallucination/deduction, etc). Includes `rosaFlores` (339688) and `carolinaDelgado` (339702), which mirror the real A360 PREVIEW_JSON payloads from the AI-Evaluations manual harness — used by `regression-scenarios` and `session-carryover`. |
| `datasets/snapshots.json` | Browser DOM snapshots for each page of each mock form. The stub `browser` tool returns these when the agent calls `action: "snapshot"`. Snapshots are keyed by suite → page name. |
| `datasets/form-fields.json` | Form-field definitions returned by the stubbed `getApricotFormFields` tool. |
| `datasets/test-cases.json` | Per-suite test case lists — input prompts, expected outputs (for tool-selection), max step counts. |
| `datasets/golden.json` | Forbidden-term lists and ideal-behavior references for the hallucination suite. |

When you change a participant's data, also update any heuristic scorer that hardcodes that participant's expected values. The `hallucination-judge` registered scorer also has Tanya Brooks's record baked into its system prompt — that's a known limitation (see the audit notes in commit `453311e`).

## AI-Evaluations cross-walk

The `regression-scenarios` and `session-carryover` suites cover the 57-step regression rubric from the manual harness at [cwilkes-npbc/AI-Evaluations](https://github.com/cwilkes-npbc/AI-Evaluations) (TC1 Rosa + TC2 Carolina). Coverage status:

| Status | Steps | Where |
|--------|-------|-------|
| Fully covered | 40 | Existing per-category suites (autonomous-progression, deduction, ask-questions, navigation, clicking-ui-interaction, verbosity, hallucination) |
| Generic scorer applies, scenario-specific fixture missing | 8 | Existing suites — could be tightened by adding Rosa/Carolina rows |
| Form-specific gaps | 5 | `regression-scenarios.eval.ts` — WIC auth (#4), applying-for-self (#13), gender→sex (#15), mother eligibility (#43), blind-from-special-needs-flag (#49) |
| Multi-turn carryover gaps | 6 | `session-carryover.eval.ts` — Q&A age (#9), same-user transitions (#10, #25, #46, #52), cross-context last name (#38) |
| Manual-only (UI-bound) | 1 | Step #23 — agent force-enabling a disabled submit DOM button. No offline equivalent; remains a manual check in the AI-Evaluations SOP. |

The original manual harness drives a live browser at the Preview / Dev / Production deployment and a human scores each step. The offline suites here run the agent against mocked tool surfaces, so they cost nothing per run, are deterministic, and can be wired into CI on every PR — at the cost of not exercising the real DOM (which is why step #23 stays manual).

## Tracing

`instrumentation.ts` at the repo root registers a `BraintrustExporter` from `@braintrust/otel` when `BRAINTRUST_API_KEY` is set. The chat route's `streamText` call has `experimental_telemetry` enabled, so AI SDK spans flow to the same `labs-asp` Braintrust project as the eval experiments.

The key is only injected in non-production environments: `terraform/cloud_run.tf` sets `BRAINTRUST_API_KEY` for `dev` and `preview` but never for `prod` (those run against the dev database). Production sessions therefore never reach Braintrust — `register()` early-returns without the key. There is no pipeline that copies real user traces into a dataset.

## Adding a new eval

1. Pick a participant in `datasets/participants.json` whose shape fits your scenario (or add a new one).
2. Add a page snapshot to `datasets/snapshots.json` under a new suite key.
3. Add test cases to `datasets/test-cases.json` under a new suite key.
4. Create `<name>.eval.ts` modeled on the closest existing suite. Extend `BaseRunState` from `helpers.ts` with whatever new tracking fields your scorers need.
5. Use `createBaseStubTools(state, { ... })` from `helpers.ts` — only override the tool callbacks that need custom behavior.
6. Call `getEvalModel()` for the agent model and `evalExperimentName("<Pretty Name>")` for the Braintrust experiment name.
7. Add heuristic scorers inline as `({ output }) => ({ name, score })`. For semantic checks too fuzzy for regex, register a new LLM-as-judge in `scorers/` and invoke it via `initFunction`.

## Adding a new registered scorer

1. Create `scorers/<name>.ts` modelled on `scorers/verbosity.ts`.
2. Keep the system prompt generic — pass any ground-truth data (e.g., the participant record) as part of `{{output}}` at invocation time rather than baking it into the prompt at registration time. (`hallucination-judge` is hardcoded to Tanya Brooks for historical reasons; new scorers should be participant-agnostic.)
3. Use `ifExists: "replace"` so re-pushing overwrites instead of failing.
4. Push it: `npx braintrust push evals/scorers/<name>.ts`.
5. Add `initFunction({ projectName: "labs-asp", slug: "<your-slug>" })` to the relevant `*.eval.ts` and call it from an async scorer entry.

## Environment

Local development (`pnpm eval` reads `.env.local`):

| Var | Required for | Notes |
|-----|--------------|-------|
| `BRAINTRUST_API_KEY` | All eval runs | Get from braintrust.dev → Settings → API keys |
| `BRAINTRUST_PARENT` | All eval runs | Defaults to `project_name:labs-asp` — only change if you want to point at a different Braintrust project |
| `EVAL_MODEL` | Optional | Defaults to `gpt-5-mini`. See prefix table above. |
| `OPENAI_API_KEY` | `gpt-*` / `o1*` / `o3*` models, all `gpt-4o` LLM judges | |
| `ANTHROPIC_API_KEY` | `claude-*` models | |
| `GOOGLE_GENERATIVE_AI_API_KEY` | `gemini-*` models | |

CI uses the same names as GitHub Actions secrets. The workflow soft-skips a matrix leg when its provider key is missing.

## Common gotchas

- **Missing `.env.local`**: `pnpm eval` will fail with a dotenv error. Use `pnpm eval:ci` instead if env vars are already in your shell.
- **Scorer rubric changed but score didn't move**: registered scorers need `npx braintrust push` to take effect. The eval file picks up the slug, but the dashboard prompt is what runs.
- **Experiments colliding in the dashboard**: every suite's `experimentName` is wrapped by `evalExperimentName()` which suffixes the active `EVAL_MODEL`. If multiple PRs are running the same matrix entry, Braintrust auto-namespaces by timestamp + git context — but very fast back-to-back runs can group as one experiment.
- **All 4 LLM judges are participant-agnostic**: each takes the participant record as part of the `{{output}}` payload at invocation time. Re-pushing them is required after any rubric change (`npx braintrust push evals/scorers/<name>.ts`).
