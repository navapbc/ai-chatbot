# Convert `web-automation.ts` Prompt into the Eve Agent (Design)

Status: approved for implementation planning
Date: 2026-07-23
Branch: `feat/eve-integration`

## Context

The live app builds its web-automation system prompt in `lib/ai/prompts/`:
`web-automation.ts` composes `application-protocol.ts` (benefits-application rules)
and `browser-and-forms.ts` (browser mechanics), and `getWebAutomationSystemPrompt()`
is injected as a `system` message by the custom `streamText` loop in
`app/(chat)/api/chat/route.ts`. A `getCurrentDateString()` helper injects today's date.

The Eve spike (see `docs/eve-spike-findings.md`) established how this maps to Eve
idioms: an always-on `agent/instructions.md`; on-demand `agent/skills/`; runtime
context via `defineDynamic` + `defineInstructions`; tools as `agent/tools/*`;
subagents as `agent/subagents/<name>/` directories; and a sandbox via
`agent/sandbox.ts`. Eve compacts context internally (`defineAgent({ compaction })`),
and its skills mechanism supersedes the `readReference` tool.

This work converts the prompt into that Eve structure as a **demonstrative reference
conversion** — the prompt CONTENT moves fully into Eve idioms, and tools/subagents/
sandbox are provided as representative, well-commented examples (not full logic ports).

## Goals

- Move the entire `web-automation.ts` prompt content into idiomatic Eve structure
  under `agent/`, split correctly between always-on instructions and on-demand skills.
- Convert the dynamic date helper to a `defineDynamic` + `defineInstructions` resolver.
- Provide representative example `defineTool`s, three subagent directories, and a
  `defineSandbox` example that demonstrate the Eve patterns the prompt implies.
- Show that Eve skills replace the `readReference` tool as the on-demand-context
  mechanism.

## Non-goals

- Porting full working tool logic (Apricot API calls, the Kernel.sh browser tool,
  the interactive card renderers). Tools are demonstrative stubs with clear comments.
- Wiring the Eve agent to the chat UI, Postgres, or the live `/api/chat` route.
- Removing or changing anything under `lib/` — the existing prompt still powers the
  live app and is left untouched (additive only).
- Deleting `agent/tools/read_reference.ts` (kept, annotated as superseded by skills).

## Decisions (confirmed with stakeholder)

- **Depth:** demonstrative reference conversion (content fully moved; tools/subagents/
  sandbox are representative examples).
- **Subagents:** all three — `database_verification`, `requirements_research`,
  `form_review`.
- **Safety-critical rules** ("NEVER submit", forbidden actions, plain-language
  communication) live in the always-on `agent/instructions.md`, not a skill, so they
  are always in context.
- `lib/ai/prompts/*` is untouched; all new work is additive under `agent/`.

## Target structure

```
agent/
  agent.ts                         # + compaction config; model stays sonnet-4.6
  instructions.md                  # REPLACE spike stub — always-on core (below)
  instructions/
    date.ts                        # defineDynamic -> defineInstructions({markdown: today's date})
  skills/
    browser-automation/
      SKILL.md                     # browser mechanics (from browser-and-forms.ts)
      field-patterns.md            # sibling file (from lib/ai/prompts/references/)
      custom-dropdowns.md          # sibling file
      browser-commands.md          # sibling file
    benefits-application/
      SKILL.md                     # application protocol (from application-protocol.ts)
  tools/
    action_label.ts                # example defineTool (zod schema)
    gap_analysis.ts                # example (stub; card render is client-side)
    form_summary.ts                # example (stub)
    check_submit_gate.ts           # example (stub; Turnstile gate)
    browser.ts                     # thin example; comment -> sub-project 3 / Kernel.sh
    read_reference.ts              # KEEP; annotate as superseded by skills
    update_working_memory.ts       # KEEP (from spike)
  subagents/
    database_verification/
      agent.ts                     # defineAgent(model)
      instructions.md              # DB retrieval+verification, data provenance, field mapping
      tools/
        get_apricot_record.ts      # example
        get_apricot_form_fields.ts # example
    requirements_research/
      agent.ts
      instructions.md              # research-requirements-upfront + web-search protocol
      tools/
        web_search.ts              # example
    form_review/
      agent.ts
      instructions.md              # review screen + formSummary spec
      tools/
        form_summary.ts            # example (subagent-scoped)
  sandbox.ts                       # defineSandbox example env (browser-automation context)
```

Exact Eve APIs (confirmed against installed `eve@0.27.0`): `defineAgent`,
`defineInstructions` (+ `defineDynamic` resolver in `agent/instructions/`),
`defineSkill`/markdown skills in `agent/skills/`, `defineTool` (zod `inputSchema`
now works — zod4 installed), `defineSandbox` at `agent/sandbox.ts`, subagents as
`agent/subagents/<name>/` directories. Any API detail not yet confirmed at author
time is verified against `node_modules/eve/dist` during implementation, not guessed.

## Content mapping (where each prompt section goes)

| Source section | Destination |
|---|---|
| Agent mission / identity | `instructions.md` |
| Applicant Identity rules | `instructions.md` (safety-critical) |
| Core Approach, Step Management | `instructions.md` |
| Communication Rules + Language | `instructions.md` (always-on) |
| Forbidden Actions / NEVER submit | `instructions.md` (always-on safety) |
| `getCurrentDateString()` | `instructions/date.ts` (defineDynamic) |
| Web Search Protocol | `subagents/requirements_research/instructions.md` (+ referenced briefly in `instructions.md`) |
| Resuming After Interruption | `skills/browser-automation/SKILL.md` |
| Action Labeling | `instructions.md` (short) + `tools/action_label.ts` |
| Database Retrieval & Verification, Data Provenance, Field Mapping | `subagents/database_verification/instructions.md` |
| Gap Analysis Protocol, Autofill, Filling Fields, No-vs-Unknown, Autonomous Progression, Review Screen, Form Completion Summary | `skills/benefits-application/SKILL.md` |
| Browser Automation, Core Workflow, Ref Format, Snapshot/Selector rules, Masked Fields, Modals, Error Recovery, Submission Protocol, Parameter Types | `skills/browser-automation/SKILL.md` |
| `field-patterns.md` / `custom-dropdowns.md` / `browser-commands.md` | skill sibling files under `skills/browser-automation/` |

## Approach notes

- **`instructions.md` vs dynamic:** if Eve allows a static `instructions.md` and a
  `defineDynamic` instruction in `agent/instructions/` to coexist, use both (static
  core + dynamic date). If it does not, fold the date into a single dynamic resolver
  that emits the static core plus the date — verified during implementation.
- **`readReference` → skills:** the `browser-automation` skill carries the three
  reference files as sibling files; the SKILL.md notes that Eve loads them on demand
  natively, so `readReference` is no longer the access path. `read_reference.ts` is
  kept (tested in the spike) but annotated as superseded.
- **Subagent tool ownership:** Apricot tools move under `database_verification`;
  web search under `requirements_research`; `form_summary` under `form_review`
  (and a top-level example remains for the main agent's own summary calls).
- **Sandbox honesty:** `agent/sandbox.ts` is a representative `defineSandbox` env.
  A comment states that today the browser runs via Kernel.sh as an app-runtime tool
  (Eve tools run in the app runtime, not the sandbox — per the spike), and the
  sandbox example shows the Eve-idiomatic alternative for isolated compute.
- **Tools are demonstrative:** each example tool has a real zod `inputSchema` and a
  minimal `execute` that returns a shaped placeholder, with a comment naming what the
  production implementation would do and where its real logic lives today (`lib/`).

## Validation

- `npx eve dev` boots with the new `agent/` structure without schema/registration
  errors (Node 24; `AI_GATEWAY_API_KEY` exported).
- A live turn shows the model can (a) read a skill on demand and (b) delegate to at
  least one subagent — demonstrating the structure works end-to-end.
- The existing spike test stays green: `pnpm exec vitest run -c vitest.config.node.mjs`.
- No `lib/` files changed; `git status` shows only additive `agent/` changes (plus
  docs).

## Risks

- **Eve beta specifics** (skill sibling-file packaging, subagent directory layout,
  `defineDynamic` import path, whether static + dynamic instructions coexist) are
  verified against the installed package during implementation; where behavior differs
  from this design, the implementation adapts and the deviation is noted.
- **Demonstrative tools are not runnable end-to-end** for real form-filling — by
  design. This is a structure/pattern deliverable, not a functional migration
  (that is migration sub-project 2).
