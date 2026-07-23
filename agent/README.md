# Eve Agent — Structure Map

This directory is a demonstrative Eve conversion of the caseworker-facing web-automation
system prompt that otherwise lives in `lib/ai/prompts/`. It exists to prove out Eve's
concepts (always-on instructions + dynamic instructions, skills, tools, subagents,
sandbox, compaction) against real prompt content from this repo — not to replace the
production Next.js agent loop in `app/(chat)/api/chat/route.ts`. Nothing under `lib/` or
`app/` was touched to build it; see "Additive-only" below.

## Tree

```
agent/
  agent.ts                                  top-level defineAgent (model + compaction)
  channels/eve.ts                           eve dev/TUI + Vercel channel wiring
  instructions.md                           always-on core instructions
  instructions/date.ts                      dynamic per-session date instruction
  sandbox.ts                                representative sandbox definition
  skills/
    benefits-application/SKILL.md           benefits-application protocol
    browser-automation/
      SKILL.md                              browser mechanics + modal handling
      field-patterns.md                     sibling reference (loaded on demand)
      custom-dropdowns.md                   sibling reference (loaded on demand)
      browser-commands.md                   sibling reference (loaded on demand)
  tools/
    action_label.ts                         example tool
    browser.ts                              example tool
    check_submit_gate.ts                    example tool
    form_summary.ts                         example tool
    gap_analysis.ts                         example tool
    read_reference.ts                       spike proof tool, superseded by skills
    update_working_memory.ts                defineState compaction prototype
  subagents/
    database_verification/                 agent.ts + instructions.md + 2 tools
    requirements_research/                  agent.ts + instructions.md + 1 tool
    form_review/                            agent.ts + instructions.md + 1 tool
```

## Instructions and Dynamic Date

`agent/instructions.md` is the always-on core. It carries identity, Applicant Identity,
Core Approach, Step Management, and Action Labeling from the top level of
`lib/ai/prompts/web-automation.ts`, but not that file's entire top level: Web Search
Protocol moved to the `requirements_research` subagent and Resuming After Interruption
moved to the `browser-automation` skill, since both are situational rather than
always-needed. It also carries two sections promoted from elsewhere for the same
reason — always-needed, not situational: Communication Rules (originally in
`lib/ai/prompts/application-protocol.ts`) and Forbidden Actions (originally in
`lib/ai/prompts/browser-and-forms.ts`). `agent/instructions/date.ts` replaces
`getCurrentDateString()` from `web-automation.ts` with a `defineDynamic` resolver on
`session.started`, so the date is computed fresh per session rather than baked in at
build time.

## Skills

`agent/skills/browser-automation/` carries the bulk of `lib/ai/prompts/browser-and-forms.ts`
(Core Workflow, Ref Format, Snapshot Modes, Selector Rules, Masked Fields, Field Type
Patterns, Custom Dropdowns, Multi-Page Forms, Dynamic/Conditional Fields, Modal Handling,
Error Recovery, Form Submission Protocol, Parameter Types) plus Resuming After
Interruption from `web-automation.ts`, minus Forbidden Actions (moved to always-on
instructions, see above). Its three sibling reference files
(`field-patterns.md`, `custom-dropdowns.md`, `browser-commands.md`) are verbatim copies
of `lib/ai/prompts/references/*.md`, loaded on demand through Eve's skill mechanism
rather than a tool call.

`agent/skills/benefits-application/` carries most of `lib/ai/prompts/application-protocol.ts`
(Autofilled Field Detection, Filling Fields, No vs Unknown Distinction, Autonomous
Progression, Review Screen, Gap Analysis Protocol, Form Completion Summary), minus
Database Retrieval & Verification, Data Provenance, and Field Mapping & Inference Rules
(all moved to the `database_verification` subagent). Review Screen and Form Completion
Summary are intentionally duplicated in the `form_review` subagent — a declared
subagent inherits no skills, so anything it needs must be copied into its own
`instructions.md`.

## Tools

The five tools under `agent/tools/` (`action_label`, `browser`, `check_submit_gate`,
`form_summary`, `gap_analysis`) are demonstrative stubs — each file names the real
implementation it mirrors in `lib/ai/tools/`. They exist to prove that Eve's
`defineTool` shape can carry this repo's tool descriptions and schemas, not to replace
the production tools, which stay wired into `app/(chat)/api/chat/route.ts` as before.
`agent/tools/read_reference.ts` is the original spike proof tool (reads
`lib/ai/prompts/references/*.md` off disk) and is superseded by the
`browser-automation` skill's sibling reference files — it is retained only as a record
of the pre-skills approach, not as the recommended pattern for new reference material.
`agent/tools/update_working_memory.ts` is not derived from any prompt file; it is a
Task 5 prototype using `defineState` from `eve/context` to persist participant/form
data outside the model's message history so it survives Eve's internal compaction —
see `docs/eve-spike-findings.md` Q2.

## Subagents

Three declared subagents each get their own `agent.ts` (description + model) and
`instructions.md`, since a subagent inherits none of the top-level instructions or
skills and must carry everything it needs. `database_verification` carries Database
Retrieval & Verification, Data Provenance, and Field Mapping & Inference Rules from
`application-protocol.ts`, plus its own copies of the `getApricotRecord` and
`getApricotFormFields` tools. `requirements_research` carries Web Search Protocol from
`web-automation.ts` plus its own `web_search` tool. `form_review` carries Review Screen
and Form Completion Summary from `application-protocol.ts` (duplicated with the
`benefits-application` skill, for the reason given above) plus its own `form_summary`
tool.

## Sandbox, Compaction, and Channel

`agent/agent.ts` is the top-level `defineAgent`, configuring the model and Eve's
built-in compaction (`thresholdPercent`) instead of porting
`lib/ai/context-compression.ts` or a `prepareStep` model-switch hook — Eve manages
context internally and exposes no such hook. `agent/sandbox.ts` is a representative
`defineSandbox` showing where per-session setup would go and how a skill's sibling
reference files are reached at runtime via `ctx.getSkill(...).file(...)`; it is not
tied to any specific prompt file. `agent/channels/eve.ts` predates Tasks 1–5 — it is
the minimal scaffold from the original spike (`eve dev` / TUI / Vercel OIDC wiring),
not something converted from the prompt.

## Additive-only

Nothing under `lib/`, `app/`, or `package.json` was modified across this whole
conversion. `git diff --stat d2bbaf0..HEAD -- lib app package.json` is empty; the
converted agent lives entirely under `agent/`, alongside the unmodified production
code it mirrors.

## Cross-Reference Note

Because prompt sections were split verbatim across `instructions.md`, the two skills,
and the three subagents, a few in-text references now point across files instead of
to a section in the same file. The source files were not reworded to fix this — the
point of the conversion was verbatim fidelity to the original prompt text — so use
this table to find the referenced section:

| Reference text | Found in | Actually lives in |
|---|---|---|
| "see Data Provenance" | `instructions.md` (Applicant Identity) | `subagents/database_verification/instructions.md` — Data Provenance (No Fabrication) |
| "see the Data Provenance section above" | `skills/benefits-application/SKILL.md` (Gap Analysis Protocol, Form Completion Summary) | `subagents/database_verification/instructions.md` — Data Provenance (No Fabrication) |
| "see the Data Provenance section above" | `subagents/form_review/instructions.md` (Form Completion Summary) | `subagents/database_verification/instructions.md` — Data Provenance (No Fabrication) |
| "follow Web Search Protocol normally" | `skills/browser-automation/SKILL.md` (Resuming After Interruption) | `subagents/requirements_research/instructions.md` — Web Search Protocol |
| "follow the Modal Handling section above" | `instructions.md` (Forbidden Actions, `evaluate` restrictions) | `skills/browser-automation/SKILL.md` — Modal Handling |

## Further Reading

- `docs/eve-spike-findings.md` — the underlying spike's findings on Eve's session
  API, context/compaction model, and durable state, with live `curl` proof of each.
- `docs/plans/2026-07-23-web-automation-prompt-to-eve.md` — the task-by-task plan
  this directory was built from.
