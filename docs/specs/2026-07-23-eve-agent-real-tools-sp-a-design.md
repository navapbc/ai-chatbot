# SP-A — Make the Eve Agent Functionally Real (Design)

Status: approved for implementation planning
Date: 2026-07-23
Branch: `feat/eve-integration`

## Context

The project is being revamped so **Eve is the real, live agent** (not the current
custom `streamText` loop in `app/(chat)/api/chat/route.ts`). The Eve spike
(`docs/eve-spike-findings.md`) proved Eve runs on this stack, and a demonstrative
conversion (`docs/specs/2026-07-23-web-automation-prompt-to-eve-design.md`) moved the
web-automation prompt into `agent/` with stub tools.

Getting from "demonstrative" to "real, live" is decomposed into sub-projects:

- **SP-A — make the Eve agent functionally real (standalone)** ← this document
- SP-B — wire the Eve runtime into the app UI (Q3 adapter route)
- SP-C — context (`defineState`) + Postgres history + cutover (remove legacy
  route/prompt/Mastra/Redis; finish archiving Apricot; add browser replay + session
  mapping back)
- SP-D — deploy to Vercel

Stakeholder constraints for the revamp: **keep Kernel.sh browser automation** (it is
the core capability), **drop Apricot** (archive it), and make Eve the definite harness.

## Purpose

Replace the demonstrative stub tools so `npx eve dev` can drive a **real browser via
Kernel.sh end-to-end**, with an **Apricot-free** agent whose data model is retargeted
to caseworker-message + inference. Validated standalone (`eve dev` + a live turn) — no
app-UI wiring, no persistence/replay, no legacy-route removal (those are SP-B/SP-C).

## Goals (exit criteria)

1. `npx eve dev` boots clean with **zero Apricot references** remaining in `agent/`
   (grep-verified) and no missing-subagent error.
2. The `browser` tool executes **real Kernel.sh commands** — a live turn navigates a
   real URL, returns a real DOM snapshot, and fills a field — resolving the Kernel
   session by a stable id from Eve's session context on each call (durable-safe; no
   reliance on a persistent in-process Playwright handle).
3. `check_submit_gate` runs the **real** DOM-probe / force-enable-submit logic against
   the same Kernel session (and still never clicks submit).
4. The agent is coherent Apricot-free: the anti-fabrication data-provenance rule is
   retained but retargeted to **caseworker + inference + missing** (no "database"
   source, no `getApricotFormFields`).

## Non-goals (deferred)

- Interactive card **rendering** / UI for `gap_analysis` / `form_summary` — SP-B.
  In SP-A these tools validate and return their structured data only.
- Wiring the Eve runtime to the chat UI / replacing `route.ts` — SP-B.
- GCS replay-video archival and the Postgres session-mapping table — SP-C.
- Removing the legacy `route.ts` / `lib/ai/prompts/*` / Mastra / Redis — SP-C cutover.
  `lib/` stays intact so the live app keeps building.
- A real web-search tool — `requirements_research` keeps its current stub / Eve
  `web_fetch`; a real search integration is a later follow-up.
- Vercel deploy — SP-D.

## Decisions (confirmed with stakeholder)

- **Tool realness:** browser + `check_submit_gate` fully real via Kernel now; card
  tools return structured data (render in SP-B); `action_label` is a lightweight real
  signal.
- **Apricot:** remove the code AND retarget the prose to caseworker + inference; keep
  the anti-fabrication framework.
- **Browser scope:** core commands only; defer replay + DB session-mapping to SP-C.
- **Provenance location:** the retargeted "Data Provenance (No Fabrication)" rule moves
  into the `benefits-application` skill (not `instructions.md`).

## Components

### 1. Real browser tool — `agent/tools/browser.ts`
Replace the stub with a real `defineTool` that dispatches structured commands
(navigate, snapshot, click, fill, type, select, check, evaluate, press, wait,
inputvalue) to a Kernel.sh browser session via the existing `@onkernel/sdk` +
`agent-browser` execution path. It runs in the app runtime (`process.env.KERNEL_API_KEY`
available). **Session identity:** derive a stable Kernel session key from Eve's session
context (`ctx.session` / `sandbox.id` — confirmed at plan time) and attach/resolve the
Kernel session by that key on each call.

**Statelessness (the crux):** the port must not depend on a persistent in-process
`BrowserManager`/Playwright `page` surviving between calls, because Eve's durable
execution can replay/resume steps. The implementation plan's FIRST step resolves whether
`agent-browser`'s `executeCommand` can attach to a Kernel session by id per call without
a persistent manager; if not, the plan adapts (e.g. a thin per-call
create-or-attach wrapper). Per-session command serialization, if still needed, is a
lightweight per-call concern, not a long-lived module mutex.

**Reuse:** prefer reusing the decoupled core of `lib/kernel/` rather than duplicating
Kernel logic; if the reusable pieces are entangled with the replay/DB extras, extract a
slim shared helper (e.g. `lib/kernel/execute.ts`) that both the Eve tool and (later) the
legacy path can call — without changing legacy behavior.

### 2. Real `check_submit_gate` — `agent/tools/check_submit_gate.ts`
Port the real logic from `lib/ai/tools/check-submit-gate.ts` (Turnstile-page DOM probe +
force-enable the stuck-disabled submit button) to operate on the SP-A Kernel session.
Returns whether it enabled the button; never clicks submit.

### 3. `action_label`, `gap_analysis`, `form_summary` (`agent/tools/*`)
`action_label` stays a lightweight real signal (records/returns the category).
`gap_analysis` and `form_summary` validate their zod input and return the structured
data; comments updated to note rendering lands in SP-B. Remove "stub" framing where the
tool's real standalone behavior is simply returning validated data.

### 4. Archive Apricot
- Delete `agent/subagents/database_verification/` in full (agent.ts, instructions.md,
  tools/get_apricot_record.ts, tools/get_apricot_form_fields.ts).
- `agent/instructions.md`: retarget the Applicant Identity line that says to confirm
  age/DOB via `getApricotFormFields` — age/DOB now comes from a caseworker message or is
  clarified with the caseworker. Update the subagent-delegation note to list only
  `requirements_research` and `form_review`.
- `agent/skills/benefits-application/SKILL.md`: drop the `database` source and
  `getApricotFormFields` references throughout (gap-analysis + form-summary specs); the
  `source` enum becomes `caseworker | inferred | missing`. Add a retargeted
  **Data Provenance (No Fabrication)** section: every filled/summarized value must trace
  to a caseworker message or an inference from one, else it is `missing`; shape is not
  identity; no fabrication.
- Note: SP-A archives Apricot within `agent/` only. Apricot code under `lib/` and its use
  by the legacy `route.ts` are removed in the SP-C cutover (removing them now breaks the
  live build).

### 5. Untouched in SP-A
`agent/subagents/requirements_research/` and `agent/subagents/form_review/`,
`agent/sandbox.ts`, `agent/agent.ts` compaction, `agent/instructions/date.ts`, the two
skills' non-Apricot content, and all of `lib/`.

## Data / control flow (SP-A, standalone)

```
POST /eve/v1/session {message: "apply for X for <person>, here is their data ..."}
  main agent (instructions + skills, Apricot-free)
    -> browser tool: navigate -> Kernel session (resolved by Eve session id) -> real DOM snapshot
    -> browser tool: fill/click/... (re-attach by id each call)
    -> check_submit_gate on a Turnstile page (real DOM probe; no submit)
    -> gap_analysis / form_summary: return structured data (no card render yet)
  reply streamed back
```

## Validation

- `npx eve dev` boots clean; `grep -ri apricot agent/` returns nothing.
- A live turn drives a real Kernel browser: navigate to a real URL, snapshot returns
  real page content, fill a field succeeds. Recorded as evidence.
- `check_submit_gate` exercised on a Turnstile page (or its real DOM-probe path
  confirmed) without clicking submit.
- The spike unit test still passes: `pnpm exec vitest run -c vitest.config.node.mjs`.
- Additive to `lib/`: the legacy build still compiles (no legacy file broken); any new
  shared Kernel helper does not change legacy behavior.

## Risks

- **BrowserManager statefulness under Eve (primary).** If `executeCommand` needs a
  persistent in-process Playwright handle, a naive port breaks under durable/replayed
  execution. Mitigation: the plan's first step is a mini-spike to confirm attach-by-id
  works per call; adapt the wrapper if not. This gates the whole sub-project.
- **Eve session id ↔ Kernel session identity.** The exact Eve context field and its
  stability are confirmed at plan time against `node_modules/eve/docs/session-context.md`.
- **Kernel availability from `eve dev`.** `KERNEL_API_KEY` is in `.env.local`; Kernel
  owns the remote browser lifecycle. Live browsing depends on Kernel being reachable.
- **Provenance rewrite is safety-relevant.** Retargeting the anti-fabrication rule must
  preserve its strictness (no invented SSNs/DOBs/etc.) while removing Apricot specifics —
  reviewed for fidelity, not just mechanically de-referenced.
