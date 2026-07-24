# Sub-Project 1 — Land on Vercel + Eve Spike (Design)

Status: approved for implementation planning
Date: 2026-07-22
Branch: `feat/eve-integration`

## Context

The `ai-chatbot` app currently runs off-Vercel: a custom `streamText` agent loop
(`app/(chat)/api/chat/route.ts`), models via Google Vertex AI (`lib/ai/providers.ts`),
browser automation on Kernel.sh, resumable streaming on Redis, and (per the abort-registry
comment in `route.ts`) deployment on Cloud Run.

The larger goal is to adopt [Eve](https://vercel.com/docs/eve) — Vercel's filesystem-first
durable-agent framework — as the agent **harness**, running the whole app natively on Vercel.
Eve's durability, sandbox, and model access are all Vercel-native (Vercel Workflows, Vercel
Sandbox, AI Gateway, Vercel Connect), so full adoption implies a platform migration.

That migration is decomposed into sequenced sub-projects (each its own spec → plan → build):

1. **Land on Vercel + Eve spike** ← this document
2. Tool + prompt/skills migration
3. Browser tool re-architecture (stateless-per-turn against Kernel.sh)
4. Context / working-memory strategy
5. UI ↔ Eve wiring + Postgres history bridge
6. Cutover + cleanup (remove custom `route.ts`, Redis, abort registry; re-point evals)

Two behaviors gate the whole migration and are currently **unknowns** against Eve's beta:
the bespoke context-compaction + working-memory system (`lib/ai/context-compression.ts`,
which rewrites the message array mid-run via `streamText`'s `prepareStep`), and the streaming
contract (the chat UI consumes AI SDK `UIMessage` SSE; Eve emits NDJSON). This spike exists to
turn those unknowns into documented facts before any production code is migrated.

## Purpose

A **de-risking spike, not a migration.** It answers three questions with evidence and running
code. It touches nothing in production: Cloud Run, Vertex, and the existing `/api/chat` route
stay live and unchanged. The output feeds sub-projects 2–6.

## Goals (exit criteria)

The spike is complete when all three questions are answered, each with a documented finding:

1. **Can Eve mount into this app on Vercel at all?**
   `agent/` compiles, `/eve/v1/session` serves a real turn end-to-end through AI Gateway, both
   locally (`pnpm dev` + curl) and on a Vercel preview deploy. Includes confirming whether Eve
   mounts cleanly into an existing **Next.js 16** App Router app or wants its own structure.

2. **How does context management work under Eve?**
   Determine whether Eve does internal context management and/or exposes any turn/step hook
   equivalent to `prepareStep`. Prototype the likely replacement for `context-compression.ts`:
   **working memory as a sandbox file** plus a **summarization subagent**. Deliver a written
   finding on how the 75%-context compaction + structured working-memory extraction re-architects
   under Eve (or a clear statement that it cannot, and why).

3. **What is the Eve → UI streaming shape?**
   Capture Eve's NDJSON lifecycle event types from `/eve/v1/session/:id/stream` and produce a
   mapping table to what the chat UI consumes today: `UIMessage` parts plus the transient data
   events `data-token-usage`, `data-compacting`, `data-checkpoint`. Deliver a decision: build an
   adapter route (Eve NDJSON → AI SDK SSE) vs. rework the UI to consume Eve's stream directly.

## Non-goals (explicitly deferred)

- Porting the full tool set (`apricotTools`, `gapAnalysis`, `formSummary`, `actionLabel`,
  `checkSubmitGate`, `browser`) — sub-project 2.
- Re-architecting the browser tool / Kernel.sh session handling — sub-project 3.
- Building the real context/working-memory system — sub-project 4 (this spike only prototypes
  enough to answer Q2).
- UI wiring and the Postgres history bridge — sub-project 5.
- Deleting the custom `route.ts`, Redis `resumable-stream`, or the abort registry;
  re-pointing evals — sub-project 6.
- Migrating the main app's models from Vertex → AI Gateway (only the Eve agent uses AI Gateway
  in this spike; the existing route keeps using Vertex).

## Approach

### Decisions (confirmed with stakeholder)

- **Spike environment:** local first (prove Eve mechanics with `pnpm dev` + curl), then a Vercel
  preview deploy to confirm on-platform. Avoids requiring Vercel provisioning before any progress.
- **Proof tool:** `readReference` — self-contained (reads local markdown under
  `lib/ai/prompts/references`, no creds, no browser, no session state). Isolates "does Eve run a
  tool at all" from external-integration noise.

### Components

**`agent/` directory (new, additive):**

- `agent/instructions.md` — a minimal always-on system prompt for the spike (not the full
  `getWebAutomationSystemPrompt()`; just enough to exercise the proof tool).
- `agent/agent.ts` — `defineAgent({ model })` with an AI Gateway model string
  (`anthropic/claude-opus-4.8` or `anthropic/claude-sonnet-4.6`; `opus-4-7` is not current on the
  gateway, so the spike uses `4.8`).
- `agent/tools/read_reference.ts` — `defineTool` port of `readReference`. The AI SDK `tool()` →
  Eve `defineTool` shape is near-1:1: `description`, zod `inputSchema`, `execute`. The runtime
  tool name comes from the filename, so the file is named `read_reference.ts`. The path-traversal
  guard and `references/` prefix handling from the original are preserved. Reference markdown is
  read from the repo's existing `lib/ai/prompts/references` directory.

**Eve routes:** mounted by the framework (`/eve/v1/session`, `/eve/v1/session/:id/stream`). No
custom route code in this spike beyond what Eve generates/mounts.

**Verification scripts:** curl scripts under a scratch/`docs` location that `POST /eve/v1/session`
with a message that forces a `read_reference` call, then attach to the stream and print NDJSON —
run against both local dev and the Vercel preview.

**Findings document:** a new doc (e.g. `docs/eve-spike-findings.md`) capturing:
- Q1: how Eve mounted into Next.js 16 (steps, gotchas, whether same project or separate).
- Q2: Eve's context-management capabilities + the working-memory-as-sandbox-file / subagent
  prototype result and recommended re-architecture.
- Q3: the NDJSON event → UI stream mapping table + adapter-vs-rework decision.
- A viability sketch for the browser-session-across-turns problem (Kernel.sh session re-resolved
  by ID each turn under Eve's replayed durable execution), informing sub-project 3.

### Data / control flow (spike)

```
curl POST /eve/v1/session {message}
  -> Eve creates a durable session (Vercel Workflows), returns x-eve-session-id + continuationToken
  -> agent turn runs: model (via AI Gateway) decides to call read_reference
  -> read_reference.execute reads lib/ai/prompts/references/<file>.md, returns { content }
  -> model produces final text
curl GET /eve/v1/session/<id>/stream
  -> NDJSON lifecycle events observed and recorded for the Q3 mapping table
```

### Error handling

Spike-level only: the proof tool keeps the original's defensive returns (`{ error: ... }` on
path-traversal or missing file). Framework/runtime errors are captured verbatim in the findings
doc rather than handled — surfacing them *is* part of the spike's value.

## Prerequisites / open items (for the implementation plan, not blockers for this spec)

- A Vercel team/project to deploy the preview into.
- AI Gateway auth locally (API key via `AI_GATEWAY_API_KEY`) vs. Vercel OIDC on the preview.
- Postgres for the Vercel preview (point at a DB branch, not production) and the auth secret /
  other env the existing app needs to boot.
- Confirm the Eve beta version pinned for the spike and note it (Eve is beta, "subject to change").

## Validation

- **Q1:** curl round-trip succeeds locally and on the Vercel preview; `read_reference` executes
  and its output appears in the streamed result.
- **Q2/Q3:** findings doc committed with the compaction re-architecture recommendation and the
  streaming mapping table + decision.
- No production changes: Cloud Run deployment, Vertex models, and `/api/chat` remain untouched;
  all spike code is additive (`agent/`, scripts, findings doc).

Formal `vitest`/Playwright coverage is intentionally deferred — the spike is validated by the
curl round-trip and the written findings. Test coverage lands with the real migration
(sub-projects 2–6).

## Risks

- **Eve beta churn** — APIs/behavior may change before GA. Mitigation: pin and record the version;
  keep the spike small.
- **Eve may not expose a compaction hook** — if Q2 concludes the bespoke `prepareStep` compaction
  cannot be faithfully expressed under Eve, that reshapes sub-project 4 (and possibly the whole
  migration's value calculus). Surfacing this early is the point of the spike.
- **Next.js 16 + Eve mounting friction** — Eve's "add to an existing app" path may not be smooth
  on Next 16 App Router. Captured as part of Q1.
