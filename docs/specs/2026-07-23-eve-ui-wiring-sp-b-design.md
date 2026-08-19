# SP-B — Wire the Eve Runtime into the App UI (Design)

Status: approved for implementation planning
Date: 2026-07-23
Branch: `feat/eve-integration`

## Context

The project is being revamped so the Eve agent (`agent/`) is the real, live agent.
SP-A (`docs/specs/2026-07-23-eve-agent-real-tools-sp-a-design.md`) made the Eve agent
functionally real standalone: real Kernel.sh browser tool, real `check_submit_gate`,
Apricot archived, data model = caseworker + inference. But Eve is only reachable via
`npx eve dev` (its own server, NDJSON, port ~2000) — it is not wired to the app's chat UI.

SP-B connects them. The spike's Q3 finding (`docs/eve-spike-findings.md`) recommends an
**adapter route**: a Next route that translates Eve's NDJSON into the AI SDK SSE the chat
client already consumes, rather than reworking `components/chat.tsx`. Two SP-A facts
constrain the design: Eve runs as a **separate server** (not mounted in Next), and Eve's
bundler **cannot import `server-only`** (so DB/auth stay on the Next side).

Sub-project sequence: SP-A (done) → **SP-B (this doc)** → SP-C (context/`defineState` +
Postgres history + cutover) → SP-D (Vercel deploy).

## Purpose

A turn typed in the existing chat UI runs on the Eve agent and streams back — text, tool
activity, and the interactive `gap_analysis`/`form_summary` cards — behind a feature flag,
with the legacy `/api/chat` path untouched.

## Goals (exit criteria)

1. With the `useEveAgent` flag ON, a message sent from `components/chat.tsx` is handled by
   a new adapter route that runs the turn on the Eve server and streams the result back
   into the UI.
2. Text streams, tool calls/results render, and `gap_analysis` / `form_summary` show as
   the **real interactive cards** (`message.tsx`'s named renderers), including the
   gap-card → caseworker-fills → follow-up-turn round-trip.
3. Multi-turn continuity works within a running server: follow-up messages continue the
   same Eve session via its `continuationToken`.
4. With the flag OFF (default), the app behaves exactly as today (legacy `/api/chat`).

## Non-goals (deferred)

- Postgres history/message persistence and the Redis resumable stream — SP-C. Eve chats
  will not survive a server restart or appear in saved history yet.
- Removing/gutting the legacy `route.ts` / `web-automation.ts` — SP-C cutover.
- Vercel deploy and the production Eve-server topology — SP-D.
- `data-compacting` / `data-checkpoint` compaction indicators — they come from Eve's
  observe-only *hooks*, not the session stream, so they are out of scope here.
- Full Eve-side **abort/stop**: the Stop button aborts the client fetch; server-side Eve
  turn cancellation is best-effort/deferred.
- Persisting the session-continuity mapping (it is in-memory only in SP-B).

## Decisions (confirmed with stakeholder)

- **Architecture:** additive adapter route that HTTP-proxies to the Eve server and
  translates its stream; flag-switched; legacy `/api/chat` stays intact.
- **Persistence:** in-memory per-chat session continuity only; defer Postgres history +
  Redis to SP-C.
- **Cards:** SP-B renders the interactive cards (tool-name mapping + round-trip).

## Components

### 1. Adapter route — `app/(chat)/api/eve-chat/route.ts` (new)
POST handler:
- `auth()` (reuse next-auth); read the same request body `chat.tsx` sends today.
- Extract the **latest user message** text (Eve maintains its own conversation history
  per session; only the new message is sent to Eve).
- Resolve per-chat continuity (component 4). New chat → `POST ${EVE_SERVER_URL}/eve/v1/session`
  with `{ message }`; continuing chat → `POST ${EVE_SERVER_URL}/eve/v1/session/:id` with
  `{ continuationToken, message }`. Capture `x-eve-session-id`.
- Attach to `GET ${EVE_SERVER_URL}/eve/v1/session/:id/stream` (NDJSON).
- Pipe through the translator (component 2) into a `createUIMessageStream` writer, then
  `JsonToSseTransformStream` → `Response` (same output shape as legacy).
- Terminate the SSE at `turn.completed` / `session.waiting`; persist the new
  `continuationToken` into the continuity store.
- On a non-OK Eve response or a stream/`error` event, surface a `ChatSDKError`-style
  error part; on Eve unreachable (dev server not running), a clear message.

### 2. Stream translator — `lib/ai/eve/stream-adapter.ts` (new, Next-side, pure/testable)
Maps Eve NDJSON events to AI SDK UIMessage parts / data events:

| Eve event | AI SDK output |
|---|---|
| `message.appended` (`messageDelta`) | text-delta part |
| `message.completed` | finalize text |
| `actions.requested` (`kind: tool-call`) | tool part `input-available` (name-mapped) |
| `action.result` (`kind: tool-result`) | tool part `output-available` (same `callId`) |
| `step.completed.usage` | `data-token-usage` (`{ inputTokens, outputTokens, cachedInputTokens }`) |
| `session.started`/`turn.*`/`step.started`/`session.waiting` | consumed for control, not emitted |

Includes a **tool-name map** so `message.tsx`'s named card renderers fire:
`gap_analysis → gapAnalysis`, `form_summary → formSummary` (extend as needed). Unmapped
tools (e.g. `browser`, `action_label`, `check_submit_gate`) pass through under their own
name to `message.tsx`'s generic tool renderer. The exact Eve **error/abort** event shape
is captured during implementation (Q3 left it unverified) so failures map to an error part.

### 3. Client flag switch — `components/chat.tsx` + `lib/feature-flags.ts`
Add a `useEveAgent` flag (env-aware default OFF; overridable via the existing dev flag
menu / `ff:` localStorage). In `chat.tsx`, the `DefaultChatTransport` `api` becomes
`useEveAgent ? '/api/eve-chat' : '/api/chat'`. This is the ONLY client change — `onData`,
tool-part rendering, and the card components are untouched, because the adapter emits the
shape the UI already consumes. The gap-card → fill → follow-up round-trip works with no
card-side change (the follow-up rides the same transport and continues the Eve session).

### 4. Session continuity — `lib/ai/eve/session-continuity.ts` (new)
An in-memory `Map` keyed by `${userId}:${chatId}` → `{ eveSessionId, continuationToken }`.
Read before proxying, written after `session.waiting`. Explicitly single-process and lost
on restart; SP-C replaces it with a Postgres-backed mapping.

### 5. Eve reachability — `EVE_SERVER_URL`
New env var (default `http://127.0.0.1:2000`), added to `.env.example`. Dev workflow runs
`npx eve dev` and `pnpm dev` side by side; the adapter proxies Next → Eve. Model/Kernel
credentials live on the Eve-server side (the `eve dev` process), not the Next side — the
adapter only proxies HTTP.

## Data flow

```
chat.tsx (useEveAgent ON) → POST /api/eve-chat { messages, chatId, ... }
  auth() → continuity[userId:chatId]?
    new  → POST  EVE/eve/v1/session            { message: <latest user msg> }
    cont → POST  EVE/eve/v1/session/:id        { continuationToken, message }
  → GET EVE/eve/v1/session/:id/stream  (NDJSON)
  → translate events → UIMessage SSE (text, tool parts w/ name map, token usage)
  → stop at turn.completed/session.waiting; save continuationToken
  → SSE Response → chat.tsx renders (gap_analysis/form_summary as interactive cards)
gap card filled → follow-up message → same flow → continues the Eve session
```

## Error handling

- Eve server unreachable / non-2xx on session create → error part + clear message (likely
  "the agent server isn't running").
- An `error`-type Eve event mid-stream → error part (shape confirmed at implementation).
- Client Stop → fetch abort stops the SSE read (existing behavior); server-side Eve
  cancellation is not wired in SP-B (documented limitation).
- Unknown Eve event types → ignored defensively (logged), never crash the translator.

## Validation

- Unit: the translator maps a captured sample of real Eve NDJSON (one text turn; one
  tool-call turn incl. `gap_analysis`) to the expected UIMessage SSE frames, including the
  snake_case→camelCase tool-name mapping. Node-config vitest (`vitest.config.node.mjs`).
- Manual e2e: flag ON, run a form-filling task; confirm text streams, the browser tool
  runs (real Kernel), a `gap_analysis` card renders, filling it sends a follow-up that
  continues the same Eve session, and `form_summary` renders. Flag OFF → legacy unchanged.
- Additive: legacy `/api/chat` + `route.ts` unmodified; flag OFF path identical to today.

## Risks

- **Eve error/abort event shape unknown** (Q3). Mitigation: capture it early (a small
  step feeds a failing turn and records the event) so the translator handles it.
- **Tool-name / tool-part-state fidelity.** `message.tsx`'s card renderers expect specific
  tool names and AI SDK v7 part states (`input-available`/`output-available`); a mismatch
  renders a card empty. Mitigation: the unit test asserts the mapped parts against what
  `message.tsx` consumes.
- **In-memory continuity is fragile** (restart/multi-instance loses it). Accepted for
  SP-B; SP-C replaces it. A server restart mid-conversation starts a fresh Eve session.
- **Two dev servers.** Developers must run `npx eve dev` alongside `pnpm dev`; if
  `EVE_SERVER_URL` is unreachable the adapter must fail with a clear message, not a hang.
- **Latest-message extraction.** Sending only the newest user message to Eve assumes Eve
  holds the rest; the adapter must correctly identify the latest user message from the AI
  SDK payload (not resend the whole history, which would duplicate context in Eve).
