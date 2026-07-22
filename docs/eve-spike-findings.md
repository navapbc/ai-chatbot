# Eve + Vercel Spike — Findings

## Eve version
`eve 0.27.0` (from `pnpm ls eve`; resolved via `"eve": "^0.27.0"` in `package.json`).

## Q1 — Can Eve mount into this app on Vercel?
### Local
- How Eve routes were mounted into the existing Next.js 16 app:
  - Eve does **not** mount into the Next.js dev server. `npx eve@latest init .`
    did not touch the `dev` script (`package.json` still has
    `"dev": "next dev --turbo"`) and did not add an `app/eve/...` route handler.
    Instead it scaffolded a standalone `agent/` app (`agent/agent.ts`,
    `agent/instructions.md`, `agent/channels/eve.ts`) that is served by Eve's
    own CLI dev server, **`npx eve dev`**, which listens on its own port
    (default **2000**, overridable with `--port`/`$PORT`). `/eve/v1/session`
    and `/eve/v1/session/:id/stream` are routes on *that* server, not on the
    Next.js app's route table. This means integrating Eve's HTTP surface into
    the existing Next app (so the chat UI can call it same-origin) is an open
    question for later tasks (proxying from a Next route handler, or calling
    the Eve server directly cross-origin) — not something `eve init` set up
    for us.
  - `eve init` did add `eve` (`^0.27.0`) and `@vercel/connect` (`0.4.0`) to
    `dependencies` and pinned `"engines": { "node": "24.x" }` in
    `package.json`, as documented in the brief.
- Any friction / manual wiring required:
  - **`eve init` created a `pnpm-workspace.yaml`** (untracked, brand new file)
    containing `minimumReleaseAgeExclude`, `allowBuilds`, and a
    `packageExtensions` compatibility shim for `eve@>=0.6.0-beta.13 <=0.7.0`.
    This file **omitted the required `packages:` field**, which made *every*
    pnpm invocation (even `pnpm --version`) in the repo root fail with
    `ERROR packages field missing or empty` under the project's pinned pnpm
    (10.13.1 via Homebrew / corepack). Fixed by adding `packages: ["."]` to
    the top of `pnpm-workspace.yaml` — a one-line manual fix, but it is a
    real (repo-breaking) papercut in the current `eve init` for a
    single-package repo and should be flagged to whoever owns the Eve CLI,
    or fixed in a follow-up before merging this spike further.
  - After that fix, `pnpm install` succeeded but reported peer-dependency
    warnings: `eve@0.27.0` wants `ai@^7.0.26` (repo has `ai@7.0.19` pinned
    transitively elsewhere) and `nitro`'s `unstorage` wants newer
    `@vercel/blob`, `@upstash/redis`, `@vercel/functions` than the repo
    currently has. These are warnings, not install failures, and did not
    block the smoke test — left as-is per "additive only," but worth a look
    before Task 2 ports a real tool.
  - No `package-lock.json` was created; only `pnpm-lock.yaml` changed.
  - Eve's own `eve dev` server prints Node 24 in its startup banner but does
    not read `.env.local` automatically — `AI_GATEWAY_API_KEY` had to be
    exported into the process environment by hand (read out of `.env.local`)
    before starting `npx eve dev`, or the live model turn would not
    authenticate.
  - Port 2000 (Eve's default) was already occupied locally by an unrelated
    `eve dev` process running out of a sibling checkout
    (`form-filling-eve/`), so the smoke test below used `--port 2001`
    instead; this has no bearing on Vercel deploy behavior, just a local
    collision to be aware of when running multiple Eve checkouts side by
    side.
- Result of `POST /eve/v1/session` locally:
  - Started the server: `npx eve dev --no-ui --port 2001` (with
    `AI_GATEWAY_API_KEY` exported from `.env.local`). Log output:
    ```
    ☰eve  v0.27.0
    [DEV] server listening at http://127.0.0.1:2001/
    ```
  - Request:
    ```
    curl -i -sS -X POST "http://127.0.0.1:2001/eve/v1/session" \
      -H 'content-type: application/json' \
      -d '{"message":"hello"}'
    ```
  - Response — `HTTP/1.1 202 Accepted` (not `200`; the brief's "expected 200"
    should be corrected to 202 for this Eve version), with the expected
    `x-eve-session-id` header and a JSON body containing `continuationToken`:
    ```
    HTTP/1.1 202 Accepted
    cache-control: no-store
    content-type: application/json
    x-eve-session-id: wrun_01KY5KZNP7AFJ87T4TQRBB513B

    {"continuationToken":"eve:38764027-b7e5-4f87-9ae3-1afc5c308ee4","ok":true,"sessionId":"wrun_01KY5KZNP7AFJ87T4TQRBB513B"}
    ```
  - Followed up with `GET /eve/v1/session/:id/stream` to confirm the live
    model turn actually runs end-to-end through the AI Gateway (not just that
    a session object was created). It streamed `session.started` →
    `turn.started` → `message.received` → `step.started` →
    `message.appended` (x2) → `message.completed` → `step.completed` →
    `turn.completed` → `session.waiting`, with the model
    (`anthropic/claude-sonnet-4.6`) replying "Hello! How can I help you
    today?" and real token/cost usage in `step.completed`
    (`inputTokens: 3558, outputTokens: 12, costUsd: 0.000189`), confirming
    `AI_GATEWAY_API_KEY` auth worked end-to-end.
  - **Task 2 update — a real authored tool executes end-to-end.** Ported
    `readReferenceFile` (pure core, unit-tested under a new
    `vitest.config.node.mjs`) into an Eve tool at
    `agent/tools/read_reference.ts`. `pnpm exec vitest run -c
    vitest.config.node.mjs tests/agent/read-reference.test.ts` went RED
    (module not found) then GREEN (4/4) per TDD.
  - **New friction found while wiring the tool: `defineTool`'s `zod`
    inputSchema path is incompatible with this repo's pinned zod.** Using
    a `z.object({...})` inputSchema (as eve.dev's docs and the task brief
    both show) crashed `npx eve dev` at startup — *before* any turn ran —
    with `Cannot read properties of undefined (reading 'input')`,
    reproduced by moving `agent/tools/` out and back in (clean boot
    without the file, crash with it). Root cause: eve's runtime tool
    registration (`serializeInputSchema` /
    `eve/dist/src/shared/tool-schema.js`) treats *any* schema object
    carrying a `~standard` key as its own extended
    "StandardJSONSchemaV1" — requiring `~standard.jsonSchema.input()` and
    `.output()` functions in addition to the plain Standard Schema v1
    `validate()`. That extension is only present in the zod v4 line eve
    bundles internally (`zod@4.4.3` under eve's own `#compiled/zod`);
    it is absent from both `zod` and `zod/v4` as resolved from this
    repo's pinned `"zod": "^3.25.76"` (confirmed directly: the repo's
    zod package has no `v4/core/json-schema-processors.js`, eve's bundled
    one does). The `^3.25.76` range cannot reach a version with this
    support — reaching it needs a major zod bump, which is out of scope
    here for the same reason the brief flags not bumping `ai`: it's a
    shared dependency used across the whole app.
    **Fix applied (wrapper-only, per the brief's own escape hatch):**
    dropped the `zod` import and passed `inputSchema` as a plain JSON
    Schema object instead (one of `defineTool`'s documented overloads).
    Eve rehydrates a JSON-Schema `inputSchema` into its own compatible
    zod instance for runtime validation, so `path` is still validated as
    a required string — `readReferenceFile` itself is unchanged. **This
    means: any future Eve tool authored in this repo that wants a zod
    `inputSchema` will hit the same crash until either eve bundles a zod
    version compatible with `^3.25.76`, or the repo's `zod` dependency is
    bumped to a v4 line with the `~standard.jsonSchema` extension — plan
    on raw JSON-Schema `inputSchema`s for Eve tools until that's
    resolved.**
  - Proved the tool runs inside a live turn: started `npx eve dev
    --no-ui --port 2001` from the repo root (Node 24, `AI_GATEWAY_API_KEY`
    exported from `.env.local`), `POST /eve/v1/session` with `{"message":
    "Read the reference field-patterns.md and tell me what it covers."}`
    returned `202` + `x-eve-session-id`, and the NDJSON stream from
    `GET /eve/v1/session/:id/stream` showed, in order: `actions.requested`
    with `{"toolName":"read_reference","input":{"path":"field-patterns.md"}}`,
    `action.result` with `output.content` containing the real contents of
    `lib/ai/prompts/references/field-patterns.md` (the `# Field Type
    Patterns` doc, verbatim), and a final `message.completed` whose text
    summarizes that content ("**field-patterns.md** covers the correct
    JSON action shapes for interacting with common form field types —
    including text fields, date fields, SSN, phone, state, native
    dropdowns, checkboxes, and radio buttons..."). Confirms a real
    authored tool executes end-to-end through the Eve runtime, not just
    that the base scaffold boots.
### Vercel preview
- (filled in Task 5)

## Q2 — Context management under Eve
- (filled in Task 4)

## Q3 — Eve → UI streaming shape

### Capture method
Added `scripts/eve-stream-capture.sh` (`POST /eve/v1/session` with a message
that triggers `read_reference`, then `GET /eve/v1/session/:id/stream`, piped
through `tee`). Ran it against a local `npx eve dev --no-ui --port 2000`
(Node 24, `AI_GATEWAY_API_KEY` exported from `.env.local`) with the message
`"Read the reference field-patterns.md and tell me what it covers."` — the
same tool-triggering prompt used in Task 2. The raw capture
(`eve-stream-capture.ndjson`, ~17 lines / one JSON object per line) is a
scratch artifact and was **not** committed; the distinct event types below
were extracted from it. Note: the SSE connection does not close after the
turn finishes — it stays open at `session.waiting`, idle, waiting for the
next user message on the same session. The capture script (and this
analysis) simply stopped reading after that event.

### Eve NDJSON event types observed (distinct types, first-occurrence order)

> **Sequencing caveat for the adapter author (sub-project 5):** this is a
> catalog of *distinct* event types in the order each first appeared — NOT a
> literal turn timeline. A tool-then-answer turn is two model steps (step 1 =
> the tool call, step 2 = the final text), so `step.started`,
> `step.completed`, and `message.appended` **recur per step**. In particular a
> `step.completed` arrives *mid-turn* after the tool step (carrying that
> step's usage), before the answer text streams in the second step — do not
> assume a single `step.completed` trailing the whole turn.

1. `session.started` — session created; `data.runtime` has `agentId`,
   `agentName`, `eveVersion`, `modelId`.
2. `turn.started` — a new turn begins (`sequence`, `turnId`).
3. `message.received` — echoes the inbound user message (`message` text +
   `parts`) that kicked off the turn.
4. `step.started` — a new step within the turn begins (`stepIndex`,
   `turnId`) — Eve's equivalent of an agent-loop step boundary.
5. `actions.requested` — the model requested one or more tool calls:
   `data.actions: [{ kind: "tool-call", toolName, input, callId }]`.
6. `action.result` — a tool finished executing:
   `data.result: { kind: "tool-result", callId, toolName, output }`,
   `data.status: "completed"`.
7. `step.completed` — step finished; carries `finishReason` and real
   `usage` (`inputTokens`, `outputTokens`, `cacheReadTokens`,
   `cacheWriteTokens`, `costUsd`) plus `providerMetadata.gateway.generationId`.
8. `message.appended` — one streaming text delta per chunk:
   `data.messageDelta` (this chunk) and `data.messageSoFar` (cumulative).
9. `message.completed` — final assistant message text for the turn plus
   `finishReason`.
10. `turn.completed` — the turn is done (`sequence`, `turnId`).
11. `session.waiting` — session now idle; `data.continuationToken` and
    `data.wait: "next-user-message"`.

No distinct "error" or "abort" event was observed — this capture only
exercised the happy path (one user turn, one tool call, one final answer).
Eve's error/cancellation event shapes are **not verified** by this task.

### What the current UI consumes
From `components/chat.tsx`:
- **Assistant text / tool calls / tool results** — not read directly in
  `chat.tsx`; the `useChat` hook (`chat.tsx:104-135`) manages the
  `ChatMessage[]` array from the AI SDK's UIMessage stream, and
  `components/message.tsx` renders it: `type === 'text'` for prose
  (`message.tsx:251`), named tool parts (`tool-getWeather` at
  `message.tsx:352`, `tool-gapAnalysis` at `message.tsx:495`, etc.), and a
  generic fallback for any other `tool-*` part (`message.tsx:532`) — each
  tool part carries AI SDK v7's `input-streaming` /
  `input-available` / `output-available` states.
- **`data-compacting`** — a transient custom event. `chat.tsx:140-142`
  (`onData`) sets `isCompacting = true` on
  `part.type === 'data-compacting'`. Written by
  `app/(chat)/api/chat/route.ts:230-235` inside `prepareStep`, only when the
  compressor's `onCompacting` callback fires:
  `{ type: 'data-compacting', data: { timestamp: Date.now() }, transient: true }`.
- **`data-checkpoint`** — a transient custom event. `chat.tsx:143-159`
  turns `isCompacting` back off and appends a `CheckpointData` entry
  (`messageId`, `stepNumber`, `summary`) for the compaction-boundary card.
  Written by `route.ts:238-248` after a real compaction happened:
  `{ type: 'data-checkpoint', data: { stepNumber, inputTokens, timestamp, summary }, transient: true }`.
- **`data-token-usage`** — a transient custom event. `chat.tsx:161-176`
  accumulates `inputTokens`/`outputTokens`/`cachedInputTokens` into the
  `tokenUsage` state exposed via `TokenUsageProvider`. Written by
  `route.ts:253-265` in `onStepEnd`:
  `{ type: 'data-token-usage', data: { inputTokens, outputTokens, cachedInputTokens }, transient: true }`
  (`cachedInputTokens` is remapped from the AI SDK's
  `usage.inputTokenDetails?.cacheReadTokens`).
- All of the above ride on `dataStream.merge(result.toUIMessageStream())`
  at `route.ts:272` — the UI only ever sees AI SDK v7 SSE frames
  (`JsonToSseTransformStream`, `route.ts:4,289,293`), never a raw model or
  tool-runtime event.

### Mapping table

| UI needs (current) | Source today | Eve NDJSON equivalent | Gap? |
|---|---|---|---|
| assistant text parts | AI SDK `UIMessage` (`result.toUIMessageStream()`, `route.ts:272`) | `message.appended` (`messageDelta`/`messageSoFar`) → `message.completed` | No — deltas map 1:1 to AI SDK text-delta parts; `messageSoFar` is redundant with client-side accumulation but harmless. |
| tool call | AI SDK `UIMessage` tool part (`input-available`) | `actions.requested` → `data.actions[].{kind:"tool-call", toolName, input, callId}` | No — `callId` maps directly to the AI SDK `toolCallId`. |
| tool result | AI SDK `UIMessage` tool part (`output-available`) | `action.result` → `data.result.{kind:"tool-result", callId, output}` | No — same `callId` correlates call and result. |
| `data-token-usage` (`route.ts:253-265`) | `route.ts` `onStepEnd` | `step.completed.data.usage.{inputTokens, outputTokens, cacheReadTokens, cacheWriteTokens, costUsd}` | Minor — all fields present, just flatter than the AI SDK's `usage.inputTokenDetails.cacheReadTokens` nesting; a straight rename, not a data gap. |
| `data-compacting` (`route.ts:230-235`) | `route.ts` `prepareStep` (compressor `onCompacting` callback) | none observed | **Yes, and open.** This event exists only because `route.ts` owns the `streamText` loop and injects it from its own `prepareStep` hook. Whether Eve exposes an equivalent per-step hook for a caller-supplied compressor is unknown from this capture — it depends on whether Eve or the app owns context management (Q4), not just on translating an event shape. |
| `data-checkpoint` (`route.ts:238-248`) | `route.ts` `prepareStep` (after a real compaction) | none observed | **Yes, and open** — same dependency as `data-compacting`: it needs a `prepareStep`-equivalent injection point in Eve, which this capture did not exercise. |
| *(reverse: no current UI consumer)* | — | `session.started`, `turn.started`/`turn.completed`, `step.started`, `session.waiting` | Eve emits agent-loop lifecycle events the current UI has no use for; an adapter would drop them on the floor. |

### Recommendation: adapter route

**Build a Next.js adapter route that reads Eve's NDJSON and re-emits the AI
SDK SSE shape `components/chat.tsx` already understands, rather than
teaching `chat.tsx` to consume Eve's stream directly.** The evidence above
supports this: Eve's text (`message.appended`/`message.completed`) and tool
(`actions.requested`/`action.result`) events map mechanically, one-to-one by
`callId`, onto the AI SDK `UIMessage` parts `message.tsx` already renders,
and `step.completed.usage` carries everything `data-token-usage` needs
(just nested differently) — none of that requires touching `chat.tsx` or
`message.tsx`. The current UI is thickly coupled to the AI SDK's `useChat`
transport contract: automatic tool-continuation
(`sendAutomaticallyWhen`/`lastAssistantMessageIsCompleteWithToolCalls`,
`chat.tsx:117-119`), resumable streaming (`resumable-stream` +
`streamContext.resumableStream`, `route.ts:284-292`), and the
`onData`/`transient` event contract for the three custom cards — all of
that machinery would have to be reimplemented against Eve's stream under a
UI-rework path, for no benefit, since Eve's own events already translate
cleanly. The one place this isn't a clean win is `data-compacting` /
`data-checkpoint`: those aren't a stream-translation problem so much as an
open dependency on Q4 — whether Eve exposes a `prepareStep`-equivalent hook
for a caller-owned compressor, or whether Eve's own context management
replaces `lib/ai/context-compression.ts` outright. That question doesn't
change the adapter-vs-rework verdict (a rework path would face the exact
same unresolved dependency, plus the cost of rewiring everything else), but
it does mean the adapter can't fully close that row of the table until Q4
lands. Two more adapter-design details fall out of the capture: it must
terminate the AI-SDK-shaped stream at `turn.completed`/`session.waiting`
rather than passing Eve's still-open connection straight through, and
error/abort event shapes remain unverified since this capture only
exercised the happy path — that should be captured before committing an
adapter to production in sub-project 5. This is the lower-risk path for
sub-project 5 because it isolates Eve behind the exact contract the UI
already speaks, is additive (no `chat.tsx`/`route.ts` changes), and fails
closed — if the adapter breaks, only the Eve-backed path degrades rather
than the shared chat UI.

## Browser session sketch
- (filled in Task 4)
