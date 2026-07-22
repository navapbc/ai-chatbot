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
- **Deferred.** This subsection was intentionally not exercised in this spike.
  Task 5 (the Vercel preview deploy) was descoped: no Vercel project has been
  provisioned for this repo yet, and there is no non-prod Postgres branch to
  point a preview build's `DATABASE_URL` at (the build runs Drizzle migrations
  — see `CLAUDE.md` — so a preview deploy needs a writable non-prod database,
  not just a Vercel project). Provisioning both is out of scope for a
  local-first, additive-only spike. Everything in the "Local" section above
  (server topology, `eve init` friction, the `pnpm-workspace.yaml` fix, live
  `POST`/`GET` session round-trips, and the `read_reference` tool proof) was
  verified end-to-end locally; only the Vercel-hosted variant of that same
  round-trip remains unverified. **Before sub-project 5 or 6 relies on a
  Vercel-hosted Eve, this needs its own short follow-up:** provision a Vercel
  project + non-prod Postgres branch, deploy, and repeat the local smoke test
  (`POST /eve/v1/session` → `GET .../stream`) against the deployed URL to
  confirm Eve's own server process boots correctly under Vercel's runtime
  (function timeouts, cold starts, and whether Eve's own server model is even
  compatible with Vercel's serverless functions are all unverified) and that
  the same-origin/cross-origin question flagged above still holds.

## Q2 — Context management under Eve

### Headline answer

**Eve manages context internally, and it is not a thin wrapper — it has its
own compaction engine that is structurally similar to the bespoke
`lib/ai/context-compression.ts`.** This is the single most important finding
of this task: it changes the shape of sub-project 4 from "port our
`prepareStep` compressor onto some Eve hook" to "delete our compressor,
configure Eve's, and rebuild only the two pieces Eve does not do — structured
working-memory extraction and pinning it outside compaction — using Eve's own
native primitives (`defineState` + dynamic instructions), not a custom hook."

**One-line disambiguation, since this question has two tiers that are easy to
conflate: there is no authorable Eve hook for context management** (compaction
is internal, `compaction.requested`/`.completed` are observe-only, per the
enumeration below) — **the only theoretical `prepareStep`-equivalent is a
generic, non-Eve AI SDK seam** (`wrapLanguageModel`/`transformParams`,
detailed below), and that seam was not built or tested live in this spike and
is not recommended even if it works. Read "no Eve hook" and "an AI-SDK seam
exists" as two different, non-conflicting answers, not a contradiction.

Cited source, not the docs site (this is internal harness code, confirmed by
reading the installed package, not eve.dev's public pages):
- `node_modules/eve/dist/src/harness/compaction.js` — `shouldCompact`,
  `compactMessages`, `resolveCompactionModel`. `compactMessages` calls the AI
  SDK's `generateText` directly with a fixed system prompt
  (`COMPACTION_SYSTEM_PROMPT` in `node_modules/eve/dist/src/harness/
  compaction-prompt.js`) to produce a handoff-style summary, escalating
  through heuristics (cap oversized tool results → summarize older region →
  degrade recent tail to text-only → shrink the window) until the result
  fits the token budget.
- `node_modules/eve/dist/src/harness/compaction.d.ts` and `node_modules/eve/
  dist/src/harness/types.d.ts` — the typed `CompactionConfig` shape
  (`threshold`, `recentWindowSize`, `lastKnownInputTokens`,
  `lastKnownPromptMessageCount`).
- `node_modules/eve/dist/src/shared/agent-definition.d.ts` (lines ~88–120,
  `PublicAgentCompactionDefinition`) — the **author-facing** config surface:
  `defineAgent({ compaction: { model, thresholdPercent,
  modelContextWindowTokens } })`. `thresholdPercent` "defaults to `0.9`."
- `node_modules/eve/dist/src/execution/session.js` — `createCompactionConfig`
  shows the hard-coded default: `recentWindowSize: 10`, `threshold:
  contextWindowTokens === undefined ? 100_000 : Math.floor(contextWindowTokens
  * thresholdPercent)`.
- `node_modules/eve/dist/src/public/definitions/hook.d.ts` — the
  `HookEventMap` includes `"compaction.requested"` and `"compaction.completed"`
  as accepted authored-hook events (`agent/hooks/*.ts`, `defineHook` from
  `eve/hooks`), and `node_modules/eve/dist/src/protocol/message.d.ts` (lines
  ~415–439) shows their payload shapes (`sessionId`, `turnId`, `modelId`,
  `usageInputTokens` on `.requested`; the same minus usage on `.completed`).
- `node_modules/eve/dist/src/public/definitions/state.d.ts` and
  `node_modules/eve/docs/guides/state.md` — `defineState(name, initial)` from
  `eve/context`, a durable per-session key/value slot ("State is durable by
  default and does not reset between turns").
- `node_modules/eve/dist/src/execution/durable-session-store.d.ts` —
  `DurableSession { history: ModelMessage[]; state?: SessionStateMap; ... }`:
  `state` (the `defineState` backing store) is a **separate top-level field**
  from `history`, and `compactMessages` (see above) only ever operates on
  `history`. So `defineState` values are structurally exempt from compaction,
  not just empirically observed to survive it.

### (a)–(e) enumeration

| # | Bespoke (`lib/ai/context-compression.ts`) | Eve native? | Verdict |
|---|---|---|---|
| (a) | Trigger at 75% of a 200K window (`COMPACT_THRESHOLD_PCT = 0.75`, `MODEL_CONTEXT_WINDOW = 200_000`, both hard-coded constants) | `defineAgent({ compaction: { thresholdPercent: 0.75 } })`; `contextWindowTokens` is read automatically from AI Gateway model metadata, not hand-maintained | **Native — becomes unnecessary to hand-roll.** Set `thresholdPercent: 0.75` on `agent/agent.ts` to match; no other code needed. |
| (b) | Haiku-summarize old messages into a session-handoff doc via `generateText` with a hand-written `COMPACTION_SYSTEM_PROMPT` | `defineAgent({ compaction: { model: 'anthropic/claude-haiku-4.5' } })` triggers Eve's own `compactMessages`, which calls `generateText` with **Eve's own fixed** `COMPACTION_SYSTEM_PROMPT` (framed almost identically: "Create a handoff summary for another LLM that will resume the task") | **Mechanically native, content NOT customizable.** The trigger/model/escalation logic is native. But Eve's compaction prompt is a hard-coded constant in `compaction-prompt.js` — there is no `defineAgent({ compaction: { prompt: ... } })` or equivalent. You cannot inject the bespoke prompt's domain-specific extraction categories (SESSION STATE / COMPLETED FIELDS / PENDING FIELDS / CASEWORKER INPUTS / GAP ANALYSIS / GAP ANSWERS / KEY DECISIONS) into Eve's summarization pass itself. See the (c)/(d) mitigation below — this is why structured data is moved out of the summary entirely rather than depending on prompt fidelity. |
| (c) | Structured working-memory extraction via a dedicated tool call (`updateWorkingMemory`, forced with `toolChoice`, run in parallel with the summary call, only at the compaction trigger) | No native equivalent — Eve's compaction produces free text only, no structured extraction step, and there is no hook that can force a tool call at `compaction.requested` (hooks are **observe-only**, see below) | **Must be rebuilt, and rebuilt differently.** Rebuilt as an always-available authored tool (`agent/tools/update_working_memory.ts`, this task) that the agent is instructed to call continuously as it learns participant data — decoupled from the compaction moment entirely, rather than a point-in-time extraction pass tied to a token threshold. This is arguably more robust than the bespoke design: it can no longer silently miss data if compaction fires at an awkward moment, because nothing depends on catching that moment. |
| (d) | A pinned working-memory message, prepended to every model call, deliberately excluded from compaction (`extractWorkingMemory`/`prepend` in `createMessageCompressor`) | No "pinned message" concept, but a **structurally stronger** equivalent: `defineDynamic` + `defineInstructions` in `agent/instructions/` (session/turn-scoped dynamic system prompt, `node_modules/eve/dist/src/public/definitions/hook.d.ts` region confirms hooks can't inject context, but `node_modules/eve/docs/concepts/context-control.md` "Dynamic context with `defineDynamic`" section confirms instructions resolvers can) reading `workingMemoryState.get()` and rendering it into the system prompt every turn | **Must be rebuilt, using a better native primitive — and empirically confirmed working, not just inferred from docs.** system-prompt content is structurally never part of `history`, so it is compaction-immune by construction, not by convention (the bespoke version relies on `compress()` remembering to re-prepend `wm` every call — a hand-maintained invariant; the Eve version can't be compacted at all because it isn't a message). Not committed as a file (out of this task's "additive only" scope — (d) is documented, not shipped), but proven live: a temporary `agent/instructions/_wm_probe.ts` (`defineDynamic({ events: { 'turn.started': () => defineInstructions({ markdown: ... }) } })`) imported the same `workingMemoryState` handle from `agent/tools/update_working_memory.ts` and called `.get()` inside the resolver. Turn 1 stored `{"probe":"hello-from-turn-1"}` via the tool; turn 2 (fresh turn, same session, no mention of "probe" in the turn-2 prompt) was asked to quote its own system instructions verbatim and replied `"[PROBE] working memory at turn.started: {\"probe\":\"hello-from-turn-1\"}"` — proof the resolver read the tool-written state back correctly. The probe file was deleted after the test (not part of the committed prototype). |
| (e) | Keep the last 8 messages verbatim after compaction (`KEEP_RECENT = 8`, with tool-call/tool-result pairing guards) | `recentWindowSize: 10`, hard-coded in `execution/session.js`'s `createCompactionConfig` — **not exposed as an author-configurable field** anywhere in `PublicAgentCompactionDefinition` | **Native but not tunable.** Eve keeps a recent window (10 vs. the bespoke 8) with its own tool-call/tool-result pairing logic (`splitMessagesForCompaction` walks back past `role === 'tool'` messages, `withResumptionGuard` avoids ending on a stray assistant turn) — functionally equivalent, slightly larger, no way to set it to exactly 8. |

### A generic AI-SDK seam that is *not* the recommendation: model middleware

The brief's Step 1 grep also asked about `middleware`, dropped from the
executive summary above because it doesn't change the recommendation — but
it deserves an honest look since it's the closest thing to a real
`prepareStep`-equivalent that exists under Eve, and it isn't Eve's own
feature. `defineAgent({ model })` accepts a bare gateway-id string **or** a
live AI SDK `LanguageModel` instance
(`PublicAgentStaticModelDefinition = string | LanguageModel`,
`node_modules/eve/dist/src/shared/agent-definition.d.ts`), and
`node_modules/eve/dist/src/runtime/agent/resolve-model.d.ts`'s
`ResolvedRuntimeModelSelection.model` is explicitly documented as a "Live
provider instance; absent for string selections, which resolve through the
reference" — i.e., an authored non-string model is carried through to
runtime as a real object, not flattened to just an id. Reading
`node_modules/eve/dist/src/harness/tool-loop.js`'s `executeStepBody`: the
resolved model (`z`) is passed directly into `new ToolLoopAgent({ model: z,
... })`, which is what actually calls the AI SDK underneath for the main
turn's model call. Since the AI SDK's own `wrapLanguageModel({ model,
middleware })` returns an object satisfying the same `LanguageModel`
interface, and its `transformParams` hook receives and can rewrite
`params.prompt` (the message array) before every `doGenerate`/`doStream`
call, wrapping the agent's model this way is, in principle, a real,
caller-owned per-call message-rewrite seam — reachable without any
Eve-specific API, simply because Eve is built on the AI SDK and doesn't
strip that capability away.

This was **not** built or tested live in this task (verifying it would mean
swapping `agent/agent.ts`'s `model` field for a wrapped instance and
confirming `transformParams` actually fires inside a live turn — a
meaningfully separate experiment from this task's Step 2/3 scope), so treat
it as a plausible reading of the type signatures and harness code above, not
an empirical result like the `defineState` proof. It is also not recommended
even if it works: (1) it would run in *addition to*, not instead of, Eve's
own native `maybeCompact` pass in `tool-loop.js` — that call happens on the
raw session history before `ToolLoopAgent` is even constructed, so a
middleware-based compressor would need explicit coordination (e.g. setting
`compaction.thresholdPercent` high enough that Eve's own pass rarely fires)
to avoid two independent compaction systems fighting over the same
transcript; (2) it reintroduces exactly the statelessness problem
`context-compression.ts`'s own comment documents working around (AI SDK
issue #9631) — a middleware closure has no more persistence guarantee across
calls than `prepareStep` did, so it would need the same kind of
module-level/`defineState`-backed workaround the bespoke code already
carries, for no benefit over just configuring Eve's built-in compaction. So:
the honest answer to "does Eve expose a `prepareStep`-equivalent hook" is
**no named Eve hook, but yes, a generic AI-SDK model-middleware seam exists
and would technically work** — and the recommendation is still to use Eve's
native compaction instead of rebuilding on top of that seam.

### Working-memory persistence prototype (Step 2)

**Persistence API used: `defineState` from `eve/context` — a real native
per-session store, not the app-runtime Postgres fallback (b) the brief
offered.** `execute(input, ctx)`'s `ctx` (`ToolContext extends SessionContext`,
`node_modules/eve/dist/src/public/definitions/tool.d.ts` and `callback-
context.d.ts`) does **not** expose a state accessor directly — `defineState`
is a **separate** public export (`eve/context`, confirmed via
`node_modules/eve/package.json`'s `exports` map: `"./context"` →
`dist/src/public/context/index.d.ts`), called at module scope, independent of
`ctx`. Built `agent/tools/update_working_memory.ts`:

```ts
export const workingMemoryState = defineState<Record<string, unknown>>(
  'labs-asp.working-memory',
  () => ({}),
);

export default defineTool({
  inputSchema: z.object({ data: z.record(z.string(), z.unknown()).optional() }).strict(),
  async execute({ data }) {
    if (data && Object.keys(data).length > 0) {
      workingMemoryState.update((current) => ({ ...current, ...data }));
    }
    return { ok: true, keys: Object.keys(workingMemoryState.get()), data: workingMemoryState.get() };
  },
});
```

(Omitting `data` reads back the current stored state; the same schema
doubles as store and recall, matching the framework `todo` tool's own
read/write convention in `runtime/framework-tools/todo.d.ts`.)

**Two-turn proof** (`npx eve dev --no-ui --port 2002`, `AI_GATEWAY_API_KEY`
exported from `.env.local`, Node 24):

Turn 1 — `POST /eve/v1/session`:
```
{"message":"Call update_working_memory with data {\"participantName\": \"Jordan Ellis\", \"formName\": \"SNAP Recertification\"}. Just call the tool, do not ask me anything."}
```
→ `202`, `x-eve-session-id: wrun_01KY5SNRXZ76S6TG0JRHD4PAWP`. Stream showed
`actions.requested` → `update_working_memory` with
`{"data":{"participantName":"Jordan Ellis","formName":"SNAP
Recertification"}}` → `action.result` with `{"ok":true,"keys":
["participantName","formName"],"data":{...}}` → `session.waiting` with
`continuationToken":"eve:723a3711-...`.

Turn 2 — `POST /eve/v1/session/wrun_01KY5SNRXZ76S6TG0JRHD4PAWP` (same
`sessionId`, in the URL path, per the pattern confirmed in Task 3) with the
**same** `continuationToken` from turn 1's `session.waiting` event:
```
{"continuationToken":"eve:723a3711-...","message":"Call update_working_memory with no data argument to read back whatever is currently stored, then tell me exactly what it returned."}
```
→ `200`. The stream for `turn_1` showed `actions.requested` →
`update_working_memory` with `input: {}` (no `data`) → `action.result` →
`{"ok":true,"keys":["participantName","formName"],"data":
{"participantName":"Jordan Ellis","formName":"SNAP Recertification"}}` — **the
exact value stored in turn 1, recalled with zero information passed in the
turn-2 prompt**, confirming the value persisted in `defineState`'s durable
store across the turn boundary, not in the model's conversation history (the
tool never received it as an argument in turn 2; it read it back from state).
Final assistant message echoed the same JSON back verbatim.

**Caveat, stated plainly:** this two-turn proof demonstrates cross-turn
persistence. It does **not** demonstrate persistence *through an actual
compaction event*, because triggering real compaction requires driving a
session to 75%+ of a 200K-token window, which is impractical to do live in
this spike. The claim that `defineState` survives compaction specifically
rests on the structural evidence above (`state` and `history` are disjoint
fields in `DurableSession`, and `compactMessages` only reads/writes `history`)
— strong circumstantial evidence from reading the actual compaction code
path, not an end-to-end empirical proof. Flagging this so sub-project 4 does
one real long-transcript test before relying on it in production.

### Summarization shape decision (Step 3): no subagent

**Conclusion: do not build `agent/subagents/summarizer/`.** Eve's own harness
already performs the trigger + summarize + recent-window-retention pass
end-to-end and automatically, on every turn, with no author-side call needed
— that is what (a), (b), and (e) above show. A subagent would only make sense
if we needed to *replace* Eve's summarization pass with our own (e.g., to get
the domain-specific extraction categories into the summary text), and Eve
gives no hook to substitute for or intercept its internal `compactMessages`
call before it runs (`compaction.requested`/`compaction.completed` hooks are
**observe-only** — `node_modules/eve/dist/src/public/definitions/hook.d.ts`:
"Handlers are observe-only: they cannot inject model context. To contribute
runtime model messages, use `defineDynamic` + `defineInstructions`."). So a
subagent could not intercept or replace Eve's compaction pass even if we
built one — there's no dispatch point for it to run at.

Instead, the (c)/(d) gap (structured, domain-specific data that needs to
survive with full fidelity) is closed **without any delegation at all**: the
`update_working_memory` tool (this task) plus a `defineDynamic` +
`defineInstructions` resolver (documented and empirically confirmed, not
committed as a file — see (d) above) keep the participant/form data entirely
outside the summarized region, so its fidelity never depends on how good
Eve's fixed compaction prompt is. `compaction.requested`/`compaction.completed`
are also the direct, native replacement for the app's own
`data-compacting`/`data-checkpoint` UI signal from `route.ts`'s `prepareStep`
— **this closes the open dependency Task 3 flagged** ("whether Eve exposes a
`prepareStep`-equivalent hook for a caller-owned compressor, or whether Eve's
own context management replaces `lib/ai/context-compression.ts` outright").
The answer is the latter. Concretely, this doesn't need a `defineHook` at
all for sub-project 5's purposes: `compaction.requested`/`compaction.completed`
are protocol-level stream events (`node_modules/eve/dist/src/protocol/
message.d.ts`, same event family as `message.appended`/`actions.requested`),
so they already arrive in the same NDJSON `GET /eve/v1/session/:id/stream`
Task 3's adapter reads — the adapter maps them straight onto
`data-compacting`/`data-checkpoint` the same mechanical way it maps
`message.appended`/`actions.requested` onto AI SDK `UIMessage` parts, no
extra in-app hook required. `defineHook` in `agent/hooks/` would only matter
for a need internal to the agent process itself (e.g. logging, a side
effect, or feeding a completely separate telemetry sink) — not for the
adapter, which already sees these events on the wire. Neither event was
observed in Task 3's capture only because that capture's single-tool-call
turn never came close to the compaction threshold, not because the events
don't exist.

### Net effect on sub-project 4

If Eve is adopted: **delete `lib/ai/context-compression.ts` and the
`prepareStep`-based compression wiring in `route.ts` entirely** — do not port
it. Replace with: `defineAgent({ compaction: { model: 'anthropic/claude-
haiku-4.5' or 'anthropic/claude-sonnet-4.6', thresholdPercent: 0.75 } })` on
`agent/agent.ts`; `agent/tools/update_working_memory.ts` (built this task)
called continuously per the agent's instructions; a `defineDynamic` +
`defineInstructions` resolver re-injecting `workingMemoryState.get()` into
the system prompt every turn (documented and empirically confirmed with a
temporary probe, not committed as a file); and sub-project 5's adapter
mapping the `compaction.requested`/`compaction.completed` stream events it
already receives onto `data-compacting`/`data-checkpoint` (no `defineHook`
needed for that). The one accepted regression: the bespoke domain-specific summary
categories (SESSION STATE / COMPLETED FIELDS / etc.) become a generic
handoff summary instead, because Eve's compaction prompt isn't
authorable — mitigated, not eliminated, by moving the fidelity-critical data
into `defineState` where prompt quality no longer matters.

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
| `data-compacting` (`route.ts:230-235`) | `route.ts` `prepareStep` (compressor `onCompacting` callback) | Eve's `compaction.requested` hook event (see Q2) | **Resolved in Q2 (mapping is possible, semantics differ).** This event isn't in the turn stream because `route.ts` injects it from its own `prepareStep` hook, but Q2 confirmed Eve owns compaction internally and emits an observe-only `compaction.requested` hook event — an adapter can map that onto `data-compacting`. The caveat is ownership, not translation: Eve decides *when* to compact (`thresholdPercent`, default 0.9), not the app. |
| `data-checkpoint` (`route.ts:238-248`) | `route.ts` `prepareStep` (after a real compaction) | Eve's `compaction.completed` hook event (see Q2) | **Resolved in Q2** — same as `data-compacting`: map Eve's observe-only `compaction.completed` event onto `data-checkpoint`. Note Eve's compaction summary text is hard-coded (not the app's domain-specific categories), so the `summary` field would differ (see Q2 limitations). |
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
`data-checkpoint`: those aren't a stream-translation problem so much as a
dependency on context ownership, which **Q2 resolved** — Eve owns context
management internally (there is no authorable `prepareStep`-equivalent hook),
so `lib/ai/context-compression.ts` is effectively replaced by Eve's built-in
compaction, and the adapter maps Eve's observe-only `compaction.requested` /
`compaction.completed` hook events onto these two UI events (with the
hard-coded-summary caveat noted in Q2). That resolution doesn't change the
adapter-vs-rework verdict (a rework path would face the same context-ownership
reality, plus the cost of rewiring everything else). Two more adapter-design
details fall out of the capture: it must
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

This is a sketch to inform sub-project 3 — not an implementation. No changes
were made to `lib/kernel/browser.ts` (read-only per the brief).

### What breaks under Eve's durable, replayed execution

Per Task 1's confirmed finding (eve.dev docs), Eve tools run in the app
runtime with full `process.env` — so a Kernel.sh-backed tool can still make
the same `kernel.browsers.*` SDK calls `lib/kernel/browser.ts` makes today.
The problem is not "can a tool reach Kernel" — it's that `lib/kernel/
browser.ts`'s two process-local mechanisms assume single-instance, same-
process continuity across a session's tool calls, and Eve's durable/
replayable session model does not guarantee that:

1. **The in-memory session cache** (`const sessions = new Map<string,
   BrowserSession>()`, keyed `cacheKey(userId, sessionId)` =
   `` `${userId}:${sessionId}` ``, `lib/kernel/browser.ts:97`). Eve sessions
   are backed by durable Workflow steps designed to "outlast crashes,
   redeploys, and days-long sessions" (`node_modules/eve/docs/guides/
   state.md`) — the whole point is that a session's tool calls are not
   pinned to one long-lived process. A tool call for the same logical
   browser session landing on a different process instance (a redeploy
   between turns, a scaled-out worker, a retried step) is a **cache miss**,
   not a hard failure — `getBrowser`/`reconnectBrowser` already have a
   fallback path that recreates from the Kernel profile
   (`ensureProfile`/`profile: { name, save_changes: true }`,
   `lib/kernel/browser.ts:169-185`). But that fallback path is written today
   as the *rare* case (`if (!session) { ... recreate ... }`); under Eve it
   would become the **common** case, and worse, a **race**: two tool calls
   for the same session landing on two different processes at nearly the
   same time would each miss the cache and each call
   `kernel.browsers.create(...)`, producing two live Kernel browsers backed
   by the same profile — profile-lock contention or silently divergent page
   state, not just a wasted-cost duplicate.

2. **The per-session mutex** (`sessionQueues` promise-chaining queue in
   `lib/ai/tools/browser.ts`, serializing calls "per session" so
   Playwright's `page` object — not concurrency-safe — never receives two
   commands at once). That queue is a `Map` in one process's memory. It only
   serializes tool calls that happen to land on the *same* process. Two
   browser-tool calls in the same Eve turn (Eve's `actions.requested` event
   carries `data.actions: [...]`, an array — Task 3's capture only ever saw
   one, but the shape supports multiple tool calls dispatched together) or
   across a step retry that resumes on a different worker would bypass the
   mutex entirely, exactly the concurrent-CDP-access scenario the mutex
   exists to prevent.

### How it would work under Eve instead

- **Re-resolve the Kernel session by its stable id at the top of every tool
  call, never trust a cached handle across calls.** Treat the in-memory
  `sessions` Map purely as a same-process, same-turn optimization (skip a
  network round-trip when you happen to still be warm), never as the
  guarantee that a live `BrowserManager`/CDP connection already exists. Every
  call should be prepared to run `reconnectBrowser`'s existing "recreate/
  reconnect from profile" path as the primary path, not the fallback —
  Kernel.sh's own profile persistence (`save_changes: true`) is already the
  right durable source of truth here; the fix is trusting it every time
  instead of trusting local memory first.
  - **One integration detail this surfaces:** the stable id today is
    `` `${chatId}-${userId}` `` (see `chatIdFromSessionId`,
    `lib/kernel/browser.ts:25-29`), derived from the *app's* chat identity.
    Eve's own session id has a different shape (`wrun_...`, confirmed live
    in this task's proof above) and is not derivable from `chatId`/`userId`.
    So the authored Eve tool's `inputSchema` (or something threaded through
    `ctx.session`) needs to carry `chatId` and `userId` explicitly on every
    call so the tool can reconstruct the exact same Kernel lookup key
    `lib/kernel/browser.ts` uses today — Eve's session identity and the
    app's chat identity are two different namespaces that need an explicit
    bridge, not an implicit one.
- **Per-turn serialization has to move to something that survives across
  processes**, because a process-local `Map<string, Promise>` mutex can no
  longer be assumed sufficient. The app already depends on Redis for
  resumable streaming (`resumable-stream` + Redis, per `route.ts` and this
  repo's `CLAUDE.md`), which is the natural place to put a cross-process
  lock: acquire a short-lived Redis lock keyed by the same stable
  `` `${chatId}-${userId}` `` id at the start of a browser tool call, release
  it (or let it lease-expire) at the end, so two calls for the same session
  — regardless of which process or Eve worker runs them — still serialize
  against the one Playwright `page`. This is new infrastructure, not a port
  of existing code: today's mutex works precisely because the app assumes
  one process per deployment (`lib/kernel/browser.ts:94`, "No Redis needed —
  this is a single Cloud Run instance talking to Kernel"); Eve's durable
  execution model removes that assumption, so the lock has to move out of
  process memory into something all instances share.
- **Never treat the `BrowserManager`/Playwright `page` object itself as
  something that can be durable or serialized.** Only the plain strings
  Kernel gives back (`cdp_ws_url`, `session_id`, `profileName`) are safe to
  treat as durable/reconstructable; `reconnectBrowser` already builds a
  fresh `BrowserManager` from `cdpWsUrl` every time it takes that path
  (`lib/kernel/browser.ts:421-427`) — that pattern is exactly right and
  should become the *only* pattern, not one of two.

Net effect for sub-project 3: this is not a rewrite of `lib/kernel/
browser.ts`'s Kernel-facing logic (creating profiles, starting replays,
standby/reconnect semantics all carry over unchanged), it's removing the
assumption that the in-memory cache and mutex are anything more than a warm-
path optimization, adding an explicit `chatId`/`userId` bridge for Eve's
different session-id namespace, and adding one new piece of infrastructure —
a cross-process lock — that the current single-instance deployment has never
needed.

## Recommendation

**Go, with eyes open — this is not a clean go.** Every question the spike set
out to answer has a concrete, source-grounded answer: Eve boots and runs a
real authored tool end-to-end locally (Q1-local), context management has a
native replacement with a defined migration path (Q2), and the UI-integration
shape is a low-risk, additive adapter route rather than a `chat.tsx` rewrite
(Q3). None of that is theoretical — it's backed by live `curl` round-trips
against a running `eve dev` server, a two-turn `defineState` persistence
proof, and a real NDJSON capture, not documentation reading. The single
biggest risk is the one Q2 surfaces: **Eve's compaction prompt is a hard-coded
constant with no authorable override, so the bespoke domain-specific summary
categories (SESSION STATE / COMPLETED FIELDS / PENDING FIELDS / CASEWORKER
INPUTS / GAP ANALYSIS / GAP ANSWERS / KEY DECISIONS) cannot be reproduced in
Eve's summarization pass.** Moving participant/form data into `defineState`
(compaction-immune by construction, not by convention) mitigates the *data*
half of that gap — anything written to working memory survives verbatim,
recall does not depend on prompt quality — but it does not eliminate the
*transcript-summary fidelity* half: whatever isn't explicitly promoted into
working memory still gets compacted through Eve's generic, non-customizable
handoff prompt, a real regression from today's hand-tuned summarizer. Two
secondary risks compound this and should not be waved off: the browser
session model needs new cross-process locking infrastructure that does not
exist today (sub-project 3), and Q1's Vercel-preview half is entirely
unverified — Eve running as its own server process is a materially different
deployment shape than "routes inside the Next app," and nothing here confirms
it behaves the same way under Vercel's serverless runtime. Recommendation:
proceed to sub-project 2 (tool migration) and a scoped Vercel-preview
follow-up in parallel, but treat sub-project 4 (context/working-memory) as
carrying real, accepted risk rather than a mechanical port, and get sign-off
from whoever owns the domain-specific summary quality bar before committing
to Eve's compaction as a full replacement.

## Open items for sub-projects 2–6

- **Sub-project 2 (tool migration):** zod is in — the repo was migrated
  v3→v4.4.3 specifically to unblock this, so `defineTool({ inputSchema: z.object(...) })`
  now works as documented (no more plain-JSON-Schema workaround; `read_reference`
  was reverted back to a zod schema after the migration). Tools run in the app
  runtime with full `process.env`, so existing tool logic (Apricot calls, DB
  queries) ports largely unchanged — the porting work is mechanically
  rewriting each `lib/ai/tools/*.ts` factory into a `defineTool` module under
  `agent/tools/`, not rewriting the tools' internals. Watch the still-open
  `ai@^7.0.26` peer-dependency gap (repo pins `ai@7.0.19`) before this
  sub-project leans harder on AI SDK internals.
- **Sub-project 3 (browser re-architecture):** the existing in-memory session
  cache and per-session mutex in `lib/kernel/browser.ts` /
  `lib/ai/tools/browser.ts` are single-process assumptions that Eve's
  durable/replayed execution model breaks — not rarely, but as the common
  case. This sub-project now needs a **new cross-process lock** (Redis is
  already in the stack via `resumable-stream`, and is the natural home for
  it), and an explicit bridge between Eve's own session-id namespace
  (`wrun_...`) and the app's `` `${chatId}-${userId}` `` cache key, since
  Eve's session id is not derivable from the app's chat identity. The
  Kernel-facing logic itself (profile creation, reconnect-from-`cdpWsUrl`)
  does not need to change.
- **Sub-project 4 (context / working memory):** delete
  `lib/ai/context-compression.ts` and the `prepareStep`-based compression
  wiring in `route.ts` outright rather than porting it — configure
  `defineAgent({ compaction: { model, thresholdPercent: 0.75 } })` and rebuild
  only structured working-memory extraction, as an always-on
  `update_working_memory` tool plus a `defineDynamic`/`defineInstructions`
  resolver re-injecting it every turn. Accept, and get explicit sign-off on,
  the caveat that Eve's compaction prompt is hard-coded: the bespoke
  domain-specific summary categories degrade to a generic handoff summary for
  anything not explicitly promoted into `defineState`. Also budget one real
  long-transcript test that drives a session past the actual compaction
  threshold — this spike's persistence proof is two-turn and structural
  (`state`/`history` are disjoint fields), not an empirical test of surviving
  a real compaction event.
- **Sub-project 5 (UI↔Eve wiring + Postgres bridge):** build a Next.js
  adapter route that translates Eve's NDJSON stream into the AI SDK SSE shape
  `chat.tsx`/`message.tsx` already consume — text and tool events map 1:1 by
  `callId`, and `step.completed.usage` covers `data-token-usage` (flatter
  nesting, same fields). `data-compacting`/`data-checkpoint` now have a
  concrete source: map them directly from the `compaction.requested`/
  `compaction.completed` protocol events already on the wire (no `defineHook`
  needed). Before committing this to production, capture Eve's error/abort
  event shapes — this spike's capture only exercised the happy path — and
  design the adapter to terminate the AI-SDK-shaped stream at
  `turn.completed`/`session.waiting` rather than passing Eve's still-open
  connection straight through.
- **Sub-project 6 (cutover):** cutover is standing up a separate Eve service
  and wiring the adapter route to it, not an in-place swap inside the
  existing Next process — `eve init` confirmed Eve runs as its own server
  (default port ~2000) and does not mount into `next dev`/`next build`, so
  deployment topology (where that second process runs, how it's reached from
  the Vercel-hosted Next app) is a real open design question, not a detail.
  Two papercuts from this spike need resolving before a real cutover, not
  just flagging: the `eve init`-generated `pnpm-workspace.yaml` omits the
  required `packages:` field and breaks every `pnpm` command in this
  single-package repo until fixed by hand, and the `ai@^7.0.26` peer-dependency
  gap against the repo's pinned `ai@7.0.19` is still an unresolved warning,
  not a confirmed-safe mismatch. `"engines": { "node": "24.x" }` (added by
  `eve init`) is a hard constraint on wherever that second process is hosted.
  The Vercel-preview half of Q1 (deferred, above) should be closed out before
  cutover planning starts in earnest, since it's the first real test of
  whether this topology works on the target platform at all.
