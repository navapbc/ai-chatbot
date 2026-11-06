# Agent Loop Implementation Plan (Plan 2 of 5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extension runs survive restarts, ask for missing data through a structured, validated question path, report structured progress, and no relay credential is ever stored by Eve.

**Architecture:** One Postgres table (`EveRun`) replaces the in-memory continuity `Map` and records each run's relay origin, pending questions and last progress report. Eve session auth carries only a non-secret run reference; the browser tool fetches a short-lived agent token from an internal Next route when it needs one. Pending `ask_question` requests are persisted and every answer is validated against them before anything reaches Eve. Agent guidance lives in a new skill so web-app behavior is unchanged.

**Tech Stack:** TypeScript, Drizzle + Postgres, Next.js route handlers, Eve 0.27 (`ask_question`, `inputResponses`, `defineTool`), zod, vitest (node config), Playwright Chromium for the extension e2e.

**Spec:** `docs/specs/2026-10-23-browser-extension-design.md`. Prior plan: `docs/plans/2026-10-23-browser-bridge.md` (merged). Tracker: `docs/browser-extension-todo.md` (plan 2 section).

## Evidence This Plan Rests On

A throwaway spike (local Postgres 14, Eve on the Postgres world, 6 model turns) found:
- Eve's Postgres world stores session auth attributes. The relay token appeared in `workflow_runs.input_cbor`, `workflow_steps.input_cbor` and `output_cbor`, and `workflow_events.payload_cbor`, as zstd-compressed CBOR (not encrypted), and was re-saved on every turn even when later requests sent no headers. A raw text search finds nothing; the check must decompress. Hence Tasks 4 and 5.
- `ask_question` calls can run in parallel (one `input.requested` with `data.requests` of length 2; `requestId` equals the tool call id). `inputResponses: [{ requestId, optionId | text }]` resumes the session with no plain message. `allowFreeform: false` is NOT enforced on structured answers; a stale or unknown `requestId` becomes an ordinary user message; `session.waiting` does not list pending requests; a bad continuation token returns HTTP 500. Hence Tasks 3 and 6.
- The first user message is stored as plaintext jsonb in `workflow_runs.attributes` (existing behavior, outside this plan; see Out of Scope).

## Global Constraints

- No relay token, relay secret or internal secret may be stored by Eve, put in a message, tool input or output, log, or error. Eve carries only a non-secret run reference.
- A free-text answer to an option-only request (`allowFreeform: false`) must never reach Eve. Sensitive values never reach the agent; plan 3 adds the classifier, this plan enforces option-only answers server-side.
- Fail closed: a relay run never silently falls back to a new session or to Kernel.
- The web app and legacy route behave exactly as before for chats without a relay origin.
- Database migrations are generated with `pnpm db:generate` and never hand-edited; check with `pnpm db:check`.
- New plan 2 surfaces are off unless `RELAY_RUNS_ENABLED` is exactly `1` (set nowhere in deployment until plan 4).
- Ask before adding dependencies. pnpm, Biome, TypeScript, Node 24. Node tests run with `vitest.config.node.mjs`. Importing `lib/db/queries.ts` needs `vi.mock('server-only', () => ({}))` in node tests.
- Agent instructions change in the Eve tree only (`agent/`), never `lib/ai/prompts/`.

## Review Focus

Each line has a test in the named task.

1. Two concurrent turns on one run: only one continuation is accepted; the other gets a conflict and Eve is not called twice (Task 3).
2. Answer integrity: stale or unknown `requestId`, text on an option-only request, a missing or duplicate answer, and a plain message while questions are pending are all rejected before Eve is called (Task 6).
3. Pending questions survive a Next restart and are cleared exactly once, only after Eve accepted the answers (Task 6).
4. `report_progress` cannot carry field values: unknown keys are stripped and oversized strings or lists rejected (Tasks 1, 7).
5. The old `x-eve-relay-token` header is ignored and the agent-token route refuses without the secret, without the flag, or for a run with no relay origin (Tasks 4, 5).
6. A service-worker restart mid-run recovers without stale element refs being reused (Task 9).

## File Structure

| Path | Responsibility |
|---|---|
| `lib/ai/eve/input-requests.ts` | Pending question types, parsing Eve's `input.requested`, answer validation |
| `lib/relay/progress.ts` | `RunProgress` schema and `sanitizeProgress` (shared by agent tool, route, panel) |
| `lib/ai/eve/run-store.ts` | `EveRunStore` interface and in-memory implementation |
| `lib/ai/eve/db-run-store.ts` | Drizzle implementation |
| `lib/db/schema.ts`, `lib/db/queries.ts`, `lib/db/migrations/` | `EveRun` table, queries, generated migration |
| `lib/relay/agent-token.ts` | Eve-side token fetch with in-memory cache |
| `app/(chat)/api/relay/agent-token/route.ts` | Internal route that mints the agent token |
| `lib/ai/eve/loopback-attributes.ts`, `agent/channels/eve.ts`, `lib/ai/eve/eve-client.ts`, `lib/relay/eve-relay-browser.ts` | Run-reference attributes in place of the token |
| `app/(chat)/api/eve-chat/route.ts`, `lib/ai/eve/stream-adapter.ts` | Store-backed turns, structured questions, answer path |
| `agent/tools/report_progress.ts`, `agent/skills/extension-run/SKILL.md` | Progress tool and extension guidance |
| `scripts/check-workflow-secrets.ts` | Verification: search Eve's Postgres world for strings |

---

### Task 1: Question and progress types

**Files:**
- Create: `lib/ai/eve/input-requests.ts`, `lib/relay/progress.ts`
- Test: `tests/agent/input-requests.test.ts`, `tests/agent/run-progress.test.ts`

**Interfaces:**
- Produces:
  ```ts
  export interface InputOption { id: string; label: string }
  export interface PendingInputRequest { requestId: string; prompt: string; display: 'text' | 'select'; options: InputOption[]; allowFreeform: boolean }
  export interface InputAnswer { requestId: string; optionId?: string; text?: string }
  export type AnswerError = 'unknown_request' | 'missing_answer' | 'duplicate_answer' | 'unknown_option' | 'text_not_allowed' | 'empty_answer';
  export function parseInputRequests(raw: unknown): PendingInputRequest[];  // from event.data.requests; drops malformed entries
  export function validateAnswers(pending: readonly PendingInputRequest[], answers: unknown): { ok: true; responses: InputAnswer[] } | { ok: false; error: AnswerError };
  // lib/relay/progress.ts
  export type SectionStatus = 'waiting' | 'filling' | 'done' | 'needs_you';
  export interface RunProgress { sections: { name: string; status: SectionStatus; filled: number; total?: number }[]; current?: string; actions?: string[] }
  export const runProgressSchema: z.ZodType<RunProgress>;
  export function sanitizeProgress(input: unknown): RunProgress | null; // strips unknown keys; null if invalid
  ```
  Limits: at most 40 sections; `name` 1-80 chars; `current` at most 120; at most 5 `actions`, each at most 120; counts are non-negative integers; text answers are trimmed and at most 2000 characters.

- [ ] **Step 1: Write failing tests.** `validateAnswers`: accepts a full valid set (select option, text on a freeform request); rejects with the exact error for each of: a requestId not pending (`unknown_request`), a pending request with no answer (`missing_answer`), the same requestId twice (`duplicate_answer`), an optionId not in the options (`unknown_option`), `text` on a request with `allowFreeform: false` (`text_not_allowed`), `optionId` together with `text` on an option-only request (`text_not_allowed`), neither field or blank text (`empty_answer`), non-array input. `parseInputRequests` handles the spiked shapes (text request with `allowFreeform: true`; select request with `allowFreeform: false` and three options) and drops entries with no `requestId`. `sanitizeProgress`: strips an extra `value` key from a section, rejects 41 sections, a 121-char `current`, a negative count, six actions; returns the clean object otherwise.
- [ ] **Step 2: Run** `pnpm exec vitest run --config vitest.config.node.mjs tests/agent/input-requests.test.ts tests/agent/run-progress.test.ts` — Expected: FAIL (modules missing).
- [ ] **Step 3: Implement** both modules as pure functions with no Eve, Next or server-only imports.
- [ ] **Step 4: Run** the same command — Expected: PASS.
- [ ] **Step 5: Commit** `feat(agent): input request and run progress types`.

### Task 2: Run table and store

**Files:**
- Modify: `lib/db/schema.ts`, `lib/db/queries.ts`
- Create: `lib/db/migrations/<generated>`, `lib/ai/eve/run-store.ts`, `lib/ai/eve/db-run-store.ts`
- Test: `tests/agent/eve-run-store.test.ts`

**Interfaces:**
- Consumes: `PendingInputRequest`, `RunProgress`.
- Produces: table `EveRun` (`eveRun` in schema): `id uuid pk`, `chatId uuid not null` (no foreign key: `eve-chat` does not create `Chat` rows today), `userId uuid not null references User`, `eveSessionId text`, `continuationToken text`, `streamIndex integer default 0`, `relayOrigin text null` (exact https origin; set only for extension runs), `pendingRequests jsonb default []`, `progress jsonb null`, `createdAt`, `updatedAt`; `unique(userId, chatId)`.
  ```ts
  export interface EveRunRecord { eveSessionId: string; continuationToken: string; streamIndex: number; relayOrigin: string | null; pendingRequests: PendingInputRequest[]; progress: RunProgress | null }
  export interface EveRunStore {
    get(userId: string, chatId: string): Promise<EveRunRecord | undefined>;
    getByChatId(chatId: string): Promise<(EveRunRecord & { userId: string }) | undefined>; // internal token route only
    create(userId: string, chatId: string, init: { eveSessionId: string; continuationToken: string; relayOrigin?: string | null }): Promise<EveRunRecord>; // throws if the row exists
    swapToken(userId: string, chatId: string, expected: string, next: string): Promise<boolean>; // compare-and-set on continuationToken
    advance(userId: string, chatId: string, update: { continuationToken: string; streamIndex: number }): Promise<void>; // only when streamIndex is greater than stored
    setPending(userId: string, chatId: string, requests: PendingInputRequest[]): Promise<void>;
    setProgress(userId: string, chatId: string, progress: RunProgress): Promise<void>;
    remove(userId: string, chatId: string): Promise<void>;
  }
  export function createMemoryRunStore(): EveRunStore;
  export function createDbRunStore(): EveRunStore;
  export function getRunStore(): EveRunStore; export function setRunStoreForTests(store: EveRunStore | null): void;
  ```

- [ ] **Step 1: Write failing tests** as one shared suite run against the memory store, and also against the Drizzle store when `TEST_DATABASE_URL` is set (skip with a logged reason otherwise): `create` then `get` round trip; `get` for another user's chat returns undefined; `create` twice throws; `swapToken` succeeds with the right expected token and returns false (changing nothing) with a wrong one; `advance` ignores a lower or equal `streamIndex` and applies a higher one; `setPending` and `setProgress` round trip and survive `get`; `getByChatId` returns the owner and `relayOrigin`; `remove` deletes.
- [ ] **Step 2: Run** `pnpm exec vitest run --config vitest.config.node.mjs tests/agent/eve-run-store.test.ts` — Expected: FAIL.
- [ ] **Step 3: Implement.** Add the table in the existing schema style; run `pnpm db:generate` and `pnpm db:check`, inspect the generated SQL and the journal diff (the migrations folder has duplicate-numbered files; if `db:generate` produces unrelated changes, stop and report). Add query functions in `lib/db/queries.ts` (`upsert`/CAS in SQL: `UPDATE ... WHERE continuationToken = $expected RETURNING`, `UPDATE ... WHERE streamIndex < $new`). `createDbRunStore` maps the interface onto them. To run the Drizzle half locally: create a scratch database, apply `DATABASE_URL=$TEST_DATABASE_URL pnpm db:migrate`, then run the suite with `TEST_DATABASE_URL` set.
- [ ] **Step 4: Run** the suite without and, if a local Postgres is available, with `TEST_DATABASE_URL` — Expected: PASS (Drizzle half reported as run or skipped).
- [ ] **Step 5: Commit** `feat(db): EveRun table and run store`.

### Task 3: Turns run on the store

**Files:**
- Modify: `app/(chat)/api/eve-chat/route.ts`
- Delete: `lib/ai/eve/session-continuity.ts`, `tests/agent/eve-session-continuity.test.ts`
- Test: `tests/agent/eve-chat-route.test.ts`

**Interfaces:**
- Consumes: `getRunStore()`, `EveRunStore`.
- Behavior: `POST` reads the run with `store.get`; a follow-up swaps the token with `swapToken(expected = stored token, next = returned token)` and answers 409-style (`ChatSDKError('offline:chat')`, nothing sent to Eve a second time) when the swap returns false; a new run calls `store.create`; the stream loop calls `store.advance` at each `session.waiting`. The run's `relayOrigin` decides the rules: for `relayOrigin !== null` a missing row or an Eve continue failure returns an error and never creates a fresh session; for `relayOrigin === null` (web chats) the previous behavior stays (missing row creates a session). Logs still carry no message text.

- [ ] **Step 1: Write failing tests** (mock `@/app/(auth)/auth`, `@/lib/ai/eve/eve-client`, inject `createMemoryRunStore` with `setRunStoreForTests`): a first turn creates a row with the session id and token; a second turn continues and stores the new token; two concurrent second turns send exactly one `continueEveSession` call and one request gets the conflict error; a relay run (`relayOrigin` set) whose Eve continue throws returns an error and leaves the row untouched; a web chat (no origin) with no row creates a session as before; the stream cursor advances monotonically across two turns and never goes backward when events replay.
- [ ] **Step 2: Run** `pnpm exec vitest run --config vitest.config.node.mjs tests/agent/eve-chat-route.test.ts` — Expected: FAIL.
- [ ] **Step 3: Implement** the route changes; remove the old module and its test; update imports. Keep the reconnect loop and logging untouched.
- [ ] **Step 4: Run** the file and `pnpm exec vitest run --config vitest.config.node.mjs tests/agent` — Expected: PASS.
- [ ] **Step 5: Commit** `feat(eve): persist run continuity in Postgres`.

### Task 4: Run reference in place of the token

**Files:**
- Modify: `lib/ai/eve/loopback-attributes.ts`, `agent/channels/eve.ts`, `lib/ai/eve/eve-client.ts`, `lib/relay/eve-relay-browser.ts`, `agent/tools/browser.ts`, `agent/tools/check_submit_gate.ts`
- Create: `lib/relay/agent-token.ts`
- Test: `tests/agent/loopback-attributes.test.ts` (update), `tests/relay/agent-token.test.ts`, `tests/relay/eve-relay-browser.test.ts` (update), `tests/relay/eve-client-relay.test.ts` (update)

**Interfaces:**
- Produces:
  ```ts
  export interface RelayRunRef { url: string; runId: string }   // replaces RelayCredentials; neither field is secret
  export type RelayCredentialState = { kind: 'none' } | { kind: 'complete'; ref: RelayRunRef } | { kind: 'partial' };
  // lib/relay/agent-token.ts
  export type AgentTokenResult = { ok: true; token: string; expiresAt: number } | { ok: false; error: string };
  export function fetchAgentToken(runId: string, opts?: { fetchImpl?: typeof fetch; baseUrl?: string; now?: () => number }): Promise<AgentTokenResult>; // never throws; caches per runId in memory until 60 s before expiry
  ```
  Attributes become `relayUrl` and `relayRunId` (from headers `x-eve-relay-url` and `x-eve-relay-run`, a UUID), granted only with loopback AND the internal secret as before. `x-eve-relay-token` is no longer read. `createEveSession(message, model?, relay?: RelayRunRef)`. `fetchAgentToken` calls `POST ${baseUrl}/api/relay/agent-token` with header `x-eve-internal-secret` and body `{ runId }`; `baseUrl` defaults to `process.env.APP_INTERNAL_URL ?? 'http://127.0.0.1:3000'`; 5 s timeout; error strings are fixed and credential-free. `runRelayCommand` fetches the token, builds the CDP URL, and passes `redact: [token, cdpUrl]` as today.

- [ ] **Step 1: Write failing tests**: `attributesFromHeaders` grants `relayUrl` and `relayRunId` with loopback, both headers and the secret; ignores `x-eve-relay-token` entirely (the sentinel string never appears in the returned attributes); refuses a non-UUID run header; refuses without the secret or off loopback. `relayCredentialState` is `partial` with only one attribute and never reads a token attribute. `fetchAgentToken` returns the token from a mocked 200, caches it (second call makes no fetch), refetches after the cache window, returns fixed errors for a non-200, a network error and a malformed body, and never includes the secret in an error. `runRelayCommand` with the mocked fetch runs the command and still redacts the token from error text; with `fetchAgentToken` failing it returns `success: false` without starting `agent-browser`. `createEveSession` sends `x-eve-relay-run` and `x-eve-relay-url`, never a token header, and throws `relay sessions require EVE_INTERNAL_SECRET` without the env secret.
- [ ] **Step 2: Run** `pnpm exec vitest run --config vitest.config.node.mjs tests/agent/loopback-attributes.test.ts tests/relay/agent-token.test.ts tests/relay/eve-relay-browser.test.ts tests/relay/eve-client-relay.test.ts` — Expected: FAIL.
- [ ] **Step 3: Implement** the signatures above; update `agent/tools/browser.ts` and `check_submit_gate.ts` for the renamed state; keep partial-credentials fail-closed behavior.
- [ ] **Step 4: Run** the same command plus `pnpm exec vitest run --config vitest.config.node.mjs tests/agent` — Expected: PASS.
- [ ] **Step 5: Commit** `refactor(relay): carry a run reference to Eve, not the token`.

### Task 5: Internal agent-token route and relay run creation

**Files:**
- Create: `app/(chat)/api/relay/agent-token/route.ts`
- Modify: `app/(chat)/api/eve-chat/route.ts`, `scripts/start-container.sh`
- Test: `tests/relay/agent-token-route.test.ts`, `tests/agent/eve-chat-route.test.ts` (extend)

**Interfaces:**
- Consumes: `EveRunStore.getByChatId`, `mintRelayToken`, `toRelayWsUrl`, `normalizeOrigin`, `RelayRunRef`.
- Produces: `POST /api/relay/agent-token` body `{ runId: uuid }`, response `{ token, expiresAt }` with `Cache-Control: no-store`. Order of checks: `RELAY_RUNS_ENABLED !== '1'` gives a bodyless 404; missing or wrong `x-eve-internal-secret` (digest comparison with `timingSafeEqual`) gives 404; invalid body gives 400; no run, or a run with `relayOrigin === null`, gives 404; missing or short `RELAY_TOKEN_SECRET` gives a credential-free 500; otherwise it mints `{ runId, role: 'agent', origin: relayOrigin }`. `eve-chat` accepts an optional body field `relay: { origin: string }` on the FIRST turn only when `RELAY_RUNS_ENABLED === '1'`, `RELAY_PUBLIC_URL` parses with `toRelayWsUrl`, and `normalizeOrigin(origin) === origin`; it then calls `createEveSession(text, model, { url: toRelayWsUrl(RELAY_PUBLIC_URL), runId: chatId })` and `store.create(..., { relayOrigin: origin })`. With the flag off, a body containing `relay` is a 400. `scripts/start-container.sh` exports `APP_INTERNAL_URL=http://127.0.0.1:${PORT:-3000}` into the Eve process environment only (Eve runs with its own `PORT`).

- [ ] **Step 1: Write failing tests**: the route refuses (404, no body, auth/store never reached for the flag case) for flag unset, wrong secret, missing secret, unknown run, run with no relay origin; returns a token that `verifyRelayToken` accepts with role `agent`, the run id and the stored origin; 400 for a non-UUID body; 500 without leaking env values. `eve-chat` with the flag on creates a run with the origin and sends the relay run reference to `createEveSession`; with the flag off and a `relay` field returns 400; with an origin that is not exact (`https://a.gov/x`, `http://a.gov`) returns 400.
- [ ] **Step 2: Run** `pnpm exec vitest run --config vitest.config.node.mjs tests/relay/agent-token-route.test.ts tests/agent/eve-chat-route.test.ts` — Expected: FAIL.
- [ ] **Step 3: Implement.** Show the `start-container.sh` change in the diff; do not set any flag in Terraform.
- [ ] **Step 4: Run** the same command — Expected: PASS.
- [ ] **Step 5: Commit** `feat(relay): internal agent-token route and relay run creation`.

### Task 6: Structured questions and the answer path

**Files:**
- Modify: `lib/ai/eve/stream-adapter.ts`, `lib/ai/eve/eve-client.ts`, `app/(chat)/api/eve-chat/route.ts`
- Test: `tests/agent/eve-stream-adapter.test.ts` (extend), `tests/agent/eve-chat-route.test.ts` (extend), `tests/agent/eve-client-input.test.ts`

**Interfaces:**
- Consumes: `parseInputRequests`, `validateAnswers`, `store.setPending`.
- Produces: `translateEveEvent` gains `ctx.structuredInput?: boolean` and result field `inputRequests?: PendingInputRequest[]`; when `structuredInput` is true an `input.requested` event writes `{ type: 'data-input-request', data: { requests } }` and no prose, otherwise behavior is unchanged. `continueEveSession(sessionId, token, input: string | { message?: string; inputResponses?: InputAnswer[] })` (string form kept). `eve-chat` request body for relay runs: `{ id, message? , inputResponses?: InputAnswer[] }`. For a run with `relayOrigin !== null`: on `inputRequests` in the stream the route calls `setPending` (with the parsed requests); a body with `inputResponses` requires non-empty pending, runs `validateAnswers`, rejects with `ChatSDKError('bad_request:api', <AnswerError>)` before calling Eve, forwards only on success, and clears pending (`setPending(..., [])`) only after `continueEveSession` resolved; a plain `message` while pending is non-empty is rejected with cause `answers_pending`; `inputResponses` with nothing pending is rejected with `unknown_request`. Web chats keep the prose behavior and ignore `inputResponses`.

- [ ] **Step 1: Write failing tests**: adapter in structured mode emits one `data-input-request` chunk with both spiked requests and no text; non-structured mode output is byte-for-byte the old text; result carries the parsed requests. Route: the stream persists two pending requests; a valid answer set is forwarded as `inputResponses` (assert the exact body Eve receives) and pending is cleared after; each rejection case from Review Focus 2 returns 400 with its cause and `continueEveSession` is never called; if Eve rejects the forward, pending stays; pending survives a fresh store read (simulating a restart by reading from the same store); an Eve 500 on continue for a relay run returns the offline error and keeps the run row.
- [ ] **Step 2: Run** `pnpm exec vitest run --config vitest.config.node.mjs tests/agent/eve-stream-adapter.test.ts tests/agent/eve-chat-route.test.ts tests/agent/eve-client-input.test.ts` — Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the same command — Expected: PASS.
- [ ] **Step 5: Commit** `feat(eve): structured needs-you questions with validated answers`.

### Task 7: Progress tool

**Files:**
- Create: `agent/tools/report_progress.ts`
- Modify: `lib/ai/eve/stream-adapter.ts`, `app/(chat)/api/eve-chat/route.ts`, `agent/tools/gap_analysis.ts`, `agent/tools/form_summary.ts`
- Test: `tests/agent/report-progress.test.ts`, `tests/agent/eve-stream-adapter.test.ts` (extend), `tests/agent/eve-chat-route.test.ts` (extend)

**Interfaces:**
- Consumes: `runProgressSchema`, `sanitizeProgress`, `store.setProgress`.
- Produces: Eve tool `report_progress` (input is `runProgressSchema`, output `{ ok: true }`, no side effects). `EVE_TOOL_NAME_MAP` maps it to `reportProgress`. When a relay run's stream carries an `actions.requested` call to it, the adapter returns `progress` (sanitized) and the route calls `store.setProgress`; invalid input is ignored, never stored. On relay sessions (use `relayCredentialState`) `gap_analysis` and `form_summary` return a refusal `{ success: false, error: 'Not available in extension runs: use ask_question for missing data and report_progress for status.' }` without running; their behavior otherwise is unchanged.

- [ ] **Step 1: Write failing tests**: the tool accepts a valid report and strips extra keys through its schema; rejects over-limit input; the adapter maps the tool name and surfaces sanitized `progress`; the route stores a valid report and does not store an invalid one; `gap_analysis` and `form_summary` refuse on a relay session (complete and partial credentials) and run normally with none.
- [ ] **Step 2: Run** `pnpm exec vitest run --config vitest.config.node.mjs tests/agent/report-progress.test.ts tests/agent/eve-stream-adapter.test.ts tests/agent/eve-chat-route.test.ts` — Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the same command plus `pnpm exec vitest run --config vitest.config.node.mjs tests/agent` — Expected: PASS.
- [ ] **Step 5: Commit** `feat(agent): report_progress tool and persistence`.

### Task 8: Extension-run guidance

**Files:**
- Create: `agent/skills/extension-run/SKILL.md`, `lib/relay/run-prompt.ts`
- Modify: `agent/instructions.md`, `agent/skills/browser-automation/browser-commands.md`, `app/(chat)/api/eve-chat/route.ts`
- Test: `tests/agent/run-prompt.test.ts`, `tests/agent/skills-guidance.test.ts`

**Interfaces:**
- Produces: `EXTENSION_RUN_MARKER = '[extension-run]'` and `buildExtensionRunMessage(text: string): string` (marker line, one sentence telling the agent to read the `extension-run` skill, then the original text). The route applies it to the first message of a relay run only. The skill states, for extension runs: ask for every known missing value in one `ask_question` batch using parallel calls; `select` options for choice fields; for sensitive fields ask with `allowFreeform: false` and exactly the options `provided` and `skip`, and never ask for or repeat the value; call `report_progress` after each section with names and counts only; before relying on cross-site iframe content, re-open the current URL once (`["open", "<current url>"]`) and then take a snapshot; address iframes with `frame @<ref>`, never a selector; do not use `gap_analysis`, `form_summary` or `check_submit_gate`; never click Submit. `agent/instructions.md` Forbidden Actions gains one explicit exception: the single re-open described in the extension-run skill, only before any field is filled. `browser-commands.md` replaces the `frame "#iframe"` selector example (agent-browser 0.33.2 cannot resolve it) with `frame @e<N>`.

- [ ] **Step 1: Write failing tests**: `buildExtensionRunMessage` output starts with the marker and ends with the original text unchanged; the route adds it on a relay run's first turn and not on follow-ups or web chats; a test reads the skill file and asserts it contains the required rules (the strings `allowFreeform: false`, `provided`, `report_progress`, `frame @`, the re-open rule) and that `browser-commands.md` no longer documents `frame "#iframe"`; `agent/instructions.md` still contains every pre-existing Forbidden Actions rule and the new exception references the skill.
- [ ] **Step 2: Run** `pnpm exec vitest run --config vitest.config.node.mjs tests/agent/run-prompt.test.ts tests/agent/skills-guidance.test.ts` — Expected: FAIL.
- [ ] **Step 3: Implement.** Match the front matter and tone of the existing skills in `agent/skills/`.
- [ ] **Step 4: Run** the same command — Expected: PASS.
- [ ] **Step 5: Commit** `feat(agent): extension-run guidance`.

### Task 9: Service-worker restart recovery

**Files:**
- Modify: `lib/relay/protocol.ts`, `relay/hub.ts`, `extension/src/bridge.ts`, `lib/relay/eve-relay-browser.ts`
- Test: `tests/relay/hub.test.ts` (extend), `tests/relay/e2e/bridge.e2e.test.ts` (extend), `tests/relay/eve-relay-browser.test.ts` (extend)

**Interfaces:**
- Produces: `ExtensionToRelay` gains `{ type: 'resumed' }`. When `resumeRun` rebuilds a run from `chrome.storage.session` and its socket registers, the extension sends `resumed` once. The hub, on `resumed` from the registered extension, closes the agent socket with reason `extension restarted` but keeps the run (a new agent attach must succeed and no tombstone is set). On the Eve side, a command failure that matches a closed or detached connection (`/closed|disconnect|not attached|ECONNRESET|EPIPE/i`) on a relay run starts a fresh agent-browser session name for the next command (`<cliSessionName>-r<n>`, counter kept in memory per run); that command is retried once only when its argv contains no `@e` ref, otherwise it returns `{ success: false, error: 'The browser connection was restored. Take a new snapshot before continuing.' }` without running. Stale element refs are never reused across a restart.

- [ ] **Step 1: Write failing tests**: hub: `resumed` from the registered extension closes the agent with `extension restarted`, the run survives and a new agent attaches and gets answers from the same extension; `resumed` from a pending or unregistered socket does nothing. Tool: a mocked closed-connection failure on `["click","@e3"]` returns the snapshot-again message and runs nothing; the same failure on `["snapshot","-i"]` is retried once with the next session name and returns its result; a non-connection failure is not retried; the counter advances per restart. E2E (Review Focus 6): start a run, fill a field, restart the extension's service worker (stop it through a browser-level CDP session, `ServiceWorker.stopAllWorkers`, and let a tab event wake it; if that cannot be made deterministic, restart it by calling the worker's own reset-and-`resumeRun` path from a test hook and say so in the report), then show that a new agent session snapshots, fills a second field with the value asserted from the page, and that the first agent connection was closed with `extension restarted`.
- [ ] **Step 2: Run** `pnpm extension:build && pnpm exec vitest run --config vitest.config.node.mjs tests/relay/hub.test.ts tests/relay/eve-relay-browser.test.ts tests/relay/e2e/bridge.e2e.test.ts` — Expected: FAIL.
- [ ] **Step 3: Implement.** Keep `chrome.debugger` attachment logic as is: after a restart the extension's `attached` is false and the next relay `attach` already detaches then attaches cleanly.
- [ ] **Step 4: Run** the same command — Expected: PASS.
- [ ] **Step 5: Commit** `feat(extension): recover a run after a service-worker restart`.

### Task 10: Verify Eve stores no credential

**Files:**
- Create: `scripts/check-workflow-secrets.ts`
- Test: `tests/agent/check-workflow-secrets.test.ts` (the pure scanner only)

**Interfaces:**
- Produces: `pnpm tsx scripts/check-workflow-secrets.ts <string>...` reads `WORKFLOW_POSTGRES_URL`, scans every text, jsonb and bytea column of the `workflow` schema, decompresses zstd frames inside bytea values (Node 24 `zlib.zstdDecompressSync`), and prints per table and column the number of rows containing each string (counts only, never row contents); exit code 1 when any given string is found. Export `scanBuffer(buffer: Buffer, needles: string[]): Map<string, number>` (decompression-aware) for the unit test. The spike's working reference is `/private/tmp/claude-502/-Users-kaylynvannorstrand-Documents-GitHub-labs-asp-form-filling/9c8edee9-166e-426d-a634-8cc814b21c97/scratchpad/pg-spike/deepsearch.mjs` if it still exists.

- [ ] **Step 1: Write the failing unit test**: `scanBuffer` finds a needle in a plain buffer, in a zstd-compressed buffer, in a buffer holding a compressed frame after a header, and returns zero for an absent needle; a control string present in the same buffer is found in all three forms.
- [ ] **Step 2: Run** `pnpm exec vitest run --config vitest.config.node.mjs tests/agent/check-workflow-secrets.test.ts` — Expected: FAIL.
- [ ] **Step 3: Implement** the scanner and the script.
- [ ] **Step 4: Run** the test — Expected: PASS.
- [ ] **Step 5: Manual verification (owner-approved, costs 3 model turns).** Start a throwaway local Postgres (Homebrew `postgresql@14`, localhost only), export `WORKFLOW_POSTGRES_URL`, `EVE_INTERNAL_SECRET` and a fake `RELAY_PUBLIC_URL`, run `pnpm tsx scripts/bootstrap-workflow-db.ts` then `pnpm eve:dev`, create an Eve session with headers `x-eve-relay-run` (a UUID), `x-eve-relay-url`, the internal secret, AND the legacy `x-eve-relay-token: SPIKE-SENTINEL-TOKEN-7f3a9c1e5b2d8046`, run two turns (the second asks for one `browser` call), then run the script with the sentinel and the internal secret value. Expected: exit 0, zero rows contain either string, while a control string from the first user message IS found (proving the scan can see inside the compressed values). Record the output in the PR. Stop Eve, stop and delete the throwaway Postgres.
- [ ] **Step 6: Update the docs.** In `docs/browser-extension-todo.md` check off the plan 2 items this plan completed (continuity, `ask_question` Needs You, progress contract, the Postgres-world verification, the iframe guidance, service-worker resume) and move the token-refresh and per-role-minting items out of plan 4 (the agent leg no longer needs them). Add `docs/changes/2026-10-26-add-agent-loop.md` in the `change-docs` format (30 to 80 lines, from the actual diff).
- [ ] **Step 7: Commit** `test(agent): script to verify Eve stores no credentials` (scanner and script) and `docs: record plan 2 changes` (tracker and change doc).

---

## Out of Scope

- The side panel, run start, the user-facing token route, and enabling any flag (plan 4). Token refresh and per-role minting are no longer needed for the agent leg because it fetches on demand.
- Sensitive-value classification, redaction, and refusing `eval`/`get` (plan 3).
- The first user message is stored as plaintext in `workflow_runs.attributes` for every Eve chat today, including participant details. It predates this plan; decide separately whether to strip it (plan 3 strips sensitive fields from the record before the message is built).
- `pnpm eve:dev` installs a `microsandbox` dev dependency on its own and edits `package.json`; revert it after a dev run (`git checkout package.json pnpm-lock.yaml`).

## Self-Review Notes

- **Spec coverage:** continuity in Postgres (Tasks 2, 3); `ask_question` as Needs You with structured, validated answers (Tasks 1, 6, 8); structured progress (Tasks 1, 7); credentials never in Eve (Tasks 4, 5, 10); iframe re-open guidance with the instruction carve-out and `frame @ref` fix (Task 8); service-worker restart (Task 9); the todo item to verify Eve's Postgres world (Task 10). Design call #6 (continuity scope) is narrowed to `eve-chat`, the only route that uses it.
- **Decision for the owner:** the extension run is identified by the chat id (`EveRun.chatId`), with no foreign key to `Chat` until plan 4 decides whether runs create chat rows.
- **Largest risk to settle early:** Task 9's end-to-end restart test may need the hook fallback described there; the report must say which path ran.
