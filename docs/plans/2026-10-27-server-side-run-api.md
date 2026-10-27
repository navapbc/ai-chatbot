# Server-Side Run API Implementation Plan (Plan 4a)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the app's API safe to put a side panel in front of: runs belong to users, the opening message is built on the server, participants come from an API, and the Jev field classifier and a model/site config are reachable.

**Architecture:** Every new route shares one guard (feature flag, signed-in, not a guest). A relay run is created by the existing `eve-chat` first turn, which now takes a participant reference instead of a client-written message, so the unstripped record never transits the browser. Relay tokens carry the user id (`sub`); the token route reads the origin from the stored run, and the hub refuses to pair an agent and an extension whose `sub` differ. The side panel itself is plan 4b.

**Tech Stack:** Next.js route handlers, Drizzle/Postgres (Neon), zod, the existing relay token and Eve run store, Jev (`lib/jev`), vitest node project (`vitest.config.node.mjs`).

**Spec:** `docs/specs/2026-10-23-browser-extension-design.md` (Security Requirements 4 and 6, Next app additions, Models). Tracker: `docs/browser-extension-todo.md`, section "Plan 4: Side Panel And App API".

## Global Constraints

- Node 24: `export PATH=/Users/kaylynvannorstrand/.nvm/versions/node/v24.20.0/bin:$PATH`. `pnpm` only. No new dependencies without asking.
- Tests: `pnpm exec vitest run --config vitest.config.node.mjs <path>` for `tests/agent` and `tests/relay`. Lint with `pnpm exec biome lint <files>`; never `pnpm lint`, never `biome format --write` on a directory.
- Never read `.env*` files or print secret values. Migrations are generated with `pnpm db:generate`, never hand-edited, and not applied by this plan (the owner applies them).
- Every new route answers a refused flag with a bodyless 404, and errors never include env values, tokens, origins from storage, or record contents. Errors use `ChatSDKError` (`lib/errors.ts`).
- Sensitive values never appear in logs, errors, spans or responses. Log lengths and counts only.
- The web app and the legacy chat route behave exactly as before: every change is gated on a relay run or a new route.
- Relay routes require `RELAY_RUNS_ENABLED === '1'`. `RELAY_TOKEN_ENDPOINT_ENABLED` stays unset in deployment config until this plan has merged and the owner decides.

## Decisions For Owner

These are defaults chosen so the work can proceed; each is one line to change.

- **P4a-a Guests.** A guest is a user whose email matches `guest-<8 chars>@preview.local` (`app/(auth)/auth.ts`). Guests are refused on every relay route (403).
- **P4a-b Run limit.** At most `RELAY_MAX_RUNS_PER_HOUR` (default 20) relay runs created per user per rolling hour, counted from `EveRun.createdAt`.
- **P4a-c Participant exposure.** The participant API returns a summary (record id, display name, county), never a full record. The full record exists only inside the server when it builds the opening message.
- **P4a-d Models.** The allowed list is the env var `RELAY_ALLOWED_MODELS` (comma-separated ids from `MODEL_MAP`), default `claude-opus-5`. Relay runs ignore the dev-only override rule: a requested model outside the list is a 400.
- **P4a-e Sites.** The Home site list is `config/relay-sites.json` (checked in, array of `{ id, name, url }`, https only, initially empty). Which sites go in it is still an owner question.
- **P4a-f Entered details.** Caseworker-entered participant details are a flat object of strings (at most 40 keys, 500 characters each). A key that the field classifier calls sensitive is refused, not stripped.

## Review Focus

- A guest session, or a signed-in user without a relay run, calling any new route: expect 403 / bodyless 404, never data.
- A second user creating a run with a chat id the first user already holds (squatting), and a leaked run id: expect no token for the squatter's row and none for the victim from it.
- Entered details with a sensitive key (`ssn`, `bank_account`), a nested object, a 10 MB body or 100 keys: expect a 400, nothing forwarded to Eve.
- Participant search with `%`, `_`, an empty string, a 5 KB string: expect a literal match or an empty list, never a full-table scan returned.
- A token minted for user A's run and presented by an extension whose agent token was minted for user B: expect the hub to refuse the second attach.
- Classify requests with 0, 1 and 51 descriptors, descriptors carrying a `value` or `url` key, and Jev failing: expect a 400, stripped input, and `sensitive: true` on failure.

## File Structure

- Create `lib/relay/route-guard.ts`: `guardRelayRoute(): Promise<{ ok: true; userId: string } | { ok: false; response: Response }>` and `isGuestEmail`.
- Create `lib/data/participant-source.ts`: the `ParticipantSource` interface, the static and database implementations.
- Create `app/(chat)/api/participants/route.ts` and `app/(chat)/api/participants/[recordId]/route.ts`.
- Create `lib/relay/run-input.ts`: validation of the first-turn `relay` field and the opening message build.
- Create `lib/relay/tool-input.ts`: client-stream redaction of tool inputs.
- Create `app/(chat)/api/relay/classify-field/route.ts` and `app/(chat)/api/relay/config/route.ts`; `lib/relay/config.ts`; `config/relay-sites.json`.
- Modify `lib/db/schema.ts`, `lib/db/queries.ts`, `lib/ai/eve/run-store.ts`, `lib/ai/eve/db-run-store.ts`, `app/(chat)/api/eve-chat/route.ts`, `app/(chat)/api/relay/token/route.ts`, `app/(chat)/api/relay/agent-token/route.ts`, `lib/relay/token.ts`, `relay/server.ts`, `relay/hub.ts`, `lib/ai/eve/stream-adapter.ts`, `lib/ai/eve/loopback-attributes.ts`, `lib/ai/eve/model-map.ts`.
- Tests go in `tests/agent/` (routes, stores, adapter) and `tests/relay/` (token, hub, server).

---

### Task 1: Route guard

**Files:**
- Create: `lib/relay/route-guard.ts`
- Test: `tests/agent/relay-route-guard.test.ts`

**Interfaces:**
- Produces: `isGuestEmail(email: string | null | undefined): boolean`; `guardRelayRoute(): Promise<{ ok: true; userId: string } | { ok: false; response: Response }>`. Order: flag off → bodyless 404; no session user id → `ChatSDKError('unauthorized:chat')`; guest → `ChatSDKError('forbidden:chat')`.

- [ ] **Step 1: Write failing tests** `isGuestEmail('guest-1a2b3c4d@preview.local') === true`; false for `guest-1a2b3c4d@example.com`, `someone@preview.local`, `Guest-1a2b3c4d@preview.local` is true (case-insensitive), `null`, and `''`. For `guardRelayRoute`, mock `@/app/(auth)/auth`: flag unset → status 404 with null body, and `auth` not called; no session → 401; guest session → 403; regular session → `{ ok: true, userId }`.
- [ ] **Step 2: Run, expect FAIL** (module missing).
- [ ] **Step 3: Implement** both exports in `lib/relay/route-guard.ts` (pattern `/^guest-[0-9a-f]{8}@preview\.local$/i`).
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** `feat(relay): add a shared guard for relay routes`.

### Task 2: Participant source and table

**Files:**
- Modify: `lib/db/schema.ts` (table `Participant`: `recordId` text primary key, `displayName` text not null, `county` text not null, `searchText` text not null, `record` jsonb not null typed `Participant`, `createdAt`), `lib/db/queries.ts` (`searchParticipantRows({ query, limit })`, `getParticipantRow(recordId)`)
- Create: `lib/data/participant-source.ts`, `scripts/seed-participants.ts`, a generated migration under `lib/db/migrations/`
- Test: `tests/agent/participant-source.test.ts`

**Interfaces:**
- Produces: `ParticipantSummary = { recordId: string; name: string; county: string }`; `interface ParticipantSource { search(query: string, limit: number): Promise<ParticipantSummary[]>; get(recordId: string): Promise<Participant | undefined> }`; `createStaticParticipantSource(records: Participant[]): ParticipantSource`; `createDbParticipantSource(): ParticipantSource`; `getParticipantSource(): ParticipantSource` (database-backed, replaceable with `setParticipantSourceForTests`); `summarize(p: Participant): ParticipantSummary`; `escapeLike(q: string): string`.
- Consumes: `PARTICIPANTS`, `getParticipantName` from `lib/data/participants.ts`.

- [ ] **Step 1: Write failing tests** against the static source: `search('celeste')` finds record `339619` case-insensitively; `search('339')` matches by record id prefix; `search('')` returns `[]`; limit is clamped to 1..20; `search('%')` and `search('_')` match nothing unless the literal character is present (assert via `escapeLike('50%_x') === '50\\%\\_x'`); `get(' 339619 ')` trims; `summarize` output has exactly the keys `recordId`, `name`, `county`.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** the source. The database implementation queries `searchText ILIKE escapeLike(q)||'%'` or `recordId` prefix, ordered by `displayName`, with the clamped limit; `searchText` is lowercase `first middle last recordId`. `scripts/seed-participants.ts` upserts every `PARTICIPANTS` entry (run with `dotenv -e .env.local -- tsx scripts/seed-participants.ts`; the plan does not run it).
- [ ] **Step 4: Run `pnpm db:generate`**, confirm the new migration only creates `Participant`, run the tests, expect PASS.
- [ ] **Step 5: Commit** `feat(data): add a participant source backed by a Neon table`.

### Task 3: Participant routes

**Files:**
- Create: `app/(chat)/api/participants/route.ts`, `app/(chat)/api/participants/[recordId]/route.ts`
- Test: `tests/agent/participants-route.test.ts`

**Interfaces:**
- Consumes: `guardRelayRoute`, `getParticipantSource`, `summarize`.
- Produces: `GET /api/participants?q=&limit=` → `{ participants: ParticipantSummary[] }` (limit default 10, max 20, `q` capped at 100 characters, longer → 400); `GET /api/participants/:recordId` → `ParticipantSummary` or `ChatSDKError('not_found:api')` style 404 with a body. Both set `Cache-Control: no-store`.

- [ ] **Step 1: Write failing tests** with a swapped static source: guard refusals pass through; `q` of 101 characters → 400; the response of the detail route has only the three summary keys (assert a record containing `calworks_id` never leaks it, by `JSON.stringify` not containing `A100121`); unknown id → 404.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** both handlers.
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** `feat(api): add participant search and summary routes`.

### Task 4: Run ownership in the store

**Files:**
- Modify: `lib/db/schema.ts` (partial unique index `EveRun_chatId_relay_unique` on `chatId` where `relayOrigin` is not null), `lib/db/queries.ts` (`getEveRunByChatId` filters `relayOrigin IS NOT NULL`; new `countRecentRelayRuns({ userId, since })`), `lib/ai/eve/run-store.ts` and `lib/ai/eve/db-run-store.ts`, plus the memory store, a generated migration
- Test: `tests/agent/eve-run-store.test.ts` (extend)

**Interfaces:**
- Produces: `EveRunStore.getByChatId` now means "the relay run with this id" and returns `undefined` for a web row; `EveRunStore.countRecentRelayRuns(userId: string, since: Date): Promise<number>`; the second relay `create` with the same `chatId` (any user) throws `EveRunExistsError`.
- The `(userId, chatId)` constraint-name check in `db-run-store.ts` must also treat the new index as a duplicate.

- [ ] **Step 1: Write failing tests** on the memory store (the contract both stores share): a web row for user B on chat X no longer hides user A's relay run on X from `getByChatId`; two users creating a relay run on X → second throws `EveRunExistsError`; `countRecentRelayRuns` counts only relay rows of that user newer than `since`.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** in all three stores, then `pnpm db:generate`; confirm the migration adds only the index.
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** `fix(relay): one relay run per chat id, looked up without web rows`.

### Task 5: Server-built opening message

**Files:**
- Create: `lib/relay/run-input.ts`
- Modify: `app/(chat)/api/eve-chat/route.ts`, `lib/ai/eve/model-map.ts` (add `isRelayModelAllowed`)
- Test: `tests/agent/run-input.test.ts`, `tests/agent/eve-chat-route.test.ts` (extend)

**Interfaces:**
- Produces: `parseRelayStart(raw: unknown): { ok: true; origin: string; participant: { recordId: string } | { details: Record<string, string> }; model?: string } | { ok: false; reason: string }` (origin exactly as today: `normalizeOrigin(origin) === origin`); `buildRelayOpeningMessage(args: { origin: string; participant: Participant | Record<string, string> }): string` using `buildApplicationPrompt` with the record passed through `stripSensitiveFields`, target = origin.
- Route contract: a relay first turn takes `relay: { origin, participant, model? }` and must not carry `message` (a present `message` → 400 `message_not_allowed`). The route resolves `recordId` through `getParticipantSource().get` (unknown id → 400 `unknown_participant`), builds the message, runs it through `relayText`, and sends it to Eve. Follow-up turns and answer turns are unchanged. The first turn also checks, before creating anything: the per-user run limit (P4a-b, `ChatSDKError('rate_limit:chat')`), the model against the allowlist (P4a-d, 400 `model_not_allowed`), and details per P4a-f (400 `sensitive_detail` / `details_too_large`).
- Consumes: Task 2 source, Task 4 `countRecentRelayRuns`.

- [ ] **Step 1: Write failing tests.** `run-input`: details key `ssn` → refused; 41 keys → refused; a nested object value → refused; a recordId the source lacks → handled in the route test. Route tests (existing harness with a mocked Eve client and the memory store): the opening message sent to Eve contains the participant's name and not `calworks_id` or its value; a body with both `relay` and `message` → 400; the 21st run in an hour → 429; model `gpt-5.4` with default allowlist → 400; with `RELAY_ALLOWED_MODELS=claude-opus-5,claude-sonnet-5` and `claude-sonnet-5` → accepted. Update the existing relay first-turn tests to the new body shape.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.** The top-of-handler check (`!isUserMessage || !text`) must not reject a relay first turn that has no message; restructure so a relay first turn is validated by `parseRelayStart` instead. Keep every refusal before `createEveSession`.
- [ ] **Step 4: Run the route file and `tests/agent`, expect PASS.**
- [ ] **Step 5: Commit** `feat(relay): build the opening message on the server`.

### Task 6: User-bound tokens

**Files:**
- Modify: `lib/relay/token.ts` (`sub` claim, required), `app/(chat)/api/relay/token/route.ts`, `app/(chat)/api/relay/agent-token/route.ts`, `relay/server.ts`, `relay/hub.ts`
- Test: `tests/relay/token.test.ts`, `tests/relay/token-route.test.ts`, `tests/agent/` agent-token route test, `tests/relay/hub.test.ts`, `tests/relay/server.test.ts`

**Interfaces:**
- Produces: `RelayClaims.sub: string` (the user id); `mintRelayToken` requires it and throws on an empty secret, a secret under 32 characters, an empty `sub`, or a non-https origin; `verifyRelayToken` returns null when `sub` is missing or empty. `POST /api/relay/token` body is `{ runId }` only; it returns `{ extensionToken, relayUrl, expiresAt }` (no agent token). The origin and owner come from `getRunStore().get(userId, runId)` (relay row required, else bodyless 404). `Hub.attachExtension` and `Hub.attachAgent` take a `sub` argument; the first attach for a run records it and a later attach with a different `sub` is refused with the same refusal path as an origin mismatch.
- The internal agent-token route mints with `sub: run.userId` (so `getByChatId` results must expose `userId`, which they already do).

- [ ] **Step 1: Write failing tests.** Token: round trip keeps `sub`; verify of a token minted without one → null. Token route: no session → 401; guest → 403; flag `RELAY_TOKEN_ENDPOINT_ENABLED` unset → bodyless 404 before auth; a run owned by another user → bodyless 404; response has no `agentToken` key; the minted token's origin equals the stored origin even when the body contains a different `origin`. Hub: extension `sub` A then agent `sub` B → agent refused and the run stays usable by A's agent. Server: a token without `sub` is rejected at upgrade.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.** Update every existing caller and test fixture that mints tokens.
- [ ] **Step 4: Run `tests/relay` (unit, `--exclude 'tests/relay/e2e/**'`) and `tests/agent`, expect PASS.** Then `pnpm extension:build && pnpm test:relay e2e/bridge.e2e.test.ts` to confirm the e2e harness mints valid tokens.
- [ ] **Step 5: Commit** `feat(relay): bind run tokens to users and check the pairing`.

### Task 7: Small route and secret hardening

**Files:**
- Modify: `lib/ai/eve/loopback-attributes.ts` (or wherever `EVE_INTERNAL_SECRET` is read; find it with `grep -rn EVE_INTERNAL_SECRET agent lib app scripts`), `app/(chat)/api/relay/agent-token/route.ts`, `app/(chat)/api/relay/token/route.ts`
- Test: `tests/agent/eve-channel-auth.test.ts`, the agent-token route test

**Interfaces:**
- Produces: an `EVE_INTERNAL_SECRET` shorter than 32 characters is treated as unset by every consumer; the agent-token route reads at most 2 KB of body (more → 400 before parsing); every error response from both token routes carries `Cache-Control: no-store`; a misconfigured secret or relay URL logs one value-free line (`[relay-token] misconfigured`).

- [ ] **Step 1: Write failing tests** for each bullet (31-character secret refused, 33 accepted; a 3 KB body → 400; headers on the 400, 401, 403, 404 and 500 paths).
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** `fix(relay): bound the agent-token body and require a long internal secret`.

### Task 8: Runtime field classification route

**Files:**
- Create: `app/(chat)/api/relay/classify-field/route.ts`
- Test: `tests/agent/classify-field-route.test.ts`

**Interfaces:**
- Consumes: `guardRelayRoute`, run store `get(userId, runId)`, `FieldDescriptor`, `sensitiveFieldQuestions`, `sensitiveFieldIsSensitive`, `describeFieldForJev`, `askJev`, `isJevEnabled('sensitive-field')`.
- Produces: `POST /api/relay/classify-field` body `{ runId, fields: FieldDescriptor[] }` (1 to 50 fields) → `{ sensitive: boolean[] }` in order. The run must be a relay run owned by the caller (else bodyless 404). Each descriptor is passed through `describeFieldForJev` before the call, so extra keys never leave the process. Jev off, failed or timed out → `true` for that field. Fields are classified in parallel, at most 8 at a time. Never logs descriptors.

- [ ] **Step 1: Write failing tests** with `askJev` mocked: three fields give three booleans in order; a descriptor with `value: '123-45-6789'` reaches `askJev` without `value` and with digits as `9`; `askJev` returning `{ ok: false }` → `true`; 0 or 51 fields → 400; another user's run → 404; flag off → every entry `true`.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.** Update the Jev section of `CLAUDE.md` in Task 11, not here.
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** `feat(jev): add the runtime sensitive-field classification route`.

### Task 9: Client-stream redaction of tool inputs

**Files:**
- Create: `lib/relay/tool-input.ts`
- Modify: `lib/ai/eve/stream-adapter.ts`
- Test: `tests/agent/tool-input.test.ts`, `tests/agent/eve-stream-adapter.test.ts` (extend)

**Interfaces:**
- Produces: `redactToolInput(toolName: string, input: unknown): unknown`. For `browser`: keep `argv[0]` (the verb) and, when `argv[1]` matches `^@e\d+$`, keep it; replace every later element with `'[hidden]'`; `open` keeps only the origin and path of its URL argument (no query, fragment or userinfo); a non-array `argv` → `{}`. For `ask_question` and `resolve_field_value`: `{}`. `report_progress` is already sanitized and passes through. Any other tool: `{}`.
- The adapter applies it to every `tool-input-available` it writes (including the synthesized empty one) when `ctx.structuredInput` is true, and not otherwise.

- [ ] **Step 1: Write failing tests.** `["fill","@e3","123-45-6789"]` → `["fill","@e3","[hidden]"]`; `["open","https://a.gov/x?ssn=1#f"]` → `["open","https://a.gov/x"]`; `["type","@e2","abc"]`; `{ argv: 'x' }` → `{}`; unknown tool → `{}`. Adapter: with `structuredInput` the streamed `fill` value never appears in the writer output; without it the input passes through unchanged.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** `fix(relay): hide tool inputs in the client stream on relay runs`.

### Task 10: Config route

**Files:**
- Create: `lib/relay/config.ts`, `config/relay-sites.json` (`[]`), `app/(chat)/api/relay/config/route.ts`
- Test: `tests/agent/relay-config.test.ts`

**Interfaces:**
- Produces: `getRelayConfig(env?: NodeJS.ProcessEnv): { sites: { id: string; name: string; url: string }[]; models: { id: string }[]; defaultModel: string }`. Sites with a non-https URL or a duplicate id are dropped. `models` is `RELAY_ALLOWED_MODELS` filtered to ids present in `MODEL_MAP`, default `['claude-opus-5']`; `defaultModel` is the first entry. `GET /api/relay/config` returns it behind the guard with `no-store`. `isRelayModelAllowed` (Task 5) reads the same list.

- [ ] **Step 1: Write failing tests** (http site dropped; duplicate dropped; unknown model id dropped; empty env → default list; guard refusals).
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** `feat(relay): serve the site list and allowed models`.

### Task 11: Docs and tracker

**Files:**
- Modify: `CLAUDE.md` (routes list, Extension runs paragraph, Jev section: the runtime call site now exists), `docs/browser-extension-todo.md` (tick the items this plan closed; move the rest to 4b), `agent/README.md` only if it describes the token route
- Create: `docs/changes/2026-10-27-add-server-side-run-api.md` (use the change-docs skill)

- [ ] **Step 1: Update the docs** from the diff, not from memory. Tracker items closed here: user binding, public token route mints extension token only, chat-id squatting, `EVE_INTERNAL_SECRET` length, bounded agent-token body, mint-side claim validation, `no-store` on errors, server-built opening message, adapter redaction of tool inputs, runtime Jev route (client side is 4b), participant API. Left for 4b: the panel, saved details, typing sensitive values and `addKnownValue`, the "Allow this site?" flow, `startRun` serialization, service-worker hook gating, Stop while reconnecting, detach decision, calling the classify route from the worker.
- [ ] **Step 2: Run the whole node suite** `pnpm exec vitest run --config vitest.config.node.mjs tests/agent tests/relay --exclude 'tests/relay/e2e/**'`, expect PASS, then the redaction e2e once.
- [ ] **Step 3: Commit** `docs: document the server-side run API`.

## Self-Review Notes

- Spec coverage: Security Requirement 4 (user-scoped tokens, origin from the stored run) is Task 6; Requirement 6 (server-side record handling) is Task 5; "Participant API", "Configuration endpoints" and the model allowlist are Tasks 2, 3, 5, 10. Chat panel, saved details and the worker-side classifier call are plan 4b by design.
- The `relayOrigin` index (Task 4) must land before the token route change (Task 6) so the lookup it relies on is unambiguous.
- The route restructuring in Task 5 is the riskiest edit: the 571-line handler has an early `!text` rejection and a first-turn branch that both assume a client message. Keep the web path byte-for-byte.
