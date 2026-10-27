# Run Plumbing Implementation Plan (Plan 4b)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Everything the side panel (plan 4c) needs underneath it: extension-to-app authentication, a run that survives the panel closing, a server-side Stop, worker-side Jev classification, and the worker lifecycle fixes the tracker lists. No panel UI.

**Architecture:** The extension authenticates with the caseworker's app session cookie, sent from extension contexts that hold host permission for the app origin; relay routes additionally check the request's `Origin`. A relay run gets an `endedAt` state on its `EveRun` row; Stop marks it, cancels the Eve turn, and every token and turn path refuses an ended run. The eve-chat route keeps consuming Eve's stream and persisting progress, pending questions and the cursor after the client disconnects, so a closed panel loses nothing and a reopened one rebuilds from a new run-state route. The worker gains a classifier hook that calls `classify-field` and can only add masking to the deterministic verdict.

**Tech Stack:** Chrome MV3 extension (esbuild bundle), Next route handlers, Drizzle/Postgres, the existing relay and redactor modules, Playwright Chromium e2e (`pnpm extension:build && pnpm test:relay`), vitest node project.

**Spec:** `docs/specs/2026-10-23-browser-extension-design.md` (Security Requirements 4 and 9, Extension, "Calls Made in Review" items 1 and 7). Tracker: `docs/browser-extension-todo.md`, "Plan 4: Side Panel And App API" and "Plan 4a follow-ups". Previous plan: `docs/plans/2026-10-27-server-side-run-api.md`.

## Global Constraints

- Node 24: `export PATH=/Users/kaylynvannorstrand/.nvm/versions/node/v24.20.0/bin:$PATH`. `pnpm` only. No new dependencies without asking.
- Tests: `pnpm exec vitest run --config vitest.config.node.mjs <path>` for `tests/agent` and `tests/relay`. Extension and redaction changes also run `pnpm extension:build && pnpm test:relay e2e/bridge.e2e.test.ts e2e/redaction.e2e.test.ts`. Lint with `pnpm exec biome lint <files>`; never `pnpm lint`; never `biome format --write` on a directory.
- Never read `.env*` files or print secret values. Migrations are generated with `pnpm db:generate`, never hand-edited, and applied by the container at start, so each new migration must be safe on a populated production table.
- Sensitive values never appear in logs, errors, spans, responses, `chrome.storage` or the worker's messages to any page. Log ids and counts only.
- Every route answers a refused flag with a bodyless 404 and sets `Cache-Control: no-store` on every response, guard refusals included.
- The web app and the legacy chat route behave exactly as before; every change is gated on a relay run or a new route. `RELAY_RUNS_ENABLED` and `RELAY_TOKEN_ENDPOINT_ENABLED` stay unset in deployment config.
- Extension code stays free of `server-only`; shared modules under `lib/relay/` stay importable by the extension build and the Eve bundle.

## Decisions For Owner

Defaults chosen so the work can proceed; each is a one-line change.

- **P4b-a Auth.** Cookie auth, as chosen: the panel and worker call the app with `credentials: 'include'`. The app origin is a build-time constant (`APP_ORIGIN`, https, exact) written into `host_permissions`. If the spike (Task 1) shows Lax cookies are not sent from the extension, the NextAuth session cookie becomes `SameSite=None; Secure` only while `RELAY_RUNS_ENABLED` is `1`.
- **P4b-b Origin check.** When `RELAY_EXTENSION_ORIGIN` is set (`chrome-extension://<id>`, derived from `extension/EXTENSION_ID`), every relay route and the participant routes refuse a request whose `Origin` header is present and different, and every state-changing relay request must carry it. Unset means no check (local development and curl).
- **P4b-c Ended runs.** Stop sets `EveRun.endedAt`; an ended run never mints a token, never continues a turn, and its state route reports `ended: true`. There is no "resume after Stop"; the caseworker starts a new run.
- **P4b-d Detach.** Any `chrome.debugger` detach whose reason is not `canceled_by_user` and that leaves the tab unattached for 5 seconds without a re-attach ends the run with reason `debugger-detached`; navigation-driven detaches that the bridge re-attaches do not.
- **P4b-e Jev outage.** When `classify-field` is unreachable or slow the worker keeps the deterministic verdict for that field (Jev only adds masking), and records one value-free counter the panel can show. The deterministic classifier already fails toward sensitive for unknown shapes.
- **P4b-f Allow-this-site.** A tab that leaves the origin still ends the run in this plan (current behavior). The approval flow needs panel UI and moves to plan 4c; this plan only adds the worker message `allowOrigin` and the bridge state it needs.

## Review Focus

- A relay route called with a wrong or missing extension `Origin`, a guest cookie, or no cookie: expect a refusal that reveals nothing.
- Stop pressed twice, during a turn, while the socket is reconnecting, and after the run already ended: expect one cancel, a persisted `endedAt`, and no token afterwards.
- The panel closing mid-turn while the agent asks a question: expect the pending question persisted, answerable after reopen, and no double-forward.
- Two `startRun` calls at once and a service-worker restart between them: expect one run, one socket.
- A `classify-field` call that times out, returns a malformed body, or is made before sign-in: expect the deterministic verdict, no value in the request, no crash.
- A sensitive value registered through `addKnownValue` followed by a worker restart: expect the value gone from worker memory and a clear signal that it must be re-entered, never a silent unmasked field.

## File Structure

- Create `lib/relay/request-origin.ts` and extend `lib/relay/route-guard.ts`: extension-origin check.
- Create `app/(chat)/api/relay/runs/[runId]/route.ts` (state) and `app/(chat)/api/relay/runs/[runId]/stop/route.ts` (stop).
- Modify `lib/db/schema.ts`, `lib/db/queries.ts`, `lib/ai/eve/run-store.ts`, `lib/ai/eve/db-run-store.ts` (`endedAt`), `lib/ai/eve/eve-client.ts` (`cancelEveTurn`), `app/(chat)/api/eve-chat/route.ts` (ended runs, background consumption), `app/(chat)/api/relay/token/route.ts`, `app/(chat)/api/relay/agent-token/route.ts`.
- Create `extension/src/app-api.ts` (typed fetch client), `extension/src/jev-classifier.ts` (worker classifier client), `extension/src/config.ts` (build constants).
- Modify `lib/relay/redactor.ts` (classifier hook), `extension/src/bridge.ts`, `extension/src/service-worker.ts`, `extension/manifest.json`, `package.json` (`extension:build`), `app/(auth)/auth.ts` (only if Task 1 requires it).
- Tests: `tests/agent/` (routes, stores, eve-client), `tests/relay/` (redactor, bridge units), `tests/relay/e2e/` (auth spike, lifecycle, classification, restart).

---

### Task 1: Authentication spike

**Files:**
- Create: `tests/relay/e2e/app-auth.e2e.test.ts`, `tests/relay/e2e/fixtures/cookie-server.ts` (a tiny local HTTP server on 127.0.0.1)
- Modify: `extension/manifest.json`, `package.json` (`extension:build` writes `host_permissions` from `APP_ORIGIN`), create `extension/src/config.ts`

**Interfaces:**
- Produces: `APP_ORIGIN` (build-time string, defined with esbuild `--define`), `host_permissions: [APP_ORIGIN + '/*']` in the built manifest; a documented answer in the test's header comment to "does a `SameSite=Lax` cookie reach the app from the extension worker and from an extension page?".

- [ ] **Step 1: Write the e2e** that loads the built extension in Chromium, serves a local app that sets a session cookie (`SameSite=Lax`, then `SameSite=None; Secure` over https if the harness already has certificates, else `SameSite=None` on localhost) and echoes whether the cookie arrived on `fetch(url, { credentials: 'include' })` from (a) the service worker and (b) an extension page. Assert and record the observed behavior for each cookie type.
- [ ] **Step 2: Run, expect FAIL** (manifest has no host permissions yet).
- [ ] **Step 3: Implement** `config.ts` exporting `APP_ORIGIN` (from `process.env.APP_ORIGIN` at build time, default `http://127.0.0.1:3000` only when `APP_ORIGIN_LOCAL=1`; an unset value fails the build) and the manifest write in the build script. Do not hand-edit `extension/dist`.
- [ ] **Step 4: Run, expect PASS.** If Lax cookies are not sent, add to the plan ledger `Ruling: P4b-a requires SameSite=None` and Task 2 gains the `auth.ts` change; if they are sent, Task 2 leaves `auth.ts` alone.
- [ ] **Step 5: Commit** `feat(extension): grant app host permission and prove cookie auth`.

### Task 2: Extension origin check on relay routes

**Files:**
- Create: `lib/relay/request-origin.ts`
- Modify: `lib/relay/route-guard.ts`, `app/(auth)/auth.ts` (only per Task 1's result), `extension/EXTENSION_ID` is read, not changed
- Test: `tests/agent/relay-request-origin.test.ts`, extend `tests/agent/relay-route-guard.test.ts`

**Interfaces:**
- Produces: `extensionOrigin(env?): string | null` (`chrome-extension://` plus the trimmed `RELAY_EXTENSION_ORIGIN` id form; any other shape is treated as unset and logged once value-free); `guardRelayRoute(request?: Request, opts?: { requireOrigin?: boolean })`. With the env set: a present `Origin` that differs -> `ChatSDKError('forbidden:chat')` (403); `requireOrigin` and no `Origin` -> 403; with it unset behavior is unchanged. All existing callers keep working with no argument; routes that mutate state pass `requireOrigin: true`.
- Consumes: Task 1's host permission (so the worker's requests carry `Origin: chrome-extension://<id>`).

- [ ] **Step 1: Write failing tests**: env unset -> unchanged for every case; env set with a matching origin -> ok; mismatched origin -> 403 with no-store and no body detail; missing origin on a `requireOrigin` call -> 403; a malformed env value -> treated as unset; the origin check runs after the flag check and before auth (so an unauthenticated foreign origin learns nothing beyond 403).
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** and thread `request` plus `requireOrigin: true` through the POST routes (`classify-field`, token, the new stop route) and `request` through the GET routes (participants, config, state).
- [ ] **Step 4: Run `tests/agent` and the relay unit suite, expect PASS.**
- [ ] **Step 5: Commit** `feat(relay): check the extension origin on relay routes`.

### Task 3: Ended runs in the store

**Files:**
- Modify: `lib/db/schema.ts` (`endedAt` timestamp, nullable), `lib/db/queries.ts` (`endEveRun({ userId, chatId })` idempotent, returns whether this call ended it), `lib/ai/eve/run-store.ts`, `lib/ai/eve/db-run-store.ts`; a generated migration
- Test: `tests/agent/eve-run-store.test.ts` (extend)

**Interfaces:**
- Produces: `EveRunRecord.endedAt: Date | null`; `EveRunStore.end(userId, chatId): Promise<boolean>` (true only for the call that set it; a second call returns false and leaves the first time). `create` starts with `endedAt: null`. The migration only adds a nullable column (safe on a populated table).

- [ ] **Step 1: Write failing tests** on the memory store: `end` returns true then false; the stored time does not move; `get` and `getByChatId` expose `endedAt`; `end` on an unknown run returns false and creates nothing.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** in both stores; DB uses `UPDATE ... SET endedAt = now() WHERE endedAt IS NULL RETURNING`. Run `pnpm db:generate`; confirm the migration only adds the column.
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** `feat(relay): record when a relay run ends`.

### Task 4: Run state and server-side Stop

**Files:**
- Create: `app/(chat)/api/relay/runs/[runId]/route.ts`, `app/(chat)/api/relay/runs/[runId]/stop/route.ts`
- Modify: `lib/ai/eve/eve-client.ts` (`cancelEveTurn`), `app/(chat)/api/relay/token/route.ts`, `app/(chat)/api/relay/agent-token/route.ts`, `app/(chat)/api/eve-chat/route.ts`
- Test: `tests/agent/relay-run-state.test.ts`, `tests/agent/relay-run-stop.test.ts`, extend the token and agent-token route tests and `tests/agent/eve-chat-route.test.ts`

**Interfaces:**
- Produces: `GET /api/relay/runs/:runId` -> `{ runId, origin, ended: boolean, progress: RunProgress | null, pending: PendingInputRequest[], updatedAt }` for the owner of a relay row, else a bodyless 404; no sensitive data beyond what the stream already carried. `POST /api/relay/runs/:runId/stop` (guard with `requireOrigin: true`) -> `{ ended: true, alreadyEnded: boolean }`: ends the row first (`store.end`), then calls `cancelEveTurn(eveSessionId)` once, only when this call ended it; a cancel failure is logged value-free and does not undo the end (the run is closed to tokens and turns either way). `cancelEveTurn(sessionId: string): Promise<'accepted' | 'no_active_turn'>` posts to Eve's `/eve/v1/session/:id/cancel` with the same loopback conventions as the other client calls and treats both statuses as success.
- Behavior change: the token route (404), the agent-token route (404) and `eve-chat` follow-up and answer turns (400 `run_ended`, before the claim) refuse an ended run.

- [ ] **Step 1: Write failing tests**: state route for owner / other user / web row / unknown run / ended; stop twice calls `cancelEveTurn` once; stop for another user's run -> 404 and nothing cancelled; cancel failure still returns ended; after stop, token route and agent-token route 404 and `eve-chat` answers 400 `run_ended`; stop and state require the extension origin when configured.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run `tests/agent` and the relay unit suite, expect PASS.**
- [ ] **Step 5: Commit** `feat(relay): add run state and a server-side stop`.

### Task 5: Keep consuming the stream when the client leaves

**Files:**
- Modify: `app/(chat)/api/eve-chat/route.ts`
- Test: `tests/agent/eve-chat-route.test.ts` (extend)

**Interfaces:**
- Contract: once a turn is accepted by Eve, the route keeps reading Eve's stream to `session.waiting` and keeps persisting `setProgress`, `setPending`, `advance` and the `heal` swap even if the response stream is cancelled or its writer throws. Client writes become best-effort (a failed write stops writing, not reading). The reconnect loop, claim handling and abort behavior for the web path are unchanged. A client abort must not cancel Eve's turn (Stop does that, Task 4).

- [ ] **Step 1: Write failing tests** with the existing mocked Eve client: cancel the response body after the first event of a turn that later raises an `input.requested` and a `session.waiting`; assert the store ends with the pending requests persisted, the cursor advanced, and the claim healed, and that a following answer turn validates against them. A second test aborts before any event and asserts the turn still completes in the store. A third asserts web chats keep today's behavior (an abort stops the read).
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** by separating "consume and persist" from "write to the client" inside `execute`, for relay runs only (`structuredInput`). Decide how to detect a cancelled client (writer throw, `request.signal`) and document it in a comment; never leave an unhandled rejection.
- [ ] **Step 4: Run the route file and `tests/agent`, expect PASS.**
- [ ] **Step 5: Commit** `fix(relay): persist run state after the client disconnects`.

### Task 6: Typed app client in the extension

**Files:**
- Create: `extension/src/app-api.ts`
- Test: `tests/relay/app-api.test.ts`

**Interfaces:**
- Produces: `createAppApi(opts: { origin: string; fetch?: typeof fetch }): AppApi` with `searchParticipants(q, signal?)`, `getParticipant(recordId)`, `getConfig()`, `getRunState(runId)`, `stopRun(runId)`, `getExtensionToken(runId)`, `classifyFields(runId, fields)`. Every request sends `credentials: 'include'`, `cache: 'no-store'`, no custom auth header, and a 10 second timeout; non-2xx maps to `AppApiError { status, code }` with no response body text (bodies can carry detail the panel must not log). A 401 is `code: 'signed_out'`. No method accepts or returns a sensitive value.

- [ ] **Step 1: Write failing tests** with an injected `fetch`: each method's URL, method and body; the timeout; 401 -> `signed_out`; a non-JSON body does not throw out of the client; the error carries no body text; `classifyFields` sends only the descriptor keys given.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** `feat(extension): add a typed client for the app API`.

### Task 7: Classifier hook in the redactor

**Files:**
- Modify: `lib/relay/redactor.ts`
- Test: `tests/relay/redactor.test.ts` (extend), `tests/relay/bridge-redaction.test.ts` as needed

**Interfaces:**
- Produces: `RedactorOptions.classifyExtra?: (descriptor: FieldDescriptor) => Promise<boolean>`. For an AX or DOM node the deterministic classifier calls ordinary, the redactor asks `classifyExtra` with the node's descriptor; `true` makes the node sensitive (masked like any other sensitive node); a throw, a rejection, a timeout (`classifyTimeoutMs`, default 2000) or a missing hook keeps the deterministic verdict. Verdicts are cached per run by the sanitized descriptor (`describeFieldForJev` output serialized), at most 256 entries, so a repeated snapshot costs no calls. At most 8 calls in flight; the per-tree deadline already bounds the whole pass. The hook is never given a value, URL or page text.

- [ ] **Step 1: Write failing tests**: a node ordinary by the deterministic rule and flagged by the hook is masked with its subtree; the hook is not called for nodes already sensitive; a hook that throws, rejects or hangs leaves the deterministic result and does not fail the reply; the cache makes a second identical node cost no call; concurrency never exceeds 8; the descriptor handed to the hook has no `value` key even when the node has one.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement**, keeping every existing fail-closed path and the 16-way describe pool untouched. This is the most delicate edit of the plan: change only the verdict step, and keep the module free of `chrome.*`.
- [ ] **Step 4: Run the redactor, bridge-redaction, screenshot-mask and sensitive-fields suites, expect PASS; then the redaction e2e once.**
- [ ] **Step 5: Commit** `feat(relay): let the redactor take an extra sensitive-field verdict`.

### Task 8: Worker Jev classification

**Files:**
- Create: `extension/src/jev-classifier.ts`
- Modify: `extension/src/bridge.ts` (build the classifier per run), `extension/manifest.json` if needed
- Test: `tests/relay/jev-classifier.test.ts`, `tests/relay/e2e/jev-classification.e2e.test.ts` (a local fake `classify-field` server)

**Interfaces:**
- Produces: `createJevClassifier(opts: { api: AppApi; runId: string; budgetPerRun?: number }): (d: FieldDescriptor) => Promise<boolean>`. It batches descriptors arriving in the same tick (up to 50 per request, the route's limit), calls `api.classifyFields`, returns the route's booleans in order, and returns `false` (no extra masking) on any failure per P4b-e, counting failures in `failures()` (a number, no detail). It stops calling after `budgetPerRun` requests (default 200) to bound cost. The run's token is not used; the call is cookie-authenticated and carries the run id.

- [ ] **Step 1: Write failing unit tests** with a fake `AppApi`: batching and order; more than 50 descriptors split; failure -> false and counted; budget exhaustion -> false without calls; no `value` key ever sent.
- [ ] **Step 2: Write the e2e**: a page with a field the deterministic classifier calls ordinary and the fake server calls sensitive; assert the snapshot the agent receives masks it, and that with the fake server returning 500 the snapshot still works with the deterministic verdict.
- [ ] **Step 3: Run, expect FAIL.**
- [ ] **Step 4: Implement** and wire `classifyExtra` when a run begins (`begin()`), not at module load.
- [ ] **Step 5: Run unit tests, then `pnpm extension:build && pnpm test:relay e2e/jev-classification.e2e.test.ts e2e/redaction.e2e.test.ts`, expect PASS.**
- [ ] **Step 6: Commit** `feat(extension): ask the app to classify fields during a run`.

### Task 9: Worker lifecycle fixes

**Files:**
- Modify: `extension/src/bridge.ts`, `extension/src/service-worker.ts`
- Test: `tests/relay/bridge-lifecycle.test.ts` (unit, with the existing chrome stubs), `tests/relay/e2e/bridge.e2e.test.ts` (extend)

**Interfaces:**
- Produces:
  - `startRun` serialized: calls queue behind one promise chain, so two concurrent calls leave exactly one run and one socket; a call made while another is in flight waits, then replaces it with reason `replaced`.
  - A `stop` issued while the socket is not open is kept and sent as the first frame once the socket reopens (or dropped when the run's token has expired); the local teardown (detach, redactor clear, storage removal) happens immediately as today.
  - Detach per P4b-d: a non-`canceled_by_user` detach starts a 5 second timer; a re-attach cancels it; expiry calls `stopRun('debugger-detached')`.
  - Test hooks `bridgeHandleMessage` and `bridgeOnDebuggerDetach` are installed on `globalThis` only when the build defines `EXTENSION_TEST_HOOKS` (the e2e build sets it; the release build does not).
  - After a worker restart the run resumes with an empty redactor: `resumeRun` marks the run `needsRegistration` and the bridge answers `getRunStatus` (a new message) with `{ active, runId, origin, attached, needsRegistration, jevFailures }` so the panel can re-enter registered values. While `needsRegistration` is true the screenshot path stays refused (known values may be missing) until the panel sends `registrationComplete` (a new message, count only, no values).
- Consumes: Task 8's classifier (`jevFailures`).

- [ ] **Step 1: Write failing tests**: two concurrent `startRun` -> one socket (count `WebSocket` constructions); stop with a closed socket sends one `stop` after reopen and none after token expiry; a detach with reason `target_closed` and no re-attach ends the run after 5 s (fake timers) and a re-attach inside 5 s does not; with `EXTENSION_TEST_HOOKS` undefined the globals are absent; after a simulated restart `getRunStatus` reports `needsRegistration: true`, screenshots are refused, and `registrationComplete` clears it.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.** Keep the existing reason strings; add `debugger-detached` to the stop reasons the relay accepts (check `lib/relay/protocol.ts` and the hub's reason handling).
- [ ] **Step 4: Run the unit suite, then the bridge e2e and the redaction e2e, expect PASS.**
- [ ] **Step 5: Commit** `fix(extension): serialize runs, send stop on reopen, and flag unregistered values after a restart`.

### Task 10: Allow-origin plumbing

**Files:**
- Modify: `extension/src/bridge.ts`, `lib/relay/protocol.ts`, `relay/hub.ts` only if the hub must learn the new origin (decide from the code: the hub's recorded origin is fixed per run today)
- Test: `tests/relay/bridge-lifecycle.test.ts` (extend), hub tests if touched

**Interfaces:**
- Produces: bridge state `pendingOrigin: string | null` set when the tab commits an https URL outside scope: instead of ending the run immediately the bridge blocks commands (`blocked = true`), records the origin, and exposes it through `getRunStatus.pendingOrigin`; `allowOrigin { origin }` message extends the run's allowed origins for this run only (exact https origin, never persisted, cleared on stop) and unblocks; if no `allowOrigin` arrives within 60 seconds the run ends with `left-origin`. Per P4b-f the panel for this lands in 4c; until then nothing sends `allowOrigin`, so the observable behavior is unchanged except for the 60 second grace.
- Constraint: the relay hub and the token still bind to the original origin; if the hub cannot accept a second origin for the agent leg, the plan's ledger records the limit and `allowOrigin` only governs the worker's own command check.

- [ ] **Step 1: Read `relay/hub.ts` origin handling and `lib/relay/policy.ts`; write the ledger note stating which of the two designs applies.**
- [ ] **Step 2: Write failing tests**: out-of-scope commit -> `blocked` plus `pendingOrigin`, run alive; `allowOrigin` with the same origin unblocks and later in-scope navigation works; a non-https or malformed origin is refused; no message within 60 s ends the run; stop clears the extension of scope.
- [ ] **Step 3: Run, expect FAIL.**
- [ ] **Step 4: Implement.**
- [ ] **Step 5: Run the unit suite and the allowlist e2e, expect PASS.**
- [ ] **Step 6: Commit** `feat(extension): hold a run on a redirect instead of ending it`.

### Task 11: Docs and tracker

**Files:**
- Modify: `CLAUDE.md`, `docs/browser-extension-todo.md`
- Create: `docs/changes/2026-10-28-add-run-plumbing.md` (use the change-docs skill)

- [ ] **Step 1: Update the docs from the diff**: auth model and `APP_ORIGIN` build constant, `RELAY_EXTENSION_ORIGIN`, the new routes, `endedAt` and the migration, the background-consumption behavior, the worker messages (`getRunStatus`, `registrationComplete`, `allowOrigin`), `EXTENSION_TEST_HOOKS`, the classifier hook and its outage behavior. Tick the tracker items this plan closed and move the remaining panel items to plan 4c.
- [ ] **Step 2: Run the whole node suite** `pnpm exec vitest run --config vitest.config.node.mjs tests/agent tests/relay --exclude 'tests/relay/e2e/**'`, then the bridge, redaction, jev-classification and allowlist e2e once; expect PASS.
- [ ] **Step 3: Commit** `docs: document the run plumbing`.

## Self-Review Notes

- Spec coverage: Security Requirement 4 (authentication of the extension to the app) is Tasks 1, 2, 6; Requirement 9 (Stop) is Tasks 4 and 9; "Calls Made in Review" 1 (redirect approval) is Task 10 up to the worker boundary, the panel half is 4c; 7 (service worker death) is Task 9; Sensitive Fields (Jev with fail-toward-sensitive) is Tasks 7 and 8. The six screens, saved details, and typing values into the page are plan 4c by design.
- Ordering: Task 3 before 4 (the state and stop routes need `endedAt`); Task 6 before 8 (the classifier uses the typed client); Task 7 before 8; Task 9 consumes Task 8's failure counter, so 9 follows 8.
- Riskiest edits: Task 5 (the eve-chat stream loop, shared with web chats; keep the web path identical) and Task 7 (the frozen redactor; touch only the verdict step).
- The cookie-versus-origin question is settled empirically in Task 1; every later task assumes only that an authenticated request from the worker reaches the routes.
