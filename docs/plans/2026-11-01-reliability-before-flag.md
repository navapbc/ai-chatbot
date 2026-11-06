# Reliability Before The Flag Implementation Plan (Plan 4d)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the items that `docs/browser-extension-todo.md` lists under "Must land before `RELAY_RUNS_ENABLED` is set anywhere", so a run neither orphans nor stalls nor silently loses a capability when the panel closes, the token refreshes, the tab holds on another site or the service worker restarts.

**Architecture:** The worker reports the real token expiry and a `restarting` flag in `getRunStatus`, so a panel reopen does not refresh the token and a refresh is not mistaken for a dead run. A same-run refresh carries its allowed extra origins as it already carries known values. `typeSensitive` attaches the debugger on demand. The panel records a run in Recents before it asks the server to create it and, when it opens, stops any server run that the worker does not hold. The server ends a run whose creating request was aborted. The Eve relay tool retries the first command lost to an unregistered extension and waits out an origin hold. The participant search moves from a query string to a POST body.

**Tech Stack:** MV3 service worker (`extension/src/bridge.ts`), the relay hub (`relay/hub.ts`) and Eve tool (`lib/relay/eve-relay-browser.ts`), the panel from plans 4c-1 to 4c-3, Next route handlers, vitest node project, Playwright Chromium e2e. No new dependencies, no schema change.

**Spec:** `docs/specs/2026-10-23-browser-extension-design.md` (Stop, Service worker death, Security Requirements 1 to 9). Tracker: `docs/browser-extension-todo.md`, "Plan 4c-1 follow-ups > Must land before `RELAY_RUNS_ENABLED`" and "Plan 4c-2 carry-forward". Previous plans: `docs/plans/2026-10-31-side-panel-sensitive-typing.md`, `docs/plans/2026-10-30-side-panel-running-view.md`, `docs/plans/2026-10-28-run-plumbing.md`.

**Scope note:** The unfilled-field highlighter, automatic submit detection and the confirmation number are plan 4c-4. Plan 5 is rollout.

## Global Constraints

- Node 24: `export PATH=/Users/kaylynvannorstrand/.nvm/versions/node/v24.20.0/bin:$PATH`. `pnpm` only. No new dependencies; if one seems necessary, stop and ask.
- Tests: `pnpm exec vitest run --config vitest.config.node.mjs <path>`; extension work also runs `pnpm extension:build && pnpm test:relay e2e/panel-shell.e2e.test.ts e2e/bridge.e2e.test.ts`. Lint with `pnpm exec biome lint <files>` (extension paths are excluded by config; then format only the files you touched by hand); never `pnpm lint`; never `biome format --write` on a directory.
- Never read `.env*` files or print secret values. Nothing in this plan adds a migration.
- The origin-lock model is unchanged: allowed extra origins are run-scoped, at most 3, in memory only, and never survive a worker restart. A refresh may carry them only for the same run id.
- The agent-facing `attached` flag is set only by the agent's relay `attach` command. A worker-initiated attach (typing) must never set it, so the Eve tool still sees `not attached` and rotates its stale daemon.
- The Stop path stays idempotent: every new stop call is the existing server stop route plus the scoped worker stop, safe to repeat and safe when the run already ended.
- Panel recents keep only `runId`, `siteName`, `startedAt` and `endedReason`; no participant data. A recorded in-flight run id is an id only.
- Everything the panel renders from the server or agent is untrusted text rendered as text only (unchanged). Fixed copy comes from `extension/panel/lib/copy.ts`.
- The web app and the legacy chat route are unchanged: `eve-chat` changes are guarded by `isRelayRun` and the `participants` route keeps its web behavior for non-relay callers. `RELAY_RUNS_ENABLED` and `RELAY_TOKEN_ENDPOINT_ENABLED` stay unset in deployment config.
- Frontend conventions from `CLAUDE.md`: Tailwind only, `handle` prefix, early returns, aria attributes, keyboard support.

## Decisions For Owner

Defaults chosen so the work can proceed; each is a one-line change.

- **P4d-a Real expiry.** `getRunStatus` returns `exp` (Unix seconds, from the token's own `claims.exp`, not secret) and the panel uses it for the rebuilt run; the 5-minute estimate remains only as a fallback when `exp` is absent.
- **P4d-b Restarting.** `getRunStatus` returns `restarting: true` while a same-run `startRun` or a service-worker resume is in progress. The panel treats an inactive status with `restarting` as not-yet-dead, capped at 30 seconds so a hung detach cannot hide a dead run.
- **P4d-c Orphan sweep.** A run is recorded in Recents (id, site name, time) before the first-turn POST, marked `start-failed` if the start fails, and on panel open the panel stops every recent with no `endedReason` that the worker does not hold, skipping recents younger than 90 seconds and treating a 404 as final only after 5 minutes. At most 10 stop calls once, then none. No server-side age reaper (an idle legitimate run cannot be told from an orphan).
- **P4d-d Abort check.** After `store.create` on a relay first turn, if the request signal has aborted, the route ends the row and cancels the Eve turn. Next 16 abort propagation is unverified; the panel sweep (P4d-c) is the backstop.
- **P4d-e Typing attach.** `typeSensitive` attaches the debugger itself when it is not attached, using the same scope checks as the relay `attach`, and never sets the agent-facing `attached`. The debugger banner may reappear while the caseworker types.
- **P4d-f Refresh on an allowed origin.** A same-run refresh carries `extraOrigins` and accepts a tab on the token origin or on a carried origin. A different run id or a worker restart never carries them.
- **P4d-g F1.** Add `extension did not answer` to `CONNECTION_LOST` so the first (retry-safe) snapshot is rotated and retried. The hub queue in `unknown` is not built (it costs a stall of up to 30 seconds on the first command, accepted for now).
- **P4d-h Hold wait.** The Eve relay tool waits out an origin hold: on the hold refusal it polls the tab URL about every 3 seconds up to about 125 seconds, then re-runs a retry-safe command or reports `CONNECTION_RESTORED` for a non-retry-safe one. It never blindly replays an action.
- **P4d-i Participant search.** `GET /api/participants?q=` is replaced by `POST /api/participants` with a JSON `{q, limit}` body, requiring the extension origin when `RELAY_EXTENSION_ORIGIN` is set. The GET is removed (only the extension uses it).

## Review Focus

- Reopening the panel while the agent is blocked in a question, while the tab is held on an allowed origin, and during a token refresh: expect no second refresh, no stop of a healthy run, no lost allowed origin, no lost scrubbing.
- Closing the panel or timing out during Starting, a worker that ends a run with the panel closed, and two panel windows: expect no live server row left behind, and no stop of a run the worker still holds.
- Typing after a refresh, a restart and a reopen: expect the fixed refusal or success, never a value typed into the wrong field, and never an agent-facing `attached` set by the typing path.
- A hold on a tab whose agent command is mid-flight: expect a bounded wait, no replay of a non-retry-safe action, and a clear message when the hold ends the run.
- The search term (a participant name) in logs: expect it only in a request body.

---

### Task 1: Report the real expiry and `restarting`

**Files:**
- Modify: `extension/src/bridge.ts` (`RunStatus`, `getRunStatus`, `startRunNow`, the service-worker resume path), `extension/src/app-api.ts` or `extension/panel/lib/worker-api.ts` (status type), `extension/panel/lib/run-session.ts` (`rebuildRun`, `onWorkerStatus`, `ESTIMATED_LEFT_S` use)
- Test: `tests/relay/bridge-lifecycle.test.ts` (extend), `tests/relay/panel-run-session.test.ts` (extend)

**Interfaces:**
- Produces: `RunStatus` gains `exp?: number` (Unix seconds) and `restarting?: boolean`. `restarting` is true from the start of a same-run `startRun` until `begin` finishes (set and cleared in a `try/finally`) and while a service-worker `resumeRun` is in flight. `rebuildRun` sets `expiresAt` from `status.exp` when present and clears `expiresAtEstimated`; the estimate stays the fallback. In `onWorkerStatus`, an inactive status with `restarting` resets `inactiveSince`, with a total cap of 30 seconds after which the existing 7-second rule applies.

- [ ] **Step 1: Write failing tests**: `getRunStatus` returns `exp` equal to the token's `exp`; `restarting` is true while a same-run `startRun` awaits (gate a storage call) and false after, including after a throw; a panel opened with `exp` 25 minutes away does not call `startRun` (no refresh) and `expiresAtEstimated` is false; with `exp` absent the old estimate and refresh remain; an inactive status with `restarting` for 10 s does not end the session, and after 30 s it does; a hung `restarting` cannot hide a dead run.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.** Update the existing `panel-run-session` tests that pinned the estimated-expiry refresh (the fact sheet names the cases near lines 1008, 1400 and 1535).
- [ ] **Step 4: Run, expect PASS** (`tests/relay`, e2e excluded).
- [ ] **Step 5: Commit** `fix(extension): report the real token expiry and a restarting flag`.

### Task 2: A refresh carries the allowed origins

**Files:**
- Modify: `extension/src/bridge.ts` (`startRunNow`)
- Test: `tests/relay/bridge-lifecycle.test.ts` (extend), `tests/relay/bridge-origin-hold.test.ts` or the existing hold test file

**Interfaces:**
- Consumes: Task 1's `restarting` handling.
- Produces: when the live run has the same run id and origin as the incoming token, `startRunNow` captures its `extraOrigins` in the same synchronous step as the carried values, accepts the tab if `inScope(url, claims.origin)` or the URL is inside a carried origin, and sets `current.extraOrigins` on the new run right after `begin` (next to `carryKnownValues`). A different run id, a resume, or a worker restart carries nothing. The cap of 3 holds.

- [ ] **Step 1: Write failing tests**: with one allowed origin, a same-run `startRun` while the tab sits on that origin succeeds and keeps the origin allowed; a refresh with the tab on an origin that was never allowed is refused as before; a different run id does not inherit; a worker-restart run has none; three origins carry and a fourth `allowOrigin` is still refused.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS** (`tests/relay`).
- [ ] **Step 5: Commit** `fix(extension): keep allowed origins across a token refresh`.

### Task 3: `typeSensitive` attaches on demand

**Files:**
- Modify: `extension/src/bridge.ts` (extract `attachTab` from the relay `attach` handler; `typeSensitive`; `live()`; `onDebuggerDetach`)
- Test: `tests/relay/bridge-type-sensitive.test.ts` (flip the pinned `not attached` test), `tests/relay/e2e/bridge.e2e.test.ts` (one real-Chromium case)

**Interfaces:**
- Produces: `attachTab(current, epoch)` is the attach core (detach any stale attach, attach, bridge-owned `Page.enable` and `Page.getFrameTree`, frame scope, epoch re-check) and does not set `attached`. The relay `attach` handler calls it and then sets `attached = true`. `typeSensitive` calls it after the value is registered when the run is not attached, sets a separate `typingAttached`, and `live()` accepts either flag. `forwardCdp` keeps refusing with `not attached` unless `attached` is true. A later agent `attach` detaches first as today. `onDebuggerDetach` clears `typingAttached`. The reply strings and the order (register, then verify, then type) are unchanged; an attach failure returns `not attached`.

- [ ] **Step 1: Write failing tests**: after a same-run refresh, `typeSensitive` attaches, types once into the masked focused field and replies `{ok: true}`; `attached` stays false afterwards (an agent command still gets `not attached`); an out-of-scope tab refuses before any attach; a failed attach returns `not attached` and registers the value as before; a later agent `attach` works and detaches the typing attach; a debugger detach clears `typingAttached`.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS** (`tests/relay`, then `pnpm extension:build && pnpm test:relay e2e/bridge.e2e.test.ts`).
- [ ] **Step 5: Commit** `fix(extension): attach on demand when typing a sensitive value`.

### Task 4: Record the run first and stop what the worker does not hold

**Files:**
- Modify: `extension/panel/lib/run-starter.ts`, `extension/panel/lib/recents.ts`, `extension/panel/app.tsx` (open-time sweep), `extension/panel/screens/home.tsx` (hide `start-failed`), `extension/panel/lib/copy.ts`
- Create: `extension/panel/lib/orphan-sweep.ts`
- Test: `tests/relay/panel-run-starter.test.ts` (extend), `tests/relay/panel-orphan-sweep.test.ts` (create), `tests/relay/panel-recents.test.ts` (extend)

**Interfaces:**
- Consumes: Task 1's `restarting`; the existing server `stopRun` and worker `stopRun(runId)`.
- Produces: `addRecent` runs before the first-turn POST with `endedReason` unset; a start failure after the POST calls the server stop and `markRecentEnded(runId, 'start-failed')`. `sweepStaleRecents(deps): Promise<void>` runs once when the panel opens: for each recent with no `endedReason`, younger than 90 s skipped, when the worker status is inactive and not `restarting` and does not hold that run id, POST the (idempotent) stop and mark it `swept`; a 404 is final only for recents older than 5 minutes (otherwise left for the next open); at most 10 stop calls, errors ignored, nothing logged. The Home list hides `start-failed`. Recents keep only the four fixed fields.

- [ ] **Step 1: Write failing tests**: a recent exists before the POST resolves; a failed start ends and marks it; closing during Starting (recent with no reason, worker holds nothing) is stopped on the next open; a recent the worker holds is never stopped; a young recent is skipped; `restarting` skips the whole sweep; 404 handling by age; the 10-call cap; a throwing stop does not break the open; two windows do not stop each other's live run (the worker holds it).
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS** (`tests/relay`; then `pnpm extension:build && pnpm test:relay e2e/panel-shell.e2e.test.ts`).
- [ ] **Step 5: Commit** `fix(extension): stop runs that nothing holds when the panel opens`.

### Task 5: End a run whose creating request was aborted

**Files:**
- Modify: `app/(chat)/api/eve-chat/route.ts` (relay first turn only)
- Test: the existing eve-chat relay route test file (extend)

**Interfaces:**
- Produces: after `store.create` on a relay first turn, a single check of `request.signal.aborted`; when aborted the route ends the row (`endedAt`) and calls the existing Eve cancel, and returns without starting the read loop. The web path and the continue and answer turns are unchanged, and the rule that an open relay run keeps reading after the client disconnects is unchanged (this check is once, right after create).

- [ ] **Step 1: Write failing tests**: an aborted signal after create ends the row and cancels the turn; a live signal behaves as before; a web chat is untouched (the flag and run store are never written); a cancel failure does not undo the end.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS** (`tests/relay` and `tests/agent`).
- [ ] **Step 5: Commit** `fix(relay): end a run whose creating request was aborted`.

### Task 6: Retry the first command lost to an unregistered extension

**Files:**
- Modify: `lib/relay/eve-relay-browser.ts` (`CONNECTION_LOST`)
- Test: `tests/relay/eve-relay-browser.test.ts` (extend)

**Interfaces:**
- Produces: `extension did not answer` joins the strings in `CONNECTION_LOST`, so the retry-safe first `snapshot` is rotated and retried and a non-retry-safe command reports `CONNECTION_RESTORED` rather than replaying.

- [ ] **Step 1: Write failing tests**: a first `snapshot` that fails with `extension did not answer` is retried once and succeeds; a `click` that fails the same way is not replayed and returns the restored message; the existing retry tests stay green.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS** (`tests/relay`).
- [ ] **Step 5: Commit** `fix(relay): retry the first command when the extension has not registered`.

### Task 7: Wait out an origin hold

**Files:**
- Modify: `lib/relay/eve-relay-browser.ts` (`runRelayCommand`), `agent/skills/extension-run/SKILL.md` (one line)
- Test: `tests/relay/eve-relay-browser.test.ts` (extend), the skills guidance test

**Interfaces:**
- Consumes: Task 6's retry classification.
- Produces: an anchored `HOLD_REFUSED` pattern for `tab is outside the allowed origin`. On a match `runRelayCommand` polls `['get','url']` (a command the policy already allows) about every 3 seconds for up to 125 seconds. When a poll answers, a retry-safe command is re-run once and a non-retry-safe command returns `CONNECTION_RESTORED` with `needsSnapshot` set; when the wait ends or the run ends, the original refusal is returned. Before choosing the bound, check Eve's tool timeout and the agent-token lifetime; shorten the wait if either is lower. The skill line: during an origin hold the tool waits and then tells the agent what to do; do not end the turn on the refusal.

- [ ] **Step 1: Write failing tests**: a hold that clears after two polls re-runs a `snapshot`; a hold that clears for a `click` returns the restored message without replaying; a hold that never clears returns the refusal after the bound; a run end during the wait stops it; the polling uses only the allowed URL command; no unrelated refusal is retried.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS** (`tests/relay` and `tests/agent`).
- [ ] **Step 5: Commit** `fix(relay): wait out an origin hold instead of failing the command`.

### Task 8: Participant search by POST

**Files:**
- Modify: `app/(chat)/api/participants/route.ts`, `extension/src/app-api.ts` (`searchParticipants`)
- Test: `tests/relay/participants-route.test.ts`, `tests/relay/relay-route-origin.test.ts`, `tests/relay/app-api.test.ts`, the panel-shell e2e fake server

**Interfaces:**
- Produces: `POST /api/participants` with body `{q: string, limit?: number}`, the same limits and result shape as the old GET, guarded by `guardRelayRoute(request, {requireOrigin: true})`; the GET handler is removed. `searchParticipants(q, limit)` sends the POST with `credentials: 'include'` and a JSON body; the query string is never used. Nothing logs the term.

- [ ] **Step 1: Write failing tests**: the POST returns summaries and honors limits; a wrong or missing extension origin is refused when `RELAY_EXTENSION_ORIGIN` is set; guests are refused; an over-long `q` is refused; the old GET returns 405; `searchParticipants` sends a body and a URL with no query; the e2e fake server serves the POST.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS** (`tests/relay`, then `pnpm extension:build && pnpm test:relay e2e/panel-shell.e2e.test.ts`).
- [ ] **Step 5: Commit** `fix(relay): search participants with a POST body`.

### Task 9: End to end

**Files:**
- Modify: `tests/relay/e2e/panel-shell.e2e.test.ts` (or a new `tests/relay/e2e/reliability.e2e.test.ts` if the file's runtime would grow more than about 30 s)

**Interfaces:**
- A scripted scenario against the real worker, a real relay and the fake app: (1) a panel opened mid-run does not refresh the token and the run keeps going (the real `exp`); (2) with an allowed origin, a refresh keeps the origin and the run continues; (3) after a refresh, with the agent silent, the sensitive card's Fill it in types into the masked field; (4) a panel closed during Starting leaves a recent that the next open stops (the fake app sees one stop for that run id and none for a run the worker holds); (5) the search request has no query string. Header states how it is driven (panel as a tab). Reuse the helpers of earlier plans.

- [ ] **Step 1: Write the e2e.**
- [ ] **Step 2: Run** `pnpm extension:build` with the hooks build, then `pnpm test:relay e2e/panel-shell.e2e.test.ts` (and the new file), then the other relay e2e files once; expect PASS. Report the added runtime.
- [ ] **Step 3: Commit** `test(extension): cover the reliability fixes end to end`.

### Task 10: Docs and tracker

**Files:**
- Modify: `CLAUDE.md`, `docs/browser-extension-todo.md`
- Create: `docs/changes/2026-11-01-add-reliability-fixes.md` (use the change-docs skill; H1 in Title Case)

- [ ] **Step 1: Update the docs from the diff**: `exp` and `restarting` in `getRunStatus`, the carried origins, typing attach, Recents before the POST and the open-time sweep, the abort check, the first-command retry, the hold wait, the POST search; tick every item this plan closed (F1, F2, I1, I2, I4, `not_attached`, the participant query string, `restarting`, the refresh-on-allowed-origin item, the hold wait/retry item) and keep the rest; add the follow-ups that surface (the 30 s first-command stall if the retry only is kept, Next abort propagation unverified, the Starting screen Cancel, the `workerGone` branch, the cross-window `worker_unreachable` run).
- [ ] **Step 2: Run the whole node suite** `pnpm exec vitest run --config vitest.config.node.mjs tests/agent tests/relay --exclude 'tests/relay/e2e/**'` and the relay e2e files once; expect PASS.
- [ ] **Step 3: Commit** `docs: document the reliability fixes`.

## Self-Review Notes

- Spec coverage: Stop and orphan runs (Tasks 4, 5), Service worker death and refresh (Tasks 1, 2), sensitive typing availability (Task 3), first-command and hold behavior (Tasks 6, 7), the participant name in logs (Task 8).
- Ordering: Task 1 before Tasks 2 and 4 (both read `restarting`); Task 2 after Task 1; Task 3 is independent of 1 and 2 but shares `bridge.ts`, so it runs after them; Task 7 after Task 6; Task 9 needs Tasks 1 to 4 and 8.
- Riskiest edits: Task 3 (a second attach path that must not set the agent-facing flag), Task 4 (a sweep that stops server runs: it must never stop a run the worker holds, a young run, or a run during `restarting`), Task 7 (a wait loop inside the tool that must not replay a non-retry-safe action).
- Left out deliberately: a hub queue for commands in `unknown`, a server-side age reaper, multi-instance gaps (plan 5), the highlighter, submit detection and the confirmation number (plan 4c-4).
