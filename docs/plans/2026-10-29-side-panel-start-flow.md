# Side Panel Foundation And Start Flow Implementation Plan (Plan 4c-1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A caseworker can open the extension's side panel, sign in state is clear, pick a site (or use the current tab), pick or enter a participant, choose a model, press Start, and see a minimal running state with Stop. The run itself is driven by the agent through the existing relay.

**Architecture:** The panel is a small React 19 app bundled by esbuild into `extension/dist/panel.js` with Tailwind CSS built from the app's own tokens; it imports only leaf shadcn components from `components/ui` through an alias. It talks to the app with the typed client from plan 4b and to the service worker with `chrome.runtime.sendMessage`. Starting a run is one orchestrated sequence: create the run through the `eve-chat` first turn, fetch an extension token, send `startRun` to the worker. The panel keeps no applicant data except what is on screen; saved details live in `chrome.storage.local` for 8 hours and never include sensitive fields.

**Tech Stack:** React 19, Tailwind 3 (CLI), esbuild, shadcn leaf components and lucide icons already in the repo, Chrome MV3 `sidePanel`, `chrome.storage`, `chrome.alarms`, vitest (node project, jsdom-free unit tests for logic), Playwright Chromium e2e. No new dependencies.

**Spec:** `docs/specs/2026-10-23-browser-extension-design.md` (Users and Flow screens 1 to 4, Saved Details, Models, Extension). Tracker: `docs/browser-extension-todo.md`, "Plan 4c carry-forward". Previous plans: `docs/plans/2026-10-27-server-side-run-api.md`, `docs/plans/2026-10-28-run-plumbing.md`. The running view, Needs You, sensitive typing, the allow-site prompt, Review and Confirmed are plan 4c-2.

## Global Constraints

- Node 24: `export PATH=/Users/kaylynvannorstrand/.nvm/versions/node/v24.20.0/bin:$PATH`. `pnpm` only. No new dependencies; if one seems necessary, stop and ask.
- Tests: `pnpm exec vitest run --config vitest.config.node.mjs <path>`; extension work also runs `pnpm extension:build && pnpm test:relay e2e/<file>`. Lint with `pnpm exec biome lint <files>`; never `pnpm lint`; never `biome format --write` on a directory.
- Never read `.env*` files or print secret values. No applicant value (a typed answer, a participant detail, a name) goes into `console.*`, error messages, `chrome.storage` (except saved details, below), the URL, or a trace.
- The panel never receives a sensitive value from the server, and in this plan never types one. Saved details exclude any key `isSensitiveField` flags and are refused before storage, not stripped.
- Frontend conventions from `CLAUDE.md`: Tailwind classes only, no inline styles, `handle` prefix for handlers, early returns, accessibility attributes and keyboard handlers on interactive elements.
- The web app (`develop` behavior) is unchanged: no edit to `components/**` outside `components/ui` additions that the web app does not import, no edit to `app/globals.css` except if strictly required and then additive. `pnpm build` and the existing browser-project tests keep passing.
- Panel code imports only: React, radix, cva, clsx, tailwind-merge, lucide (already dependencies), leaf files from `components/ui`, `lib/utils` `cn` is replaced by a local copy if `lib/utils` pulls Next or server code, `lib/relay/*` and `lib/data` pure modules, and `extension/src/*`. A test greps the bundle's import graph for `next/`, `server-only`, `@/lib/db`, `drizzle`, `@typesafe-ai`, `ai` and fails.
- `RELAY_RUNS_ENABLED` and `RELAY_TOKEN_ENDPOINT_ENABLED` stay unset in deployment config.

## Decisions For Owner

Defaults chosen so the work can proceed; each is a one-line change.

- **P4c-a Opening the panel.** Clicking the toolbar icon opens the side panel (`chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true })`), no keyboard shortcut.
- **P4c-b Recent runs.** Home shows the last 10 runs as `{ runId, siteName, startedAt, endedReason }` kept 24 hours in `chrome.storage.local`. No participant name, id or detail is stored for recents (spec: participants are not stored).
- **P4c-c Saved details.** The enter-details form keeps non-sensitive field values for 8 hours in `chrome.storage.local`, checked on read and by a `chrome.alarms` sweep, with a visible "Clear saved details" action. The fields are a fixed list (first name, middle name, last name, date of birth, phone, email, street, unit, city, state, county, zip); anything else a caseworker needs is asked by the agent during the run.
- **P4c-d Fonts.** The panel uses the system font stack with `font-inter` falling back to it; web fonts are not bundled.
- **P4c-e Token lifetime.** Tokens last 30 minutes. The panel refreshes at 5 minutes remaining with a same-run-id `startRun`, and tells the caseworker if a refresh is refused (the tab is on an allowed extra origin): the run then ends at the 30 minute mark with a plain message.
- **P4c-f Stop.** The Stop button calls the server stop route and the worker `stopRun`, in that order, and shows the run as ended even if one of them fails.
- **P4c-g Site list.** Sites come from `GET /api/relay/config` (`config/relay-sites.json`, still empty: which sites belong there is the owner's open question). With an empty list Home shows only "Use this tab".

## Review Focus

- The panel opened while signed out, offline, or with the app returning 5xx: a clear plain-language state and a retry, never a blank screen or a stack trace.
- A saved-details record that is 9 hours old, one with a sensitive key someone planted in storage, and one written by an older build: expect expiry and refusal on read, never display.
- Start pressed twice, the tab closed between Start and the worker call, the app answering 429 or 403, and the worker refusing `startRun`: expect one run at most, no stuck "starting" state, a specific message for each.
- A participant search typed quickly (race between responses), with an empty query, and with `%` or an emoji: expect the latest query's results only and no crash.
- The panel closed and reopened mid-start and mid-run: expect it to rebuild from the worker status and the run-state route, not start a second run.
- Anything the panel renders or logs that came from the server (error text, participant names): expect it rendered as text, never as HTML, and never logged.

## File Structure

- Create `extension/panel/`: `index.html`, `main.tsx`, `app.tsx` (state machine), `screens/` (`home.tsx`, `participant.tsx`, `details.tsx`, `ready.tsx`, `running.tsx`, `signed-out.tsx`, `error.tsx`), `lib/` (`worker-api.ts`, `saved-details.ts`, `recents.ts`, `run-starter.ts`, `run-session.ts`, `copy.ts`), `panel.css`, `tailwind.config.ts`.
- Modify `scripts/build-extension.ts` and `scripts/extension-build-config.ts` (panel bundle, CSS, `panel.html`), `extension/manifest.json` (`side_panel`, `action`, permissions `sidePanel`, `alarms`), `extension/src/service-worker.ts` (action click, alarm sweep), `extension/src/bridge.ts` (runId check on `registrationComplete` and `addKnownValue`).
- Tests: `tests/relay/panel-*.test.ts` (logic), `tests/relay/e2e/panel.e2e.test.ts`.

---

### Task 1: Panel build pipeline

**Files:**
- Create: `extension/panel/index.html`, `extension/panel/main.tsx` (renders a placeholder `<App />`), `extension/panel/panel.css`, `extension/panel/tailwind.config.ts`
- Modify: `scripts/build-extension.ts`, `scripts/extension-build-config.ts`, `extension/manifest.json`
- Test: `tests/relay/extension-build.test.ts` (extend), `tests/relay/panel-bundle.test.ts`

**Interfaces:**
- Produces: `extension:build` also writes `panel.html`, `panel.js`, `panel.css` into the output directory; the manifest gains `side_panel: { default_path: 'panel.html' }`, `action: { default_title: 'Labs ASP' }`, permissions `sidePanel` and `alarms`. The CSS is built with the Tailwind CLI (`node_modules/.bin/tailwindcss`) from `panel.css` and a panel Tailwind config whose `content` is `extension/panel/**` and `components/ui/**` and whose theme extends the root `tailwind.config.ts` so the tokens match. esbuild bundles `main.tsx` with `--alias:@=<repo root>`, automatic JSX, `--define` for `process.env.NODE_ENV`.
- The existing worker bundle command, `APP_ORIGIN` handling, `EXTENSION_RELEASE`, `EXTENSION_TEST_HOOKS` and `--out` behavior are unchanged; a plain build still works with no new env.

- [ ] **Step 1: Write failing tests**: the build writes the three panel files and a manifest with the side panel and the two permissions; `panel.html` loads only `panel.js` and `panel.css` (no inline script, which MV3 CSP forbids); the bundle contains no string from the forbidden-import list (a graph check using esbuild's metafile, failing on `next/`, `server-only`, `@/lib/db`, `drizzle`, `@typesafe-ai`, `node_modules/ai/`); the release build refuses nothing new.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.** Reuse root tokens from `app/globals.css` by importing its `:root` variable blocks in `panel.css` (`@import` of the file is acceptable if it carries no Next-only rules; otherwise extract the variables into `extension/panel/tokens.css` generated from the same source and test they match). Do not edit `app/globals.css` or the root Tailwind config.
- [ ] **Step 4: Run `pnpm extension:build`, load the panel page in a Playwright Chromium test and assert it renders the placeholder with the app's primary color applied; expect PASS.**
- [ ] **Step 5: Commit** `feat(extension): build a side panel bundle`.

### Task 2: Worker side of the panel contract

**Files:**
- Modify: `extension/src/service-worker.ts`, `extension/src/bridge.ts`
- Test: `tests/relay/bridge-lifecycle.test.ts` (extend), `tests/relay/service-worker.test.ts` (new, with chrome stubs)

**Interfaces:**
- Produces: `chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true })` at worker start (errors swallowed and logged value-free); `registrationComplete` and `addKnownValue` accept `runId: string` and refuse (`{ ok: false, error: 'wrong run' }`) when it is missing or differs from the active run's id; an `alarms` listener for a named alarm `saved-details-sweep` that does nothing in the worker (the sweep runs in the panel, Task 6) but is registered so a future owner is explicit; `getRunStatus` keeps its shape.
- This closes the tracker item "the bridge must itself check `runId`".

- [ ] **Step 1: Write failing tests**: both messages refused without `runId` and with another run's id, accepted with the right one; existing tests updated to send the id; the panel behavior call is made once at startup and its rejection does not break worker start.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the relay unit suite, then the bridge and redaction e2e, expect PASS.**
- [ ] **Step 5: Commit** `feat(extension): bind registration messages to the run and open the panel on click`.

### Task 3: Panel shell and worker client

**Files:**
- Create: `extension/panel/app.tsx`, `extension/panel/screens/signed-out.tsx`, `extension/panel/screens/error.tsx`, `extension/panel/lib/worker-api.ts`, `extension/panel/lib/copy.ts`
- Test: `tests/relay/panel-worker-api.test.ts`, `tests/relay/panel-copy.test.ts`

**Interfaces:**
- Produces: `createWorkerApi(sendMessage?: typeof chrome.runtime.sendMessage): WorkerApi` with `startRun(args)`, `stopRun()`, `addKnownValue(runId, value)`, `registrationComplete(runId, count)`, `allowOrigin(origin)`, `getRunStatus()` returning typed results; a worker reply `{ ok: false, error }` becomes a typed `WorkerError`. `copy.ts` exports `messageFor(error: AppApiError | WorkerError | unknown): { title: string; detail: string; retryable: boolean }`, a fixed table of plain-language messages for: signed out, forbidden, rate limited, offline, server error, relay unreachable, tab closed, tab left the site, run limit reached, token expired, generic. `App` renders a screen from a small state machine (`home | participant | details | ready | starting | running | signedOut | error`); the sign-in probe is `getConfig()`: `signed_out` -> the signed-out screen with a button that opens `APP_ORIGIN` in a new tab and a "Try again" button.
- No error screen shows server-provided text; copy comes only from the table.

- [ ] **Step 1: Write failing tests**: each worker message shape; `WorkerError` mapping; `messageFor` for every code including an unknown error (generic, retryable) and that no server text can pass through (feed an error carrying a marker string and assert it never appears in the output).
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** the shell with the state machine and the two screens; other screens are stubs returning to Home until their tasks land.
- [ ] **Step 4: Run, expect PASS; load the panel in the Playwright test with a fake app answering 401 and assert the signed-out screen and its button.**
- [ ] **Step 5: Commit** `feat(extension): add the panel shell, worker client and error copy`.

### Task 4: Home and recent runs

**Files:**
- Create: `extension/panel/screens/home.tsx`, `extension/panel/lib/recents.ts`
- Test: `tests/relay/panel-recents.test.ts`

**Interfaces:**
- Produces: `loadRecents(now?: number): Promise<Recent[]>`, `addRecent(r: Recent)`, `markRecentEnded(runId, reason)`; `Recent = { runId: string; siteName: string; startedAt: number; endedReason?: string }`; at most 10 entries, entries older than 24 hours dropped on read; the stored value is validated on read (wrong shape or extra keys dropped, never rendered). Home lists configured sites (each opens the site in a new tab and continues to the participant screen with that tab) and a "Use this tab" action (reads the active tab's origin; https only, otherwise a plain message that the agent works on https pages only), plus Recent. No participant data in recents.

- [ ] **Step 1: Write failing tests**: cap, expiry, validation of malformed and planted storage, the stored JSON has only the four keys, no participant fields.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** the module and screen. "Use this tab" uses `chrome.tabs.query`; the chosen `{ tabId, origin, siteName }` goes into the state machine.
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** `feat(extension): add the panel Home screen and recent runs`.

### Task 5: Participant search

**Files:**
- Create: `extension/panel/screens/participant.tsx`
- Test: `tests/relay/panel-participant.test.ts` (logic hooks tested without a DOM: the search controller)

**Interfaces:**
- Produces: `createSearchController(api: AppApi, onResult: (s: SearchState) => void, opts?: { debounceMs?: number })` with `setQuery(q)` and `dispose()`: debounces 250 ms, aborts the previous request, ignores any response that is not for the latest query, empty or whitespace query -> idle (no request), trims to 100 characters, maps errors through `messageFor`. The screen lists `ParticipantSummary` rows (name, county, record id) rendered as text, with a "Enter details instead" action.

- [ ] **Step 1: Write failing tests** with fake timers and a fake client: debounce, abort of the earlier request, a slow early response arriving after a later one is ignored, empty query, 101-character query trimmed, error state, `dispose` stops callbacks.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** controller and screen (keyboard navigable list, `aria-live` for result count).
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** `feat(extension): add participant search to the panel`.

### Task 6: Entered details and saved details

**Files:**
- Create: `extension/panel/screens/details.tsx`, `extension/panel/lib/saved-details.ts`
- Modify: `extension/panel/main.tsx` (alarm sweep)
- Test: `tests/relay/panel-saved-details.test.ts`

**Interfaces:**
- Produces: `DETAIL_FIELDS` (the fixed list in P4c-c with labels); `loadSavedDetails(now?): Promise<Record<string,string> | null>` (null when absent, older than 8 hours, malformed, or containing any key outside `DETAIL_FIELDS` or any key `isSensitiveField({ name: key, label: key })` flags; an invalid record is deleted); `saveDetails(values, now?)` refuses (throws `SensitiveDetailError`) if any key is outside the list or sensitive, drops empty values, caps value length at 500 and the record at 12 keys; `clearSavedDetails()`; `sweepSavedDetails(now?)` deletes an expired record, called on panel start and on the `saved-details-sweep` alarm (created with `chrome.alarms.create(name, { periodInMinutes: 30 })`). The screen has "Reuse saved details" at the top when a valid record exists, the form with labeled inputs, a "Save these for 8 hours" checkbox (off by default), and "Clear saved details". Values go to the next step in memory only. The submit payload is the flat `{ [key]: string }` object that `eve-chat` accepts as `participant.details`.

- [ ] **Step 1: Write failing tests**: 9-hour-old record -> null and deleted; planted sensitive key (`ssn`, `calworks id`) -> null and deleted; `save` refusal for a sensitive or unknown key; value and key caps; clear; sweep; storage holds only the record and its timestamp; the saved JSON never contains a value that was not saved (checkbox off writes nothing).
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** `feat(extension): add entered details with eight-hour saved details`.

### Task 7: Ready screen

**Files:**
- Create: `extension/panel/screens/ready.tsx`
- Test: `tests/relay/panel-ready.test.ts`

**Interfaces:**
- Produces: `ReadyModel = { siteName; origin; participant: { kind: 'record'; recordId; name } | { kind: 'details'; details }; models: { id }[]; modelId: string }`; a pure `buildStartRequest(model: ReadyModel): { origin: string; participant: { recordId } | { details }; model?: string }`. The screen shows the participant name (record) or "Details you entered", the site, a model picker populated from `getConfig().models` (default `defaultModel`), and three plain statements of what the agent will do (works only on this site; asks you for anything missing; stops before Submit). A Start button is disabled while a start is in flight (guards a double press).

- [ ] **Step 1: Write failing tests**: `buildStartRequest` shapes for both participant kinds, model omitted when it equals the config default, origin must be an https origin (else throws), details never include keys outside `DETAIL_FIELDS`.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** `feat(extension): add the panel Ready screen`.

### Task 8: Run starter

**Files:**
- Create: `extension/panel/lib/run-starter.ts`
- Test: `tests/relay/panel-run-starter.test.ts`

**Interfaces:**
- Produces: `startRun(deps: { api: AppApi; worker: WorkerApi; fetch: typeof fetch; appOrigin: string; newId: () => string; now: () => number }, args: { tabId: number; request: StartRequest; siteName: string }): Promise<{ ok: true; runId: string; expiresAt: number } | { ok: false; error: unknown; runId?: string }>` implementing the sequence: (1) check the tab is still open and on the request origin (`chrome.tabs.get`); (2) `runId = newId()` (a UUID); (3) POST `${appOrigin}/api/eve-chat` with `{ id: runId, relay: request }`, `credentials: 'include'`, `redirect: 'error'`, reading the response only until the first chunk or until a 4xx/5xx status (the route keeps consuming the turn after the client leaves, so the panel cancels the stream reader after the first chunk and does not wait for the turn); a non-2xx maps through the error codes (400 with `message_not_allowed`/`details_*`/`model_not_allowed`/`sensitive_detail` -> generic bad request copy, 403 -> forbidden, 429 -> run limit, 401 -> signed out); (4) `api.getExtensionToken(runId)`; (5) `worker.startRun({ relayUrl, token, tabId })`; (6) `registrationComplete(runId, 0)` because a fresh run has no registered values; (7) `addRecent`. A failure after step 3 calls the server stop route (`api.stopRun(runId)`) so no orphan run keeps working, then returns the error. A second `startRun` while one is in flight returns `{ ok: false, error: new StartInFlightError() }` without any request.
- The agent may call the browser before the extension is attached: step 4 and 5 run as soon as step 3's first chunk arrives; the plan's Context note: the Eve tool retries `not attached` (verify in `agent/skills/extension-run/SKILL.md` and `lib/relay/eve-relay-browser.ts`) and the task's report records what it found.

- [ ] **Step 1: Write failing tests** with fakes: the call order and bodies; the reader is cancelled after the first chunk; failure at each step (tab gone, 403, 429, 500, token 404, worker refusal) returns the specific error and calls server stop exactly when a run was created; double start -> second refused with no requests; no applicant detail appears in any thrown error message or console call (spy); a details payload is sent verbatim as `participant.details` and never logged.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.** Do not log or retain the request body after the call.
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** `feat(extension): start a run from the panel`.

### Task 9: Run session: status, refresh, stop

**Files:**
- Create: `extension/panel/lib/run-session.ts`, `extension/panel/screens/running.tsx` (minimal)
- Test: `tests/relay/panel-run-session.test.ts`

**Interfaces:**
- Produces: `createRunSession(deps, run: { runId; expiresAt }): RunSession` with `subscribe(cb)`, `stop()`, `dispose()`. It polls the worker `getRunStatus` every 2 seconds and the server `getRunState` every 5 seconds; derives `state: 'running' | 'ended'` and `endedReason`; refreshes the token when 5 minutes remain (P4c-e: `getExtensionToken` then a same-run-id `worker.startRun`, followed by `registrationComplete(runId, 0)`; a refusal sets `refreshFailed` and shows a notice); when the worker reports no active run while the server row is live (worker-initiated end: left-origin, token-expired, debugger-canceled, debugger-detached, tab closed), calls the server stop route once and marks the run ended with the reason; `stop()` calls `api.stopRun(runId)` then `worker.stopRun()` (each failure ignored but recorded), marks ended, `markRecentEnded`. `dispose()` stops all timers. The minimal Running screen shows site name, an elapsed timer, the pending-hold notice if `getRunStatus.pendingOrigin` is set ("The page moved to another site. Waiting for your decision": the decision UI is 4c-2), and a Stop button.
- Per owner decision P4c-f and the tracker: tokens minted before Stop stay valid up to 1800 s; the panel cannot fix that.

- [ ] **Step 1: Write failing tests** with fake timers and fakes: polling cadence; refresh at 5 minutes remaining and its refusal path; a worker-initiated end triggers exactly one server stop; Stop order and double Stop; a worker that answers errors does not end the run; `dispose` leaves no timers; server `ended: true` ends the session; polling pauses cleanly when the panel unmounts.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** `feat(extension): track a run from the panel and stop it on the server`.

### Task 10: Panel end-to-end

**Files:**
- Create: `tests/relay/e2e/panel.e2e.test.ts`

**Interfaces:**
- Drives the built extension in Chromium (reuse the harness and the fake-app server pattern of `app-auth` and `jev-classification`): a fake app serving config, participants, `eve-chat` (a stream that emits one chunk then stays open), `relay/token`, `runs/:id`, `runs/:id/stop`; open `panel.html` as an extension page, and assert: signed-out state with a 401, then the Home screen after sign-in; search results render and a stale response is ignored; entering details, saving them, reloading the panel shows "Reuse saved details", and with the clock moved past 8 hours (inject the clock through the page's `Date` override in the test) they are gone; starting a run issues the fake app calls in order and the worker receives `startRun` (read through the test hooks); Stop issues the server stop and the worker stop; closing and reopening the panel mid-run rebuilds the Running screen from the worker status and the state route without a second `eve-chat` call.

- [ ] **Step 1: Write the e2e.** If an interaction cannot be driven headlessly (the side panel UI itself), drive the same `panel.html` as a normal extension page tab and say so in the test header.
- [ ] **Step 2: Run `pnpm extension:build` with the hooks build used by the other e2e suites and `pnpm test:relay e2e/panel.e2e.test.ts`, expect PASS; run the other relay e2e files once to confirm nothing regressed.**
- [ ] **Step 3: Commit** `test(extension): cover the panel start flow end to end`.

### Task 11: Docs and tracker

**Files:**
- Modify: `CLAUDE.md`, `docs/browser-extension-todo.md`
- Create: `docs/changes/2026-10-29-add-side-panel-start-flow.md` (use the change-docs skill)

- [ ] **Step 1: Update the docs from the diff**: the panel's location and build, what it may import, the screens that exist, saved details rules, recents, the start sequence, the worker messages that gained `runId`, how to build and load the extension for a manual check. Tick the carry-forward items this plan closed (call both Stops, worker-initiated ends, token refresh, the `runId` check, hide nothing yet since pending questions are 4c-2) and move the rest to 4c-2.
- [ ] **Step 2: Run the whole node suite** `pnpm exec vitest run --config vitest.config.node.mjs tests/agent tests/relay --exclude 'tests/relay/e2e/**'` and the relay e2e files once; expect PASS.
- [ ] **Step 3: Commit** `docs: document the side panel start flow`.

## Self-Review Notes

- Spec coverage: screens 1 to 4 (Home, Participant from the system, Participant entered, Ready) and Saved Details are Tasks 4 to 7; "the extension locates and starts runs" is Task 8; the carry-forward contracts for Stop, worker-initiated ends, token refresh and the `runId` check are Tasks 2 and 9. Running, Needs You, sensitive typing, allow-this-site, Review and Confirmed are plan 4c-2 by design.
- Ordering: Task 1 before everything (the bundle); Task 3 before the screens; Task 8 needs Tasks 3 and 7; Task 9 needs Task 8; the e2e (Task 10) needs all of them.
- Riskiest edits: Task 1 (a second build target and Tailwind inside the extension build, which every e2e suite runs) and Task 8 (the first time the panel creates a run: the order of the app calls against what the agent expects).
- The panel is tested mostly through pure logic modules (controllers, storage, starter, session) so tests stay fast and DOM-free; React components stay thin.
