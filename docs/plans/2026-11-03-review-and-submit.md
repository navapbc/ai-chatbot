# Review And Submit Implementation Plan (Plan 4c-4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When the agent stops, the caseworker can press "Show me the unfilled field" and see the next required field that is still empty highlighted in their own tab. When the caseworker submits the form on the site, the panel notices and moves to Confirmed, showing the confirmation number when the page displays one. The manual "I submitted it" button stays as the fallback.

**Architecture:** Two new worker messages do the page work, because only the extension service worker can reach the tab. `showUnfilledField` runs one read-only script in the isolated `relay-focus` world, counts empty required controls, classifies the first few with the redactor and highlights the first with `Overlay.highlightNode`; it returns counts only. `armSubmitWatch` and `disarmSubmitWatch` turn on detection of a form-submission navigation from raw debugger events while the panel is in Review; a detected submission is reported through `getRunStatus`, with a confirmation number read from the page by a labelled pattern, before the panel stops the run. The panel adds the Review button, a "did you submit?" prompt for an unconfirmed detection, and the confirmation number on Confirmed.

**Tech Stack:** MV3 service worker (`extension/src/bridge.ts`, `extension/src/typing-scripts.ts` pattern), `chrome.debugger`, the panel from plans 4c-1 to 4d, vitest node project, Playwright Chromium e2e. No new dependencies, no manifest permission change, no server or schema change.

**Spec:** `docs/specs/2026-10-23-browser-extension-design.md` (Review and Confirmed, Sensitive Value Redaction). Tracker: `docs/browser-extension-todo.md`, "Plan 4c-2 carry-forward". Previous plans: `docs/plans/2026-11-01-reliability-before-flag.md`, `docs/plans/2026-10-31-side-panel-sensitive-typing.md`, `docs/plans/2026-10-30-side-panel-running-view.md`.

**Spec corrections:** the spec says to "observe the page's submit event or a click on a submit-type control". The extension has no content script, no `scripting` permission and no host permission, so that is not available. The only signals are `chrome.debugger` events; this plan uses `Page.frameRequestedNavigation` (reason `formSubmissionGet` or `formSubmissionPost`) followed by a main-frame `Page.frameNavigated` that stays in scope. A submission that fires no navigation (an XHR or single-page app) is not detected, and the manual button covers it.

## Global Constraints

- Node 24: `export PATH=/Users/kaylynvannorstrand/.nvm/versions/node/v24.20.0/bin:$PATH`. `pnpm` only. No new dependencies; if one seems necessary, stop and ask.
- Tests: `pnpm exec vitest run --config vitest.config.node.mjs <path>`; extension work also runs `pnpm extension:build && pnpm test:relay e2e/panel-shell.e2e.test.ts e2e/bridge.e2e.test.ts`. Lint with `pnpm exec biome lint <files>` (extension paths are excluded by config; then format only the files you touched by hand); never `pnpm lint`; never `biome format --write` on a directory.
- Never read `.env*` files or print secret values. No migration, no server change.
- Worker-initiated CDP skips the agent's command policy, so every new worker page read re-applies the checks the typing path applies: run live and not stopped or blocked, tab in scope (`requireTabInScope`, `requireSameScope`), the per-tab lock (wait out a screenshot, hold shared), bounded steps (`withTimeout`), and `needsRegistration` refuses. A new isolated world goes into `BRIDGE_WORLDS`; reuse `relay-focus`.
- The agent-facing `attached` flag is set only by the agent's relay `attach`. A worker-initiated attach (typing, the scan, the submit watch) never sets it, and the Eve tool still sees `not attached` on that path. `Overlay.*` and `Accessibility.getPartialAXTree` stay out of the agent allowlist, and `Overlay.*` events stay dropped.
- What a worker page read may return to the panel: counts and flags, never a field value, name, label or page text, except the confirmation token below. Everything a read returns passes through the redactor's checks (`masksFilledField`, `hasSensitiveShape`, `hasKnownValue`) first, and any failure counts as sensitive or refuses.
- The confirmation number goes only to the panel (React text). It is never sent to the server, the relay, the agent, recents, storage or a log, and it is read before the run is stopped, because stop clears the redactor and detaches.
- The panel keeps recents at exactly `runId`, `siteName`, `startedAt`, `endedReason`. Fixed copy comes from `extension/panel/lib/copy.ts`; worker error text is mapped to codes and never shown.
- Panel code keeps the 4c-1 import rules and the bundle guard test stays green. Frontend conventions from `CLAUDE.md` apply: Tailwind only, `handle` prefix, early returns, aria attributes, keyboard support.
- `RELAY_RUNS_ENABLED` and `RELAY_TOKEN_ENDPOINT_ENABLED` stay unset in deployment config. The web app and the legacy chat route are unchanged.

## Decisions For Owner

Defaults chosen so the work can proceed; each is a one-line change.

- **P4c4-a Locator.** The worker finds unfilled fields itself (the agent's question carries no field identity). A control counts as unfilled when it is visible, enabled and either fails `validity.valueMissing` or has `aria-required="true"` with no value (an unchecked radio group or checkbox group, an empty select, an empty text control). Up to 12 candidates are classified; a classification failure counts as sensitive.
- **P4c4-b What the panel learns.** Only `{ok, found, remaining, sensitive?, capped?}`: whether a field was highlighted, how many remain, whether the highlighted one is sensitive, whether the scan stopped at its cap. No label, name or value leaves the worker. The highlight is on the caseworker's own page, so a sensitive field may be highlighted; the panel says so in fixed words.
- **P4c4-c Highlight.** `DOM.scrollIntoViewIfNeeded` then `Overlay.highlightNode` for about 6 seconds, cleared on timer, navigation, detach and stop. It has no DOM footprint, so the agent's views and screenshot masking are unaffected. If Overlay is unavailable the worker scrolls and focuses the field instead.
- **P4c4-d Submit detection.** Armed only while the panel is in Review. A main-frame navigation in scope that follows a `formSubmission*` request is reported as `submitted`. The panel treats it as submitted only when a confirmation number was found; otherwise it asks "Did you submit the form?" with Yes and No, because a "Next" click in a multi-step form looks the same. "I submitted it" stays.
- **P4c4-e Attach in Review.** While the agent is attached the watch uses its attach. When it is not (after a token refresh), the worker attaches on demand as typing does, with a flag `reviewAttached` that never sets `attached`, and detaches when the watch is disarmed.
- **P4c4-f Confirmation number.** Read once the page has loaded after the submission (up to 3 tries, 1 second apart) by an in-page labelled pattern: `confirmation number`, `confirmation #`, `confirmation code`, `reference number`, `application number`, `tracking number`, followed by a token of 4 to 32 characters that contains a digit. `case number` is not used: it is on the sensitive-field list. The worker drops the token if it has a sensitive shape (9 or more digits), matches a registered value, or its label is sensitive.

## Review Focus

- A form with required fields of every kind (text, select, radio group, checkbox, a sensitive field, a hidden or disabled one, one inside an iframe): expect the first visible empty one highlighted, a correct remaining count, nothing read from a sensitive field.
- A "Next" click, a validation error that does not navigate, a page reload, an XHR submit and a cross-origin redirect after submit: expect no false confirmed state, and the manual button still works.
- A confirmation page that echoes the applicant's SSN or shows a 10-digit number, a "Case number", or no label at all: expect no number shown, never the SSN.
- The panel closed or reopened in Review, a token refresh in Review, the agent attached or not: expect the watch to resume or be re-armed and no stuck debugger attach.
- Stop pressed during a scan or a detection: expect nothing left highlighted and no leaked state.

---

### Task 1: The worker finds and highlights the next unfilled field

**Files:**
- Modify: `extension/src/bridge.ts` (`BridgeMessage`, `handleMessage`, a new `showUnfilledFieldNow`), `extension/src/typing-scripts.ts` (the in-page scan script)
- Test: `tests/relay/bridge-unfilled-field.test.ts` (create), `tests/relay/e2e/bridge.e2e.test.ts` (extend)

**Interfaces:**
- Consumes: the typing path's discipline (`queuedAttach`/`attachForTyping`/`releaseTypingAttach`, `lockFor`, `untilNoScreenshot`, `holdShared`, `requireTabInScope`, `requireSameScope`, `live()`, `sessionSend`, the `relay-focus` world, `masksFilledField`).
- Produces: message `{type: 'showUnfilledField', runId: string}` replying `{ok: true, found: boolean, remaining: number, sensitive?: boolean, capped?: boolean}` or `{ok: false, error}` with the same fixed strings as `typeSensitive`'s `TypeRefusal` where they apply (`no active run`, `wrong run`, `not attached`, `run is held`, `field out of scope`) plus `registration pending`. The in-page script `SCAN_UNFILLED` returns, per candidate, only an index and kind flags, never a value or label. After counting, the worker classifies up to 12 candidates with `masksFilledField` and `isSensitiveField` (any failure means sensitive), highlights the first with `DOM.scrollIntoViewIfNeeded` and `Overlay.highlightNode` (about 6 s, then `Overlay.hideHighlight`; fall back to scroll plus focus when Overlay is unavailable), and clears the highlight on navigation, detach and stop. Read-only: it types nothing and registers nothing.

- [ ] **Step 1: Write failing tests** with the fake `chrome.debugger`: success counts and highlights once; no candidates gives `{ok: true, found: false, remaining: 0}`; a sensitive first candidate sets `sensitive: true`; a classification failure counts as sensitive; each refusal string for its condition and no CDP call on a refusal; `needsRegistration` refuses; a screenshot in flight delays the scan; the reply never contains a value, label or page text; the highlight is cleared on stop, navigation and timeout; Overlay unavailable falls back to scroll plus focus. Real Chromium: a page with a text field, a select, a radio group, a checkbox, a disabled field, a hidden field and a sensitive field returns the right count and highlights the first visible empty one.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement `showUnfilledFieldNow(runId: string): Promise<BridgeResponse>`** and the `SCAN_UNFILLED` script.
- [ ] **Step 4: Run, expect PASS** (`tests/relay`, then `pnpm extension:build && pnpm test:relay e2e/bridge.e2e.test.ts`).
- [ ] **Step 5: Commit** `feat(extension): find and highlight the next unfilled field`.

### Task 2: The Review button for the unfilled field

**Files:**
- Modify: `extension/panel/lib/worker-api.ts`, `extension/panel/lib/copy.ts`, `extension/panel/screens/review.tsx`, `extension/panel/screens/running.tsx`
- Test: `tests/relay/panel-worker-api.test.ts` (extend), `tests/relay/panel-review.test.ts` (extend)

**Interfaces:**
- Consumes: Task 1's message and reply.
- Produces: `WorkerApi.showUnfilledField(runId: string): Promise<{ok: true, found: boolean, remaining: number, sensitive?: boolean, capped?: boolean} | {ok: false, code: UnfilledFieldCode}>` (never throws; worker text is never kept; codes map as for `typeSensitive`). A "Show me the unfilled field" button above "I submitted it" on Review; the result appears in a `role="status"` line from fixed copy (`found`: "Showing the next field that is still empty. N left."; `found` with `sensitive`: adds that it is a sensitive field to fill in yourself; not found: "No empty required fields found. Check the page before you submit."; `capped`: "Showing the first of many."; each refusal code has its own sentence). The button is disabled while a call is in flight (a ref guards a double press) and while the run is held.

- [ ] **Step 1: Write failing tests**: each worker string maps to its code; an unknown or thrown error maps to `worker_error`; the reply has no label or value property; the button renders on Review only; the status line shows each copy case; a double press makes one call; the button is disabled during a hold; the call is made with the run id.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS** (`tests/relay`, bundle guard test, `tsc --noEmit` with only the three known error files).
- [ ] **Step 5: Commit** `feat(extension): add show me the unfilled field to Review`.

### Task 3: The worker watches for a submission

**Files:**
- Modify: `extension/src/bridge.ts` (`BridgeMessage`, `handleMessage`, `onDebuggerEvent`, `RunStatus`, the attach helpers)
- Test: `tests/relay/bridge-submit-watch.test.ts` (create), `tests/relay/e2e/bridge.e2e.test.ts` (extend)

**Interfaces:**
- Consumes: the attach helpers and the per-run attach queue from plan 4d, `mainFrameNavigationUrl`, `inRunScope`.
- Produces: messages `{type: 'armSubmitWatch', runId}` and `{type: 'disarmSubmitWatch', runId}` replying `{ok: true}` or `{ok: false, error}` (`no active run`, `wrong run`, `not attached`, `run is held`). While armed the worker taps raw debugger events before the early return for an unattached run: a `Page.frameRequestedNavigation` for the main frame with reason `formSubmissionGet` or `formSubmissionPost` arms a pending submission, and the next main-frame `Page.frameNavigated` in scope and not held, to a different URL or the same URL after a POST, sets `RunStatus.submitted = {at: number}` (no URL, no query). When the agent is not attached, arming attaches on demand with a `reviewAttached` flag that never sets `attached`, and the typing attach helpers treat it as held; disarming, stop and a detach clear it. A navigation out of scope follows the existing hold and leave-origin rules unchanged. Disarm clears `submitted`. `getRunStatus` returns `submitted` only while armed.

- [ ] **Step 1: Write failing tests**: a POST submission followed by a main-frame navigation sets `submitted`; a GET form submission does the same; a bare navigation, a same-document navigation and an iframe navigation do not; a navigation to a not-allowed origin holds the run as before and does not set `submitted`; disarm clears it; arming while unattached attaches on demand and `attached` stays false (an agent command still gets `not attached`); the typing attach is refused or waits while `reviewAttached`; a detach clears the watch; arming twice is idempotent; `needsRegistration` is not a refusal for arming (no page text is read). Real Chromium: a form POST to a same-origin confirmation page sets `submitted`; a validation re-render with no navigation does not.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS** (`tests/relay`, then the bridge e2e).
- [ ] **Step 5: Commit** `feat(extension): watch for a form submission in Review`.

### Task 4: The worker reads the confirmation number

**Files:**
- Modify: `extension/src/bridge.ts` (the read after a detected submission), `extension/src/typing-scripts.ts` (the in-page pattern script)
- Test: `tests/relay/bridge-confirmation.test.ts` (create), `tests/relay/e2e/bridge.e2e.test.ts` (extend)

**Interfaces:**
- Consumes: Task 3's detection, `hasSensitiveShape`, the redactor's `hasKnownValue`, `isSensitiveField`.
- Produces: after a detected submission the worker waits for the page to load and reads the confirmation page up to 3 times, 1 second apart, in the isolated world. The in-page script matches only a labelled pattern (`confirmation number`, `confirmation #`, `confirmation code`, `reference number`, `application number`, `tracking number`, case-insensitive) followed by a token of 4 to 32 characters that contains a digit and only `A-Za-z0-9-_/.`, and returns that token or nothing. The worker drops the token when it has a sensitive shape (`hasSensitiveShape`), matches a registered value, its label is sensitive per `isSensitiveField`, `needsRegistration` is set, or the run is held or stopped; then `RunStatus.submitted` is `{at, confirmation?: string}`. Nothing else from the page text is returned or logged, and the token goes nowhere but `getRunStatus`.

- [ ] **Step 1: Write failing tests**: a labelled alphanumeric token is returned; a numeric token of 9 or more digits is dropped; a token equal to a registered value is dropped; `Case number: 4455` is not used; an echoed SSN on the page is never returned; no label gives no confirmation; a token over 32 characters is dropped; the read stops at 3 tries; a stop during the read returns nothing and clears state; the status never contains the page text. Real Chromium: a confirmation page with `Confirmation number: AB-1234` returns it, one with `Confirmation number: 123456789` and one with an echoed SSN do not.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS** (`tests/relay`, then the bridge e2e).
- [ ] **Step 5: Commit** `feat(extension): read the confirmation number after a submission`.

### Task 5: The panel moves to Confirmed on a detected submission

**Files:**
- Modify: `extension/panel/lib/worker-api.ts`, `extension/panel/lib/run-session.ts`, `extension/panel/lib/state.ts`, `extension/panel/app.tsx`, `extension/panel/screens/running.tsx`, `extension/panel/screens/review.tsx`, `extension/panel/screens/confirmed.tsx`, `extension/panel/lib/copy.ts`
- Test: `tests/relay/panel-run-session.test.ts`, `tests/relay/panel-review.test.ts`, `tests/relay/panel-state.test.ts` (extend)

**Interfaces:**
- Consumes: Tasks 3 and 4 (`armSubmitWatch`, `disarmSubmitWatch`, `RunStatus.submitted`).
- Produces: `WorkerApi.armSubmitWatch(runId)` and `disarmSubmitWatch(runId)` (never throw). The run session arms the watch when the phase becomes `review` and disarms when it leaves `review` or the run ends. When a polled status carries `submitted`: with a `confirmation`, it captures the number first, then calls `stop('submitted')` as "I submitted it" does, and `runConfirmed` carries `{stopFailed, how: 'detected', confirmation}`; without one, the Review screen shows "Did you submit the form?" with Yes (same path, `how: 'detected'`) and No (disarms and re-arms the watch, clearing the prompt). `State.confirmed` gains `how` and optional `confirmation`. `confirmed.tsx` shows "Confirmation number" and the token as React text when present, and the wording differs by how it was confirmed (detected: "The form was submitted."; manual: "You said you submitted."). `ENDED_TEXT.submitted` and the Review copy no longer say submission is always manual. The number is held only in component or reducer state for the Confirmed screen, never stored, and is dropped on `startAnother` and `clearTarget`.

- [ ] **Step 1: Write failing tests**: arm on entering review and disarm on leaving it or ending; a detected submission with a number goes to Confirmed with the number after one stop (server then worker); one without a number shows the prompt, Yes confirms, No re-arms and clears the prompt; the number is captured before the stop (the stop does not lose it); a failed stop still shows Confirmed with the existing notice; the number never reaches `recents`, storage, the answer client or the server stop call (assert by serializing); `startAnother` clears it; a double detection does not stop twice; a refresh in Review re-arms.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS** (`tests/relay`, bundle guard test, `tsc --noEmit`; then `pnpm extension:build && pnpm test:relay e2e/panel-shell.e2e.test.ts`).
- [ ] **Step 5: Commit** `feat(extension): confirm a submission from the page`.

### Task 6: End to end

**Files:**
- Modify: `tests/relay/e2e/panel-shell.e2e.test.ts` (or a new `tests/relay/e2e/review-and-submit.e2e.test.ts` if the file's runtime would grow more than about 30 s)

**Interfaces:**
- A scripted scenario against the real worker, a real relay and the fake app, in a real page: a form with required fields of every kind; the agent's turn ends and Review shows; "Show me the unfilled field" highlights the first empty field and the status line counts the rest (assert through the page's own DOM that nothing about the highlight is in the agent-facing snapshot and the screenshot mask still works); filling the fields and pressing the page's submit button POSTs to a confirmation page with `Confirmation number: AB-1234`; the panel moves to Confirmed showing `AB-1234` with exactly one server stop and one worker stop; a second run: a confirmation page with no label shows the "Did you submit?" prompt and Yes confirms; a third: an echoed SSN and a 10-digit number show no confirmation number and never the SSN anywhere in the panel DOM, console, storage or fake-app requests. Header states how it is driven (panel as a tab, per 4c-2).

- [ ] **Step 1: Write the e2e.**
- [ ] **Step 2: Run** `pnpm extension:build` with the hooks build, then `pnpm test:relay e2e/panel-shell.e2e.test.ts` (and any new file), then the other relay e2e files once; expect PASS. Report the added runtime against the relay-e2e job's 20 minute limit.
- [ ] **Step 3: Commit** `test(extension): cover review and submit end to end`.

### Task 7: Docs and tracker

**Files:**
- Modify: `CLAUDE.md`, `docs/browser-extension-todo.md`
- Create: `docs/changes/2026-11-03-add-review-and-submit.md` (use the change-docs skill; H1 in Title Case)

- [ ] **Step 1: Update the docs from the diff**: the two worker messages and their refusals, what the scan returns and never returns, the highlight and its fallback, submit detection (armed in Review, POST or GET form submission plus an in-scope main-frame navigation, what it cannot see), the confirmation number rules and where it goes, the Review and Confirmed copy and the "Did you submit?" prompt, the spec correction (no content script); tick the 4c-4 items; remove every "plan 4c-4 remains" sentence; add the follow-ups that surface (XHR and single-page submits are not detected, a multi-step form's "Next" shows the prompt, Overlay is not verified on every Chrome version, the infobar while the review attach is held, unverified real benefit-site behavior).
- [ ] **Step 2: Run the whole node suite** `pnpm exec vitest run --config vitest.config.node.mjs tests/agent tests/relay --exclude 'tests/relay/e2e/**'` and the relay e2e files once; expect PASS.
- [ ] **Step 3: Commit** `docs: document review and submit`.

## Self-Review Notes

- Spec coverage: "Show me the unfilled field" (Tasks 1, 2), submit detection with a manual fallback (Tasks 3, 5), the confirmation number on Confirmed (Tasks 4, 5), tests that put sensitive content on the confirmation page (Tasks 4, 6). The spec's content-script detection is replaced (see Spec corrections).
- Ordering: Task 1 before Task 2; Task 3 before Task 4; Tasks 3 and 4 before Task 5; Task 6 needs all of them. Tasks 1 and 3 both touch `bridge.ts`, so they run one after the other.
- Riskiest edits: Task 3 (a third worker-initiated attach path next to the agent's and typing's, tapping raw events) and Task 4 (the first time the worker reads page text and returns a piece of it to the panel).
- Left out deliberately: detecting XHR and single-page submits, reading any other page text, sending the confirmation number anywhere but the panel, a hub or server change, multi-site testing (plan 5).
