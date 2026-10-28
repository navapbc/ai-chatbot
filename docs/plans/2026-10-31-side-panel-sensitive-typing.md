# Side Panel Sensitive Typing Implementation Plan (Plan 4c-3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When the agent asks the caseworker for a sensitive value (SSN, account number and similar), the caseworker can type it into the side panel and the extension types it into the page for them, so the value never reaches the model, the server, Eve state or traces. The agent is told only "provided".

**Architecture:** A new worker message `typeSensitive` takes the value from the panel, checks that the field the caseworker has focused in the tab is one the redactor masks, registers the value for scrubbing, then types it with `Input.insertText`. The worker also keeps its registered values across a same-run token refresh, so a refresh cannot silently drop scrubbing. The panel's sensitive question card gains a masked input with an eye toggle and a "Fill it in" button; on success it clears the value and sends the existing `provided` answer.

**Tech Stack:** MV3 service worker (`extension/src/bridge.ts`), `chrome.debugger`, React 19 panel from plans 4c-1 and 4c-2, vitest node project, Playwright Chromium e2e. No new dependencies, no server or schema change.

**Spec:** `docs/specs/2026-10-23-browser-extension-design.md` (Needs You, Sensitive Value Redaction). Tracker: `docs/browser-extension-todo.md`, "Plan 4c-2 carry-forward". Previous plans: `docs/plans/2026-10-30-side-panel-running-view.md`, `docs/plans/2026-10-29-side-panel-start-flow.md`.

**Scope note:** The unfilled-field highlighter ("Show me the unfilled field"), automatic submit detection and the confirmation number are plan 4c-4, not this plan.

## Global Constraints

- Node 24: `export PATH=/Users/kaylynvannorstrand/.nvm/versions/node/v24.20.0/bin:$PATH`. `pnpm` only. No new dependencies; if one seems necessary, stop and ask.
- Tests: `pnpm exec vitest run --config vitest.config.node.mjs <path>`; extension work also runs `pnpm extension:build && pnpm test:relay e2e/panel-shell.e2e.test.ts e2e/bridge.e2e.test.ts`. Lint with `pnpm exec biome lint <files>` (some extension paths are excluded by config; then format only the files you touched); never `pnpm lint`; never `biome format --write` on a directory.
- Never read `.env*` files or print secret values.
- A sensitive value lives only in the typing component's React state and in worker memory. It is never put in the panel reducer or any `Draft`, `localStorage`, `chrome.storage`, a URL, an error string, a log line, a test name or a snapshot. It is cleared from the component in a `finally` after the worker call. It is never passed to `answer-client.ts` or any HTTP call. The only thing the server sees is `optionId: 'provided'`.
- Order inside the worker is fixed: register the value for scrubbing (refuse to type if registration is refused), then verify the target, then type. If the target check fails, the value stays registered (scrubbing more is safe) and nothing is typed.
- The worker types only into a field it can prove is in scope and masked by the redactor (the same verdict the redactor uses to hide a field's value). A field the classifier does not flag is refused with a fixed message; the caseworker types it themselves.
- Worker-initiated CDP skips the agent's command policy, so the new helper re-implements the checks the agent path applies: run active and attached, not blocked or stopped, tab in scope, the per-tab lock (wait out any screenshot), the focused-frame guard, a timeout. Any new isolated world must be added to `BRIDGE_WORLDS`.
- Everything the panel renders from the server or agent is untrusted text, rendered as text only (unchanged from 4c-2). Fixed copy comes from `extension/panel/lib/copy.ts`-style tables; worker error text is mapped to codes and never shown.
- Panel code keeps the 4c-1 import rules and the bundle guard test stays green. Frontend conventions from `CLAUDE.md` apply: Tailwind only, `handle` prefix, early returns, aria attributes, keyboard support.
- `RELAY_RUNS_ENABLED` and `RELAY_TOKEN_ENDPOINT_ENABLED` stay unset in deployment config. The web app and legacy chat route are unchanged.

## Decisions For Owner

Defaults chosen so the work can proceed; each is a one-line change.

- **P4c3-a Field locator.** The caseworker clicks the field in the page, then presses "Fill it in". The worker types into the focused field only if the redactor masks it. The spec's "label and role the agent supplies" cannot work: the agent's question has no such fields and the agent cannot see a sensitive field's label. (Owner chose this over a worker scan or a structured locator.)
- **P4c3-b One message.** `typeSensitive {runId, value}` registers, checks and types in the worker, so the value crosses the extension boundary once. The existing `addKnownValue` message stays but the panel does not call it.
- **P4c3-c Refresh.** The worker carries its registered values across a same-run `startRun` (token refresh), in memory only. A worker restart still loses them; the panel keeps today's fail-closed gate and field masking, and the todo records the residual. Values are never persisted.
- **P4c3-d Short values.** A value under 3 characters is typed but cannot be scrubbed (the redactor stores nothing for it); field masking still applies. The card does not block it.
- **P4c3-e Limits.** The redactor holds at most 32 values of at most 512 characters per run. When it refuses, the card says the value could not be typed here and offers "I'll type it myself".
- **P4c3-f Free-text sensitive question.** A free-text question whose prompt is flagged sensitive gets the same typing card. Its answer is the text `provided`, and `agent/skills/extension-run/SKILL.md` gains one line: a text answer of exactly `provided` means the caseworker entered the value and the agent must not ask again.

## Review Focus

- Typing into the wrong field: focus moved by the page, an unmasked field, a field inside an out-of-scope iframe, a hidden or read-only field, no focus at all. Expect a fixed refusal, nothing typed.
- The value echoed by the page into an unclassified element, a URL, a review text or a screenshot after typing: expect scrubbing and mask to cover it in every reply and event the agent can see.
- A token refresh, a second panel window or an origin hold in the middle of typing: expect no typing into a held tab and no loss of scrubbing.
- The value left behind: in the input after success or failure, in React devtools state, in the reducer, in an error message, in a log line, in the DOM after the card unmounts.
- A paste with spaces or newlines, a 600-character paste, a value under 3 characters, 33 values in one run, Fill it in pressed twice, Stop pressed while typing.

---

### Task 1: The worker keeps known values across a same-run refresh

**Files:**
- Modify: `extension/src/bridge.ts` (`startRunNow`, `begin`, `endRun`, the `addKnownValue` path), `lib/relay/redactor.ts` only if a getter is needed
- Test: `tests/relay/bridge-known-values.test.ts` (create or extend the existing bridge unit test file that covers `addKnownValue`)

**Interfaces:**
- Produces: the run keeps an in-memory list of the values `addKnownValue` accepted (a `string[]` beside the redactor, because the redactor exposes no getter). A same-run `startRun` re-registers them on the new redactor before `begin` returns, sets `hasKnownValues` accordingly and does not set `needsRegistration` for lost values it carried. Every other end (`stopRun`, `endRun` without replacement, a different run id) clears the list. Nothing logs or stores the list.

- [ ] **Step 1: Write failing tests**: after `addKnownValue(runId, 'v1')` and a same-run `startRun`, a reply containing `v1` is still scrubbed and `getRunStatus` has `needsRegistration` false; after `stopRun` and a new run, a reply containing `v1` is not scrubbed and the list is empty; a different run id does not inherit; the 32-value cap holds across a refresh; a value of under 3 characters is carried as the redactor treats it today.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** the list, the carry in `startRunNow` when `sameRun`, and the clear in `endRun` for non-replacing ends. `valuesLost` only becomes true for a restart (no list available).
- [ ] **Step 4: Run, expect PASS** (`tests/relay` whole directory).
- [ ] **Step 5: Commit** `fix(extension): keep known values across a token refresh`.

### Task 2: The worker `typeSensitive` message

**Files:**
- Modify: `extension/src/bridge.ts` (`BridgeMessage`, `handleMessage`, a new `typeSensitiveNow`), `extension/src/test-hooks.ts` only if hooks need the new message
- Test: `tests/relay/bridge-type-sensitive.test.ts` (create)

**Interfaces:**
- Consumes: Task 1's list; the existing focus read (`focusedFrameId`, the `relay-focus` world), `describeVia`, the redactor's field verdict, the per-tab lock helpers (`lockFor`, `untilNoScreenshot`, `holdShared`), `requireTabInScope`.
- Produces: message `{type: 'typeSensitive', runId: string, value: string}` replying `{ok: true}` or `{ok: false, error: <one of the fixed strings below>}`. Error strings (exact): `no active run`, `wrong run`, `not attached`, `run is held`, `value not accepted` (empty after trim, over 512 characters or the redactor refused), `no focused field`, `field not sensitive`, `field out of scope`, `type failed`. The reply never contains the value, the field's name or label, or any page text.
- Order: validate run and state; register the value (`addKnownValue` semantics, so a `false` returns `value not accepted` and nothing is typed); wait out a screenshot and hold the shared tab lock; read the focused element in the isolated world (through open shadow roots and same-origin iframes, as `focusedFrameId` does); refuse when there is none, it is not a text-entry element, it is read-only or disabled, its frame is out of scope, or the redactor does not mask it; then `Input.insertText` with a timeout. The element is described through the same path the redactor uses, so a Jev `classifyExtra` verdict counts. No new world unless unavoidable; if one is added it goes into `BRIDGE_WORLDS`.

- [ ] **Step 1: Write failing tests** with the existing fake `chrome.debugger`: success types once and the fake CDP sees exactly one `Input.insertText` whose text equals the value; each error string for its condition; an `Input.insertText` is never sent on any refusal; the reply and every event forwarded up the relay after typing contain no occurrence of the value (page echo in a DOM or AX reply is scrubbed); a screenshot in flight delays typing; Stop during typing leaves no value in the reply; a 600-character value and 33 distinct values return `value not accepted`; a hold (`blocked`) returns `run is held`.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement `typeSensitiveNow(runId: string, value: string): Promise<BridgeResponse>`** in `bridge.ts` and the `handleMessage` case.
- [ ] **Step 4: Run, expect PASS** (`tests/relay`).
- [ ] **Step 5: Commit** `feat(extension): type a sensitive value into the focused field for the caseworker`.

### Task 3: The panel's worker API and fixed copy

**Files:**
- Modify: `extension/panel/lib/worker-api.ts`, `extension/panel/lib/copy.ts`
- Test: `tests/relay/panel-worker-api.test.ts` (extend)

**Interfaces:**
- Consumes: Task 2's message and error strings.
- Produces: `WorkerApi.typeSensitive(runId: string, value: string): Promise<{ok: true} | {ok: false, code: TypeSensitiveCode}>` where `TypeSensitiveCode` is one of `not_attached`, `run_held`, `value_not_accepted`, `no_focused_field`, `field_not_sensitive`, `field_out_of_scope`, `type_failed`, `worker_error`; `no active run` and `wrong run` map to `worker_error`. Fixed copy per code in `copy.ts` (plain sentences telling the caseworker what to do, e.g. `no_focused_field`: "Click the field in the page first, then press Fill it in."; `field_not_sensitive`: "The extension does not treat that field as sensitive, so it will not type there. Type it in the page yourself."). Worker text is never shown, and the value is never part of any error object or copy.

- [ ] **Step 1: Write failing tests**: each worker string maps to its code; an unknown string maps to `worker_error`; a thrown `chrome.runtime.sendMessage` error maps to `worker_error` and the thrown error text (set it to contain the value) is not in the result; the reply object has no `value` property.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** the method and the copy table.
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** `feat(extension): add typeSensitive to the panel's worker API`.

### Task 4: The masked input on the sensitive question card

**Files:**
- Create: `extension/panel/screens/sensitive-fill.tsx`
- Modify: `extension/panel/screens/needs-you.tsx` (sensitive branch)
- Test: `tests/relay/panel-sensitive-fill.test.ts`, extend `tests/relay/panel-needs-you.test.ts`

**Interfaces:**
- Consumes: Task 3's `typeSensitive`; the existing `sensitiveAnswer(req, 'provided' | 'skip')` and the card's send path from plan 4c-2 Task 4; the run id from the session.
- Produces: `SensitiveFill` props `{runId: string, onFilled: () => void, onTypeMyself: () => void}`. It renders a `type="password"` input (`autoComplete="off"`, `spellCheck={false}`, no `name`, no `id` derived from the question), an eye toggle (`aria-pressed`, label "Show what I typed"), the instruction "Click the field in the page, then press Fill it in.", a "Fill it in" button (disabled while empty or in flight, so a second press cannot type twice) and the existing "I'll type it myself" action. On `ok` it clears the value, then calls `onFilled`, which sends `provided` through the existing answer path. On a refusal it shows the fixed copy for the code, keeps the card, and keeps the value only while the card stays mounted (the caseworker may focus the right field and retry); the value is cleared on unmount and in a `finally` when the call settles with success. The value never enters `Draft`, the reducer, `answerRequests` or any prop other than the input's own state.

- [ ] **Step 1: Write failing tests**: value lives only in the input (after success the input is empty and `typeSensitive` was called once); the reducer state and the `answerRequests` payload never contain it (assert by serializing them); eye toggle switches the input type; Fill it in is disabled when empty and while in flight; each refusal code shows its fixed copy with `role="alert"` and is not the worker text; pressing Fill it in twice quickly calls the worker once; unmount clears the value; a pasted value with surrounding spaces or newlines is sent trimmed only at the ends; `provided` is sent only after `ok`; "I'll type it myself" sends `provided` without calling the worker.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** the component and wire it into the sensitive branch of `needs-you.tsx` (keep the batch Send behaviour: a sensitive question answered by typing counts as answered `provided`).
- [ ] **Step 4: Run, expect PASS**; run the bundle guard test and `tsc --noEmit` (only the three known pre-existing error files).
- [ ] **Step 5: Commit** `feat(extension): fill a sensitive value in from the panel`.

### Task 5: A flagged free-text sensitive question gets the same card

**Files:**
- Modify: `extension/panel/lib/question-kind.ts`, `extension/panel/screens/needs-you.tsx`, `agent/skills/extension-run/SKILL.md`
- Test: `tests/relay/panel-question-kind.test.ts` (extend), `tests/relay/panel-needs-you.test.ts` (extend), the existing skill-text test if one pins the SKILL wording

**Interfaces:**
- Consumes: Task 4's `SensitiveFill`.
- Produces: `classifyQuestion` still returns `sensitive` for a flagged free-text prompt; the card renders `SensitiveFill` for it instead of the dead end, and its answer is the text `provided` (a text answer, not an option). The server's prose guard sees only `provided`. The skill line: a text answer of exactly `provided` means the caseworker entered the value in the page; do not ask again and do not read it.

- [ ] **Step 1: Write failing tests**: a flagged free-text card renders the typing card and no text box that would be sent; the answer payload is exactly the text `provided`; an ordinary free-text card is unchanged; the skill text contains the new line.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS** (`tests/relay` and `tests/agent`).
- [ ] **Step 5: Commit** `feat(extension): let the caseworker fill a flagged free-text sensitive question`.

### Task 6: End to end with the real worker

**Files:**
- Modify: `tests/relay/e2e/panel-shell.e2e.test.ts` (or a new `tests/relay/e2e/sensitive-typing.e2e.test.ts` if the file's runtime would grow more than about 30 s)

**Interfaces:**
- A scripted scenario against the real worker, a real relay and a fake app: a page with a masked SSN field and an ordinary field; a sensitive question arrives (Needs You); the caseworker clicks the SSN field in the page, types in the panel, presses Fill it in; assert the page field holds the value, the fake app saw only `optionId: 'provided'` (the value appears nowhere in any request body, URL, or console output of the panel or worker), the agent-facing relay frames (snapshot, `DOM` reply, an echoed review text, a screenshot request) contain no occurrence of the value, a refusal when the ordinary field is focused (nothing typed), then a token refresh in the middle (extend or reuse the refresh hook) and the value is still scrubbed afterwards. Header states how it is driven (panel as a tab, per 4c-2).

- [ ] **Step 1: Write the e2e.**
- [ ] **Step 2: Run** `pnpm extension:build` with the hooks build, then `pnpm test:relay e2e/panel-shell.e2e.test.ts` (or the new file), then the other relay e2e files once; expect PASS. Report the added runtime.
- [ ] **Step 3: Commit** `test(extension): cover typing a sensitive value end to end`.

### Task 7: Docs and tracker

**Files:**
- Modify: `CLAUDE.md`, `docs/browser-extension-todo.md`
- Create: `docs/changes/2026-10-31-add-sensitive-value-typing.md` (use the change-docs skill; H1 in Title Case)

- [ ] **Step 1: Update the docs from the diff**: the `typeSensitive` message and its refusals, the locator decision (caseworker focuses the field), the carry across refresh, the 32/512 limits, the value hygiene rules, the new `provided` text convention; fix every "plan 4c-3" reference that now means 4c-4 (field highlighter, automatic submit detection, confirmation number); tick "Typing sensitive values" and the dead-end card item; add the worker-restart residual (values typed before a restart are not scrubbed; field masking still applies) and the `registrationComplete` count that the worker never verifies.
- [ ] **Step 2: Run the whole node suite** `pnpm exec vitest run --config vitest.config.node.mjs tests/agent tests/relay --exclude 'tests/relay/e2e/**'` and the relay e2e files once; expect PASS.
- [ ] **Step 3: Commit** `docs: document typing sensitive values`.

## Self-Review Notes

- Spec coverage: the Needs You sensitive card (Tasks 3 to 5), typing through the debugger with the agent told only "provided" (Task 2, Task 4), known-value scrubbing registered before typing and surviving a refresh (Tasks 1 and 2), tests with fixtures echoing the value (Tasks 2 and 6). The spec's locator-from-agent-label is replaced by P4c3-a. The highlighter and submit detection are plan 4c-4.
- Ordering: Task 1 and Task 2 (worker) before Task 3; Task 3 before Task 4; Task 5 after Task 4; the e2e (Task 6) needs all of them.
- Riskiest edits: Task 2 (a worker-initiated CDP path outside the agent policy that types a secret) and Task 4 (the first place a secret exists in panel memory). Both get the most reviewer attention.
- Left out deliberately: persisting values anywhere, a worker-restart recovery, the highlighter, submit detection, the confirmation number, the `registrationComplete` count check.
