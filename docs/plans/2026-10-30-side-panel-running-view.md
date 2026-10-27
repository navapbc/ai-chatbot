# Side Panel Running View Implementation Plan (Plan 4c-2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** While a run is in progress the caseworker sees what the agent is doing, answers the agent's questions for ordinary fields, decides whether the agent may follow the tab to another site, reviews the result when the agent stops, and marks the form as submitted. Sensitive values are still entered on the page by the caseworker; typing them for the caseworker is plan 4c-3.

**Architecture:** The server records when an agent turn has ended (`turnIdle` on the `EveRun` row) and the run-state route derives a `phase` (`working`, `needs_you`, `review`, `ended`). The panel's run session polls that state and exposes `progress`, `pending`, `phase` and an in-memory action log. The Running view renders sections, a progress bar and the actions; a Needs You card answers `ask_question` requests through the `eve-chat` answer turn (`inputResponses`); an allow-this-site card drives the worker's `allowOrigin`; a Review screen lists what is left and ends the run when the caseworker says they submitted.

**Tech Stack:** React 19 panel from plan 4c-1, Next route handlers, Drizzle/Postgres (one nullable-safe migration), vitest node project, Playwright Chromium e2e. No new dependencies.

**Spec:** `docs/specs/2026-10-23-browser-extension-design.md` (Running view, Needs You, Review and Confirmed). Tracker: `docs/browser-extension-todo.md`, "Plan 4c-2 carry-forward" and "Plan 4c-1 follow-ups". Previous plans: `docs/plans/2026-10-29-side-panel-start-flow.md`, `docs/plans/2026-10-28-run-plumbing.md`.

## Global Constraints

- Node 24: `export PATH=/Users/kaylynvannorstrand/.nvm/versions/node/v24.20.0/bin:$PATH`. `pnpm` only. No new dependencies; if one seems necessary, stop and ask.
- Tests: `pnpm exec vitest run --config vitest.config.node.mjs <path>`; extension work also runs `pnpm extension:build && pnpm test:relay e2e/panel-shell.e2e.test.ts e2e/bridge.e2e.test.ts`. Lint with `pnpm exec biome lint <files>`; never `pnpm lint`; never `biome format --write` on a directory.
- Never read `.env*` files or print secret values. Migrations are generated with `pnpm db:generate`, never hand-edited, and applied by the container at start: each must be safe on a populated production table.
- Everything the panel renders from the server or the agent (progress names, action labels, questions, option labels) is untrusted text: render as text only, cap lengths, never as HTML or a URL, never log it. Fixed copy comes from `extension/panel/lib/copy.ts`-style tables.
- A sensitive value is never typed into the panel in this plan: for a sensitive question the card only offers "I entered it myself" and "Skip" (the answers `provided` and `skip`). Answers for ordinary fields go to the server and the agent, so the panel refuses to send text that the server's guard would refuse and shows the fixed refusal copy.
- Panel code keeps the 4c-1 import rules (leaf `components/ui`, pure `lib/relay/*`, `extension/src/*`); the bundle guard test must stay green.
- Frontend conventions from `CLAUDE.md`: Tailwind classes only, `handle` prefix for handlers, early returns, aria attributes and keyboard support, no inline styles.
- The web app (`develop` behavior) and the legacy chat route are unchanged. `RELAY_RUNS_ENABLED` and `RELAY_TOKEN_ENDPOINT_ENABLED` stay unset in deployment config.

## Decisions For Owner

Defaults chosen so the work can proceed; each is a one-line change.

- **P4c2-a Polling.** The run session polls the server state every 3 seconds (was 5) so questions and progress feel live; the worker poll stays at 2 seconds.
- **P4c2-b "I submitted it".** Submission is confirmed by the caseworker pressing "I submitted it" on the Review screen, which ends the run (server stop then worker stop) and shows Confirmed. Automatic submit detection is plan 4c-3; the confirmation number is not read from the page in this plan.
- **P4c2-c Action log.** The agent reports at most 5 recent actions per progress update, so "What the agent did" is a session-local log built from successive updates (de-duplicated, capped at 200 entries, never stored). A panel opened mid-run shows only what it has seen.
- **P4c2-d Sensitive questions.** A question is treated as sensitive when its options are exactly `provided` and `skip` and free text is not allowed (the convention in `agent/skills/extension-run/SKILL.md`), or when `isSensitiveField` flags its prompt. The card says "This one is sensitive. Type it in the page yourself, then press the button." Typing for the caseworker arrives in plan 4c-3.
- **P4c2-e Allow-this-site prompt.** It shows the new origin and the tab's actual URL as origin plus path (never the query or fragment), lasts as long as the hold (60 seconds), and is never auto-approved. After an allow, the card warns that the run now ends at its expiry time because a refresh is refused on an allowed extra origin.
- **P4c2-f Review wording.** "Ready for your review" appears when the agent's turn has ended with no pending question; the panel does not claim the form is complete, only that the agent stopped.

## Review Focus

- A question the agent asks twice, answered twice, or answered while the run ended: expect one answer sent, no double submit, and a clear message if the server refuses.
- An answer containing something the server's guard refuses (an SSN typed into an ordinary field): expect a fixed refusal message, nothing logged, the text not retained.
- Progress with 40 sections, long names, names containing markup, zero sections, and counts that go backwards: expect a stable, readable view, no crash, text only.
- The panel opened while the agent is mid-turn, waiting on a question, and after the turn ended: expect the right phase each time, no second run.
- The allow-site card while the tab moves again, after the 60 second hold ends, and after the run is stopped: expect no stale "Allow" button that acts on a dead hold.
- A run that ends (Stop, token expiry, worker end) while a question is open: expect the card disappears and the ended panel shows.

## File Structure

- Modify `lib/db/schema.ts`, `lib/db/queries.ts`, `lib/ai/eve/run-store.ts`, `lib/ai/eve/db-run-store.ts` (`turnIdle`), `app/(chat)/api/eve-chat/route.ts` (set and clear it), `app/(chat)/api/relay/runs/[runId]/route.ts` (`phase`), a generated migration.
- Create `extension/panel/lib/answer-client.ts`, `extension/panel/lib/action-log.ts`, `extension/panel/screens/` (`needs-you.tsx`, `allow-site.tsx`, `review.tsx`, `confirmed.tsx`, `progress-view.tsx`), `extension/panel/lib/question-kind.ts`.
- Modify `extension/panel/lib/run-session.ts`, `extension/panel/screens/running.tsx`, `extension/panel/lib/state.ts`, `extension/panel/app.tsx`, `extension/panel/lib/copy.ts`.
- Tests: `tests/agent/` (store, routes), `tests/relay/panel-*.test.ts`, `tests/relay/e2e/panel-shell.e2e.test.ts`.

---

### Task 1: Turn status on the server

**Files:**
- Modify: `lib/db/schema.ts` (`turnIdle` boolean not null default false on `EveRun`), `lib/db/queries.ts` (`setEveRunTurnIdle({ userId, chatId, idle })`), `lib/ai/eve/run-store.ts`, `lib/ai/eve/db-run-store.ts`; a generated migration
- Modify: `app/(chat)/api/eve-chat/route.ts`, `app/(chat)/api/relay/runs/[runId]/route.ts`
- Test: `tests/agent/eve-run-store.test.ts`, `tests/agent/eve-chat-route.test.ts`, `tests/agent/relay-run-state.test.ts` (extend)

**Interfaces:**
- Produces: `EveRunRecord.turnIdle: boolean` (false on create); `EveRunStore.setTurnIdle(userId, chatId, idle: boolean): Promise<void>` (no-op when no row); the state route response gains `phase: 'working' | 'needs_you' | 'review' | 'ended'` derived as: ended row -> `ended`; else `pending.length > 0` -> `needs_you`; else `turnIdle` -> `review`; else `working`. `turnIdle` is NOT exposed on its own. The migration adds a boolean with a default, safe on a populated table.
- Route behavior (relay runs only): set `turnIdle = false` when a follow-up or answer turn is claimed (right after the live claim succeeds) and when the first turn row is created (default); set `turnIdle = true` at the turn's `session.waiting`, in the same place the cursor is advanced, after pending is persisted; a persistence failure for this flag is logged value-free and never fails the stream. Web chats are untouched.

- [ ] **Step 1: Write failing tests**: store `setTurnIdle` and `create` default (memory suite shared with the DB suite); the route: a relay first turn that ends with `session.waiting` and no questions leaves `turnIdle` true and the state route answers `review`; a turn that ends with a question answers `needs_you` (pending wins); an answer turn sets it false at claim and the state route answers `working` until the next waiting; an ended run answers `ended`; the client disconnecting mid-turn still ends with the right flag (plan 4b persistence); a web chat never touches it.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement**, then `pnpm db:generate`; confirm the migration only adds the column with its default.
- [ ] **Step 4: Run `tests/agent` and the relay unit suite, expect PASS.**
- [ ] **Step 5: Commit** `feat(relay): record when an agent turn has ended`.

### Task 2: Run session carries progress, questions and phase

**Files:**
- Create: `extension/panel/lib/action-log.ts`
- Modify: `extension/panel/lib/run-session.ts`, `extension/panel/lib/state.ts`
- Test: `tests/relay/panel-run-session.test.ts`, `tests/relay/panel-action-log.test.ts` (extend/new)

**Interfaces:**
- Produces: `RunSessionState` gains `progress: RunProgress | null`, `pending: PendingInputRequest[]` (empty when the run ended), `phase: 'working' | 'needs_you' | 'review' | 'ended'`, `actions: string[]` (the session-local log). The server poll runs every 3 s (P4c2-a). `createActionLog(max = 200)` with `add(actions: string[])` that appends only entries not already at the tail of the log (a new update repeating the last N entries adds nothing; entries are trimmed, capped at 120 characters, empty ones dropped) and `entries(): string[]`; never persisted. `phase` comes from the new server field; an unknown value maps to `working`.
- The existing end-detection, refresh and stop behavior of the session is unchanged.

- [ ] **Step 1: Write failing tests** with fake timers and a fake client: the 3 s cadence; state carries the latest progress and pending; `ended` clears pending; the action log de-duplication (the same five actions twice add nothing, a sliding window adds only the new ones), caps, trimming; an unknown phase; progress that goes backwards or repeats does not crash the log.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.** Keep the at-most-once stop and the 7 s rule exactly as they are.
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** `feat(extension): carry progress, questions and phase in the run session`.

### Task 3: Running view

**Files:**
- Create: `extension/panel/screens/progress-view.tsx`
- Modify: `extension/panel/screens/running.tsx`, `extension/panel/lib/copy.ts`
- Test: `tests/relay/panel-progress-view.test.ts` (pure view-model tests)

**Interfaces:**
- Produces: `buildProgressView(progress: RunProgress | null, actions: string[]): ProgressViewModel` (pure) with `bar: { filled: number; total?: number; label: string }` (label "N of M fields" when every section has a `total`, else "N fields so far"; "Section X of Y" when sections exist), `sections: { name, status, filled, total?, open: boolean }[]` (finished sections collapsed, the active one, i.e. `filling` or the section named by `current`, open with the last actions, waiting muted, `needs_you` marked), and a flat `steps` list when there are no sections. Names are trimmed and cut to 80 characters; counts are clamped to non-negative integers and `filled` never exceeds a known `total`; at most 40 sections. The Running screen renders the view model: a `components/ui/progress` bar, section cards with count lines, the active section's actions in the small mono style of the existing tool-call cards (`font-ibm-plex-mono`), the Stop button and elapsed timer from plan 4c-1, and the phase-specific cards from the next tasks above the sections. Text only.

- [ ] **Step 1: Write failing tests**: 40 sections, 0 sections (flat steps), long names cut, markup in names stays text, counts backwards or above total clamped, `current` naming no section, all sections done, mixed totals (label falls back).
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** the pure builder and the view.
- [ ] **Step 4: Run, expect PASS; render the Running screen in the panel e2e with canned progress from the fake app and assert the bar, one collapsed and one open section.**
- [ ] **Step 5: Commit** `feat(extension): show progress and actions in the Running view`.

### Task 4: Needs You for ordinary fields

**Files:**
- Create: `extension/panel/lib/question-kind.ts`, `extension/panel/lib/answer-client.ts`, `extension/panel/screens/needs-you.tsx`
- Modify: `extension/panel/screens/running.tsx`, `extension/panel/lib/copy.ts`
- Test: `tests/relay/panel-question-kind.test.ts`, `tests/relay/panel-answer-client.test.ts`

**Interfaces:**
- Produces: `classifyQuestion(req: PendingInputRequest): 'sensitive' | 'choice' | 'text'`: `sensitive` when the options are exactly `provided` and `skip` and free text is not allowed, or when `isSensitiveField({ label: req.prompt, name: req.prompt })` is true; `choice` when options exist; else `text`. `answerRequests(deps, run: { runId }, answers: InputAnswer[]): Promise<{ ok: true } | { ok: false; error: unknown }>` posts `{ id: runId, inputResponses }` to `${appOrigin}/api/eve-chat` with the same options and first-chunk handling as the run starter (credentials include, no-store, redirect error, 15 s timeout, read only to the first chunk, cancel the reader); statuses map as in the starter, plus 400 with `answers_pending`/`unknown_request` -> a "this question was already answered" copy, 400 `sensitive_value_in_message` and the size refusals -> a fixed "We can't send that here" copy; `run_ended` -> the ended panel. A second answer for the same request while one is in flight returns without a request. The text is capped at 2000 characters, trimmed, and an empty answer is refused client-side. Nothing about an answer is logged, stored or retained after the call.
- The Needs You card (pinned above the sections whenever `phase === 'needs_you'`): one input per `text` question (a labeled textarea or input), one radio/select group per `choice` question, a Send button for the batch (disabled until every question is answered or skipped; a skip action is offered per text question and sends the text "skip"), and for `sensitive` questions the fixed card from P4c2-d with "I entered it myself" (answer `optionId: 'provided'`) and "Skip" (`skip`). Questions are rendered as text (the prompt may carry agent wording; it is capped at 300 characters in the view). After Send the card shows "Sent" and the session waits for the next poll; the card is keyed by `requestId`s so a new question set resets it.

- [ ] **Step 1: Write failing tests**: `classifyQuestion` for the skill's sensitive shape, a prompt that `isSensitiveField` flags without those options, a choice, a text question, malformed requests; `answerRequests` body shape for text, option and mixed answers (exact JSON), first-chunk handling and reader cancel, every status mapping, double send guard, trim/cap/empty refusal, nothing logged (spy console) and no answer text in any returned error (JSON string check).
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS; extend the panel e2e with a fake app that reports one pending text question and one sensitive question: typing an answer and pressing Send posts the exact `inputResponses` once, and the sensitive card posts `provided`; a 400 `sensitive_value_in_message` from the fake shows the fixed refusal.**
- [ ] **Step 5: Commit** `feat(extension): answer the agent's questions from the panel`.

### Task 5: Allow this site

**Files:**
- Create: `extension/panel/screens/allow-site.tsx`
- Modify: `extension/panel/lib/run-session.ts` (expose `pendingOrigin` with the hold's start time), `extension/panel/screens/running.tsx`, `extension/panel/lib/copy.ts`
- Test: `tests/relay/panel-allow-site.test.ts`

**Interfaces:**
- Produces: `describeTabUrl(url: string | undefined): string` returning `origin + pathname` (path cut to 80 characters) or an "unknown page" fixed text, never the query or fragment; `allowPending(session, deps)`: reads `chrome.tabs.get(tabId)` for the actual URL, shows the card only while the worker still reports `pendingOrigin` equal to the origin the card was built for; the Allow button calls `worker.allowOrigin(origin)` exactly once (a second press, or a press after the hold ended, shows the fixed "The hold ended" message and does nothing), Stop is always present. The card text: "The page moved to <new origin>. The agent is paused until you decide." with the tab's actual URL line, "Allow this site for this run" and "Stop the run"; after an allow it shows the P4c2-e note with the run's expiry time. The card is never shown when `pendingOrigin` is absent, and it disappears the moment the worker clears it or the run ends.

- [ ] **Step 1: Write failing tests**: `describeTabUrl` (query, fragment, credentials, a long path, a non-URL, undefined); the card's lifecycle with fakes: shown for a pending origin, hidden when it clears, a changed pending origin rebuilds the card (the Allow button now names the new origin), Allow once, Allow after the hold ended is refused without a call, the worker refusing `allowOrigin` shows a fixed message, the card is never auto-approved (no code path calls `allowOrigin` without a user event).
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS; e2e: with the real worker and fixture pages, redirect the tab to the second fixture origin, assert the card shows both the origin and the tab URL, press Allow and assert the worker's run continues on the new origin (`getRunStatus`), and that doing nothing ends the run after the (shortened) hold.**
- [ ] **Step 5: Commit** `feat(extension): ask before following the tab to another site`.

### Task 6: Review and Confirmed

**Files:**
- Create: `extension/panel/screens/review.tsx`, `extension/panel/screens/confirmed.tsx`
- Modify: `extension/panel/screens/running.tsx`, `extension/panel/lib/state.ts`, `extension/panel/app.tsx`, `extension/panel/lib/copy.ts`
- Test: `tests/relay/panel-review.test.ts`, `tests/relay/panel-state.test.ts` (extend)

**Interfaces:**
- Produces: when `phase === 'review'` the Running screen shows the Review panel: "Ready for your review" (P4c2-f wording: the agent stopped), a list of sections whose status is `needs_you` or whose `filled` is below a known `total` ("Left for you"), "What the agent did" (the session's action log, collapsed by default), a hint "Check the form on the page, then submit it there yourself.", and two actions: "I submitted it" (calls the session's `stop()`, i.e. server stop then worker stop, then dispatches `runConfirmed` and shows Confirmed) and "Stop without submitting" (the existing Stop, ending at the ended panel). There is no "Show me the unfilled field" button in this plan (plan 4c-3 adds it with the field locator); the Review panel says nothing about it. `Confirmed` shows "Submitted" with the site name and two actions: "Start another form for this person" (keeps `target` and `participant` in memory and goes to Ready; clears `run`) and "Back to Home" (clears everything). Confirmed does not claim the submission succeeded on the site, only that the caseworker said they submitted; no confirmation number is shown.
- State: `State.screen` gains `confirmed`; actions `runConfirmed`, `startAnother`; reducer tests.

- [ ] **Step 1: Write failing tests**: the Review model builder from progress (left-for-you list with `needs_you`, with `filled < total`, none), the reducer transitions (`runConfirmed` -> `confirmed` and clears `run`, `startAnother` -> `ready` keeping `target` and `participant`, `goHome` clears), "I submitted it" calls stop once and then confirms even when a stop call fails (the failure is shown as a small notice), double press guarded.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS; e2e: the fake app reports an idle turn (`phase: review`): the Review panel shows, pressing "I submitted it" calls the server stop and the worker stop once each and shows Confirmed; "Start another form for this person" returns to Ready with the same participant.**
- [ ] **Step 5: Commit** `feat(extension): add Review and Confirmed to the panel`.

### Task 7: Reopen and rebuild with phases

**Files:**
- Modify: `extension/panel/lib/run-session.ts` (`rebuildRun`), `extension/panel/app.tsx`
- Test: `tests/relay/panel-run-session.test.ts` (extend)

**Interfaces:**
- Consumes: Task 2's phase. A panel opened mid-run rebuilds `state.run` as today and lands on the Running screen showing the right card for the phase: questions pending -> Needs You, `review` -> Review, a worker hold -> the allow card. The action log starts empty ("What the agent did" notes that the panel opened mid-run), progress shows immediately from the first poll. A run that has `ended` on the server is not rebuilt. No second `eve-chat` call is ever made by a rebuild.

- [ ] **Step 1: Write failing tests** for each phase on rebuild, the empty-log note, and no `eve-chat` call.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** `feat(extension): rebuild the right screen when the panel reopens`.

### Task 8: End to end

**Files:**
- Modify: `tests/relay/e2e/panel-shell.e2e.test.ts`

**Interfaces:**
- A scripted scenario against the real worker, a real relay and a fake app whose `runs/:id` answers change over time: working with progress (sections and actions) -> a text question and a sensitive question appear (Needs You) -> answers posted once -> working again -> the tab is redirected (allow card, Allow) -> the turn idles (`review`) -> "I submitted it" -> Confirmed -> "Start another form for this person" reaches Ready. Assert the fake app saw each call once and in order, no applicant text in the console output, no stale cards after each transition.

- [ ] **Step 1: Write the e2e.** If a step cannot be driven headlessly, drive the same `panel.html` as a tab and say so in the test header.
- [ ] **Step 2: Run `pnpm extension:build` with the hooks build used by the other e2e suites and `pnpm test:relay e2e/panel-shell.e2e.test.ts`, expect PASS; run the other relay e2e files once.**
- [ ] **Step 3: Commit** `test(extension): cover the running view end to end`.

### Task 9: Docs and tracker

**Files:**
- Modify: `CLAUDE.md`, `docs/browser-extension-todo.md`
- Create: `docs/changes/2026-10-30-add-side-panel-running-view.md` (use the change-docs skill)

- [ ] **Step 1: Update the docs from the diff**: the `turnIdle` column and migration (applied at container start, safe on a populated table), the `phase` field of the state route, the screens that now exist, the answer flow and its refusals, the allow-site card rules, Review and Confirmed wording, what the panel does not do yet (typing sensitive values, "show me the unfilled field", automatic submit detection, confirmation number: plan 4c-3). Tick the carry-forward items this plan closed (hide pending when ended, the allow-origin prompt, wait-and-retry note stays an agent-side item) and keep the rest. 
- [ ] **Step 2: Run the whole node suite** `pnpm exec vitest run --config vitest.config.node.mjs tests/agent tests/relay --exclude 'tests/relay/e2e/**'` and the relay e2e files once; expect PASS.
- [ ] **Step 3: Commit** `docs: document the side panel running view`.

## Self-Review Notes

- Spec coverage: Running view (Tasks 2, 3), Needs You for ordinary fields (Task 4; sensitive typing is plan 4c-3), the allow-site carry-forward (Task 5), Review and Confirmed (Task 6; "Show me the unfilled field" and automatic submit detection are 4c-3), reopen (Task 7).
- Ordering: Task 1 (server) before Task 2; Task 2 before Tasks 3 to 6; Task 7 after 2 to 6; the e2e (Task 8) needs all of them.
- Riskiest edits: Task 1 (the `eve-chat` stream loop is shared with web chats; keep the web path identical and the flag write best-effort) and Task 4 (the first time the panel sends free text to the agent: the refusal handling and no-retention rules).
- Things this plan deliberately leaves out: typing sensitive values for the caseworker, the field locator and highlighter, automatic submit detection, the confirmation number, the plan 4d reliability fixes. The panel tells the caseworker only what it knows.
