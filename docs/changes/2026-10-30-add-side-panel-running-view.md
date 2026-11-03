# Add Side Panel Running View

## Summary

The side panel now shows a run while it works: progress, an action log, Needs You cards the caseworker answers, the allow-this-site prompt, Review and Confirmed. The state route gained a `phase` field, backed by a new `EveRun.turnIdle` column. This is plan 4c-2 (`docs/plans/2026-10-30-side-panel-running-view.md`).

## Changes made

- Server: `lib/db/schema.ts`, `lib/ai/eve/run-store.ts`, `lib/ai/eve/db-run-store.ts` and `lib/db/queries.ts` add `EveRun.turnIdle` (migration `0011_majestic_moonstone.sql`, boolean default false, one `ADD COLUMN`, safe on a populated table, applied at container start). `app/(chat)/api/eve-chat/route.ts` sets it when a relay turn reaches `session.waiting` and clears it when a turn starts; the writes are best-effort and the web chat path is unchanged.
- `app/(chat)/api/relay/runs/[runId]/route.ts` returns `phase`: `ended`, else `needs_you` when questions are pending, else `review` when `turnIdle`, else `working`. The flag itself is not returned.
- `extension/src/app-api.ts` accepts `phase`; an unknown or missing value is treated as `working`.
- `extension/panel/lib/run-session.ts` polls the server every 3 s (was 5 s), keeps progress, pending, phase and a session-local action log (`action-log.ts`, 200 entries, never stored). A failed poll keeps the last phase.
- Screens in `extension/panel/screens/`: `running.tsx` and `progress-view.tsx` (sections, counts, current step, log), `needs-you.tsx`, `allow-site.tsx`, `review.tsx`, `confirmed.tsx`. Supporting modules in `extension/panel/lib/`: `progress-view`, `question-kind`, `answer-client`, `eve-chat-post`, `allow-site`, `tab-url`, `review-model`; fixed copy in `copy.ts`. Everything from the agent renders as text, length-capped.
- `scripts/build-extension.ts` and `scripts/extension-build-config.ts` pass the hold length into the panel build.
- `agent/skills/extension-run/SKILL.md`: a text answer of exactly `skip` means leave the field alone.
- Tests: unit tests under `tests/relay/panel-*.test.ts` and `tests/agent/`, and new scenarios in `tests/relay/e2e/panel-shell.e2e.test.ts`.

## What it improves or fixes

Before, the panel showed Running with only a Stop button, so a caseworker could not see progress, answer the agent's questions or react to a redirect. Now:

- Needs You: questions come from the state route. Answers go to `eve-chat` as `inputResponses`. A question with exactly the options `provided` and `skip` and no free text is sensitive; the card offers only those two options and never takes typed text. A free-text question whose prompt `isSensitiveField` flags is a dead end (no input). The panel maps server refusals to fixed copy: `unknown_request`, `answers_pending` and `missing_answer` as already answered, `sensitive_value_in_message` as refused, `message_too_large` as too long, `run_ended` as ended.
- Allow this site: shown when the worker holds the run for a new https origin. It shows the origin and the tab's URL as origin plus path (no query or fragment), a countdown that is informational only (the worker decides), and is never auto-approved. After an allow it notes that the run ends at its expiry time because token refresh is refused on an allowed extra origin. (Superseded in plan 4d: a same-run refresh now keeps allowed origins, so this note was removed.)
- Review: "Ready for your review" appears when the turn ended with no pending question. It says only that the agent stopped. "I submitted it" is a manual press: it stops the run on the server, then the worker, and shows Confirmed. Nothing detects the submit and no confirmation number is read.
- Reopen: the panel rebuilds the right screen (phase card, hold prompt, Review) from the worker and the server. A panel opened mid-run shows only the steps it has seen.

## How to run / verify

```bash
export PATH=/Users/kaylynvannorstrand/.nvm/versions/node/v24.20.0/bin:$PATH
pnpm exec vitest run --config vitest.config.node.mjs tests/agent tests/relay --exclude 'tests/relay/e2e/**'
pnpm extension:build && pnpm test:relay e2e/panel-shell.e2e.test.ts e2e/bridge.e2e.test.ts
```

The panel-shell e2e takes about 154 s; the relay-e2e CI job has a 20 minute limit.

Not in this change (plan 4c-3): typing sensitive values for the caseworker, "show me the unfilled field", automatic submit detection and the confirmation number. Follow-ups are in `docs/browser-extension-todo.md`.
