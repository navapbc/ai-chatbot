# Add Reliability Fixes

## Summary

Closed the reliability gaps that had to be fixed before `RELAY_RUNS_ENABLED` is set anywhere (plan 4d): orphan relay runs, token refreshes that dropped state, sensitive typing that failed after a refresh, a stalled first agent command, and a participant name in a URL.

## Changes made

- `extension/src/bridge.ts`: `getRunStatus` returns `exp` (the token's own expiry) and `restarting` (a same-run `startRun` or a service-worker resume is in progress). A same-run refresh carries known values and allowed origins and accepts a tab on a carried origin. `typeSensitive` attaches the debugger on demand; that attach never sets the agent-facing `attached` flag. Attach and let-go run through a per-run queue with bounded steps (3 s typing, 10 s agent).
- `extension/panel/lib/run-session.ts`: a rebuilt run uses the real `exp`; `restarting` holds the 7 s inactive rule for at most 30 s.
- `extension/panel/lib/recents.ts`, `run-starter.ts`: the recent is written before the first-turn POST; new end reasons `start-failed` and `swept`.
- `extension/panel/lib/orphan-sweep.ts` (new), `app.tsx`, `screens/home.tsx`: on open, stop recents that have no end reason and that the worker does not hold. Skips recents younger than 90 s, runs the worker holds, and the whole sweep while the worker is `restarting` or unreadable. A 404 is final only after 5 minutes. At most 10 stops, under a cross-window `navigator.locks` lock.
- `app/(chat)/api/eve-chat/route.ts`: on a relay first turn, a request that aborted before `store.create` returned ends the row, cancels the Eve turn and answers 499.
- `lib/relay/eve-relay-browser.ts`: `extension did not answer` counts as a lost connection, so the first retry-safe command is rotated and retried. During an origin hold the tool polls `get url` every 3 s for about 125 s, then re-runs a retry-safe command or reports `CONNECTION_RESTORED`; it never replays a non-retry-safe command. `agent/skills/extension-run/SKILL.md` describes this.
- `app/(chat)/api/participants/route.ts`, `extension/src/app-api.ts`: search is `POST /api/participants` with a JSON body; the GET is removed. The extension origin is required when `RELAY_EXTENSION_ORIGIN` is set.
- Removed the "run ends at its expiry" note from the Allow this site card (`extension/panel/lib/allow-site.ts`, `copy.ts`).
- Tests: new `tests/relay/panel-orphan-sweep.test.ts`; extended bridge, run-session, run-starter, recents, eve-relay-browser, eve-chat route and participants route tests; panel-shell and bridge e2e scenarios.
- `CLAUDE.md` and `docs/browser-extension-todo.md` updated; items this plan closed are ticked and the parked findings are under Plan 4d Follow-ups.

## What it improves or fixes

- Orphan runs: before, a run whose panel closed during Starting, or whose worker run ended while the panel was closed, stayed live on the server (token minting, hourly cap). Now the next panel open stops it.
- Before, opening the panel mid-run refreshed the token at once, detaching the debugger and dropping allowed origins. Now the rebuild uses the real expiry, and a refresh keeps both.
- Before, "Fill it in" failed with `not_attached` after a refresh or reopen while the agent waited on `ask_question`. Now typing attaches itself.
- Before, the first agent command could fail with `extension did not answer` and was not retried; the Eve tool also returned the out-of-scope refusal during an origin hold. Now both are retried or waited out.
- Before, the participant name appeared in a GET query string (and so in logs). Now it is in the request body.

Not fixed: the first command can still stall up to 30 s before its retry (no hub queue, decision P4d-g), and Next 16 abort propagation for the 499 path is unverified, so the sweep is the backstop.

## How to run / verify

```bash
export PATH=/Users/kaylynvannorstrand/.nvm/versions/node/v24.20.0/bin:$PATH
pnpm exec vitest run --config vitest.config.node.mjs tests/agent tests/relay --exclude 'tests/relay/e2e/**'
pnpm extension:build && pnpm test:relay e2e/panel-shell.e2e.test.ts e2e/bridge.e2e.test.ts
```

The panel-shell e2e takes about 237 s.
