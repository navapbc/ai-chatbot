# Add the extension side panel and start flow

## Summary
Added a Chrome side panel to the extension. A caseworker can pick a participant, start an agent run on the current tab, watch it run, and stop it. Needs You cards, typing sensitive values, the allow-this-site prompt, Review and Confirmed are not built yet (plan 4c-2).

## Changes made
- Panel (`extension/panel/`): React and Tailwind, built into `panel.html`, `panel.js` and `panel.css` by `pnpm extension:build` (`scripts/build-extension.ts`, `scripts/extension-build-config.ts`). Design tokens are extracted from `app/globals.css` at build time. `tests/relay/panel-bundle.test.ts` fails the build if the bundle imports `next/`, `server-only`, `@/lib/db`, drizzle, `@typesafe-ai` or `ai`.
- Manifest: permissions `sidePanel` and `alarms`, `side_panel.default_path`, `action.default_title`. The toolbar icon opens the panel (`extension/src/panel-behavior.ts`).
- Screens: Home with recent runs and site list, participant search, entered details, Ready (site and model), Starting, Running with Stop, signed-out, and error screens with fixed copy (`panel/lib/copy.ts`).
- Recents (`panel/lib/recents.ts`): `runId`, `siteName`, `startedAt` and `endedReason` only, at most 10, 24 hours, in `chrome.storage.local`. Anything else read back is dropped.
- Saved details (`panel/lib/saved-details.ts`): a fixed list of 12 non-sensitive fields kept 8 hours. Other or sensitive keys are refused, not stripped. Expiry is checked on read and by a `saved-details-sweep` alarm owned by the service worker. No participant record is stored and nothing is logged.
- Worker (`extension/src/bridge.ts`, `service-worker.ts`): `registrationComplete` and `addKnownValue` are bound to `runId`; `stopRun` takes an optional `runId` and ignores a stop for another run; `getRunStatus` also returns `tabId`.
- Tests: unit suites `tests/relay/panel-*.test.ts`, `extension-build.test.ts`, `service-worker.test.ts`, `bridge-lifecycle.test.ts`; e2e `panel-shell`, `panel-build`, plus updated `bridge` and `redaction`.
- Docs: `CLAUDE.md` side panel paragraph; `docs/browser-extension-todo.md` ticks plan 4c items and adds 4c-1 follow-ups and 4c-2 carry-forward.

## What it improves or fixes
Before, the only caller of the run routes and worker messages was test code. Now a caseworker can start and stop a run from the browser. Stop and worker-side ends were carry-forward items from plan 4b: a worker-initiated end (left origin, token expiry, debugger detach) left the server row live and the Eve turn running.

Start sequence (`panel/lib/run-starter.ts`): check the tab, `POST eve-chat` for the first turn and read to its first chunk (15 s), fetch the extension token, worker `startRun`, `registrationComplete(runId, 0)`, then save the recent. On failure the server run is stopped and the worker run is stopped by run id.

Run tracking (`panel/lib/run-session.ts`): polls the worker every 2 s and the server every 5 s; refreshes the token at 5 minutes left (a same-run-id `startRun`, which detaches the debugger and clears known values); treats 7 s of continuous worker inactivity for its run as an end and then stops the server run. Stop calls the server route, then the worker, scoped by run id. Reopening the panel mid-run rebuilds from the worker and server without a second `eve-chat` call.

Known limits:
- `registrationComplete` always sends count 0 until 4c-2 types values.
- F1: the agent's first command before the extension registers is not retried (30 s `extension did not answer`).
- F2: if the 15 s timeout fires before `eve-chat` answers, the server may still create a run the panel cannot stop.
- I1: a run the worker ends while the panel is closed is never stopped on the server (worker-initiated end handling above runs only inside an open panel; the server row, the Eve turn and token minting continue).
- I2: closing the panel during Starting orphans the run: the page unloads after the `eve-chat` POST but before `worker.startRun`, so the start cleanup never runs.
- I4: every panel open mid-run (or a second window) refreshes the token at once, detaching the debugger under any in-flight command, clearing known values and dropping allowed origins.
- F1, F2, I1, I2, I4 and the name in the GET query string must be settled before `RELAY_RUNS_ENABLED` is set anywhere; it stays unset in deployment.
- One worker run at a time: a start in a second window replaces the first window's run.
- The participant name is sent in a GET query string, which server logs may capture.
- The Starting screen has no Cancel. Full list: `docs/browser-extension-todo.md`, "Plan 4c-1 follow-ups".

## How to run / verify
Manual check (Node 24):
1. Sign in to the app in a normal tab.
2. `APP_ORIGIN=<app origin> pnpm extension:build`
3. In `chrome://extensions`, enable developer mode and load `extension/dist` unpacked.
4. Click the toolbar icon. Home lists recents and sites; with no sites configured only "Use this tab" shows. Pick a participant or enter details, choose a model on Ready, and start. Running shows Stop.

Tests:
```bash
pnpm exec vitest run --config vitest.config.node.mjs tests/agent tests/relay --exclude 'tests/relay/e2e/**'
pnpm extension:build && pnpm test:relay e2e/panel-shell.e2e.test.ts e2e/panel-build.e2e.test.ts e2e/bridge.e2e.test.ts
pnpm test:relay   # whole relay suite
```
