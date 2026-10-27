# Add extension run plumbing

## Summary

Added the app-side and worker-side plumbing the side panel (plan 4c) needs: cookie authentication from the extension to the app, an extension-origin check, ended runs with state and Stop routes, run persistence after a client disconnect, a Jev field classifier in the worker, worker lifecycle fixes, and a hold-and-allow flow when a tab leaves the run's origin. No panel UI is included.

## Changes made

- Auth: `scripts/build-extension.ts` and `scripts/extension-build-config.ts` generate `host_permissions` from the build-time `APP_ORIGIN` (exact https origin; unset builds without it; `EXTENSION_RELEASE=1` makes unset or non-https fatal). `extension/src/app-api.ts` is the worker's app client (`credentials: 'include'`).
- Origin check: `lib/relay/request-origin.ts` and `lib/relay/route-guard.ts` compare `Origin` with `RELAY_EXTENSION_ORIGIN` on every relay and participant route (`guardRelayRoute(request, { requireOrigin })`).
- Ended runs: `EveRun.endedAt` (migration `0010_needy_quasimodo.sql`), `store.end`, and `run_ended` (400) from `eve-chat`. Token routes answer an ended run with the bodyless 404.
- New routes: `GET /api/relay/runs/[runId]` (origin, progress, `pending`, `ended`, `endedAt`; `pending` is `[]` once ended) and `POST /api/relay/runs/[runId]/stop`. Both are user-bound, `no-store`, and answer unknown, foreign and web-chat runs with a bodyless 404.
- Stop: ends the row first, then calls Eve's cancel through `cancelEveTurn` (5 s timeout). Every call attempts a cancel, including a repeat; a cancel failure is logged value-free and does not undo the end. `eve-chat` re-reads the row after a continue and cancels if Stop landed meanwhile.
- `eve-chat` (`app/(chat)/api/eve-chat/route.ts`): on relay runs a throwing client write no longer ends the read; pending questions, cursor and token still persist, and the Eve body is closed when the read ends. Web chats are unchanged.
- Redactor: `lib/relay/redactor.ts` gains a `classifyExtra` hook; the worker's Jev classifier (`extension/src/jev-classifier.ts`) can only add masking (OR with the deterministic verdict). It rejects on failure so nothing is cached, batches requests, and counts failures.
- Worker lifecycle (`extension/src/bridge.ts`): `startRun`, `stopRun` and `resumeRun` are serialized; a Stop issued while the socket is down is queued and delivered on reopen (30 s limit); a debugger detach unrecovered for 5 s ends the run (`debugger-detached`); test hooks exist only in `EXTENSION_TEST_HOOKS=1` builds.
- Worker messages: `getRunStatus` (active, origin, attached, `needsRegistration`, `jevFailures`), `registrationComplete` (optional `count`), `allowOrigin`.
- Restart: after a worker restart the run starts with `needsRegistration`, and the bridge refuses every CDP command that can return page text until `registrationComplete`.
- Origin hold: a tab that leaves the origin to another https origin holds the run for 60 s (120 s cap) instead of ending it; `allowOrigin` adds an origin for this run (max 3). Other schemes end the run at once.

## What it improves or fixes

- Before: nothing authenticated the extension to the app, and the extension origin was unchecked. After: a Lax session cookie reaches the app from the worker and from extension pages once `host_permissions` covers the app origin. A GET carries no `Origin`; a POST carries `chrome-extension://<id>`. The check therefore refuses a present, different `Origin` and requires it only on state-changing routes.
- Before: Stop only closed the relay session and Eve kept running. After: Stop ends the run and cancels the turn.
- Before: a reopened panel could not read progress or pending questions. After: the state route returns them.
- Before: closing the browser mid-turn could drop pending questions and the stream cursor. After: they persist.
- Before: a restart lost registered values silently. After: the run stalls fail-closed until re-registration.
- Before: leaving the origin ended the run, which broke login redirects. After: the run holds for an allow decision.

## How to run / verify

New environment variables:
- `RELAY_EXTENSION_ORIGIN`: full `chrome-extension://<32 chars a-p>`. Any other shape is treated as unset (one value-free warning). Unset means no check. Set it at deploy or the check is off.
- Build time: `APP_ORIGIN` (must equal the deployed app origin exactly, or cookies stop arriving), `EXTENSION_RELEASE=1`, `EXTENSION_TEST_HOOKS=1` (refused with a release build), and the test-only `EXTENSION_TEST_ORIGIN_HOLD_MS`.

Migration 0010 adds a nullable `endedAt timestamp` to `EveRun`. The container applies it at start; it is safe on a populated table.

```bash
export PATH=/Users/kaylynvannorstrand/.nvm/versions/node/v24.20.0/bin:$PATH
pnpm exec vitest run --config vitest.config.node.mjs tests/agent tests/relay --exclude 'tests/relay/e2e/**'
pnpm extension:build && pnpm test:relay e2e/bridge.e2e.test.ts e2e/redaction.e2e.test.ts e2e/jev-classification.e2e.test.ts e2e/allowlist.e2e.test.ts e2e/app-auth.e2e.test.ts
```

The e2e harness rebuilds the extension itself with the test hooks. Plan 4c carry-forward items and deferred minors are in [the extension tracker](../browser-extension-todo.md).
