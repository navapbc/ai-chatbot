# Add the browser bridge for the Chrome extension

## Summary

Added a token-authenticated WebSocket relay, an MV3 extension service worker, and a relay route for the Eve `browser` tool, so agent-browser can drive a caseworker's own Chrome tab. Deployment of the relay is opt-in and nothing has been applied.

## Changes made

- Added `lib/relay/`: signed run tokens (`token.ts`), origin and CDP-method policy plus event filtering (`policy.ts`, `allowed-cdp-methods.ts`), relay/extension message types (`protocol.ts`), and a shared relay URL helper (`url.ts`).
- Added `relay/`: the relay service (`server.ts`, `hub.ts`, `cdp-shim.ts`, `bus.ts`, `env.ts`, `main.ts`) and `relay/Dockerfile`. One agent socket and one extension socket are paired per run; the relay refuses unauthenticated, wrong-role, wrong-origin and wrong-host connections before upgrade.
- Added `extension/`: manifest with a pinned extension ID and a service worker (`src/bridge.ts`, `src/service-worker.ts`) that attaches `chrome.debugger` to one tab, enforces the origin on every command, and ends the run on Stop, infobar cancel or leaving the origin.
- Changed the Eve tool path: `agent/tools/browser.ts` uses the relay when the session carries relay credentials and never falls back to Kernel; `agent/tools/check_submit_gate.ts` refuses on relay sessions; `agent/channels/eve.ts` and `lib/ai/eve/eve-client.ts` pass credentials only on loopback with a shared secret; `lib/kernel/cli.ts` gained env-based CDP and redaction before telemetry.
- Added `app/(chat)/api/relay/token/route.ts`, disabled unless `RELAY_TOKEN_ENDPOINT_ENABLED=1`.
- Added `terraform/relay.tf` and edited `terraform/cloud_run.tf`, `main.tf`, `outputs.tf`, `variables.tf`: relay service, dedicated service account, request-log exclusion, and app env wiring, all gated on a relay image URL.
- Changed `.github/workflows/deploy.yml` and `promote.yml` to build and pass the relay image only when the repository variable `RELAY_ENABLED` is `true`, and `test.yml` to run the relay unit and e2e suites.
- Added `tests/relay/` (unit and Chromium e2e suites) and `docs/plans/2026-10-23-browser-bridge.md`; added dependencies `ws`, `esbuild`, `@types/ws`, `@types/chrome`.

## What it improves or fixes

Before: the Eve `browser` tool could only drive a remote Kernel browser, so the caseworker watched a live-view pane instead of working in their own tab. After: when a session carries relay credentials, the same agent-browser commands run against the caseworker's tab through the relay. The Kernel path is unchanged.

The relay path is deny by default: exact-origin lock, a measured CDP method allowlist enforced in both relay and extension, cookie and header stripping on network events, tokens kept out of logs and the command line, and the token route off by default. With `RELAY_ENABLED` unset, existing Terraform and workflows plan as before.

Known limits are listed in [the todo tracker](../browser-extension-todo.md): single relay instance only, no redaction of sensitive values yet (plan 3), no panel UI (plan 4), and a service-worker restart does not resume a run.

## How to run / verify

```bash
pnpm extension:build && pnpm test:relay
pnpm exec vitest run --config vitest.config.node.mjs tests/agent
terraform -chdir=terraform init -backend=false && terraform -chdir=terraform validate
```

The relay suite includes browser tests (Playwright Chromium, `openssl`, and the `agent-browser` binary from `node_modules`). The app's own `pnpm exec vitest run --project node` had two failing tests that were not investigated when this was written.
