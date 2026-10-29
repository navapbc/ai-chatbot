# Add Pilot Rollout

## Summary

Prepared the browser extension and relay for a managed pilot: the extension token moves off the URL on the extension leg, the relay logs and caps outbound buffers, Terraform and the pipelines gain default-off switches, the extension can be packaged for force-install, and the panel explains the debugger bar and Chrome policy refusals. Nothing was applied or deployed, and no flag is on anywhere.

## Changes made

- Token transport: `relay/server.ts` reads the extension token from the subprotocol `relay.v1, <token>` on `/extension`, selects only `relay.v1` and never echoes the token. The query token on `/extension` remains as a transition fallback; `/cdp` stays query-only.
- Relay observability: `relay/server.ts` and `relay/hub.ts` add `severity` to every log line, `origin` on the accepted line, socket errors logged by code only (10 per 60 s), and a 64 MiB outbound cap (2x the inbound ceiling) that closes the run with `send buffer full`. The `command` line is flat `{event:'command', durationMs}`.
- Terraform (`terraform/variables.tf`, `relay.tf`, `cloud_run.tf`): new variables `relay_min_instances`, `relay_runs_enabled`, `relay_token_endpoint_enabled`, `relay_extension_origin`, `relay_allowed_models`, `relay_allowed_hosts`, `relay_allowed_host_suffixes` and `relay_command_duration_metric_enabled`, all defaulting to current behavior; a plan-time check of `eve-internal-secret-<env>`; preconditions on the host list and extension origin; an optional log-based metric.
- Pipelines: `deploy.yml` deploys the relay only to dev and prod when the repository variable `RELAY_ENABLED` is `true`. `promote.yml` fails when the relay is enabled and the promoted SHA has no relay image.
- Packaging: `scripts/package-extension.ts` and `package-extension-lib.ts` produce `extension.zip`, optional `updates.xml` and a locally signed CRX3 (`EXTENSION_PEM_PATH`); `.github/workflows/extension-release.yml` builds the zip for inspection; `deploy/chrome-policy/` holds the policy template and README. `extension/manifest.json` sets `minimum_chrome_version` 120.
- Extension and panel: `extension/src/bridge.ts` sets `RunStatus.policyRestricted` when Chrome refuses an attach by policy (a refused CDP command or screenshot only gives the agent a fixed message) and returns `restricted by policy` from the panel-facing typing, highlight and submit-watch refusals; `extension/panel/lib/copy.ts`, `worker-api.ts`, `run-session.ts` and the Ready and Running screens show fixed Chrome-policy and debugger-bar copy. `agent/skills/extension-run/SKILL.md` has a matching line.
- Docs: `docs/rollout-pilot.md` (runbook), `CLAUDE.md`, and `docs/browser-extension-todo.md` (Plan 5 items ticked, a Plan 5 Follow-ups section added, the token replay limit corrected).
- Tests: new and extended files under `tests/relay/` and `tests/agent/relay-deploy-config.test.ts`.

## What it improves or fixes

- Before: the extension token appeared in the `/extension` request URL and therefore in request logs. After: it travels in a subprotocol offer. The agent leg still uses the query because agent-browser cannot send a header.
- Before: socket errors were swallowed and relay lines had no severity. After: errors are visible by code and severity filters work in Cloud Logging. A peer that stops draining can no longer grow memory without bound.
- Before: `promote.yml` warned on a missing relay image, and `RELAY_ENABLED` applied to every preview. After: promotion fails and the relay exists only in dev and prod.
- Before: a Chrome policy refusal surfaced as a generic error. After: a refused attach shows fixed panel copy; a refused CDP command or screenshot gives the agent a fixed message and no panel notice. Nothing was run on a managed Chrome.
- Corrected: the closed-run tombstone refuses a re-dial with the same token for the token TTL, in memory and after terminal closes only; the old "replayable for 30 minutes" limit was wrong.
- Unverified: what a relay restart or the one-hour socket close does to a live run. The worker reconnects with the same token and the panel shows no reconnecting state; the runbook has the operator record the outcome.

## How to run / verify

```bash
export PATH=/Users/kaylynvannorstrand/.nvm/versions/node/v24.20.0/bin:$PATH
pnpm exec vitest run --config vitest.config.node.mjs tests/agent tests/relay --exclude 'tests/relay/e2e/**'
pnpm extension:build && pnpm test:relay e2e/panel-shell.e2e.test.ts e2e/bridge.e2e.test.ts
pnpm exec tsx scripts/package-extension.ts --app-origin https://app.example.org --out /tmp/ext-out
```

Rollout steps and the managed-Chrome checks are in [the runbook](../rollout-pilot.md).
