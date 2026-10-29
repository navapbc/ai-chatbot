# Pilot Rollout Implementation Plan (Plan 5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Everything in the repository that a single-instance pilot needs: the extension token out of the URL on the extension leg, relay logs and a metric an operator can use, a deploy pipeline that fails loudly, Terraform wiring that is off by default, a release build with packaging and Chrome policy templates for managed force-install, the policy-error and banner copy the spec requires, and a runbook for the steps only the owner can take.

**Architecture:** No new service. The relay keeps one instance. The extension offers its token as a WebSocket subprotocol and the relay echoes a fixed protocol name. Relay log lines gain `severity` and `origin`, and socket errors are logged by code. Terraform gains variables (all defaulting to off) for the app's relay flags and the exact relay host. The extension is built in release mode per environment, zipped, and described by an update manifest and a policy template. The remaining work is documentation of owner actions.

**Tech Stack:** Node relay (`relay/`), the MV3 extension, Terraform, GitHub Actions, vitest node project. No new runtime dependencies. This plan runs no `terraform apply`, no deploy and no `gcloud` command that changes anything.

**Spec:** `docs/specs/2026-10-23-browser-extension-design.md` (Managed Chrome Requirements, The debugging banner, Constraints, Open Items). Tracker: `docs/browser-extension-todo.md` ("Plan 5: Rollout", "Owner Actions Before The Relay Is Enabled", "Open Questions"). Previous plans: `docs/plans/2026-11-03-review-and-submit.md`, `docs/plans/2026-11-01-reliability-before-flag.md`.

## Global Constraints

- Node 24: `export PATH=/Users/kaylynvannorstrand/.nvm/versions/node/v24.20.0/bin:$PATH`. `pnpm` only. No new dependencies; if one seems necessary, stop and ask.
- Tests: `pnpm exec vitest run --config vitest.config.node.mjs <path>`; extension work also runs `pnpm extension:build && pnpm test:relay e2e/panel-shell.e2e.test.ts e2e/bridge.e2e.test.ts`. Lint with `pnpm exec biome lint <files>` (extension paths are excluded by config; format touched files by hand); never `pnpm lint`; never `biome format --write` on a directory.
- Never read `.env*` files or print secret values. Never run `terraform apply`, `terraform plan` against a real backend, `gcloud` or `gh` commands that change state, or a deploy. `terraform fmt -check` and `terraform validate` (after `terraform init -backend=false`) are allowed if the tool is installed; say so if it is not.
- Every new Terraform variable and GitHub Actions input defaults to the current behavior (relay off, flags unset). Nothing in this plan turns a flag on in any environment. `RELAY_RUNS_ENABLED` and `RELAY_TOKEN_ENDPOINT_ENABLED` stay unset in deployment config until the owner sets them following the runbook.
- No secret, token or participant value is written to a log line, a metric label, a workflow log or a doc. Relay logs never include a URL, a query string, a header or page content.
- The web app and the legacy chat route are unchanged. The release extension build refuses test hooks (`EXTENSION_RELEASE=1`, unchanged).
- Frontend conventions from `CLAUDE.md` apply to panel changes: Tailwind only, `handle` prefix, early returns, aria attributes.

## Decisions For Owner

Defaults chosen so the work can proceed; each is a one-line change.

- **P5-a Token transport.** The extension leg offers `['relay.v1', <token>]` as `Sec-WebSocket-Protocol` and the relay echoes `relay.v1`. During the transition the relay accepts the token from either the subprotocol or the query on that leg, and the extension sends only the subprotocol. The agent leg keeps its query token: agent-browser takes a plain `ws://` URL and cannot send a header, and the request-log exclusion covers it for a single instance. A single-use token (jti) is rejected: reconnects and the agent's cached token reuse the same token.
- **P5-b Token lifetime.** Unchanged (1800 s, both roles). Shortening the agent token needs the hub tombstone kept at the longer TTL and a run longer than the TTL tested first; deferred.
- **P5-c Relay sizing.** One instance, minimum 1 (a cold start drops live sockets), 512Mi, concurrency 80, request timeout 3600 s as today. Run state and tombstones live in memory, so a relay restart or the one-hour socket limit ends live runs; the panel already shows that as a normal end.
- **P5-d Where the relay exists.** `RELAY_ENABLED` is a repository-wide variable today, so every preview environment would get a relay. The pipeline scopes it to dev and production.
- **P5-e Exact relay host.** `RELAY_ALLOWED_HOSTS` replaces the `.run.app` suffix, in two applies (first set both, then remove the suffix), because a wrong host makes every request answer 421.
- **P5-f Minimum Chrome.** The manifest gets `minimum_chrome_version`. Default `120` (side panel and debugger are older than that); the owner confirms against the managed fleet's version.
- **P5-g Extension identity.** The pilot uses the existing manifest `key` (id `ppcenadbeddfnlgklnccjkgjegmbkkkl`) with a self-hosted update URL. A Chrome Web Store listing may assign a different id, which would change `RELAY_EXTENSION_ID` and `RELAY_EXTENSION_ORIGIN`; the runbook says to re-derive both if the id changes. The private key (`.pem`) is never in the repository.
- **P5-h Observability.** Structured logs plus one log-based metric on command duration; no OpenTelemetry spans in the relay (deferred). Outbound backpressure gets a byte cap.

## Review Focus

- The extension still connects after the token moves: a missing subprotocol echo makes Chrome fail the handshake. Expect the handshake to be tested with a real client, an old query client, and a wrong token.
- A relay log line, a metric label or a workflow log that contains a token, URL, query, header, origin of a participant site beyond the allowed one, or page content: expect none.
- A Terraform change that turns something on by default, or an exact-host change that locks the relay out: expect none.
- A policy template that blocks the extension's own host access or leaves the debugger banner behavior undefined.
- A release build that bakes the wrong `APP_ORIGIN`, ships test hooks, or leaks the key.

---

### Task 1: The extension token moves to the subprotocol

**Files:**
- Modify: `relay/server.ts` (the `/extension` upgrade: `handleProtocols`, token read), `extension/src/bridge.ts` (the two extension-leg connect sites: the run socket and the stop socket)
- Test: `tests/relay/relay-server.test.ts` (extend or the existing server test file), `tests/relay/bridge-lifecycle.test.ts` (extend), `tests/relay/e2e/bridge.e2e.test.ts` (one real-Chromium handshake)

**Interfaces:**
- Produces: on `/extension` the server reads the token from `Sec-WebSocket-Protocol` (offered as `relay.v1, <token>`, comma separated) and, for the transition, from the query when no subprotocol token is offered. It selects the fixed protocol `relay.v1` through `handleProtocols`; a client that offers no `relay.v1` is refused as before. The query token on `/cdp` (agent leg) is unchanged. The extension builds its socket with `new WebSocket(url, ['relay.v1', token])` and a URL with no query. The token is never logged, and a refused handshake logs only the existing fixed reason.

- [ ] **Step 1: Write failing tests**: a client offering `['relay.v1', validToken]` connects and the server echoes `relay.v1`; a client offering only a query token still connects on `/extension` (transition); an invalid or expired token in the subprotocol is refused with the existing reason and no token in any log line; a client offering `relay.v1` with no token is refused; `/cdp` still requires the query token; the bridge's two socket constructions pass the subprotocol and no query (assert the constructor arguments); a real-Chromium handshake against the test relay succeeds with the subprotocol token.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS** (`tests/relay`, then `pnpm extension:build && pnpm test:relay e2e/bridge.e2e.test.ts`).
- [ ] **Step 5: Commit** `feat(relay): carry the extension token in the subprotocol`.

### Task 2: Relay logs, errors and backpressure

**Files:**
- Modify: `relay/server.ts` (`ignore` at the socket error sites, the `accepted` log line), `relay/hub.ts` (`sendTo`, the `PeerSocket` interface), `relay/main.ts` (log shape)
- Test: `tests/relay/relay-server.test.ts`, `tests/relay/hub.test.ts` (extend)

**Interfaces:**
- Produces: every relay log line gains `severity` (`INFO`, `WARNING`, `ERROR`) so Cloud Logging reads it; the `accepted` line gains `origin` (the run's allowed origin, never a participant value); socket `error` events are logged by `code` only (never the message, URL or stack) instead of ignored; the log-based-metric field `command.durationMs` is documented in a comment next to its emit site. Outbound backpressure: an optional `bufferedAmount` getter on `PeerSocket` and a cap (default 8 MiB, constant in `hub.ts`) checked in `sendTo`; over the cap the send is dropped, the run is closed with a fixed reason and one `WARNING` line is logged. Inbound `maxPayload` stays at its current value.

- [ ] **Step 1: Write failing tests**: each log line is JSON with a `severity`; the accepted line has `origin` and no token, URL or query; a socket error logs its `code` and nothing else; a peer whose `bufferedAmount` exceeds the cap gets the run closed with the fixed reason and one warning, and a peer under the cap is unaffected; a missing `bufferedAmount` getter is treated as under the cap.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS** (`tests/relay`).
- [ ] **Step 5: Commit** `feat(relay): add severity, origin and backpressure to the relay`.

### Task 3: Pipeline and Terraform, off by default

**Files:**
- Modify: `.github/workflows/promote.yml`, `.github/workflows/deploy.yml`, `terraform/relay.tf`, `terraform/cloud_run.tf`, `terraform/variables.tf`, `terraform/outputs.tf` only if needed
- Test: `tests/agent/check-workflow-secrets.test.ts` (extend if it pins workflow shape); `terraform fmt -check` and `terraform validate` if available

**Interfaces:**
- Produces: `promote.yml` exits 1 when the relay image for the promoted SHA is missing while the relay is enabled for that environment (today it only warns). `relay.tf` gets a data source for `eve-internal-secret-<env>` beside the existing one for `relay-token-secret`. New variables, each defaulting to today's behavior: `relay_runs_enabled` (sets `RELAY_RUNS_ENABLED=1` on the app only when true), `relay_token_endpoint_enabled` (sets `RELAY_TOKEN_ENDPOINT_ENABLED=1`, and the stale comment in `cloud_run.tf` is corrected), `relay_extension_origin` (sets `RELAY_EXTENSION_ORIGIN` when non-empty), `relay_allowed_models` (sets `RELAY_ALLOWED_MODELS` when non-empty), `relay_allowed_hosts` (sets `RELAY_ALLOWED_HOSTS` on the relay when non-empty, alongside the existing suffix), `relay_min_instances` (default 0, the pilot sets 1). `deploy.yml` creates the relay only for dev and production (not preview). A log-based metric on `command.durationMs` is declared in `relay.tf` behind a variable defaulting to off. No plan-time default turns the relay or any flag on.

- [ ] **Step 1: Write failing checks**: a workflow lint test (or a script assertion) that `promote.yml` contains `exit 1` on the missing relay image path and no longer only warns; `terraform fmt -check`; `terraform validate`; a unit or script check that every new variable has a default equal to the current behavior (parse `variables.tf`).
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS** (the checks above and `tests/agent`; say clearly if Terraform is not installed and the checks were skipped).
- [ ] **Step 5: Commit** `ci: fail promotion without a relay image; wire relay settings, off by default`.

### Task 4: Release build, packaging and Chrome policy templates

**Files:**
- Modify: `extension/manifest.json` (`minimum_chrome_version`), `scripts/build-extension.ts` only if the version check needs it
- Create: `scripts/package-extension.ts`, `deploy/chrome-policy/extension-policy.template.json`, `deploy/chrome-policy/README.md`, `.github/workflows/extension-release.yml`
- Test: `tests/relay/package-extension.test.ts` (create)

**Interfaces:**
- Produces: `minimum_chrome_version` in the manifest (P5-f). `scripts/package-extension.ts` (`tsx scripts/package-extension.ts --app-origin https://… --out <dir> [--update-url https://…/updates.xml] [--version x.y.z]`): runs the release build (`EXTENSION_RELEASE=1`, refuses test hooks and a non-https origin), verifies the manifest has the `key`, `minimum_chrome_version` and no `host_permissions` beyond the app origin, writes `extension.zip`, and, when `--update-url` is given, an `updates.xml` Chrome update manifest naming the extension id (derived from the `key`), the version and the hosted package URL. A CRX is produced only when `EXTENSION_PEM_PATH` points to a key outside the repository; the script never reads or writes the `.pem` otherwise, and never prints a key. `deploy/chrome-policy/extension-policy.template.json` is a policy template with placeholders: `ExtensionInstallForcelist` (`<id>;<update url>`), `ExtensionSettings` for the id with `installation_mode: force_installed`, no `runtime_blocked_hosts` that blocks the app or the allowed sites, and a comment block (in the README, since JSON has no comments) on `DeveloperToolsAvailability` and the debugger infobar flag (`--silent-debugger-extension-api`) with what the spec says about each and what must be tested on a managed machine. `extension-release.yml` is a manually dispatched workflow (inputs: environment, app origin, version) that builds and uploads the zip and `updates.xml` as artifacts only; it publishes nothing and holds no signing key.

- [ ] **Step 1: Write failing tests**: the package script refuses a missing `APP_ORIGIN`, a non-https origin and test hooks; the zip contains `manifest.json`, `service-worker.js`, `panel.html`, `panel.js`, `panel.css` and nothing else (no `test-page.html`, no maps); the manifest in the zip has the `key` and `minimum_chrome_version` and host permissions equal to the app origin only; `updates.xml` carries the id derived from the `key`, the version and the URL; no `.pem` content appears in the output or the logs; the policy template is valid JSON and contains the placeholders.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS** (`tests/relay`, `pnpm extension:build` still works).
- [ ] **Step 5: Commit** `feat(extension): add release packaging and Chrome policy templates`.

### Task 5: The banner sentence and policy-error copy

**Files:**
- Modify: `extension/src/bridge.ts` (map the two Chrome policy errors to a fixed worker refusal), `extension/panel/lib/worker-api.ts`, `extension/panel/lib/copy.ts`, `extension/panel/screens/ready.tsx`, `extension/panel/screens/running.tsx` (where a run ends or a step is refused for policy)
- Test: `tests/relay/bridge-policy-errors.test.ts` (create), `tests/relay/panel-worker-api.test.ts`, `tests/relay/panel-copy.test.ts` (extend)

**Interfaces:**
- Consumes: the spec's lines on the Chrome 155 restrictions and the debugging banner (the fact sheet at `scratchpad/p5-facts.md` quotes them; re-read `docs/specs/2026-10-23-browser-extension-design.md` lines 398-454).
- Produces: the Ready screen shows one fixed sentence that Chrome will show a "is being debugged" bar while the agent works and that it is expected. A `chrome.debugger` attach or command error whose text contains `Host access is restricted by policy` or `Screenshot capture is restricted by policy` maps to a fixed worker refusal (`restricted by policy`) that never carries the Chrome text; the panel maps it to a fixed explanation and what to do (ask IT to allow the extension on the site). The worker text is never shown.

- [ ] **Step 1: Write failing tests**: each Chrome phrase maps to the fixed refusal on attach, on a CDP send and on a screenshot; an unrelated debugger error does not; the refusal string carries no Chrome text; the panel maps it to fixed copy; the Ready sentence renders; the copy never contains the Chrome text.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS** (`tests/relay`, bundle guard test, `tsc --noEmit` with only the three known error files).
- [ ] **Step 5: Commit** `feat(extension): explain the debugger bar and Chrome policy restrictions`.

### Task 6: The pilot runbook

**Files:**
- Create: `docs/rollout-pilot.md`

**Interfaces:**
- Produces: a runbook for an operator, in order, with who does each step: (1) decisions to confirm (sites list, allowed models, minimum Chrome, extension hosting); (2) secrets: create `relay-token-secret-<env>` and `eve-internal-secret-<env>` in Secret Manager (generate with `openssl rand -hex 32`; never paste a value into a ticket or chat); (3) the logging exclusion permission (`roles/logging.configWriter` or accept request logs knowingly), with how to check sinks; (4) repository variables and Terraform variables to set, in order (relay on for one environment, `relay_min_instances=1`, `relay_allowed_hosts` in two applies to avoid a 421 lockout); (5) the migration order and the pre-deploy duplicate check from the todo; (6) build, package and host the extension (release build per environment, update manifest, policy template), including the extension-id check against `RELAY_EXTENSION_ID` and `RELAY_EXTENSION_ORIGIN`; (7) the managed-Chrome tests: banner suppression flag, host access policy, Chrome 155 behavior, with the exact things to look for; (8) the smoke checklist after the first apply: `/healthz` 200, a wrong Host answers 421, a run end to end, timings against the spike (56 ms round trip, about 600 ms snapshot, about 240 ms fill), a search of the Eve Postgres world for token strings (`eyJ`), a Logs Explorer check that no relay request logs with tokens exist plus the sink list, a mid-run relay restart; (9) turning the flags on last (`RELAY_RUNS_ENABLED`, `RELAY_TOKEN_ENDPOINT_ENABLED`) and the rollback (flags off first; the relay and its deletion-protection caveat); (10) what is deliberately not covered (multi-instance, Redis, OpenTelemetry spans, submit-detection tuning on real portals) and the IT questions still open (which Chrome channel and version, whether a flag can be set on the managed fleet, the site list). It states the known limits plainly: run state and tombstones are in memory, a relay restart or the one-hour socket limit ends live runs, one instance, the agent token still travels in a URL query on the agent leg.

- [ ] **Step 1: Write the runbook** from the fact sheet and the diff, checking each command against the repository (variable and secret names, workflow inputs, file paths).
- [ ] **Step 2: Verify** every file path, variable name and command in it exists in the repository (a small script or grep list in the report), and that no step needs a secret value in the doc.
- [ ] **Step 3: Commit** `docs: add the pilot rollout runbook`.

### Task 7: Docs and tracker

**Files:**
- Modify: `CLAUDE.md`, `docs/browser-extension-todo.md`
- Create: `docs/changes/2026-11-04-add-pilot-rollout.md` (use the change-docs skill; H1 in Title Case)

- [ ] **Step 1: Update the docs from the diff**: the subprotocol transport and the transition, relay logging and backpressure, the new Terraform variables and their defaults, the pipeline change, the release packaging and policy templates, the panel copy for the banner and Chrome policy errors, the runbook; tick the Plan 5 items this plan closes (banner copy, token exposure on the extension leg, `promote.yml` fail-hard, the eve-internal-secret data source, the exact relay host variable, observability and backpressure) and keep the rest (multi-instance gaps, Redis, spans, banner suppression test on a managed Chrome); correct the todo's "token replayable 30 minutes after close" limit (the hub's closed-run tombstone refuses a re-dial for the token TTL, in memory and after terminal closes only).
- [ ] **Step 2: Run the whole node suite** `pnpm exec vitest run --config vitest.config.node.mjs tests/agent tests/relay --exclude 'tests/relay/e2e/**'` and the relay e2e files once; expect PASS.
- [ ] **Step 3: Commit** `docs: document the pilot rollout changes`.

## Self-Review Notes

- Spec coverage: the debugging banner and Chrome 155 policy copy (Task 5), managed force-install artifacts and the managed-Chrome checks (Tasks 4 and 6), token exposure (Task 1), relay observability (Task 2), the deploy pipeline and exact host (Task 3). Open items that only the owner can answer (site list, allowed models, fleet Chrome version, flag on the fleet) are in the runbook (Task 6).
- Ordering: Task 1 before Task 6 (the runbook describes the transport); Tasks 3 and 4 before Task 6; Task 5 independent; Task 7 last. Tasks 1, 2 and 5 touch different files except `bridge.ts` (Tasks 1 and 5), so they run one after the other.
- Riskiest edits: Task 1 (a handshake that a real Chrome must accept: the protocol echo) and Task 3 (infrastructure that must not change behavior by default and must not lock the relay out).
- Left out deliberately: Redis and multi-instance gaps, relay OpenTelemetry spans, single-use tokens, splitting `hub.ts` and `bridge.ts`, a Web Store listing, any deploy or apply.
