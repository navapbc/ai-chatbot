# Browser Bridge Implementation Plan (Plan 1 of 5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The Eve agent fills a form in a real Chrome tab through a separately deployed relay, under token auth, an origin lock and a CDP method allowlist.

**Architecture:** A new Cloud Run service (`relay/`) accepts two WebSockets per run: `/extension` (the caseworker's service worker) and `/cdp` (agent-browser, run by the Eve process). It presents a browser-level CDP endpoint backed by `chrome.debugger`. A minimal MV3 extension (`extension/`) forwards CDP and enforces scope. Policy and token code in `lib/relay/` is shared by the relay, the extension and the Next app. The Eve `browser` tool uses the relay when the session carries relay credentials, and Kernel otherwise.

**Tech Stack:** Node 24, TypeScript, `ws`, `redis` (already a dependency), Chrome MV3, agent-browser 0.33.2, vitest (node config) with Playwright Chromium for end-to-end tests, Terraform.

**Spec:** `docs/specs/2026-10-23-browser-extension-design.md`. Reference code: `spike/cdp-relay/` (protocol shim, re-attach, child sessions). The spike is a reference, not a base to extend.

## Plan Series

1. **Browser bridge** (this plan). 2. Agent loop (Postgres session continuity, `ask_question` as Needs You, progress contract). 3. Sensitive-value redaction. 4. Side panel and app API. 5. Rollout on managed Chrome.

**Out of scope here:** redaction and the `eval`/`get` command refusals (plan 3), the panel UI and run start (plan 4), Postgres continuity (plan 2), managed-Chrome banner and policy testing (plan 5), OpenTelemetry spans for the relay (plan 5; plan 1 emits structured log lines).

## Prerequisites (owner decisions, before Task 1)

- [ ] **Approve new dependencies** (repo rule: ask first): `ws` and `@types/ws` (relay sockets), `esbuild` (extension bundling, devDependency), `@types/chrome` (devDependency). `redis` is already a dependency.
- [ ] **Redis for the bus.** The repo has only Upstash REST secrets, which cannot pub/sub. Plan 1 ships the relay with an optional bus and Terraform `max_instances = 1` until a TCP `REDIS_URL` secret named `relay-redis-url-<env>` exists. Provide it when multi-instance is wanted.
- [ ] **Create Secret Manager secret** `relay-token-secret-<env>` (32+ random bytes, hex) for each environment.

## Global Constraints

- HTTPS and `wss` only. Origins are compared exactly (scheme, host, port).
- Tokens are short-lived, scoped to one run, one role and one origin. No shared secret between runs.
- Fail closed: no configured origin means no tab exposed. Unknown CDP method means refused.
- Final submit is human: no code path in this plan clicks a submit control.
- Tokens never appear in logs, error messages, traces or HTTP responses.
- Ask before adding dependencies. pnpm, Biome, TypeScript.
- The web app on `develop` and the Kernel path keep working unchanged.
- Tests: node tests in `vitest.config.node.mjs`; end-to-end tests need Playwright Chromium.

## Review Focus

Each line below has a test in the named task.

1. Relay restarts mid-run: the extension reconnects and a new agent session sees the iframes again (Task 6).
2. A link click or redirect takes the tab off-origin mid-run: the tab detaches and the next command fails with a clear message (Task 6).
3. The tab is closed mid-run: in-flight commands reject promptly instead of hanging (Tasks 4, 6).
4. A 5 MB CDP response (large snapshot) passes through intact (Task 5).
5. Token strings never appear in relay log output (Task 5).

## File Structure

| Path | Responsibility |
|---|---|
| `lib/relay/token.ts` | Mint and verify signed run tokens (used by Next and the relay) |
| `lib/relay/policy.ts` | Origin scope and method allowlist checks (used by relay and extension) |
| `lib/relay/allowed-cdp-methods.ts` | The allowlist data |
| `lib/relay/protocol.ts` | Message types between relay and extension |
| `lib/relay/eve-relay-browser.ts` | Eve-side credentials lookup and command runner |
| `lib/kernel/session-identity.ts` | `browserIdentity` extracted from `eve-browser.ts` (pure) |
| `relay/bus.ts` | Local and Redis pub/sub |
| `relay/cdp-shim.ts` | Browser-level CDP emulation |
| `relay/hub.ts` | Per-run socket pairing, forwarding, timeouts |
| `relay/server.ts`, `relay/main.ts` | HTTP/WS server and entrypoint |
| `relay/Dockerfile` | Relay image |
| `extension/manifest.json`, `extension/src/*.ts` | MV3 service worker (bridge, scope, reconnect) |
| `app/(chat)/api/relay/token/route.ts` | Mints extension and agent tokens |
| `terraform/relay.tf` | Relay Cloud Run service |
| `tests/relay/**` | Unit and end-to-end tests |

---

### Task 1: Run tokens

**Files:**
- Create: `lib/relay/token.ts`
- Test: `tests/relay/token.test.ts`
- Modify: `vitest.config.node.mjs` (add `'tests/relay/**/*.test.ts'` to `include`), `package.json` (add `"test:relay": "vitest run --config vitest.config.node.mjs tests/relay"`)

**Interfaces:**
- Produces:
  ```ts
  export type RelayRole = 'extension' | 'agent';
  export interface RelayClaims { runId: string; role: RelayRole; origin: string; exp: number } // exp = epoch seconds
  export const RELAY_TOKEN_TTL_SECONDS = 1800; // plan default, tunable
  export function mintRelayToken(claims: Omit<RelayClaims, 'exp'>, secret: string, opts?: { now?: number; ttlSeconds?: number }): string;
  export function verifyRelayToken(token: string, secret: string, opts?: { now?: number }): RelayClaims | null;
  ```

- [ ] **Step 1: Write failing tests** in `tests/relay/token.test.ts`: round trip returns the claims with `exp = now + 1800`; returns `null` for each of: wrong secret, flipped payload byte, flipped signature byte, expired (`now = exp`), missing dot, non-JSON payload, `role: 'admin'`, `origin: 'http://example.com'` (not https), empty `runId`.
- [ ] **Step 2: Run** `pnpm test:relay -- token` — Expected: FAIL (module missing).
- [ ] **Step 3: Implement** as `base64url(JSON claims) + '.' + base64url(HMAC-SHA256)` using `node:crypto` and `timingSafeEqual` (compare hashes of equal length). No new dependency.
- [ ] **Step 4: Run** the same command — Expected: PASS.
- [ ] **Step 5: Commit** `feat(relay): signed run tokens`.

### Task 2: Policy (scope and method allowlist)

**Files:**
- Create: `lib/relay/policy.ts`, `lib/relay/allowed-cdp-methods.ts`, `lib/relay/protocol.ts`
- Test: `tests/relay/policy.test.ts`

**Interfaces:**
- Produces:
  ```ts
  export function normalizeOrigin(url: string): string | null; // https only, else null (about:blank, http:, chrome: -> null)
  export function inScope(url: string, origin: string): boolean; // exact origin match
  export function isMethodAllowed(method: string): boolean;
  export type CommandCheck = { ok: true } | { ok: false; reason: string };
  export function checkCommand(cmd: { method: string; params?: unknown }, ctx: { origin: string }): CommandCheck;
  ```
  `protocol.ts` exports `RelayToExtension` (`listTargets | attach {targetId} | createTarget {url?} | closeTarget {targetId} | cdp {sessionId, method, params?}`, each with `id: string`) and `ExtensionToRelay` (`register {runId, tabId, origin} | reply {id, result?, error?} | event {sessionId?, method, params} | tabClosed`).

- [ ] **Step 1: Write failing tests**: `normalizeOrigin('https://a.gov/x?y')` is `'https://a.gov'`; `http://a.gov`, `about:blank`, `chrome://x`, `''` are `null`; `inScope` is false for a different port, a subdomain, and a lookalike host; `checkCommand` refuses `Page.navigate` to an out-of-scope URL, `Target.createTarget` out of scope, `Network.getAllCookies`, `Network.getCookies`, `Storage.getCookies`, `Fetch.enable`, `Page.setDownloadBehavior`, and an unknown method; allows `Accessibility.getFullAXTree` and `Input.insertText`; `Page.navigate` to an in-scope URL is allowed.
- [ ] **Step 2: Run** `pnpm test:relay -- policy` — Expected: FAIL.
- [ ] **Step 3: Implement.** Seed `ALLOWED_CDP_METHODS: ReadonlySet<string>` with the methods in `spike/cdp-relay` relay logs (Runtime, Page, DOM, Accessibility, Input, Network.enable, Target.setAutoAttach, Emulation reads). Task 8 replaces the seed with a measured set.
- [ ] **Step 4: Run** — Expected: PASS.
- [ ] **Step 5: Commit** `feat(relay): scope and CDP method policy`.

### Task 3: Bus

**Files:**
- Create: `relay/bus.ts`
- Test: `tests/relay/bus.test.ts`

**Interfaces:**
- Produces:
  ```ts
  export interface Bus {
    publish(channel: string, message: string): Promise<void>;
    subscribe(channel: string, handler: (message: string) => void): Promise<() => Promise<void>>; // returns unsubscribe
    close(): Promise<void>;
  }
  export function createLocalBus(): Bus;
  export function createRedisBus(url: string): Promise<Bus>; // uses the `redis` package already in dependencies
  ```
  Channel names are built by `channelFor(runId: string, direction: 'to-ext' | 'from-ext' | 'to-agent'): string`, also exported.

- [ ] **Step 1: Write failing tests**: local bus delivers to every subscriber on the same channel, not to others, and stops after unsubscribe; the same suite runs against `createRedisBus` when `RELAY_TEST_REDIS_URL` is set (skipped otherwise, with a logged reason).
- [ ] **Step 2: Run** `pnpm test:relay -- bus` — Expected: FAIL.
- [ ] **Step 3: Implement.** Redis bus uses one publisher and one subscriber client (`duplicate()`), as `spike/cdp-relay/relay.mjs` does.
- [ ] **Step 4: Run**, then once with `RELAY_TEST_REDIS_URL=redis://127.0.0.1:6390` against `docker run -d --rm -p 6390:6379 redis:7-alpine` — Expected: PASS both.
- [ ] **Step 5: Commit** `feat(relay): local and Redis bus`.

### Task 4: CDP shim and run hub

**Files:**
- Create: `relay/cdp-shim.ts`, `relay/hub.ts`
- Test: `tests/relay/hub.test.ts`

**Interfaces:**
- Consumes: `checkCommand`, `RelayToExtension`, `ExtensionToRelay`, `Bus`, `channelFor`.
- Produces:
  ```ts
  export interface PeerSocket { send(data: string): void; close(code?: number, reason?: string): void; on(event: 'message', cb: (data: string) => void): void; on(event: 'close', cb: () => void): void }
  export type DistributiveOmit<T, K extends PropertyKey> = T extends unknown ? Omit<T, K> : never;
  export type ExtensionCall = (msg: DistributiveOmit<RelayToExtension, 'id'>) => Promise<unknown>; // sends to the run's extension, resolves with its reply result
  export class RunHub {
    constructor(opts: { bus: Bus; commandTimeoutMs?: number /* default 30000 */; resumeWindowMs?: number /* default 300000 */; log: (line: Record<string, unknown>) => void });
    attachExtension(runId: string, origin: string, socket: PeerSocket): void; // replaces an earlier extension socket for the run
    attachAgent(runId: string, origin: string, socket: PeerSocket): 'ok' | 'busy';  // one agent socket per run
    closeRun(runId: string, reason: string): void;
  }
  ```
  `cdp-shim.ts` exports `handleBrowserLevel(method: string, params: unknown, ext: ExtensionCall): Promise<unknown>` covering `Browser.getVersion`, `Target.getTargets`, `Target.attachToTarget` (also emits `Target.attachedToTarget`), `Target.createTarget`, `Target.closeTarget`, `Target.getTargetInfo`; other browser-level methods return `{}` and are logged.

- [ ] **Step 1: Write failing tests** with fake `PeerSocket`s: a session-scoped command goes to the extension only after `checkCommand` passes; a denied command gets a CDP error reply and never reaches the extension; `Target.getTargets` returns only the tab the extension registered; the extension's events reach the agent socket; a second agent socket for the same run returns `'busy'`; with `commandTimeoutMs = 50`, an unanswered command rejects with `"extension did not answer"`; when the extension socket closes mid-command, pending commands reject at once with `"tab or extension disconnected"`; with two hubs on one local bus, an agent on hub A reaches an extension on hub B; when the extension socket drops, the run is held for `resumeWindowMs` (agent socket stays open, commands wait) and resumes if a new extension socket attaches within it, and with `resumeWindowMs = 50` and no reattach the agent socket is closed with reason `"extension did not return"`.
- [ ] **Step 2: Run** `pnpm test:relay -- hub` — Expected: FAIL.
- [ ] **Step 3: Implement** both modules. Port the browser-level handling and session routing from `spike/cdp-relay/relay.mjs`; ids are `<instanceId>:<counter>` so replies find their issuer across instances.
- [ ] **Step 4: Run** — Expected: PASS.
- [ ] **Step 5: Commit** `feat(relay): CDP shim and run hub`.

### Task 5: Server

**Files:**
- Create: `relay/server.ts`, `relay/main.ts`
- Test: `tests/relay/server.test.ts`

**Interfaces:**
- Consumes: `verifyRelayToken`, `RunHub`, `Bus`.
- Produces:
  ```ts
  export interface RelayServerOptions { secret: string; allowedHostSuffixes: string[]; allowedHosts: string[]; extensionId?: string; bus: Bus; log: (line: Record<string, unknown>) => void; maxPayloadBytes?: number /* default 32 MiB */; recordMethods?: boolean /* logs {event:'method', method} once per distinct forwarded method; env RELAY_RECORD_METHODS=1; used only by Task 8 */ }
  export function createRelayServer(opts: RelayServerOptions): import('node:http').Server;
  ```
  `main.ts` reads `PORT`, `RELAY_TOKEN_SECRET`, `RELAY_ALLOWED_HOST_SUFFIXES`, `RELAY_ALLOWED_HOSTS`, `RELAY_EXTENSION_ID`, `REDIS_URL` (optional) and listens on `0.0.0.0`.
  Routes: `GET /healthz` (200, no token); `/extension?token=` requires role `extension` and an `Origin` of `chrome-extension://<RELAY_EXTENSION_ID>` (prefix only when no id configured); `/cdp?token=` requires role `agent` and no `Origin` header. Any other path, bad `Host`, or failed token: socket destroyed, one log line without the token.

- [ ] **Step 1: Write failing tests** against a real server on an ephemeral port: no token, wrong secret, expired token, wrong role for the path, a web `Origin` on `/cdp`, a bad `Host`, and a wrong extension ID are each rejected; valid connections pair into a run; `/healthz` answers 200; a 5 MB payload sent extension → agent arrives intact; the captured log output never contains the token string, even for rejected connections.
- [ ] **Step 2: Run** `pnpm test:relay -- server` — Expected: FAIL.
- [ ] **Step 3: Implement** `createRelayServer` with `ws` (`noServer` upgrade handling, `maxPayload` set). Log fields: event, runId, role, durationMs; never the URL.
- [ ] **Step 4: Run** — Expected: PASS.
- [ ] **Step 5: Commit** `feat(relay): authenticated WebSocket server`.

### Task 6: Extension service worker and end-to-end

**Files:**
- Create: `extension/manifest.json`, `extension/src/service-worker.ts`, `extension/src/bridge.ts`, `tests/relay/helpers/harness.ts`, `tests/relay/fixtures/form.html`, `tests/relay/fixtures/oopif.html`, `tests/relay/e2e/bridge.e2e.test.ts`
- Modify: `package.json` (add `"extension:build": "esbuild extension/src/service-worker.ts --bundle --format=esm --outfile=extension/dist/service-worker.js && cp extension/manifest.json extension/dist/manifest.json"`), `.gitignore` (add `extension/dist/`)

**Interfaces:**
- Consumes: `inScope`, `checkCommand`, `normalizeOrigin`, `ExtensionToRelay`, `RelayToExtension`.
- Produces: `chrome.runtime.onMessage` handler `{ type: 'startRun'; relayUrl: string; token: string; tabId: number }` and `{ type: 'stopRun' }`. The service worker registers `{ runId, tabId, origin }` after the socket opens, where `origin` comes from the tab's current URL, which must normalize to https. Test helpers: `startRelay(): Promise<{ url: string; secret: string; stop(): Promise<void> }>`, `launchExtensionBrowser(): Promise<{ context; sw; stop(): Promise<void> }>`, `agentBrowser(args: string[], cdpUrl: string): Promise<unknown>` (spawns the repo's `agent-browser` with `--cdp`).
- Manifest: `permissions: ["debugger", "tabs", "storage"]`, a `key` that pins the extension ID (generate an RSA key with `openssl`, store only the public key in the manifest; record the resulting ID in `extension/EXTENSION_ID`).

Behavior the worker must have (each is asserted by the tests below):
- Exposes only the registered tab; refuses `createTarget` outside scope.
- Checks the tab's current URL on every command, detaches and replies with an error when out of scope (`"tab is outside the allowed origin"`).
- Detaches before attaching on each new relay client (the spike's re-attach fix) and forwards child (iframe) sessions.
- Refuses a `relayUrl` that is not `wss:` unless its host is `127.0.0.1` or `localhost` (tests only).
- On socket loss, reconnects with the same token while it is valid. The relay holds the run for its `resumeWindowMs` (Task 4); after that the run stops.

- [ ] **Step 1: Write failing end-to-end tests** (the harness serves fixtures over HTTPS from a certificate it generates per run with `openssl req -x509 -newkey rsa:2048 -nodes -subj /CN=localhost -days 1`, and launches Chromium with `--ignore-certificate-errors`; the relay itself listens on plain `ws://127.0.0.1` in tests): `snapshot -i` lists the form and the same-origin and cross-origin iframe fields; `fill`, `select`, `check` and `batch` land values; `eval document.title` works; raw CDP `Network.getAllCookies` is refused; `open` to another origin is refused and the tab stays put; `tab new` is refused; `startRun` with `relayUrl: 'ws://example.com/x'` is refused while `ws://127.0.0.1:<port>` is accepted.
  Review Focus 1: restart the relay mid-run, then a fresh agent-browser session lists the cross-origin iframe field again.
  Review Focus 2: click a link to another origin (page-initiated); the next command fails with the out-of-scope message and the debugger is detached.
  Review Focus 3: close the tab; an in-flight and a following command both fail within five seconds, not at the 30 second timeout.
- [ ] **Step 2: Run** `pnpm extension:build && pnpm test:relay -- e2e` — Expected: FAIL.
- [ ] **Step 3: Implement** the worker; port from `spike/cdp-relay/extension/sw.js`, replacing the file-based config with the `startRun` message and persisting the run (token, relay URL, tab id) in `chrome.storage.session` so a service worker restart can resume.
- [ ] **Step 4: Run** the same command — Expected: PASS.
- [ ] **Step 5: Commit** `feat(extension): MV3 bridge with scope enforcement`.

### Task 7: Eve browser tool over the relay

**Files:**
- Create: `lib/kernel/session-identity.ts` (move `browserIdentity` from `lib/kernel/eve-browser.ts`; re-export it there), `lib/relay/eve-relay-browser.ts`
- Modify: `agent/tools/browser.ts`, `agent/channels/eve.ts`, `lib/ai/eve/eve-client.ts`, `tests/agent/eve-browser-identity.test.ts` (import path)
- Test: `tests/relay/eve-relay-browser.test.ts`

**Interfaces:**
- Produces:
  ```ts
  export interface RelayCredentials { url: string; token: string }
  export function relayCredentialsFrom(ctx: ToolContext): RelayCredentials | null; // reads auth attributes relayUrl, relayToken (initiator first, then current)
  export function runRelayCommand(ctx: ToolContext, command: readonly string[]): Promise<{ success: boolean; data?: unknown; error?: string | null; liveViewUrl: null }>;
  ```
  `runRelayCommand` calls `runCommand` from `lib/kernel/cli.ts` with `cdpUrl = <url>/cdp?token=<token>` and the same per-session queue and `cliSessionName(userId, sessionId)` as the Kernel path. `agent/channels/eve.ts` replaces `modelAttributeAuth` with one function that, on loopback requests only, exposes `x-eve-model`, `x-eve-relay-url` and `x-eve-relay-token` as attributes `eveModel`, `relayUrl`, `relayToken`. `createEveSession` gains an optional `relay?: RelayCredentials` that sets those headers.

- [ ] **Step 1: Write failing tests**: `relayCredentialsFrom` returns `null` with no attributes, credentials when both are present, prefers the initiator's, and returns `null` when only one of the two is set; `runRelayCommand` against the Task 6 harness fills a fixture field end to end (no model needed); the Kernel path is used when credentials are absent (mock `runBrowserCommand` called).
- [ ] **Step 2: Run** `pnpm test:relay -- eve-relay-browser` and `pnpm exec vitest run --config vitest.config.node.mjs tests/agent/eve-browser-identity.test.ts` — Expected: FAIL.
- [ ] **Step 3: Implement** the files above. In `agent/tools/browser.ts` change "remote Kernel browser" in the description to "the browser".
- [ ] **Step 4: Run** both commands plus the rest of `tests/agent` — Expected: PASS.
- [ ] **Step 5: Verify the credential is not persisted.** Run `pnpm eve:dev`, create a session with a recognizable fake token through `createEveSession`, run one browser command, then `grep -R "<fake token>" .eve .eve/.workflow-data 2>/dev/null` and check the Braintrust span output locally. Expected: no match. If a match appears, stop and report it; do not ship.
- [ ] **Step 6: Commit** `feat(agent): route Eve browser tool through the relay`.

### Task 8: Allowlist from measurement

**Files:**
- Modify: `lib/relay/allowed-cdp-methods.ts`
- Create: `tests/relay/e2e/allowlist.e2e.test.ts`

**Interfaces:**
- Consumes: the Task 6 harness. The relay gains a `RELAY_RECORD_METHODS=1` mode that logs each distinct method it forwards (log line `{event: 'method', method}`), used only by this test.

- [ ] **Step 1: Write the failing test**: run every command in the Eve browser tool description (`open`, `snapshot` and its flags, `click`, `fill`, `type`, `select`, `find ... fill|click`, `press`, `hover`, `check`, `uncheck`, `scrollintoview`, `wait`, `get`, `scroll`, `screenshot`, `batch`, `back`, `forward`, `eval`, `tab`, `dialog`, `frame`) against the fixture with recording on, with the allowlist temporarily set to "allow all". Assert that the recorded set equals `ALLOWED_CDP_METHODS` (nothing missing, nothing extra beyond a short listed `KEPT_FOR_RELAY` set).
- [ ] **Step 2: Run** `pnpm extension:build && pnpm test:relay -- allowlist` — Expected: FAIL, printing the missing methods.
- [ ] **Step 3: Update** `ALLOWED_CDP_METHODS` to the measured set. Keep `Network.getAllCookies`, `Network.getCookies`, `Storage.*`, `Fetch.*` and `Page.setDownloadBehavior` out; if agent-browser genuinely sends one, stop and raise it, because the spec denies them.
- [ ] **Step 4: Run** the whole `pnpm test:relay` — Expected: PASS.
- [ ] **Step 5: Commit** `feat(relay): CDP allowlist from measured traffic`.

### Task 9: Token endpoint

**Files:**
- Create: `app/(chat)/api/relay/token/route.ts`
- Test: `tests/relay/token-route.test.ts`

**Interfaces:**
- Consumes: `mintRelayToken`, `auth` from `@/app/(auth)/auth`, `ChatSDKError`.
- Produces: `POST /api/relay/token` body `{ runId: string (uuid); origin: string }`, response `{ extensionToken: string; agentToken: string; relayUrl: string; expiresAt: number }`. 401 without a session, 400 for a non-https or non-origin value, 500 (`ChatSDKError`) when `RELAY_TOKEN_SECRET` or `RELAY_PUBLIC_URL` is unset. Ownership of `runId` is not checked in plan 1; plan 4 binds runs to users.

- [ ] **Step 1: Write failing tests** (mock `auth`): unauthenticated is 401; `origin: 'http://a.gov'` is 400; a good request returns two tokens that verify with the right roles, the same `runId` and the normalized origin, and `relayUrl` equals `RELAY_PUBLIC_URL`; missing env is a 500 that does not echo secrets.
- [ ] **Step 2: Run** `pnpm test:relay -- token-route` — Expected: FAIL.
- [ ] **Step 3: Implement** the route following the style of `app/(chat)/api/chat/route.ts` (zod schema, `ChatSDKError`).
- [ ] **Step 4: Run** — Expected: PASS.
- [ ] **Step 5: Commit** `feat(api): relay token endpoint`.

### Task 10: Deployment

**Files:**
- Create: `relay/Dockerfile`, `terraform/relay.tf`
- Modify: `terraform/variables.tf` (add `relay_image_url` with an empty-string default so `promote.yml`, which passes only `environment` and `chatbot_image_url`, keeps planning; `relay_max_instances` default `1`), `terraform/cloud_run.tf` (add env `RELAY_TOKEN_SECRET` from Secret Manager and `RELAY_PUBLIC_URL` = relay service URI to `ai_chatbot`), `.github/workflows/deploy.yml` (build and push the relay image next to the chatbot image; pass `relay_image_url`)

**Interfaces:**
- Relay service: name `labs-asp-relay-<environment>`, 1 vCPU, 512 MiB, min 0, max `var.relay_max_instances`, `session_affinity = false`, `timeout = "3600s"`, the existing `google_service_account.cloud_run`, env `RELAY_TOKEN_SECRET` and `REDIS_URL` from Secret Manager (`REDIS_URL` omitted while `relay_max_instances = 1`), `RELAY_ALLOWED_HOST_SUFFIXES=.run.app`, `RELAY_EXTENSION_ID` from `extension/EXTENSION_ID`. `allUsers` invoker, because the extension cannot send Google identity tokens; token auth is the gate. Skip the service when `relay_image_url` is empty (`count`).

- [ ] **Step 1: Write the image and run it locally**: `docker build -f relay/Dockerfile -t relay:local .`; run with `PORT=8080 RELAY_TOKEN_SECRET=<test> RELAY_ALLOWED_HOST_SUFFIXES=.localhost,127.0.0.1 ...`. Expected: `GET /healthz` returns 200 and `pnpm test:relay -- e2e` passes when pointed at the container (`RELAY_E2E_URL=http://127.0.0.1:8080`).
- [ ] **Step 2: Add the Terraform** and run `terraform -chdir=terraform fmt -check && terraform -chdir=terraform validate` — Expected: success.
- [ ] **Step 3: Run `terraform plan`** for the dev environment as CI does; Expected: relay resources added, no changes to existing services except the two new env vars. Do not apply; deployment is the owner's call.
- [ ] **Step 4: Commit** `feat(deploy): relay Cloud Run service`.

### Task 11: Cloud check (owner-approved, manual)

- [ ] After the owner applies the dev environment, run the Task 6 end-to-end suite against the deployed relay (`RELAY_E2E_URL=https://<relay>.run.app`) from a home network, and record median round trip, snapshot time and one-fill time next to the spike's 56 ms, 600 ms and 240 ms in `docs/specs/2026-10-23-browser-extension-design.md` under "What the Spike Proved". Expected: within about 50% of the spike numbers. Larger regressions are reported before plan 2 starts.
- [ ] Run one agent smoke test with credentials in `.env.local` (not read by the agent): mint tokens with the route, start an Eve session with the relay headers, and ask the agent to fill the fixture form. Record the result in the PR; this step needs a model key and is not part of CI.

---

## Self-Review Notes

- **Spec coverage** (spec sections this plan covers): origin lock, tab scope, CDP allowlist, authentication with short-lived scoped tokens, extension ID pin and `Origin`/`Host` rules, no token in responses, relay as a separate Cloud Run service with a Redis bus, Eve browser tool over the relay, run pause on socket loss, hard checks on tab leaving scope. Deferred to later plans as listed under Out of scope.
- **Proportion:** each task carries signatures, test names and assertions, not bodies; the port from the spike is referenced by file instead of repeated.
- **Open risk the executor must raise, not guess:** whether the relay credential in the agent-browser argv (`--cdp ...?token=`) is visible to other processes in the Eve container and whether agent-browser has an environment-variable alternative; check `agent-browser --help` in Task 7 and record the answer.
