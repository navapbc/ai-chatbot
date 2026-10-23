# CDP Relay Spike

Reference only. Not production code, not wired into the app, and not
reviewed for production use.

Question: can agent-browser drive the caseworker's own Chrome tab through a
browser extension (`chrome.debugger`) plus a server-side relay that looks like
a browser-level CDP endpoint, so the existing prompts, skills and evals keep
working unchanged?

Answer: yes, with the caveats under Not Proven.

## Layout

- `relay.mjs` — WebSocket server. `/extension` is the extension, `/cdp` is
  agent-browser. Optional Redis bus for multi-instance routing. Token-protected
  `/ping` and `/bench` endpoints for measurement.
- `extension/` — MV3 extension that attaches `chrome.debugger` and forwards
  CDP. `config.js` (token, relay URL, tab origin) is generated and gitignored.
- `launch.mjs` — headless Chromium with the extension loaded. `EXT_DIR` and
  `FORM_URL` override the defaults.
- `bench.mjs`, `run-bench.sh`, `delay.mjs` — timing harness and a TCP proxy
  that delays each direction.
- `Dockerfile.cloud` — container recipe. It expects a build directory holding
  `relay.mjs`, `bench.mjs`, `form.html`, `oopif.html` and a `bin/` directory
  with the agent-browser Linux binaries copied from
  `node_modules/agent-browser/bin`.

## Findings

Verified with agent-browser 0.33.2 against the relay, in Playwright Chromium
and in a desktop Chrome:

- `snapshot -i`, `fill`, `type`, `select`, `check`, `find label … fill`,
  `batch`, `eval`, `get`, `screenshot`, `open` to another origin and
  `tab new` all worked, including fields inside same-origin and cross-origin
  iframes.
- The extension must detach and re-attach on each new relay client. Otherwise a
  relay restart yields "Another debugger is already attached" and drops
  cross-origin iframe sessions.
- Two relay instances sharing a Redis pub/sub bus route correctly: the
  extension on one instance, agent-browser on the other. No measurable cost.

Real Cloud Run (single instance, extension on a home network, agent-browser in
the container):

| Measurement | Result |
|---|---|
| Round trip, extension to relay | 56 ms median |
| Snapshot | ~600 ms |
| One fill | ~240 ms (about 4 CDP calls, each a round trip) |
| Ten fills in one `batch` | ~2.3 s, no faster than sequential |
| Socket after 6 idle minutes | Survived |

Latency scales with round trips, not with command count. `batch` does not help
because agent-browser still awaits each CDP reply in sequence.

Security controls the spike proved out:

- Loopback bind locally; per-run random token on both sockets (constant-time
  compare); `Origin` rules (extension only on `/extension`, no `Origin` on
  `/cdp`); Host header allowlist; the token is never returned over HTTP.
- Extension enforces an origin lock on every command against the tab's current
  URL, blocks off-origin `Page.navigate` and new tabs, and detaches when a tab
  leaves scope.

## Not Proven

- More than one Cloud Run instance (routing was proven only locally over Redis).
- A signed-in desktop Chrome profile against a remote relay, backgrounded
  windows, and runs longer than about ten minutes.
- Page-initiated navigation (link click, redirect) mid-run; only commands the
  client sent were exercised.
- Behavior of the extension's service worker across a restart mid-run.
- The production token model: this spike uses one shared token per run.
  Production needs a short-lived per-session token tied to the caseworker login.

## Known Gaps (do not copy into production)

- **CDP method allowlist is missing.** The extension forwards any CDP method
  once a tab is in scope. A token holder could read cookies for the whole
  profile (`Network.getAllCookies`), navigate through `Runtime.evaluate`, or use
  `Fetch.*`, `Storage.*` and `Target.*`. The origin lock only checks the tab's
  current URL, not what a command does. Production needs an explicit allowlist
  derived from the methods agent-browser actually sends, with these families
  denied by default. Child (iframe) sessions need the same check against their
  own target URL, not just the owning tab's.
- One shared token per run, no per-session identity.
- No audit log of which commands ran on which origin.

## Running locally

```bash
node spike/cdp-relay/relay.mjs
```

Load `spike/cdp-relay/extension` unpacked, reload it after every relay start
(the token changes), then point agent-browser at the `cdp=` URL the relay
prints.
