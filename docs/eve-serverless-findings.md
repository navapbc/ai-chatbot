# Eve Serverless Findings

Exploration on branch `explore/eve-serverless`, 2026-08-31. Question: is there a
serverless way to run Eve, instead of the `npx eve dev` sidecar + `EVE_SERVER_URL`
topology that SP-B wired up?

Short answer: **yes for the agent loop, no for the browser tool as currently
written.** The mounting story is better than expected and deletes a lot of SP-B.
The browser tool has a hard dependency on process locality that serverless
breaks. That second half is the blocker, and it is not a routing problem.

Eve version under test: `0.27.13`. `agent-browser` `0.33.2`. Node 24.19.0.

## The serverless path exists: `withEve()`

`eve/next` exports `withEve`, which mounts the Eve runtime into the Next.js app
as a single project — no sidecar, no CORS, no `EVE_SERVER_URL`.

```ts
// next.config.ts
import { withEve } from 'eve/next';
export default withEve(nextConfig);
```

Verified against this repo (this branch carries the one-line change):

- `pnpm dev` boots the Eve dev server as a child and rewrites `/eve/v1/**` to it.
- `curl localhost:3000/eve/v1/health` → `{"ok":true,"status":"ready","workflowId":"workflow//eve//workflowEntry"}`
  on the **Next origin (:3000)**, identical to the Eve child's own port.
- No compile errors from the existing `agent/` tree.

**The Vercel shape below is doc-sourced, not verified here.** What was tested is
`pnpm dev`, which Eve's own docs describe as a *different* topology (child process
+ rewrites). Per the docs, on Vercel this deploys as one project: `withEve()`
writes Build Output `services` for Eve plus `routes` sending `/eve/v1/**` to that
service ahead of filesystem routing; durability moves from the on-disk local
Workflow world (`.eve/.workflow-data`) to Vercel Workflow; authored schedules
become Vercel Cron jobs. That would be genuinely serverless — no long-running
process to operate. Confirming it needs an actual preview deploy, which is still
the deferred Task 5 from the original spike.

### What this deletes from SP-B

`withEve()` + `useEveAgent` from `eve/react` makes most of the SP-B adapter
redundant: `lib/ai/eve/session-continuity.ts`, `stream-adapter.ts`,
`eve-client.ts`, the NDJSON→SSE translation in `app/(chat)/api/eve-chat/route.ts`,
and the `EVE_SERVER_URL` env var.

**But not for free.** `useEveAgent` surfaces Eve tool calls as `dynamic-tool`
parts. `components/message.tsx` renders on *typed* parts — it matches
`tool-gapAnalysis` (lines 108, 495) and `tool-formSummary` (line 513). The SP-B
adapter exists partly to do that snake_case→camelCase renaming. Dropping the
adapter means reworking those renderers to read `dynamic-tool` + `toolName`, or
keeping a thin projection. Not verified how the cards land in practice.

## Blocker: `@eN` refs do not survive an instance hop

`agent-browser` is a native CLI whose **daemon** holds the CDP connection and the
ref table between invocations, keyed by `--session` over a unix socket
(`--namespace` isolates "daemon sockets"; `--idle-timeout` defaults to 1h). SP-A
proved refs survive an ~11-minute durable park — but that was under one
long-lived `eve dev` process, where the daemon never went away.

On Vercel the Eve service runs as Functions. Instance reuse is likely under Fluid
Compute but never guaranteed, and a parked workflow resuming on a different
instance is by design.

Tested directly, against one persistent Chrome over CDP, using `--namespace` to
simulate a fresh instance (new daemon socket, same remote browser):

```
# daemon A: open + snapshot -i  →  refs {e1: heading, e2: link "Learn more"}
# CONTROL — daemon A, click @e2
{"success":true,"data":{"clicked":"@e2", ...}}

# TEST — fresh daemon B, same CDP target, same ref
{"success":false,"data":null,"error":"Unknown ref: e2"}
```

`Unknown ref: e2`. The ref table is daemon-local memory, not page state. A cold
instance between `snapshot` and `click` breaks the snapshot-first discipline the
entire browser prompt is built on.

Recovery is possible — a fresh `snapshot` re-derives refs — but the agent has no
signal that it hopped instances, and the obvious mitigation is worse than it
looks. Catching `Unknown ref:` inside the tool and transparently re-snapshotting
**cannot** then reissue `click @e2`: a fresh snapshot may bind `e2` to a
different element, so a blind retry silently clicks the wrong thing — strictly
worse than failing. Any sound recovery has to hand the new snapshot back to the
model and let it re-choose, which makes this a turn-level concern, not a
tool-level one. That is a substantially harder fix than a latency cost.

### Second, quieter problem: duplicate Kernel browsers

`lib/kernel/eve-browser.ts` caches sessions in a module-scope
`Map` (`sessions`, `pendingCreations`, `sessionQueues`). On a cache miss
`getOrCreateEveBrowser` calls `kernel.browsers.create(...)`. On serverless, a
cold instance is a guaranteed cache miss — so every cold start **creates a new
Kernel browser and orphans the previous one**, rather than reattaching. This is a
code-read finding, not measured; it needs a cross-instance store (Redis/Postgres)
keyed by `cacheKey(userId, sessionId)` holding `cdpWsUrl` + `kernelSessionId`.

## The `server-only` barrier is unchanged

SP-A's #1 SP-C risk survives `withEve()`. A probe tool importing
`@/lib/db/queries` (whose line 1 is `import 'server-only'`) fails Eve's discovery
step:

```
[eve:dev] rebuild failed: Failed to evaluate authored module:
  agent/tools/probe_serveronly.ts
```

Confirmed the cause is the barrier itself and not a missing `DATABASE_URL` —
under plain Node resolution (no `react-server` condition), `import('server-only')`
throws *"This module cannot be imported from a Client Component module."*

`withEve()` does not set that condition for the Eve compile. So the SP-C Postgres
history bridge still needs one of SP-A's two fixes: a bundler-level alias for
`server-only`, or lifting the DB/GCS calls out of `lib/kernel/browser.ts`'s
module-load path.

## Fallback: self-host on Cloud Run

Not built — noted so it isn't re-derived. `eve build` writes a Nitro Node server
to `.output/` (`eve start`). That runs on Cloud Run as a container, but two things
must change:

- **Workflow state.** The default local world persists to disk under
  `.eve/.workflow-data`, which does not survive Cloud Run's ephemeral, scaling
  instances. `@workflow/world-postgres@5.0.0-beta.38` exists on npm and matches
  Eve's required `5.0.0-beta` protocol line; select it via
  `experimental.workflow.world` in `agent/agent.ts`.
- **Routing.** Both `/eve/` *and* `/.well-known/workflow/` must be forwarded
  unrewritten. A proxy that only forwards `/eve/` lets a session start and then
  stalls when the workflow callback can't get home.

Note this fallback has the *same* browser-daemon problem the moment Cloud Run
scales past one instance — it is not a way to dodge the blocker, only to defer it.

## Where this leaves the migration

1. `withEve()` is the right topology and is proven to mount on this repo.
2. The browser tool needs cross-instance state before Eve can go serverless:
   a shared session store, and either ref-recovery-on-miss or a move off refs.
3. `server-only` is still unsolved and still gates the Postgres history bridge.

The honest framing: serverless Eve is reachable, but the browser tool — the thing
this product is — is the part that isn't ready, and no amount of deployment
configuration fixes it.

## Reproducing

```bash
git checkout explore/eve-serverless
nvm use 24
set -a; . ./.env.local; set +a
pnpm dev
curl localhost:3000/eve/v1/health
```

Ref-survival experiment: launch Chrome with `--remote-debugging-port=9333`, then
run `snapshot -i` under `--namespace nsA` and `click @e2` under `--namespace nsB`
against the same `--cdp` URL.
