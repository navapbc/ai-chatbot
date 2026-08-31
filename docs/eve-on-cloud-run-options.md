# Running Eve in the Current Cloud Run Deployment

Branch `explore/eve-serverless`, 2026-08-31. Question: can Eve run inside the
deployment this app already has, rather than needing its own server?

**Yes, and it needs no infrastructure change.** Eve's `withEve()` spawns the Eve
runtime as a child process of `next start` — one container, one Cloud Run
service, one port. The work is three lines of Dockerfile and a workflow-state
decision.

## Correcting the previous finding

`docs/eve-serverless-findings.md` concluded the browser daemon blocks serverless
Eve. **That conclusion was scoped to Vercel Functions and does not apply here.**

This deployment already solves daemon locality — [`terraform/cloud_run.tf:369`](terraform/cloud_run.tf:369):

```hcl
# Pin each user to the same instance so in-memory BrowserManager
# CDP connections persist across tool calls
session_affinity = true
```

The `Unknown ref: e2` result still stands as a fact about `agent-browser` — refs
live in daemon-local memory — but session affinity is exactly the mitigation, and
it is already in production. It is a caveat below, not a blocker.

## What the current deployment is

| | |
|---|---|
| Platform | Cloud Run, one service, container port 3000 |
| Scaling | `min_instance_count = 2`, `max = 20`, `session_affinity = true` |
| Resources | 2 vCPU, 8Gi, request timeout 3600s |
| Database | Cloud SQL Postgres over unix socket at `/cloudsql` |
| Image | `node:24-slim`, `next build` at image build, `pnpm tsx lib/db/migrate && pnpm start` at boot |
| Browser | `agent-browser` native binary at `/usr/local/bin`, `HOME=/tmp` for its daemon socket |

Root filesystem is writable (no `read_only` set), backed by Cloud Run's
in-memory tmpfs — anything written at runtime counts against the 8Gi.

## The mechanism: `next start` spawns Eve

From `eve/next`'s compiled `server.js`, `startEveProductionServer`:

```js
const entry = join(appRoot, '.output', 'server', 'index.mjs');
if (existsSync(entry))
  return startServerProcess({
    command: process.execPath, args: [entry], cwd: appRoot,
    env: { HOST, NITRO_HOST, NITRO_PORT: port, PORT: port },
  });
```

When `NODE_ENV=production`, `VERCEL` is unset, and `EVE_NEXT_PRODUCTION_ORIGIN`
is unset, Next.js **spawns** `node .output/server/index.mjs` on
`127.0.0.1:4274` and rewrites `/eve/v1/**` to it. If `.output/` is missing it
silently returns `undefined` and the rewrite points at a dead port — so the build
step is mandatory, and its absence fails at request time, not build time.

Two things verified for this repo:

- **`eve build` needs no runtime secrets.** Run with `env -i` (no `DATABASE_URL`,
  no Vertex credentials, no `KERNEL_API_KEY`, no `AI_GATEWAY_API_KEY`): exit 0,
  writes `.output/server/index.mjs`. 11.4 MB total. It works in the Docker builder
  stage as it stands.
- **Workflow callbacks stay internal.** `execution/workflow-callback-url.js`
  resolves the callback base to `WORKFLOW_LOCAL_BASE_URL` or the server's own
  origin. In a same-container setup the Eve process calls itself on localhost, so
  Next only has to proxy `/eve/v1/**` — which `withEve()` already does. The
  self-hosting docs' warning about also routing `/.well-known/workflow/` applies
  when Eve is the *public* service (Option C below), not here.

## The decision that actually matters: workflow state

Process topology is the easy axis. The one that discriminates is **where durable
workflow state lives.**

Eve's default local world writes `.eve/.workflow-data` (`events/`, `runs/`,
`steps/`, `hooks/`, `streams/`) to container-local disk. On this service that
means **per-instance, in-memory, and lost when the instance goes away.**

Session affinity does not rescue this. Cloud Run affinity is best-effort — it
breaks on instance replacement, scale-down, and every deploy. Today that degrades
gracefully: a `BrowserManager` cache miss just makes a new browser. With Eve on
local disk it does not — a durable session whose state lived on a dead instance
is simply gone, which defeats the reason to adopt Eve at all.

**`@workflow/world-postgres@5.0.0-beta.38`** is the fix, and it fits unusually
well:

- Matches Eve's required `5.0.0-beta` protocol line.
- Reads `WORKFLOW_POSTGRES_URL`, **falling back to `DATABASE_URL`** — already set.
- Standard `pg` connection string, so the existing `/cloudsql` unix socket works
  (`postgres:///db?host=/cloudsql/INSTANCE`). Can also share an existing `pg.Pool`.
- Built on `drizzle-orm` + `graphile-worker`, both compatible with this stack.

Costs: a schema bootstrap step (`workflow-postgres-setup`, alongside the existing
`lib/db/migrate` in `CMD`), and a `graphile-worker` polling loop per instance
(normal for that library — it is designed for multiple workers).

Selected in `agent/agent.ts`:

```ts
export default defineAgent({
  experimental: { workflow: { world: '@workflow/world-postgres' } },
});
```

## The options

### Option A — same container (recommended)

`next start` spawns Eve. No new service, no new networking, no new IAM.

Changes:
1. `withEve(nextConfig)` in `next.config.ts` (already on this branch).
2. `RUN pnpm eve build` in the Dockerfile builder stage; `.output/` is gitignored,
   so the image must generate it.
3. Copy `.output/` into the runtime stage (it lives under `/app/client`).
4. Add the Postgres world (see above) + its bootstrap to `CMD`.

Cost: Eve and Next share one instance's 2 vCPU / 8Gi and scale as one unit.

### Option B — sidecar container, same Cloud Run service

Cloud Run supports multiple containers per service. Eve runs as a sidecar on
`localhost:4274`; Next reaches it via `EVE_NEXT_PRODUCTION_ORIGIN=http://127.0.0.1:4274`.

Buys separate resource limits and independent restarts. Costs a second image, a
second build pipeline, and container startup ordering. **No advantage over A
unless Eve needs its own memory ceiling** — the two still scale together, since
sidecars share the service's instance count.

### Option C — separate Cloud Run service

`eve build && eve start` in its own service; Next points at it with
`EVE_NEXT_PRODUCTION_ORIGIN=https://eve-agent-...run.app`.

Buys independent scaling and makes Eve reachable by non-Next clients. Costs the
most:

- Must route **both** `/eve/` and `/.well-known/workflow/` unrewritten — a proxy
  that forwards only `/eve/` lets sessions start and then stall.
- `placeholderAuth()` in `agent/channels/eve.ts` rejects browser traffic in
  production, so this needs a real auth policy before it can serve anything.
- Postgres world becomes mandatory, not optional.
- New service, IAM, VPC, and domain wiring.

Worth it only if Eve must scale separately from the web app.

### Which combinations are coherent

| | Local disk state | Postgres world |
|---|---|---|
| **A** same container | ⚠️ Only at `min = max = 1`. At the current 2–20 it boots and looks fine, then loses sessions on any instance churn. **The trap.** | ✅ **Recommended** |
| **B** sidecar | ⚠️ Same trap | ✅ Works, more moving parts than A |
| **C** separate service | ❌ Incoherent | ✅ Works, most infrastructure |

## Standing constraints (independent of the option)

- **`server-only` barrier.** `lib/db/queries.ts` starts with `import 'server-only'`,
  which throws under Eve's bundler resolution. Any Eve tool that needs app DB
  access still fails discovery. Unchanged by `withEve()` and by all three options
  — see `docs/eve-serverless-findings.md`.
- **Browser refs on affinity break.** With the Postgres world, the workflow
  survives instance loss — but the `agent-browser` daemon and
  `lib/kernel/eve-browser.ts`'s module-scope `Map` do not. A resumed session on a
  new instance gets `Unknown ref` and creates a duplicate Kernel browser. Today
  affinity makes this rare and the session dies with the instance anyway; with
  durable sessions the window gets *larger*, because the session now survives to
  hit it. Needs the cross-instance session store noted in the previous findings.
- **tmpfs.** Whatever state stays on local disk counts against the 8Gi.

## Recommendation

**Option A with the Postgres world.** It reuses the container, the service, the
scaling config, and the database that already exist; the only genuinely new
infrastructure is a set of tables in a Postgres instance already mounted into the
container. Options B and C add operational surface without solving a problem this
deployment has.

Sequence: Dockerfile `eve build` → Postgres world + bootstrap → deploy to dev →
confirm `/eve/v1/health` through the Cloud Run URL → run a real turn → then
revisit the browser session store.
