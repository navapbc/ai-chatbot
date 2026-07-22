# Eve + Vercel Spike — Findings

## Eve version
`eve 0.27.0` (from `pnpm ls eve`; resolved via `"eve": "^0.27.0"` in `package.json`).

## Q1 — Can Eve mount into this app on Vercel?
### Local
- How Eve routes were mounted into the existing Next.js 16 app:
  - Eve does **not** mount into the Next.js dev server. `npx eve@latest init .`
    did not touch the `dev` script (`package.json` still has
    `"dev": "next dev --turbo"`) and did not add an `app/eve/...` route handler.
    Instead it scaffolded a standalone `agent/` app (`agent/agent.ts`,
    `agent/instructions.md`, `agent/channels/eve.ts`) that is served by Eve's
    own CLI dev server, **`npx eve dev`**, which listens on its own port
    (default **2000**, overridable with `--port`/`$PORT`). `/eve/v1/session`
    and `/eve/v1/session/:id/stream` are routes on *that* server, not on the
    Next.js app's route table. This means integrating Eve's HTTP surface into
    the existing Next app (so the chat UI can call it same-origin) is an open
    question for later tasks (proxying from a Next route handler, or calling
    the Eve server directly cross-origin) — not something `eve init` set up
    for us.
  - `eve init` did add `eve` (`^0.27.0`) and `@vercel/connect` (`0.4.0`) to
    `dependencies` and pinned `"engines": { "node": "24.x" }` in
    `package.json`, as documented in the brief.
- Any friction / manual wiring required:
  - **`eve init` created a `pnpm-workspace.yaml`** (untracked, brand new file)
    containing `minimumReleaseAgeExclude`, `allowBuilds`, and a
    `packageExtensions` compatibility shim for `eve@>=0.6.0-beta.13 <=0.7.0`.
    This file **omitted the required `packages:` field**, which made *every*
    pnpm invocation (even `pnpm --version`) in the repo root fail with
    `ERROR packages field missing or empty` under the project's pinned pnpm
    (10.13.1 via Homebrew / corepack). Fixed by adding `packages: ["."]` to
    the top of `pnpm-workspace.yaml` — a one-line manual fix, but it is a
    real (repo-breaking) papercut in the current `eve init` for a
    single-package repo and should be flagged to whoever owns the Eve CLI,
    or fixed in a follow-up before merging this spike further.
  - After that fix, `pnpm install` succeeded but reported peer-dependency
    warnings: `eve@0.27.0` wants `ai@^7.0.26` (repo has `ai@7.0.19` pinned
    transitively elsewhere) and `nitro`'s `unstorage` wants newer
    `@vercel/blob`, `@upstash/redis`, `@vercel/functions` than the repo
    currently has. These are warnings, not install failures, and did not
    block the smoke test — left as-is per "additive only," but worth a look
    before Task 2 ports a real tool.
  - No `package-lock.json` was created; only `pnpm-lock.yaml` changed.
  - Eve's own `eve dev` server prints Node 24 in its startup banner but does
    not read `.env.local` automatically — `AI_GATEWAY_API_KEY` had to be
    exported into the process environment by hand (read out of `.env.local`)
    before starting `npx eve dev`, or the live model turn would not
    authenticate.
  - Port 2000 (Eve's default) was already occupied locally by an unrelated
    `eve dev` process running out of a sibling checkout
    (`form-filling-eve/`), so the smoke test below used `--port 2001`
    instead; this has no bearing on Vercel deploy behavior, just a local
    collision to be aware of when running multiple Eve checkouts side by
    side.
- Result of `POST /eve/v1/session` locally:
  - Started the server: `npx eve dev --no-ui --port 2001` (with
    `AI_GATEWAY_API_KEY` exported from `.env.local`). Log output:
    ```
    ☰eve  v0.27.0
    [DEV] server listening at http://127.0.0.1:2001/
    ```
  - Request:
    ```
    curl -i -sS -X POST "http://127.0.0.1:2001/eve/v1/session" \
      -H 'content-type: application/json' \
      -d '{"message":"hello"}'
    ```
  - Response — `HTTP/1.1 202 Accepted` (not `200`; the brief's "expected 200"
    should be corrected to 202 for this Eve version), with the expected
    `x-eve-session-id` header and a JSON body containing `continuationToken`:
    ```
    HTTP/1.1 202 Accepted
    cache-control: no-store
    content-type: application/json
    x-eve-session-id: wrun_01KY5KZNP7AFJ87T4TQRBB513B

    {"continuationToken":"eve:38764027-b7e5-4f87-9ae3-1afc5c308ee4","ok":true,"sessionId":"wrun_01KY5KZNP7AFJ87T4TQRBB513B"}
    ```
  - Followed up with `GET /eve/v1/session/:id/stream` to confirm the live
    model turn actually runs end-to-end through the AI Gateway (not just that
    a session object was created). It streamed `session.started` →
    `turn.started` → `message.received` → `step.started` →
    `message.appended` (x2) → `message.completed` → `step.completed` →
    `turn.completed` → `session.waiting`, with the model
    (`anthropic/claude-sonnet-4.6`) replying "Hello! How can I help you
    today?" and real token/cost usage in `step.completed`
    (`inputTokens: 3558, outputTokens: 12, costUsd: 0.000189`), confirming
    `AI_GATEWAY_API_KEY` auth worked end-to-end.
### Vercel preview
- (filled in Task 5)

## Q2 — Context management under Eve
- (filled in Task 4)

## Q3 — Eve → UI streaming shape
- (filled in Task 3)

## Browser session sketch
- (filled in Task 4)
