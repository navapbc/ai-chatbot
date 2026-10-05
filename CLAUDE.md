# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This is the `ai-chatbot` Next.js application. In the Labs ASP project it is consumed as a Git submodule (tracking the `develop` branch), but it also lives standalone at [navapbc/ai-chatbot](https://github.com/navapbc/ai-chatbot) — keep changes here self-contained to the app.

## Commands

```bash
pnpm dev              # next dev --turbo
pnpm eve:dev          # dotenv -e .env.local -- eve dev — local Eve agent server (see Eve agent below)
pnpm build            # runs lib/db/migrate THEN next build
pnpm lint             # biome lint --write --unsafe
pnpm format           # biome format --write
pnpm test             # vitest (browser mode, chromium via Playwright)
pnpm test:playwright  # Playwright e2e tests (sets PLAYWRIGHT=True)

# Database (Drizzle)
pnpm db:generate      # generate migrations from schema
pnpm db:migrate       # apply migrations (npx tsx lib/db/migrate.ts)
pnpm db:studio        # open Drizzle Studio (read-only browsing)
pnpm db:push          # push schema without migration files
pnpm db:check         # check migration consistency

# Braintrust evals (evals/*.eval.ts — see Evals & observability below)
pnpm eval                  # run every evals/*.eval.ts locally (needs .env.local)
pnpm eval:ci                # same, without dotenv (CI already has the env)
pnpm eval:tool-selection    # run a single eval file
pnpm eval:check-harness     # sanity-check the browser eval harness itself
pnpm eval:online:apply      # apply online scoring rules to a live Braintrust project
```

`.github/workflows/evals.yml` (PR-triggered) gates every matrix leg on its provider API key being present as a secret and skips the leg (green, not passing) when it isn't — a green Evals check does not by itself mean the evals ran; check the job logs for `::warning::...skipping` before trusting it.

Run a single unit test: `pnpm exec vitest run <path>` (e.g. `pnpm exec vitest run tests/client/some.test.tsx`).

Node-only tests under `tests/agent/**` (anything touching `node:fs`, `eve/*` tools, etc.) are excluded from the default browser-mode `pnpm test` because importing them crashes the browser runner — run them explicitly with `pnpm exec vitest run --config vitest.config.node.mjs`.

Always use `pnpm` (packageManager is pinned). Ask before installing any new dependency.

## Stack

Next.js 16 (App Router) · React 19 · Vercel AI SDK v7 (`ai` package) · Zod v4 · [Eve](https://eve.dev/docs) (durable agent runtime, mounted via `withEve()`) · Drizzle ORM + Postgres · next-auth v5 (beta, with guest auth) · Biome (lint/format, 2-space indent, 80-col) · Tailwind · TypeScript path alias `@/*` → repo root.

## Architecture

**Two agent implementations, both live in this repo.** Production chat traffic is served by the legacy `streamText` loop in `app/(chat)/api/chat/route.ts` — the `useEveAgent` feature flag that would route a chat through the Eve agent instead defaults to `false` (`lib/feature-flags.ts`). But the Eve *runtime* itself is not a side experiment: `next.config.ts` wraps the app in `withEve()`, and in deployment (`scripts/start-container.sh`, `terraform/cloud_run.tf`) Eve runs as a sibling process inside the same Cloud Run container with its durable session state backed by Postgres. The practical consequence: agent behavior (prompts, tools, compaction) is currently **duplicated across two trees** — `lib/ai/prompts/` + `lib/ai/tools/` for the legacy route, `agent/instructions.md` + `agent/skills/` + `agent/tools/` for Eve — and a change to "how the agent behaves" needs to either target the right one or update both. Don't trust `agent/README.md`'s framing of itself as a "demonstrative conversion" that doesn't replace production — that predates the Cloud Run/Postgres wiring in `agent/agent.ts` and is stale.

**Routes** — `app/` uses App Router route groups: `(auth)` (login/register/next-auth handlers), `(chat)` (main chat UI + APIs), `(landing)`. API routes live under `app/(chat)/api/` (`chat` — legacy agent loop, `eve-chat` — Eve adapter route, `document`, `files`, `history`, `suggestions`, `vote`). `app/api/kernel-browser` sits outside the `(chat)` group and serves the live browser-view URL to the UI regardless of which transport produced it (see Browser automation below).

**Agent loop (legacy, default path)** — `app/(chat)/api/chat/route.ts` runs `streamText` as a multi-step agent (`stopWhen: [stepCountIs(500), abort]`) with `prepareStep` switching the model mid-run. `maxDuration = 300` (5 min) for long web-automation tasks. Resumable streaming is wired via `resumable-stream` + Redis. In-flight chats can be cancelled through `lib/chat-abort-registry.ts` (the `/api/chat/stop` route). Context is trimmed by `lib/ai/context-compression.ts`.

**Eve agent (opt-in path)** — `agent/` defines a second, parallel agent via Eve: `agent.ts` (top-level `defineAgent`), `instructions.md` + `instructions/date.ts` (always-on instructions), `skills/browser-automation` and `skills/benefits-application` (situational reference material, loaded on demand — the Eve equivalent of the legacy `readReference` tool), `tools/` (real implementations, not stubs: `browser`, `check_submit_gate`, `action_label`, `form_summary`, `gap_analysis`, plus a `defineState`-based `update_working_memory` prototype), and `subagents/requirements_research` + `subagents/form_review` (each carries its own copies of instructions/tools since Eve subagents inherit nothing from the parent). `app/(chat)/api/eve-chat/route.ts` is the adapter: it creates/continues an Eve session, translates Eve's NDJSON event stream into AI SDK SSE (`lib/ai/eve/stream-adapter.ts`, `eve-client.ts`), and tracks per-chat session continuity in an in-memory `Map` (`lib/ai/eve/session-continuity.ts` — not yet Postgres-backed, so it doesn't survive an instance restart; a lost entry starts a fresh Eve session rather than corrupting one). Models are called directly against Vertex AI rather than through the AI SDK Gateway, defaulting to `claude-opus-5` — a different default than the legacy route's `claude-opus-4-7` (`agent/agent.ts`); compaction uses Eve's built-in `thresholdPercent` (no `prepareStep`-equivalent hook exists, so this does not use `lib/ai/context-compression.ts`). Eve's durable workflow state uses `@workflow/world-postgres` when `WORKFLOW_POSTGRES_URL` is set (deployment), falling back to Eve's local file world otherwise (local dev) — see the extensive comments in `agent/agent.ts` before touching this, the package is pinned to a specific version for World-spec-compatibility reasons.

**Models / providers** — `lib/ai/providers.ts`. Web automation uses `webAutomationModel = vertexAnthropic('claude-opus-4-7')` via Google Vertex AI; `prepareStepModel` uses `claude-haiku-4-5`. Both the legacy route and the Eve agent require `GOOGLE_VERTEX_LOCATION=global` — the opus models have no quota on any regional Vertex endpoint on this project, only `global`; a 429 there almost always means this env var, not a real quota exhaustion. A `customProvider` exposes selectable dev-only models (GPT and Claude variants, hidden in production; the dev model picker can also drive the Eve agent's model per-session via `lib/ai/eve/model-map.ts`). In test env, models are swapped for mocks from `lib/ai/models.test.ts` — **that file is not a test**; it exports mocks and is excluded from the vitest run.

**Tools** — `lib/ai/tools/` (legacy route). Tools are factory functions bound to a session/user where needed (e.g. `createBrowserTool(sessionId, userId)`, `createCheckSubmitGateTool`) or to the participant record (`createGapAnalysisTool`, `createFormSummaryTool`, `createResolveFieldValueTool` — see Jev below). Wired into the agent in the chat route: `browser`, `gap-analysis`, `form-summary`, `resolve-field-value`, `check-submit-gate`, `action-label`, `read-reference`, plus document/suggestion tools. The Eve agent's equivalents live under `agent/tools/` (see above) and are separate implementations, not shared code.

**Browser automation (Kernel.sh)** — `lib/kernel/browser.ts` creates and owns remote browser sessions on Kernel.sh via `@onkernel/sdk` (profiles, replay recording, standby) for the legacy route; `lib/kernel/eve-browser.ts` is a separate, minimal reimplementation for the Eve agent (Eve's bundler cannot import anything that transitively pulls in `server-only`, which `browser.ts` does via `lib/db/queries.ts` — known duplication debt, not yet consolidated). `agent-browser` is a **native Rust CLI**, not a library: `lib/kernel/cli.ts` runs it as a subprocess and attaches it to the Kernel browser with `--cdp <cdp_ws_url>`, keeping session lifecycle on our side. Its daemon is keyed by `--session` (see `cliSessionName`) and holds the CDP connection between calls, so `@eN` refs survive across commands. Sessions live in an **in-memory cache** keyed `${userId}:${sessionId}` — this assumes a single server instance; Kernel.sh owns lifecycle/timeout. `lib/kernel/session-store.ts` holds pure, dependency-free helpers factored out so session-status logic can be unit-tested without pulling in `@onkernel/sdk`. `lib/kernel/telemetry.ts` bridges Kernel's own browser telemetry (console errors, CDP connect/disconnect, captcha results, OOM kills) into OpenTelemetry spans; the `network` category is deliberately left off because request data can contain applicant PII. The `browser` tool (`lib/ai/tools/browser.ts`) serializes commands per session via a mutex queue so a snapshot's refs still describe the page when the next command runs. The agent emits the CLI's own argv (`["snapshot", "-i"]`, `["click", "@e1"]`) rather than a translated action vocabulary, so new agent-browser commands work on upgrade; `snapshot` first is the expected discipline. Session IDs are `${chatId}-${userId}`. Because the Eve agent runs in a separate process from Next, its browser's live-view URL can't be read back out of `lib/kernel/browser.ts`'s in-memory map — `lib/kernel/live-view-store.ts` is a second in-memory store the `eve-chat` route writes into and `app/api/kernel-browser` reads from, so the live browser-view UI works the same regardless of which transport is driving.

**Participant data** — there is no participant database (an earlier Apricot client-database integration was fully removed). `lib/data/participants.ts` holds the participant records and `buildApplicationPrompt` turns one into a chat message, so the agent receives the participant JSON inline in the caseworker's first message. Both the suggested-action buttons (`components/suggested-actions.tsx`) and the landing page Client ID lookup (`components/benefit-applications-landing.tsx`) go through that helper.

**Database** — `lib/db/schema.ts` (Drizzle, Postgres). Use the current tables `Message_v2` (`message`) and `vote`, not the deprecated `Message` (`messageDeprecated`) / `voteDeprecated`. Queries are centralized in `lib/db/queries.ts`; migrations in `lib/db/migrations/` (Biome-ignored — don't hand-edit). Config in `drizzle.config.ts`.

**Artifacts** — interactive side-panel artifacts in `artifacts/` with shared logic in `lib/artifacts/`: `browser` (live Kernel session viewer — `client.tsx`, `client-kernel.tsx`, `server.ts`), `code`, `image`, `sheet`, `text`. The `artifacts/session_*` directories are runtime scratch output, not source.

**Prompts** — legacy route: `lib/ai/prompts/` (`web-automation.ts`, `browser-and-forms.ts`, `application-protocol.ts`) compose the system prompt; markdown references the model can read on demand are in `lib/ai/prompts/references/`. Eve agent: `agent/instructions.md` (always-on) plus `agent/skills/*` (loaded on demand) carry the equivalent content — see `agent/README.md`'s section-by-section mapping if reconciling the two, but not its overall "demonstrative" framing (stale, see above).

**Jev (TypeSafe System One)** — `lib/jev/` wraps the `@typesafe-ai/sdk` client for small, fast, typed judgments (choice / noul / score with calibrated probabilities) where code needs semantic understanding. The design rule: **questions are written in code** (`lib/jev/questions.ts`) and called from code, never composed by the model at runtime — fixed questions can be versioned and regression-scored. `lib/jev/client.ts` is deliberately free of `server-only` so the Eve runtime could import it. `askJev` never throws: any failure (no key, rate limit, timeout, abort) returns `{ ok: false }` and the caller falls back to its pre-Jev behavior, so nothing blocks a form run on TypeSafe being reachable.

Three call sites, all **annotate-only** — they add a `jev` field to rows the agent already produced and never drop or rewrite one (dropping a row from a benefits application needs a threshold chosen from observed data, which does not exist yet): `gapAnalysis` rows get "does the record already answer this?", `formSummary` rows get "is this value grounded, and is the agent's own `source` label accurate?", and `resolveFieldValue` picks a verbatim value from record-derived candidates. Gated by `JEV_FEATURES` (comma-separated, or `all`; empty disables everything) — a server-side env var, **not** `lib/feature-flags.ts`, because that resolves overrides from `localStorage` and yields only `defaultValue` on the server. The participant record reaches these tools via `lib/jev/participant.ts`, which parses it back out of the caseworker's opening message (there is no other carrier). An earlier generic "ask Jev anything" tool was removed: letting the model author its own questions defeats the point, since such questions cannot be versioned or regression-scored.

**Both agent trees are wired**, sharing everything under `lib/jev/`. Eve needs one extra hop because its tools are module-level `defineTool` default exports with no construction step, and `ToolContext` exposes no message history — so the record cannot be passed in or looked up. Instead `agent/hooks/participant.ts` lifts it out of `message.received` into a `defineState` slot (`agent/lib/participant-state.ts`), and `agent/lib/jev.ts` reads it back. A subagent inherits no state, so `agent/subagents/form_review/tools/form_summary.ts` degrades to un-annotated rows rather than erroring. Eve counterparts: `agent/tools/gap_analysis.ts`, `agent/tools/form_summary.ts`, `agent/tools/resolve_field_value.ts`. `JEV_FEATURES` and `TYPESAFE_API_KEY` reach Eve without extra wiring: `scripts/start-container.sh` starts it as a plain child process, which inherits the container env.

`lib/jev/telemetry.ts` traces every call: one `jev <feature> batch` span per card tool (carrying
`rows_attempted` vs `rows_annotated`, and `enabled: false` when the flag is off) with one
`jev <feature>` child per request. Both exporters run `filterAISpans: true`, so the tracer scope
`labs-asp.jev` has to stay in the `KEEP_SCOPES` set in **both** `instrumentation.ts` and
`agent/instrumentation.ts` or the spans are silently dropped in that process. The same module
prints a JSON line per batch to stdout, which is the only signal where no exporter is configured.

Braintrust also supports Jev as an online-eval judge — see `evals/README.md`.

**Feature flags** — `lib/feature-flags.ts`. Env-aware defaults, overridable per-browser via `localStorage` (`ff:` prefix) through a dev-only menu. Current flags: `declutterToolCalls` (show only value-bearing tool calls in prod) and `useEveAgent` (route chat through the Eve adapter route instead of the legacy loop; defaults `false`).

**Deployment** — Cloud Run, not Vercel (an earlier Vercel migration was explored and abandoned once a working Cloud Run deployment already existed). `scripts/start-container.sh` is the container entrypoint: it applies DB migrations, conditionally bootstraps the Eve workflow Postgres schema (`scripts/bootstrap-workflow-db.ts`, only when `WORKFLOW_POSTGRES_URL` is set), then starts the Eve runtime (`.output/server/index.mjs`, built by `eve build`) and `next start` as sibling processes, supervising both — either one dying takes the container down so Cloud Run replaces the instance. `terraform/cloud_run.tf` sets `WORKFLOW_POSTGRES_URL` against the same Cloud SQL instance the app already uses, because Eve's default local-file workflow world doesn't survive this service's routine instance churn (`min_instance_count = 2`, best-effort session affinity).

**Evals & observability** — `evals/*.eval.ts` is a Braintrust eval suite (run via `pnpm eval*`, see Commands) covering tool selection, navigation, gap-analysis, hallucination, verbosity, and other scenario/regression checks against the web-automation agent; `evals/online/` applies scoring rules to live production traces rather than a fixed dataset. Traces reach Braintrust via OpenTelemetry: the root `instrumentation.ts` covers the Next process, and `agent/instrumentation.ts` separately exports Eve's agent spans — required because the Eve agent runs in its own server process, which the root instrumentation file never sees. Both use `@braintrust/otel` + `@vercel/otel`. `lib/observability/browser-telemetry.ts` is the shared span-collector interface `lib/kernel/telemetry.ts` implements.

**Errors** — throw `ChatSDKError` (`lib/errors.ts`) for API/route errors rather than raw `Error`s.

## Conventions

**React/frontend (`.cursor/rules/react.mdc`):**
- Tailwind classes only for styling — no inline styles, no separate CSS. Add a Tailwind variable if a new style property is needed.
- Event handlers use a `handle` prefix (`handleClick`, `handleKeyDown`); prefer typed `const` arrow functions over `function`.
- Use early returns; include accessibility attributes (`aria-label`, `tabIndex`, keyboard handlers).

**Testing:** Default to `vitest --browser` (Playwright + `vitest-browser-react`); only write node/jsdom tests if explicitly asked. Import components under test from source; import test helpers from `vitest-browser-react`. Prefer `getByRole`/`findByRole` (name with regex) over `getByTestId`. Use `findBy*`/`waitFor` for async (no `setTimeout` polling), MSW for HTTP, and local `vi.mock` for module mocks. Test layout: `tests/client/` (component tests, browser-mode by default — but a file named `*.node.test.ts` inside `tests/client/` runs in `pnpm test`'s separate `node` project instead, for anything that can't resolve in a browser bundle, e.g. `browser-telemetry.node.test.ts`, `kernel-telemetry.node.test.ts`); `tests/agent/` (a fully separate node-mode suite for Eve/tool-adjacent tests, NOT part of `pnpm test` at all — run via `vitest.config.node.mjs`, see Commands); `tests/e2e/` + `tests/routes/` + `tests/pages/` (Playwright).

**Documentation:** Write for engineers — avoid marketing language ("powerful", "out-of-the-box", "production-ready", "makes it easy", "Check out"). H1 headings use Title Case.
