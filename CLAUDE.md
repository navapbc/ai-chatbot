# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This is the `ai-chatbot` Next.js application. In the Labs ASP project it is consumed as a Git submodule (tracking the `develop` branch), but it also lives standalone at [navapbc/ai-chatbot](https://github.com/navapbc/ai-chatbot) — keep changes here self-contained to the app.

## Commands

```bash
pnpm dev              # next dev --turbo
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
```

Run a single unit test: `pnpm exec vitest run <path>` (e.g. `pnpm exec vitest run tests/client/some.test.tsx`).

Always use `pnpm` (packageManager is pinned). Ask before installing any new dependency.

## Stack

Next.js 16 (App Router) · React 19 · Vercel AI SDK v6 (`ai` package) · Drizzle ORM + Postgres · next-auth v5 (beta, with guest auth) · Biome (lint/format, 2-space indent, 80-col) · Tailwind · TypeScript path alias `@/*` → repo root.

## Architecture

**Routes** — `app/` uses App Router route groups: `(auth)` (login/register/next-auth handlers), `(chat)` (main chat UI + APIs), `(landing)`. API routes live under `app/(chat)/api/` (`chat`, `document`, `files`, `history`, `suggestions`, `vote`).

**Agent loop** — `app/(chat)/api/chat/route.ts` is the core. It runs `streamText` as a multi-step agent (`stopWhen: [stepCountIs(500), abort]`) with `prepareStep` switching the model mid-run. `maxDuration = 300` (5 min) for long web-automation tasks. Resumable streaming is wired via `resumable-stream` + Redis. In-flight chats can be cancelled through `lib/chat-abort-registry.ts` (the `/api/chat/stop` route). Context is trimmed by `lib/ai/context-compression.ts`.

**Models / providers** — `lib/ai/providers.ts`. Web automation uses `webAutomationModel = vertexAnthropic('claude-opus-4-7')` via Google Vertex AI; `prepareStepModel` uses `claude-haiku-4-5`. A `customProvider` exposes selectable dev-only models (GPT and Claude variants, hidden in production). In test env, models are swapped for mocks from `lib/ai/models.test.ts` — **that file is not a test**; it exports mocks and is excluded from the vitest run.

**Tools** — `lib/ai/tools/`. Tools are factory functions bound to a session/user where needed (e.g. `createBrowserTool(sessionId, userId)`, `createCheckSubmitGateTool`). Wired into the agent in the chat route: `apricotTools`, `browser`, `gap-analysis`, `form-summary`, `check-submit-gate`, `action-label`, `read-reference`, plus document/suggestion tools.

**Browser automation (Kernel.sh)** — `lib/kernel/browser.ts` manages remote browser sessions on Kernel.sh using `agent-browser`'s `BrowserManager` + `executeCommand` (in-process, no CLI subprocess). Sessions live in an **in-memory cache** keyed `${userId}:${sessionId}` — this assumes a single server instance; Kernel.sh owns lifecycle/timeout. The `browser` tool (`lib/ai/tools/browser.ts`) serializes commands per session via a mutex queue because Playwright's `page` object is not concurrency-safe. The agent sends structured JSON commands (`navigate`, `snapshot`, `click`, `fill`, `type`, …); `snapshot` first is the expected discipline. Session IDs are `${chatId}-${userId}`. The `NEXT_PUBLIC_USE_AI_SDK_AGENT` flag toggles this Kernel path vs. legacy Mastra WebSocket streaming.

**Apricot/Apricot360 integration** — `lib/apricot-api.ts` + `lib/ai/tools/apricot/` fetch participant/form data from the Apricot API (OAuth client-credentials). Uses the `sandbox` environment everywhere except production (`api`). Response types are in `lib/models/`.

**Database** — `lib/db/schema.ts` (Drizzle, Postgres). Use the current tables `Message_v2` (`message`) and `vote`, not the deprecated `Message` (`messageDeprecated`) / `voteDeprecated`. Queries are centralized in `lib/db/queries.ts`; migrations in `lib/db/migrations/` (Biome-ignored — don't hand-edit). Config in `drizzle.config.ts`.

**Artifacts** — interactive side-panel artifacts in `artifacts/` with shared logic in `lib/artifacts/`: `browser` (live Kernel session viewer — `client.tsx`, `client-kernel.tsx`, `server.ts`), `code`, `image`, `sheet`, `text`. The `artifacts/session_*` directories are runtime scratch output, not source.

**Prompts** — `lib/ai/prompts/` (`web-automation.ts`, `browser-and-forms.ts`, `application-protocol.ts`) compose the system prompt; markdown references the model can read on demand are in `lib/ai/prompts/references/`.

**Feature flags** — `lib/feature-flags.ts`. Env-aware defaults, overridable per-browser via `localStorage` (`ff:` prefix) through a dev-only menu. Current flag: `declutterToolCalls`.

**Errors** — throw `ChatSDKError` (`lib/errors.ts`) for API/route errors rather than raw `Error`s.

## Conventions

**React/frontend (`.cursor/rules/react.mdc`):**
- Tailwind classes only for styling — no inline styles, no separate CSS. Add a Tailwind variable if a new style property is needed.
- Event handlers use a `handle` prefix (`handleClick`, `handleKeyDown`); prefer typed `const` arrow functions over `function`.
- Use early returns; include accessibility attributes (`aria-label`, `tabIndex`, keyboard handlers).

**Testing:** Default to `vitest --browser` (Playwright + `vitest-browser-react`); only write node/jsdom tests if explicitly asked. Import components under test from source; import test helpers from `vitest-browser-react`. Prefer `getByRole`/`findByRole` (name with regex) over `getByTestId`. Use `findBy*`/`waitFor` for async (no `setTimeout` polling), MSW for HTTP, and local `vi.mock` for module mocks. Test layout: `tests/client/` (component), `tests/e2e/` + `tests/routes/` + `tests/pages/` (Playwright).

**Documentation:** Write for engineers — avoid marketing language ("powerful", "out-of-the-box", "production-ready", "makes it easy", "Check out"). H1 headings use Title Case.
