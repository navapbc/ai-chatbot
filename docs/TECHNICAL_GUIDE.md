# ASP Form-Filling Assistant — Technical Guide

A guide for developers and technical contributors working with the ASP Form-Filling Assistant. This covers architecture, configuration, extending the application, and troubleshooting.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Authentication](#authentication)
- [Browser Automation](#browser-automation)
- [The Eve Agent (Opt-In)](#the-eve-agent-opt-in)
- [AI Tools](#ai-tools)
- [Database & Schema](#database--schema)
- [Shared Links & Redis](#shared-links--redis)
- [Feature Flags](#feature-flags)
- [Evals & Observability](#evals--observability)
- [Deployment](#deployment)
- [Adding a New AI Tool](#adding-a-new-ai-tool)
- [Adding a New Authentication Provider](#adding-a-new-authentication-provider)
- [Rate Limiting](#rate-limiting)
- [Environment Variables Reference](#environment-variables-reference)
- [Troubleshooting](#troubleshooting)

---

## Architecture Overview

The application is a Next.js App Router application that connects users to an AI agent capable of filling out benefit application forms via browser automation.

There are currently **two agent implementations** in this repo — see [The Eve Agent (Opt-In)](#the-eve-agent-opt-in) for why. Production traffic goes through the legacy path by default.

```
User (Browser)
  |
  v
Next.js App (App Router)
  |
  +-- Auth (NextAuth / Auth.js)
  |     Google OAuth, Microsoft Entra ID, Credentials
  |
  +-- AI Agent — legacy path (default, app/(chat)/api/chat/route.ts)
  |     Vercel AI SDK `streamText` loop · Anthropic Claude (via Vertex AI)
  |
  +-- AI Agent — Eve path (opt-in, `useEveAgent` flag; app/(chat)/api/eve-chat/route.ts)
  |     Durable Eve agent runtime (agent/), mounted into the same Next process via withEve()
  |
  +-- Browser Automation (Kernel.sh)
  |     Remote Chromium, Playwright-based agent-browser CLI
  |
  +-- Participant Data (lib/data/participants.ts)
  |     Bundled demo dataset — no live external database by default
  |
  +-- Database (Neon Serverless Postgres / Drizzle ORM)
  |     Users, Chats, Messages, Votes
  |
  +-- Shared Links (Upstash Redis)
        AES-256-GCM encrypted, TTL-based expiration
```

---

## Tech Stack

| Layer | Technology |
| --- | --- |
| Framework | [Next.js](https://nextjs.org) 16 (App Router) |
| AI SDK | [Vercel AI SDK v7](https://sdk.vercel.ai/docs) (`ai` package) |
| Agent runtime | [Eve](https://eve.dev/docs) — opt-in second agent path, see below |
| Schema validation | [Zod v4](https://zod.dev) |
| UI Components | [shadcn/ui](https://ui.shadcn.com) + [Radix UI](https://radix-ui.com) |
| Styling | [Tailwind CSS](https://tailwindcss.com) |
| Authentication | [Auth.js](https://authjs.dev) (NextAuth v5) |
| Database | [Neon](https://neon.tech) Serverless Postgres |
| ORM | [Drizzle ORM](https://orm.drizzle.team) |
| Cache / Links | [Upstash Redis](https://upstash.com) |
| Browser Automation | [Kernel.sh](https://onkernel.com) + [agent-browser](https://www.npmjs.com/package/agent-browser) |
| Evals / tracing | [Braintrust](https://www.braintrust.dev) |
| Package Manager | pnpm |

---

## Project Structure

This repo's own root — not nested under a `client/` folder, even though the Labs ASP monorepo mounts it as a submodule:

```
app/                    # Next.js App Router pages and API routes
  (auth)/               # Login and registration pages
  (chat)/               # Chat interface pages + API routes (chat, eve-chat, document, files, history, ...)
  api/                  # API routes outside the chat group (e.g. kernel-browser, link)
components/             # React components (UI, chat, browser view)
agent/                  # Eve agent definition (opt-in path) — instructions, skills, tools, subagents
lib/
  ai/
    tools/              # AI tool definitions for the legacy route (browser, gap analysis, ...)
    prompts/            # System prompt content for the legacy route
    eve/                # Adapter glue between the Next app and the Eve runtime
  data/                 # lib/data/participants.ts — the bundled demo participant dataset
  db/                   # Database schema and queries (Drizzle)
  kernel/               # Kernel.sh browser session management
  models/               # Legacy external-API client code (currently unused, kept in case it's needed again)
evals/                  # Braintrust eval suite for the web-automation agent
terraform/              # Cloud Run deployment infrastructure
docs/                   # Documentation (you are here)
```

---

## Authentication

Authentication is managed by [Auth.js](https://authjs.dev) with three providers:

### Google OAuth

Requires `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET`. Configure the OAuth consent screen in Google Cloud Console with the callback URL:

```
https://your-domain.com/api/auth/callback/google
```

### Microsoft Entra ID

Requires `AUTH_MICROSOFT_ENTRA_ID_ID`, `AUTH_MICROSOFT_ENTRA_ID_SECRET`, and `AUTH_MICROSOFT_ENTRA_ID_ISSUER`. Register the application in Azure Portal with the callback URL:

```
https://your-domain.com/api/auth/callback/microsoft-entra-id
```

### Credentials

Email/password authentication for users provisioned directly in the database.

### Email Domain Filtering

Set `ALLOWED_EMAIL_DOMAINS` to a comma-separated list of domains to restrict who can sign in:

```env
ALLOWED_EMAIL_DOMAINS=example.com,another.org
```

If unset, all email domains are allowed.

### Guest Login

Set `USE_GUEST_LOGIN=true` to enable a guest sign-in option. This is intended for preview/demo environments only. Guest users receive isolated sessions.

---

## Browser Automation

The browser automation stack is the core of the application and consists of three layers:

### 1. Kernel.sh — Remote Browser Provider

[Kernel.sh](https://onkernel.com) provisions remote Chromium browser instances. The SDK (`@onkernel/sdk`) handles session creation and lifecycle.

**Key files:**
- `lib/kernel/browser.ts` — session management for the legacy route (create, cache, destroy remote browsers; profiles, replay recording, standby)
- `lib/kernel/eve-browser.ts` — a separate, minimal reimplementation for the Eve path. Eve's bundler can't import anything that transitively pulls in `server-only` (which `browser.ts` does via `lib/db/queries.ts`), so this isn't just calling into `browser.ts` — it's known duplication, not yet consolidated.
- `lib/kernel/session-store.ts` — pure, dependency-free session-status helpers factored out so they're unit-testable without `@onkernel/sdk`
- `lib/kernel/telemetry.ts` — bridges Kernel's own browser telemetry (console errors, CDP connect/disconnect, captcha results, OOM kills) into OpenTelemetry spans. The `network` category is deliberately left off — request bodies can contain applicant PII.

Sessions are cached per-chat (keyed `${chatId}-${userId}`) to avoid creating duplicate browsers. Idle handling is a multi-stage policy (`lib/kernel/session-config.ts`): after 12 minutes of inactivity a warning appears, then a 3-minute countdown before the session disconnects to standby (no further cost, state preserved, reconnectable from the UI); a session also has a hard 60-minute lifetime cap regardless of activity. Kernel itself additionally reaps a session after 5 minutes of network inactivity as a cost backstop, independent of the app's own timers.

### 2. agent-browser — a native CLI, not a library

[`agent-browser`](https://www.npmjs.com/package/agent-browser) is a native Rust binary, invoked as a subprocess (`lib/kernel/cli.ts`) and attached to the Kernel browser over CDP. Its daemon holds the CDP connection between calls (keyed by `--session`), so element refs (`@e1`, `@e2`, ...) survive across commands.

### 3. AI Browser Tool

**Key file:** `lib/ai/tools/browser.ts`

The tool's input is **agent-browser's own CLI argv**, not a fixed action enum — e.g. `["snapshot", "-i"]`, `["click", "@e1"]`, `["fill", "@e1", "John"]`. This means new agent-browser commands work automatically on a CLI upgrade, with no tool-schema change needed. `snapshot` first (to get current `@eN` refs) is the expected discipline before acting on the page. Commands for a given session are serialized through a per-session mutex queue, since interleaved parallel tool calls would scramble which refs are still valid.

### Live Streaming

Browser activity streams to the user in real time via Kernel's embedded live view. Because the Eve path runs the browser in a separate process from Next, its live-view URL can't be read out of the legacy in-memory session map — `lib/ai/eve/live-view-store.ts` is a second in-memory store that the `eve-chat` route writes into and `app/api/kernel-browser` reads from, so the live-view UI works the same regardless of which path is driving.

### Configuration

```env
KERNEL_API_KEY=your-kernel-api-key
```

---

## The Eve Agent (Opt-In)

Alongside the legacy `streamText` loop, this repo also defines a second agent using [Eve](https://eve.dev/docs), Vercel's durable agent runtime. It's real, not a demo: `next.config.ts` mounts it into the app via `withEve()`, and in deployment it runs as a sibling process to Next inside the same Cloud Run container, with durable session state backed by Postgres. But it is **not what production chat traffic uses by default** — the `useEveAgent` feature flag that routes a chat through it defaults to `false` (see [Feature Flags](#feature-flags)).

**The practical consequence for contributors:** agent behavior (prompts, tools, compaction) is currently duplicated across two trees. Changing "how the agent behaves" means figuring out which path you're changing, or changing both:

| | Legacy path (default) | Eve path (opt-in) |
| --- | --- | --- |
| Entry point | `app/(chat)/api/chat/route.ts` | `app/(chat)/api/eve-chat/route.ts` (adapter) |
| Prompts/instructions | `lib/ai/prompts/` | `agent/instructions.md` + `agent/skills/` |
| Tools | `lib/ai/tools/` | `agent/tools/` |
| Context management | `lib/ai/context-compression.ts` (custom `prepareStep` hook) | Eve's built-in compaction (`thresholdPercent`, no `prepareStep`-equivalent hook) |
| Model calls | Direct to Vertex AI (`lib/ai/providers.ts`) | Direct to Vertex AI (`agent/agent.ts`) — bypasses the AI SDK Gateway on both paths |

`agent/README.md` has the full structure map and a detailed walkthrough of running Eve locally, the model-selection wiring, and the Vertex `global`-region quota gotcha (see [Environment Variables Reference](#environment-variables-reference) below for the short version).

---

## AI Tools

**Legacy path** (`lib/ai/tools/`) — tools wired into `app/(chat)/api/chat/route.ts`:

| Tool | Purpose |
| --- | --- |
| `browser` | Remote browser automation (see above) |
| `gapAnalysis` | Surfaces a card of form fields that couldn't be filled from available data |
| `formSummary` | Summarizes what was filled, for caseworker review before submission |
| `actionLabel` | Labels/categorizes a browser action for the UI |
| `checkSubmitGate` | Checks whether the form's submit control is actually clickable (handles Turnstile-style gates) |
| `readReference` | Reads a markdown reference file on demand |

`lib/ai/tools/apricot/` and `lib/models/apricot-models.ts` are an earlier client-database integration that isn't wired into any route today (kept in place rather than deleted, in case a similar integration is needed again — see [Adding a New AI Tool](#adding-a-new-ai-tool) for the current pattern instead).

**Eve path** (`agent/tools/`, `agent/subagents/*/tools/`) — separate implementations of most of the same capabilities, plus a `defineState`-based `update_working_memory` prototype tool. Not shared code with the legacy tools above; see [The Eve Agent (Opt-In)](#the-eve-agent-opt-in).

Participant data itself isn't a tool call — `lib/data/participants.ts` is read directly and its data is folded into the caseworker's first chat message (`buildApplicationPrompt`), so the agent has it inline rather than fetching it mid-conversation.

---

## Database & Schema

The database uses Neon Serverless Postgres with Drizzle ORM for schema management and queries.

### Tables

| Table | Purpose | Key Columns |
| --- | --- | --- |
| `User` | User accounts | email, name, image |
| `Chat` | Chat sessions | title, userId, visibility |
| `Message_v2` | Chat messages | chatId, role, parts (multi-part content) |
| `Vote` | Message feedback | messageId, chatId, isUpvoted |

Use `Message_v2` (`message`) and `Vote` (`vote`) — the older `Message`/`Vote` tables (`messageDeprecated`/`voteDeprecated` in the schema) are deprecated; don't write to them.

### Migrations

```bash
# Generate a migration after changing schema files
pnpm db:generate

# Apply pending migrations
pnpm db:migrate
```

Schema definitions live in `lib/db/schema.ts`. Query functions are in `lib/db/queries.ts`. Migration files in `lib/db/migrations/` are generated output — don't hand-edit them.

---

## Shared Links & Redis

Shared links allow users to send pre-populated session content to others.

### How it works

1. Content is serialized to JSON
2. Encrypted with AES-256-GCM using a key derived from `AUTH_SECRET`
3. Stored in Upstash Redis with a TTL (time-to-live)
4. A short 8-character token is generated as the link ID
5. Recipients access `/link/[token]`, which decrypts and loads the content

### Configuration

```env
UPSTASH_REDIS_REST_URL=https://your-instance.upstash.io
UPSTASH_REDIS_REST_TOKEN=your-token
```

**Key files:**
- `app/api/link/route.ts` — Link creation endpoint
- `app/link/[token]/route.ts` — Link resolution route

---

## Feature Flags

The app's own feature-flag system (`lib/feature-flags.ts`) — env-aware defaults, overridable per-browser via `localStorage` (`ff:` prefix) through a dev-only menu:

| Flag | Default | Description |
| --- | --- | --- |
| `declutterToolCalls` | `true` in production, `false` otherwise | Show only value-bearing tool calls in the chat UI |
| `useEveAgent` | `false` | Route a chat through the Eve agent adapter route instead of the legacy loop — see [The Eve Agent (Opt-In)](#the-eve-agent-opt-in) |

These are distinct from plain environment variables that also gate behavior, like `USE_GUEST_LOGIN` and `ENVIRONMENT` — see [Environment Variables Reference](#environment-variables-reference).

---

## Evals & Observability

`evals/*.eval.ts` is a [Braintrust](https://www.braintrust.dev) eval suite that runs the web-automation agent against fixed scenarios with mocked tool surfaces — tool selection, navigation, gap analysis, hallucination, verbosity, and more. `evals/online/` separately applies scoring rules to live production traces. See `evals/README.md` for the full suite list, scoring model, and how to add a new eval — it's kept up to date and is the source of truth here, not this guide.

```bash
pnpm eval                # run every suite locally (reads .env.local)
pnpm eval:ci              # same, for CI (env already in the shell)
pnpm eval:tool-selection  # run one suite
```

The PR-triggered `evals.yml` GitHub Actions workflow gates each matrix leg on its provider API key being present as a secret, and skips (green) rather than fails when a key is missing — a green Evals check does not by itself mean the evals ran; check the job logs before trusting it.

Traces reach Braintrust via OpenTelemetry (`@braintrust/otel` + `@vercel/otel`): the root `instrumentation.ts` covers the Next process, and `agent/instrumentation.ts` separately exports the Eve agent's spans, since Eve runs in its own server process that the root instrumentation never sees. The Braintrust key is only set for `dev`/`preview` in Terraform, never `prod` — production sessions don't reach Braintrust.

---

## Deployment

The app deploys to **Cloud Run**, not Vercel. `terraform/` holds the infrastructure config.

`scripts/start-container.sh` is the container entrypoint: it applies database migrations, conditionally bootstraps the Eve workflow schema in Postgres (only when `WORKFLOW_POSTGRES_URL` is set), then starts the Eve runtime and `next start` as sibling processes inside the same container and supervises both — either one dying takes the whole container down so Cloud Run replaces the instance.

Eve's durable session state needs a real database in deployment because this service runs with `min_instance_count = 2` and best-effort session affinity — routine instance churn would otherwise silently drop in-flight Eve sessions on the default local-file world. `terraform/cloud_run.tf` points `WORKFLOW_POSTGRES_URL` at the same Cloud SQL instance the app already uses.

---

## Adding a New AI Tool

To give the legacy-path agent a new capability:

1. Create a tool file in `lib/ai/tools/`:

```typescript
// lib/ai/tools/my-tool.ts
import { tool } from "ai";
import { z } from "zod";

export const myTool = tool({
  description: "Describe what this tool does",
  inputSchema: z.object({
    input: z.string().describe("What the input represents"),
  }),
  execute: async ({ input }) => {
    // Tool logic here
    return { result: "..." };
  },
});
```

2. Register the tool in the `tools` object passed to `streamText` in `app/(chat)/api/chat/route.ts` (see the other entries there for the pattern — some tools are plain exports, others are factory functions like `createBrowserTool(sessionId, userId)` for session-scoped state).

To add the equivalent capability to the Eve path instead, see `agent/README.md`'s "Tools" section — Eve tools are separate files under `agent/tools/` and register under a snake_case file slug, not the tool's exported name.

---

## Adding a New Authentication Provider

1. Install the provider package if needed
2. Add the provider to the Auth.js configuration in `app/(auth)/auth.ts`
3. Set the required environment variables
4. Add the callback URL to the provider's developer console:
   ```
   https://your-domain.com/api/auth/callback/[provider-id]
   ```
5. Update the login page UI in `app/(auth)/login/page.tsx`

---

## Rate Limiting

Rate limits are enforced per user, per entitlement tier:

- Default: **100 messages per day**
- Tracked in the database
- Rate limit errors return a clear message to the user

To adjust limits, modify the entitlement configuration in `lib/ai/entitlements.ts`.

---

## Environment Variables Reference

### Required

| Variable | Description |
| --- | --- |
| `DATABASE_URL` | PostgreSQL connection string |
| `AUTH_SECRET` | NextAuth session encryption secret |

### Authentication

| Variable | Description |
| --- | --- |
| `GOOGLE_CLIENT_ID` | Google OAuth client ID |
| `GOOGLE_CLIENT_SECRET` | Google OAuth client secret |
| `AUTH_MICROSOFT_ENTRA_ID_ID` | Microsoft Entra ID app ID |
| `AUTH_MICROSOFT_ENTRA_ID_SECRET` | Microsoft Entra ID app secret |
| `AUTH_MICROSOFT_ENTRA_ID_ISSUER` | Microsoft Entra ID issuer URL |
| `ALLOWED_EMAIL_DOMAINS` | Comma-separated allowed email domains |

### AI & Automation

| Variable | Description |
| --- | --- |
| `GOOGLE_VERTEX_PROJECT` | GCP project for Vertex AI |
| `GOOGLE_VERTEX_LOCATION` | GCP region for Vertex AI. **Must be `global`** for the opus models — they have no quota on any regional Vertex endpoint on this project. A 429 on an opus call almost always means this var, not real quota exhaustion. |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to GCP service account JSON |
| `KERNEL_API_KEY` | Kernel.sh API key for browser automation |
| `WORKFLOW_POSTGRES_URL` | Optional. Backs Eve's durable session state with Postgres; unset means Eve uses its local file world (fine for local dev, not for a multi-instance deployment). Must be a direct connection, not a pooled one — Eve's workflow schema needs `LISTEN`/`NOTIFY` and session-scoped advisory locks. |

### Storage & Services

| Variable | Description |
| --- | --- |
| `UPSTASH_REDIS_REST_URL` | Upstash Redis URL for shared links |
| `UPSTASH_REDIS_REST_TOKEN` | Upstash Redis auth token |

### Optional

| Variable | Description |
| --- | --- |
| `OPENAI_API_KEY` | OpenAI API key (if using OpenAI models) |
| `ANTHROPIC_API_KEY` | Anthropic API key (if using direct API, e.g. in evals CI) |
| `BRAINTRUST_API_KEY` | Enables eval runs and OpenTelemetry trace export to Braintrust |
| `ENVIRONMENT` | `dev`, `prod`, or `preview-*` |
| `USE_GUEST_LOGIN` | Enable guest login for previews |

See [`.env.example`](../.env.example) for the full list.

---

## Troubleshooting

### "Rate limit exceeded" error

The user has exceeded their daily message limit (default 100). This resets daily. To adjust, modify entitlements in `lib/ai/entitlements.ts`.

### An opus model call 429s / "RESOURCE_EXHAUSTED"

Check `GOOGLE_VERTEX_LOCATION` first — it needs to be `global`. The opus models have no quota on any regional Vertex endpoint on this project; other models (sonnet, haiku) do work regionally, which is what makes this one easy to miss locally.

### Browser automation not working

1. Verify `KERNEL_API_KEY` is set and valid
2. Check server logs for Kernel session creation errors

### Authentication failures

1. Verify OAuth callback URLs match your deployment domain exactly
2. Check that the user's email domain is in `ALLOWED_EMAIL_DOMAINS` (if set)
3. Confirm client ID and secret are correct for the provider
4. For Microsoft Entra ID, verify the issuer URL matches your tenant

### Database connection errors

1. Verify `DATABASE_URL` is correct and the database is accessible
2. Run `pnpm db:migrate` to apply any pending migrations
3. For Neon, ensure the database hasn't been suspended due to inactivity

### Shared links not working

1. Verify `UPSTASH_REDIS_REST_URL` and `UPSTASH_REDIS_REST_TOKEN` are set
2. Check that `AUTH_SECRET` is consistent across deployments (it's used for encryption)
3. Links expire — the recipient may need a fresh link

### Eve agent turn fails or won't respond (`useEveAgent` flag on)

1. Confirm the page was reloaded after toggling the flag — it's read once at first render, not reactively
2. Locally, confirm `pnpm dev` is running (it boots Eve as a child process via `withEve()`) and check `curl localhost:3000/eve/v1/health`
3. See `agent/README.md` for the full manual verification checklist and known gotchas (model selection, Vertex region, Postgres world setup)
