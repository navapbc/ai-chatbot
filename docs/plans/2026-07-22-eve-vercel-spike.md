# Eve + Vercel De-Risking Spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove Eve can mount into this Next.js 16 app on Vercel and answer three migration-gating questions (Eve-on-Vercel viability, context-management strategy, Eve→UI streaming shape) with running code and documented findings — changing nothing in production.

**Architecture:** Add an `agent/` directory (Eve's filesystem-first layout) alongside the existing app. Scaffold a minimal Eve agent with one self-contained proof tool (`read_reference`), exercise it via `/eve/v1/session`, capture Eve's runtime behavior, and write findings. The existing `app/(chat)/api/chat/route.ts` loop, Vertex providers, Redis, and Cloud Run deployment are untouched.

**Tech Stack:** Eve (beta) · Vercel AI Gateway (`@ai-sdk/gateway`, already a dependency) · Next.js 16 App Router · TypeScript · zod 3 · vitest · pnpm.

## Global Constraints

- Use `pnpm` only (packageManager is pinned). Ask before adding any dependency other than `eve` itself (approved as the point of this spike).
- **Additive only.** Do NOT modify `app/(chat)/api/chat/route.ts`, `lib/ai/providers.ts`, `lib/ai/context-compression.ts`, `lib/kernel/browser.ts`, or any existing tool. Cloud Run / Vertex / the live `/api/chat` route stay live and unchanged.
- **Pin the Eve beta version** installed (record the exact version in the findings doc). Eve is beta and "subject to change" — treat every Eve API used here as "per Eve docs as of the pinned version; verify against what is installed."
- Spike models resolve through AI Gateway using the string `anthropic/claude-sonnet-4.6` (sufficient for a tool-calling proof; cheaper than opus). All gateway slugs are **dot-versioned** (`claude-sonnet-4.6`, `claude-haiku-4.5`), NOT the dashed Vertex IDs.
- Tool filenames must be **snake_case ASCII** — the filename becomes the tool name the model sees (`read_reference.ts` → `read_reference`).
- Per Eve docs: **tools run in your app runtime with full `process.env`, not inside the sandbox.** (This is why a Kernel.sh call from a tool is viable, and why "working memory as a sandbox file" must be verified against Eve's actual sandbox API rather than assumed.)
- Biome formatting: 2-space indent, 80-col. TypeScript path alias `@/*` → repo root.
- The repo's default vitest runs in **browser mode (Playwright/chromium)** — Node-only tests (`node:fs`, `process.cwd()`) cannot run there and need a dedicated Node config (added in Task 2). Do not change the default browser config; `pnpm test` must stay browser-mode.
- Never commit secrets. New env keys go to `.env.example` (empty values) and `.env.local` (real values, gitignored).
- The proof tool reads the existing reference markdown at `lib/ai/prompts/references/` (`browser-commands.md`, `custom-dropdowns.md`, `field-patterns.md`) — do not duplicate those files.

---

### Task 1: Install Eve and scaffold a minimal agent that mounts locally

**Files:**
- Modify: `package.json` (add `eve` dependency)
- Create: `agent/agent.ts`
- Create: `agent/instructions.md`
- Create: `.env.example` entry `AI_GATEWAY_API_KEY=` (append; keep existing keys)
- Create: `docs/eve-spike-findings.md` (skeleton — sections filled by later tasks)

**Interfaces:**
- Consumes: nothing (first task).
- Produces: a booting Eve runtime that serves `POST /eve/v1/session` and `GET /eve/v1/session/:id/stream`; a findings doc at `docs/eve-spike-findings.md` with headed sections `## Q1`, `## Q2`, `## Q3`, `## Browser session sketch`, and `## Eve version`.

- [ ] **Step 1: Initialize Eve into the existing app**

Run Eve's initializer from the repo root (it detects the existing `package.json`, adds the `eve`, `ai`, and `zod` deps, pins `engines.node` to 24+, and scaffolds the `agent/` files):
```bash
npx eve@latest init .
```
This may create/modify `agent/agent.ts`, `agent/instructions.md`, an example tool, and possibly the `dev` script in `package.json` and/or a config file. **Review the diff before continuing** (`git status`, `git diff package.json`) so you know exactly what Eve wired in — Steps 2–3 then overwrite the scaffolded `agent.ts`/`instructions.md` with the spike's versions. If `init` refuses because `agent/` will conflict, install manually with `pnpm add eve ai zod` and create the files by hand.

Then record the resolved version:
```bash
pnpm ls eve
```
Expected: `eve` in `dependencies`, and a concrete version printed (e.g. `eve 0.x.y`). Note it — it goes in the findings doc in Step 6. Convert any `npm`/`package-lock` artifacts Eve's initializer created back to pnpm (delete a stray `package-lock.json`; ensure `pnpm-lock.yaml` is the lockfile that changed).

- [ ] **Step 2: Create the agent config**

Create `agent/agent.ts`:
```ts
import { defineAgent } from 'eve';

// Model resolves through Vercel AI Gateway. Locally this uses
// AI_GATEWAY_API_KEY; on Vercel it uses OIDC (see Task 5).
export default defineAgent({
  model: 'anthropic/claude-sonnet-4.6',
});
```

- [ ] **Step 3: Create the always-on instructions**

Create `agent/instructions.md`:
```md
You are a spike agent used to verify the Eve runtime.

When the user asks you to load or read a reference document, call the
`read_reference` tool with the filename they mention (for example
`field-patterns.md`) and then summarize what it contains in one sentence.

Keep every response short.
```

- [ ] **Step 4: Add the AI Gateway env key**

Append to `.env.example`:
```
AI_GATEWAY_API_KEY=
```
Add the real value to `.env.local` (gitignored). Obtain the key from the Vercel dashboard (AI Gateway) or `vercel ai-gateway` — the implementer supplies it; do not invent one.

- [ ] **Step 5: Boot locally and confirm Eve mounts**

First determine how Eve runs in this app. Inspect the `dev` script after `init`:
```bash
grep -E '"(dev|eve)":' package.json
```
Eve may (a) run its own dev server via `npx eve dev` (default port **2000**), or (b) hook `/eve/*` routes into the existing `next dev --turbo` server (port **3000**). Use whichever `init` configured. Start it:
```bash
# whichever applies:
pnpm dev        # if init wired eve into the Next dev server
# or
npx eve dev     # if Eve runs its own server (port 2000)
```
In a second shell, create a session (use the port your dev server printed — 2000 or 3000):
```bash
PORT=2000  # or 3000
curl -i -X POST "http://127.0.0.1:$PORT/eve/v1/session" \
  -H 'content-type: application/json' \
  -d '{"message":"hello"}'
```
Expected (per Eve docs; confirm against the installed version): HTTP `200`, a response header `x-eve-session-id: <id>`, and a JSON body containing a `continuationToken`. Record in the findings doc (Q1-local) exactly how Eve mounted (own server vs. into Next, and which port), since this directly informs the Task 5 Vercel deploy and the sub-project 5 UI wiring.

- [ ] **Step 6: Create the findings doc skeleton and record Q1-local + version**

Create `docs/eve-spike-findings.md`:
```md
# Eve + Vercel Spike — Findings

## Eve version
<exact version from `pnpm ls eve`>

## Q1 — Can Eve mount into this app on Vercel?
### Local
- How Eve routes were mounted into the existing Next.js 16 app:
- Any friction / manual wiring required:
- Result of `POST /eve/v1/session` locally:
### Vercel preview
- (filled in Task 5)

## Q2 — Context management under Eve
- (filled in Task 4)

## Q3 — Eve → UI streaming shape
- (filled in Task 3)

## Browser session sketch
- (filled in Task 4)
```
Fill the `## Eve version` and `## Q1 — Local` sections now with what Step 1 and Step 5 produced.

- [ ] **Step 7: Commit**

```bash
git add package.json pnpm-lock.yaml agent/agent.ts agent/instructions.md .env.example docs/eve-spike-findings.md
git commit -m "feat(eve-spike): scaffold minimal Eve agent mounting locally

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Port `readReference` as an Eve tool (TDD on the core function)

**Files:**
- Create: `agent/tools/read_reference.ts`
- Create: `vitest.config.node.mjs` (Node-environment config; the default config is browser-only)
- Test: `tests/agent/read-reference.test.ts`

**Interfaces:**
- Consumes: the booting Eve runtime from Task 1; the existing reference dir `lib/ai/prompts/references/`.
- Produces: `readReferenceFile(path: string): Promise<{ content: string } | { error: string }>` (named export) and a default `defineTool` export named `read_reference` at the runtime (filename-derived).

- [ ] **Step 1: Add a Node-environment vitest config**

The default `vitest.config.mjs` runs in browser mode (Playwright/chromium), where `node:fs` and `process.cwd()` do not work. Create `vitest.config.node.mjs` for Node-only tests, mirroring the `@` alias:
```js
import { defineConfig } from 'vitest/config';
import path from 'node:path';

export default defineConfig({
  resolve: {
    alias: {
      '@': path.resolve(process.cwd(), './'),
    },
  },
  test: {
    environment: 'node',
    globals: true,
    include: ['tests/agent/**/*.test.ts'],
  },
});
```
This is additive — `pnpm test` still uses the browser config; the agent tests run explicitly with `-c vitest.config.node.mjs`.

- [ ] **Step 2: Write the failing test**

Create `tests/agent/read-reference.test.ts`:
```ts
import { describe, it, expect } from 'vitest';
import { readReferenceFile } from '@/agent/tools/read_reference';

describe('readReferenceFile', () => {
  it('reads an existing reference file', async () => {
    const result = await readReferenceFile('field-patterns.md');
    expect(result).toHaveProperty('content');
    if ('content' in result) {
      expect(result.content.length).toBeGreaterThan(0);
    }
  });

  it('strips a leading references/ prefix', async () => {
    const result = await readReferenceFile('references/field-patterns.md');
    expect(result).toHaveProperty('content');
  });

  it('denies path traversal outside the references dir', async () => {
    const result = await readReferenceFile('../../package.json');
    expect(result).toEqual({ error: 'Access denied: path must be within references' });
  });

  it('returns a not-found error for a missing file', async () => {
    const result = await readReferenceFile('does-not-exist.md');
    expect(result).toEqual({ error: 'File not found: does-not-exist.md' });
  });
});
```

- [ ] **Step 3: Run the test to verify it fails**

Run:
```bash
pnpm exec vitest run -c vitest.config.node.mjs tests/agent/read-reference.test.ts
```
Expected: FAIL — cannot resolve `@/agent/tools/read_reference` (module does not exist yet).

- [ ] **Step 4: Write the tool (core function + defineTool wrapper)**

Create `agent/tools/read_reference.ts`:
```ts
import { defineTool } from 'eve/tools';
import { z } from 'zod';
import { readFile } from 'node:fs/promises';
import { resolve, normalize, join } from 'node:path';

const REFERENCES_DIR = normalize(
  join(process.cwd(), 'lib/ai/prompts/references'),
);

// Pure, unit-testable core. The defineTool wrapper below delegates to this
// so the file-reading logic is verifiable without the Eve runtime.
export async function readReferenceFile(
  filePath: string,
): Promise<{ content: string } | { error: string }> {
  const cleaned = filePath.replace(/^references\//, '');
  const resolved = resolve(REFERENCES_DIR, cleaned);
  if (
    !resolved.startsWith(`${REFERENCES_DIR}/`) &&
    resolved !== REFERENCES_DIR
  ) {
    return { error: 'Access denied: path must be within references' };
  }
  try {
    const content = await readFile(resolved, 'utf-8');
    return { content };
  } catch {
    return { error: `File not found: ${filePath}` };
  }
}

export default defineTool({
  description:
    'Load a reference document. Use the path the instructions tell you to load (e.g. "field-patterns.md", "custom-dropdowns.md", "browser-commands.md").',
  inputSchema: z.object({
    path: z
      .string()
      .describe(
        'Filename within lib/ai/prompts/references (e.g. "field-patterns.md")',
      ),
  }),
  // Eve passes the validated input object; destructure it directly.
  async execute({ path }: { path: string }) {
    return readReferenceFile(path);
  },
});
```
Note: `defineTool`'s call shape (`description`, `inputSchema`, `async execute(input)`) is confirmed from eve.dev docs for the current beta. If the installed package differs, adjust the wrapper only — leave `readReferenceFile` (and therefore the passing unit tests) unchanged.

- [ ] **Step 5: Run the test to verify it passes**

Run:
```bash
pnpm exec vitest run -c vitest.config.node.mjs tests/agent/read-reference.test.ts
```
Expected: PASS (4 passing).

- [ ] **Step 6: Prove the tool executes inside an Eve turn**

With the dev server running (port from Task 1 Step 5 — 2000 or 3000), create a session that forces the tool:
```bash
PORT=2000  # or 3000, matching Task 1
curl -i -X POST "http://127.0.0.1:$PORT/eve/v1/session" \
  -H 'content-type: application/json' \
  -d '{"message":"Read the reference field-patterns.md and tell me what it covers."}'
```
Grab the `x-eve-session-id` from the response header, then attach to the stream:
```bash
curl -N "http://127.0.0.1:$PORT/eve/v1/session/<sessionId>/stream"
```
Expected: the NDJSON stream shows a tool call to `read_reference` and a final assistant message that references the file's content. If the model does not call the tool, tighten `agent/instructions.md` wording and retry. Record in the findings doc (Q1-local) that a tool executed end-to-end.

- [ ] **Step 7: Commit**

```bash
git add agent/tools/read_reference.ts vitest.config.node.mjs tests/agent/read-reference.test.ts docs/eve-spike-findings.md
git commit -m "feat(eve-spike): port read_reference as an Eve tool with unit tests

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Capture Eve's streaming shape and map it to the current UI (Q3)

**Files:**
- Create: `scripts/eve-stream-capture.sh`
- Modify: `docs/eve-spike-findings.md` (fill `## Q3`)
- Read (do not modify): `components/chat.tsx`, `app/(chat)/api/chat/route.ts`

**Interfaces:**
- Consumes: the working tool turn from Task 2.
- Produces: a raw NDJSON capture and a mapping table in the findings doc; a documented decision (adapter route vs. UI rework).

- [ ] **Step 1: Write a capture script**

Create `scripts/eve-stream-capture.sh`:
```bash
#!/usr/bin/env bash
# Captures the raw NDJSON event stream from an Eve session for analysis.
# Set BASE to match the dev server from Task 1 (port 2000 for `eve dev`,
# 3000 if Eve is mounted into Next), or a Vercel preview URL in Task 5.
# Usage: BASE=http://127.0.0.1:2000 ./scripts/eve-stream-capture.sh
set -euo pipefail
BASE="${BASE:-http://127.0.0.1:2000}"
OUT="${OUT:-eve-stream-capture.ndjson}"

resp=$(curl -sD - -o /dev/null -X POST "$BASE/eve/v1/session" \
  -H 'content-type: application/json' \
  -d '{"message":"Read the reference field-patterns.md and tell me what it covers."}')
sid=$(printf '%s' "$resp" | tr -d '\r' | awk -F': ' 'tolower($1)=="x-eve-session-id"{print $2}')
echo "session: $sid"
curl -N "$BASE/eve/v1/session/$sid/stream" | tee "$OUT"
```
Make it executable:
```bash
chmod +x scripts/eve-stream-capture.sh
```

- [ ] **Step 2: Capture a real stream**

With `pnpm dev` running:
```bash
./scripts/eve-stream-capture.sh
```
Expected: `eve-stream-capture.ndjson` contains one JSON object per line covering the session lifecycle (session start, model/text deltas, the `read_reference` tool call + result, usage/token info if present, and completion). This file is a scratch artifact — do NOT commit it; extract the event types into the findings doc instead.

- [ ] **Step 3: Identify what the current UI consumes**

Read `components/chat.tsx` and note every stream part it reads. Confirmed consumers to look for: the AI SDK `UIMessage` parts (text, tool calls/results) plus the transient custom data events emitted by `route.ts`: `data-token-usage`, `data-compacting`, `data-checkpoint`. Record each with the exact shape `route.ts` writes (see `app/(chat)/api/chat/route.ts` lines that call `dataStream.write({ type: 'data-...' })`).

- [ ] **Step 4: Write the mapping table and decision**

Fill `## Q3` in `docs/eve-spike-findings.md` with a table of the form:

| UI needs (current) | Source today | Eve NDJSON equivalent | Gap? |
|---|---|---|---|
| assistant text parts | AI SDK UIMessage | `<eve event type>` | |
| tool call / result | AI SDK UIMessage | `<eve event type>` | |
| `data-token-usage` | `route.ts` onStepEnd | `<eve usage event or none>` | |
| `data-compacting` | `route.ts` prepareStep | `<none — Eve-specific>` | |
| `data-checkpoint` | `route.ts` prepareStep | `<none — Eve-specific>` | |

Then write a one-paragraph recommendation: **adapter route** (a Next route that reads Eve NDJSON and re-emits the AI SDK SSE shape `components/chat.tsx` already understands) vs. **UI rework** (teach `components/chat.tsx` to consume Eve's stream directly). State which is lower-risk for sub-project 5 and why.

- [ ] **Step 5: Commit**

```bash
git add scripts/eve-stream-capture.sh docs/eve-spike-findings.md
git commit -m "docs(eve-spike): capture Eve streaming shape + UI mapping decision (Q3)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Investigate context management + prototype the working-memory replacement (Q2)

**Files:**
- Create: `agent/tools/update_working_memory.ts`
- Create: `agent/subagents/summarizer/agent.ts` (only if Task 4 concludes a subagent is the right shape — see Step 3)
- Modify: `docs/eve-spike-findings.md` (fill `## Q2` and `## Browser session sketch`)
- Read (do not modify): `lib/ai/context-compression.ts`, `lib/kernel/browser.ts`

**Interfaces:**
- Consumes: the working agent from Tasks 1–2; Eve's sandbox filesystem tools (`bash`, `read_file`, `write_file` per Eve concepts).
- Produces: a documented answer to "can the `prepareStep` compaction be expressed under Eve, and how"; a runnable working-memory-file prototype; a browser-session viability sketch.

- [ ] **Step 1: Investigate Eve's context/turn hooks**

Inspect the installed `eve` package for any turn/step hook or context configuration on `defineAgent` (search the package's type definitions):
```bash
grep -rEn "prepareStep|onStep|context|compact|maxTokens|trim|history" node_modules/eve/dist 2>/dev/null | head -40
```
Also re-read `lib/ai/context-compression.ts` to enumerate exactly what must be replaced: (a) trigger at 75% of a 200K window, (b) Haiku summarization into a session-handoff doc, (c) structured working-memory extraction via a tool call, (d) a pinned working-memory message that is never compacted, (e) keeping the last 8 messages. Record in `## Q2` which of (a)–(e) Eve provides natively, which need re-building, and which become unnecessary under Eve's durable model.

- [ ] **Step 2: Prototype cross-turn working-memory persistence**

Per Eve docs, **tools run in your app runtime with full `process.env`, not inside the sandbox** — so a tool cannot assume a `ctx.sandbox` writer. First determine what Eve gives a tool for durable per-session state: inspect the `execute` signature and any second `context` argument in the installed types:
```bash
grep -rEn "execute|context|session|state|store|sandbox|writeFile" node_modules/eve/dist 2>/dev/null | head -40
```
Then prototype persistence using whichever is real. Two candidate shapes — pick the one the installed API supports and delete the other from the findings:
- **(a) Eve session/state API** if `execute` receives a context with a per-session store.
- **(b) App-runtime store** (the tool writes to the app's existing Postgres via `lib/db/queries`, keyed by session id) — always available since the tool runs in the app runtime with `process.env`.

Create `agent/tools/update_working_memory.ts` implementing the chosen shape, e.g. the runtime-agnostic (b) fallback:
```ts
import { defineTool } from 'eve/tools';
import { z } from 'zod';

// Prototype for sub-project 4: persist structured participant/session data
// OUTSIDE the model context so it survives across turns. Eve tools run in
// the app runtime (not the sandbox), so this uses app-runtime persistence.
// If the installed Eve `execute` context exposes a native per-session store,
// prefer that and record it in the findings; otherwise persist via the
// existing app database keyed by session id.
export default defineTool({
  description:
    'Persist structured working-memory data (form state, completed fields, caseworker answers) so it survives across turns.',
  inputSchema: z.object({
    data: z.record(z.string(), z.unknown()),
  }),
  async execute({ data }: { data: Record<string, unknown> }) {
    // Prototype sink — replace with the real store confirmed above.
    // The point is to prove a tool CAN durably persist state across turns.
    globalThis.__eveWorkingMemory = data;
    return { ok: true, keys: Object.keys(data) };
  },
});
```
Run a session that asks the agent to store some data, then a follow-up turn (`POST /eve/v1/session/<id>` with the `continuationToken`) that asks it to recall it. Confirm the value persisted across the two turns. Record in `## Q2` the actual persistence API used and whether Eve offers a native per-session store.

- [ ] **Step 3: Decide summarization shape and (conditionally) prototype a subagent**

Based on Step 1: if Eve has no in-run compaction hook, the summarization must be triggered explicitly — most naturally as a **subagent** the main agent delegates to when context grows. If Step 1 shows a subagent is the right shape, create `agent/subagents/summarizer/agent.ts`:
```ts
import { defineAgent } from 'eve';

// A focused child agent that condenses a transcript into a session-handoff
// summary. Runs with its own fresh context (Eve subagent semantics), so it
// does not inflate the parent's context window.
export default defineAgent({
  // Dot-versioned gateway slug. Verify `anthropic/claude-haiku-4.5` is listed
  // by `vercel ai-gateway models ls`; if not, fall back to
  // `anthropic/claude-sonnet-4.6` for the summarizer.
  model: 'anthropic/claude-haiku-4.5',
});
```
with `agent/subagents/summarizer/instructions.md` containing the extraction rules copied from `COMPACTION_SYSTEM_PROMPT` in `lib/ai/context-compression.ts`. Subagent declaration (`agent/subagents/<name>/`) is per the Eve concepts doc; confirm the exact directory/config layout against the installed version. If Step 1 instead shows Eve manages context internally (making the subagent unnecessary), do NOT create these files — record that conclusion in `## Q2` and skip. Either outcome is a valid, documented answer.

- [ ] **Step 4: Write the browser-session viability sketch**

Read `lib/kernel/browser.ts` and note the two things that break under Eve's replayed durable execution: the in-memory session cache keyed `${userId}:${sessionId}` and the per-session mutex around a long-lived Playwright `page`. In `## Browser session sketch`, write how a Kernel.sh-backed tool would work under Eve instead: re-resolve the Kernel session by its stable ID (`${chatId}-${userId}`) at the start of each tool call rather than relying on in-process state, and where per-turn serialization would live. This is a sketch to inform sub-project 3, not an implementation.

- [ ] **Step 5: Commit**

```bash
git add agent/tools/update_working_memory.ts docs/eve-spike-findings.md
# include the subagent files only if Step 3 created them:
git add agent/subagents 2>/dev/null || true
git commit -m "docs(eve-spike): context-management findings + working-memory prototype (Q2)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Deploy a Vercel preview and confirm Eve on-platform (Q1 remote)

**Prerequisite (blocker for this task only):** a Vercel team/project to deploy into. If none exists yet, stop and get it provisioned — Tasks 1–4 and 6 do not depend on this task and can complete first.

**Files:**
- Create: `vercel.json` (only if Eve/Next require build config not inferred by the preset — otherwise skip)
- Modify: `docs/eve-spike-findings.md` (fill `## Q1 — Vercel preview`)

**Interfaces:**
- Consumes: the local-proven agent from Tasks 1–2.
- Produces: a live preview URL where `POST /eve/v1/session` + stream work through AI Gateway via OIDC.

- [ ] **Step 1: Link the project**

Run:
```bash
vercel link
```
Follow the prompts to link to the (non-production) Vercel project. This creates `.vercel/` (gitignored).

- [ ] **Step 2: Set preview environment variables**

The existing app must boot, so set at minimum the env the app reads on startup. Point `DATABASE_URL` at a **non-production** Postgres branch — never production.
```bash
vercel env add DATABASE_URL preview
vercel env add AUTH_SECRET preview
```
Add any other keys the app needs to boot that are present in `.env.example` (e.g. Apricot, Kernel, Redis) as required to get past startup. AI Gateway on Vercel uses OIDC automatically — do NOT set `AI_GATEWAY_API_KEY` in the Vercel env unless the deploy proves OIDC is unavailable (record which was needed).

- [ ] **Step 3: Deploy a preview**

**Heads-up:** this repo's build is `tsx lib/db/migrate && next build`, so `vercel deploy` **runs Drizzle migrations against `DATABASE_URL` during the build.** The preview `DATABASE_URL` must therefore point at a writable, migration-compatible **non-production** branch — if it points at a read-only or schema-mismatched DB, the build fails before Eve is ever exercised (and if it pointed at prod, it would mutate prod schema). Double-check the Step 2 value before deploying.

Run:
```bash
vercel deploy
```
Expected: a preview URL (e.g. `https://<hash>.vercel.app`) and a successful build. If the build fails, capture the error verbatim into `## Q1 — Vercel preview` — whether it's the migration step or the Eve compile step, build friction is a real spike finding.

- [ ] **Step 4: Run the curl round-trip against the preview**

```bash
BASE=https://<preview-url> ./scripts/eve-stream-capture.sh
```
Expected: same behavior as local — a session is created and the `read_reference` tool executes end-to-end, now with models resolved via AI Gateway OIDC on Vercel. Also check Agent Runs in the Vercel dashboard shows the session/turn/tool call.

- [ ] **Step 5: Record findings and commit**

Fill `## Q1 — Vercel preview` with: whether it deployed, any config required (and whether a `vercel.json` was needed), OIDC-vs-key outcome, and the Agent Runs observation.
```bash
git add docs/eve-spike-findings.md vercel.json 2>/dev/null || git add docs/eve-spike-findings.md
git commit -m "docs(eve-spike): confirm Eve on a Vercel preview (Q1 remote)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Verify exit criteria and finalize the findings doc

**Files:**
- Modify: `docs/eve-spike-findings.md` (add `## Recommendation` and `## Open items for sub-projects 2–6`)

**Interfaces:**
- Consumes: all prior findings.
- Produces: a decision-ready findings doc.

- [ ] **Step 1: Check every exit criterion is answered**

Confirm the findings doc has concrete content (not skeleton placeholders) for: Q1 local, Q1 Vercel preview (or a clear "blocked on Vercel provisioning" note if Task 5 was deferred), Q2, Q3, and the browser sketch. Grep for leftover angle-bracket placeholders:
```bash
grep -nE "<[a-z].*>|\(filled in Task" docs/eve-spike-findings.md || echo "no placeholders"
```
Expected: `no placeholders` (except Q1-preview if intentionally deferred, which must say so explicitly).

- [ ] **Step 2: Write the recommendation**

Add `## Recommendation`: a 3–5 sentence go/no-go on the full migration, calling out the single biggest risk surfaced (most likely the context-compaction re-architecture from Q2), and `## Open items for sub-projects 2–6` listing what each downstream spec now knows that it didn't before.

- [ ] **Step 3: Run the full test suite to confirm nothing regressed**

Run:
```bash
pnpm exec vitest run tests/agent/read-reference.test.ts
```
Expected: PASS. (The spike added only additive tests; existing suites are unaffected since no existing file was modified.)

- [ ] **Step 4: Commit**

```bash
git add docs/eve-spike-findings.md
git commit -m "docs(eve-spike): finalize findings + migration recommendation

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review notes

- **Spec coverage:** Q1 → Tasks 1, 2, 5; Q2 → Task 4; Q3 → Task 3; browser sketch → Task 4; "additive only / no prod changes" → Global Constraints + enforced by touching only `agent/`, `tests/agent/`, `scripts/`, `docs/`, and the additive `vitest.config.node.mjs`; "local-first then preview" → Tasks 1–4 local, Task 5 preview; "readReference proof tool" → Task 2; findings deliverable → Tasks 1/3/4/5/6.
- **Grounded vs. discovery:** the core Eve APIs are confirmed from eve.dev docs — `npx eve init .` for existing apps, `defineAgent({ model })`, `defineTool({ description, inputSchema, async execute(input) })`, snake_case tool filenames, tools running in the app runtime (not the sandbox), and the `/eve/v1/session` routes. Genuine unknowns that the spike must resolve at runtime (and are marked as such): how Eve mounts into Next 16 (own server on 2000 vs. into Next on 3000), whether Eve exposes a `prepareStep`-equivalent context hook, the native per-session state API, the exact subagent directory layout, and the NDJSON event shapes. These are the point of the spike, not placeholders.
- **Environment fixes:** Node-only tests use the added `vitest.config.node.mjs` (default vitest is browser-mode); the Vercel build runs Drizzle migrations, so the preview `DATABASE_URL` must be a writable non-prod branch (Task 5).
- **Type consistency:** `readReferenceFile` is defined in Task 2 and referenced only by its own test; `read_reference` (filename-derived tool name) is used consistently in Tasks 1–3; `update_working_memory` and the `summarizer` subagent are self-contained in Task 4. All model slugs are dot-versioned (`claude-sonnet-4.6`, `claude-haiku-4.5`).
