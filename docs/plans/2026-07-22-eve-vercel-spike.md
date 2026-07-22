# Eve + Vercel De-Risking Spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove Eve can mount into this Next.js 16 app on Vercel and answer three migration-gating questions (Eve-on-Vercel viability, context-management strategy, Eve→UI streaming shape) with running code and documented findings — changing nothing in production.

**Architecture:** Add an `agent/` directory (Eve's filesystem-first layout) alongside the existing app. Scaffold a minimal Eve agent with one self-contained proof tool (`read_reference`), exercise it via `/eve/v1/session`, capture Eve's runtime behavior, and write findings. The existing `app/(chat)/api/chat/route.ts` loop, Vertex providers, Redis, and Cloud Run deployment are untouched.

**Tech Stack:** Eve (beta) · Vercel AI Gateway (`@ai-sdk/gateway`, already a dependency) · Next.js 16 App Router · TypeScript · zod 3 · vitest · pnpm.

## Global Constraints

- Use `pnpm` only (packageManager is pinned). Ask before adding any dependency other than `eve` itself (approved as the point of this spike).
- **Additive only.** Do NOT modify `app/(chat)/api/chat/route.ts`, `lib/ai/providers.ts`, `lib/ai/context-compression.ts`, `lib/kernel/browser.ts`, or any existing tool. Cloud Run / Vertex / the live `/api/chat` route stay live and unchanged.
- **Pin the Eve beta version** installed (record the exact version in the findings doc). Eve is beta and "subject to change" — treat every Eve API used here as "per Eve docs as of the pinned version; verify against what is installed."
- Spike models resolve through AI Gateway using the string `anthropic/claude-sonnet-4.6` (sufficient for a tool-calling proof; cheaper than opus).
- Biome formatting: 2-space indent, 80-col. TypeScript path alias `@/*` → repo root.
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

- [ ] **Step 1: Install Eve**

Run:
```bash
pnpm add eve
```
Then record the resolved version:
```bash
pnpm ls eve
```
Expected: `eve` appears in `dependencies` in `package.json` and `pnpm ls eve` prints a concrete version (e.g. `eve 0.x.y`). Note this version — it goes in the findings doc in Step 6.

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

Run the dev server:
```bash
pnpm dev
```
In a second shell, create a session:
```bash
curl -i -X POST http://127.0.0.1:3000/eve/v1/session \
  -H 'content-type: application/json' \
  -d '{"message":"hello"}'
```
Expected (per Eve docs; confirm against the installed version): HTTP `200`, a response header `x-eve-session-id: <id>`, and a JSON body containing a `continuationToken`. If the route 404s, the "add Eve to an existing Next.js app" wiring is missing — this is a Q1 discovery point: consult the installed Eve package's README / `eve` CLI for how it registers routes in an existing Next 16 App Router project, apply it, and record exactly what was required in the findings doc.

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
- Test: `tests/agent/read-reference.test.ts`

**Interfaces:**
- Consumes: the booting Eve runtime from Task 1; the existing reference dir `lib/ai/prompts/references/`.
- Produces: `readReferenceFile(path: string): Promise<{ content: string } | { error: string }>` (named export) and a default `defineTool` export named `read_reference` at the runtime (filename-derived).

- [ ] **Step 1: Write the failing test**

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

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
pnpm exec vitest run tests/agent/read-reference.test.ts
```
Expected: FAIL — cannot resolve `@/agent/tools/read_reference` (module does not exist yet).

- [ ] **Step 3: Write the tool (core function + defineTool wrapper)**

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
  async execute(input: { path: string }) {
    return readReferenceFile(input.path);
  },
});
```
Note: `defineTool`'s exact call shape is from Eve docs as of the pinned version. If the installed package expects a different key than `execute` or a different `inputSchema` type, adjust the wrapper only — leave `readReferenceFile` (and therefore the passing unit tests) unchanged.

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
pnpm exec vitest run tests/agent/read-reference.test.ts
```
Expected: PASS (4 passing).

- [ ] **Step 5: Prove the tool executes inside an Eve turn**

With `pnpm dev` running, create a session that forces the tool:
```bash
curl -i -X POST http://127.0.0.1:3000/eve/v1/session \
  -H 'content-type: application/json' \
  -d '{"message":"Read the reference field-patterns.md and tell me what it covers."}'
```
Grab the `x-eve-session-id` from the response header, then attach to the stream:
```bash
curl -N http://127.0.0.1:3000/eve/v1/session/<sessionId>/stream
```
Expected: the NDJSON stream shows a tool call to `read_reference` and a final assistant message that references the file's content. If the model does not call the tool, tighten `agent/instructions.md` wording and retry. Record in the findings doc (Q1-local) that a tool executed end-to-end.

- [ ] **Step 6: Commit**

```bash
git add agent/tools/read_reference.ts tests/agent/read-reference.test.ts docs/eve-spike-findings.md
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
# Usage: BASE=http://127.0.0.1:3000 ./scripts/eve-stream-capture.sh
set -euo pipefail
BASE="${BASE:-http://127.0.0.1:3000}"
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

- [ ] **Step 2: Prototype working memory as a sandbox file**

Create `agent/tools/update_working_memory.ts`:
```ts
import { defineTool } from 'eve/tools';
import { z } from 'zod';

// Prototype: persist structured participant/session data as a JSON file in
// the agent sandbox, so it survives across turns without living in the
// model context. Proves the "working memory as a file" re-architecture for
// sub-project 4. Path/API of the sandbox write is per Eve docs as of the
// pinned version; adjust to the installed sandbox API if it differs.
export default defineTool({
  description:
    'Persist structured working-memory data (form state, completed fields, caseworker answers) to the agent sandbox.',
  inputSchema: z.object({
    data: z.record(z.string(), z.unknown()),
  }),
  async execute(input: { data: Record<string, unknown> }, ctx: any) {
    // Preferred: use the sandbox write helper Eve exposes on the tool
    // context. If the context does not expose one, document what IS
    // available and fall back to the framework `write_file` tool.
    await ctx.sandbox.writeFile(
      'working-memory.json',
      JSON.stringify(input.data, null, 2),
    );
    return { ok: true, keys: Object.keys(input.data) };
  },
});
```
Run a session that asks the agent to store some data, then confirm the file exists in the sandbox (via Eve's `read_file`/`bash` framework tool, or the dashboard). If `ctx.sandbox` is not the real API, record the actual sandbox-write API in `## Q2` and update this file to match — the deliverable is a *working* write, however Eve exposes it.

- [ ] **Step 3: Decide summarization shape and (conditionally) prototype a subagent**

Based on Step 1: if Eve has no in-run compaction hook, the summarization must be triggered explicitly — most naturally as a **subagent** the main agent delegates to when context grows. If Step 1 shows a subagent is the right shape, create `agent/subagents/summarizer/agent.ts`:
```ts
import { defineAgent } from 'eve';

// A focused child agent that condenses a transcript into a session-handoff
// summary. Runs with its own fresh context (Eve subagent semantics), so it
// does not inflate the parent's context window.
export default defineAgent({
  model: 'anthropic/claude-haiku-4-5',
});
```
with `agent/subagents/summarizer/instructions.md` containing the extraction rules copied from `COMPACTION_SYSTEM_PROMPT` in `lib/ai/context-compression.ts`. If Step 1 instead shows Eve manages context internally (making the subagent unnecessary), do NOT create these files — record that conclusion in `## Q2` and skip. Either outcome is a valid, documented answer.

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

Run:
```bash
vercel deploy
```
Expected: a preview URL (e.g. `https://<hash>.vercel.app`) and a successful build. If the build fails on the Eve step, capture the error verbatim into `## Q1 — Vercel preview` — build friction is a real spike finding.

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

- **Spec coverage:** Q1 → Tasks 1, 2, 5; Q2 → Task 4; Q3 → Task 3; browser sketch → Task 4; "additive only / no prod changes" → Global Constraints + enforced by touching only `agent/`, `tests/agent/`, `scripts/`, `docs/`; "local-first then preview" → Tasks 1–4 local, Task 5 preview; "readReference proof tool" → Task 2; findings deliverable → Tasks 1/3/4/5/6.
- **Beta honesty:** every Eve API used (`defineAgent`, `defineTool`, sandbox write, subagents, route mounting, NDJSON shape) is marked "per Eve docs as of the pinned version; verify against installed" because the framework is beta — these are genuine discovery points, which is the spike's purpose, not placeholders.
- **Type consistency:** `readReferenceFile` is defined in Task 2 and referenced only by its own test; `read_reference` (filename-derived tool name) is used consistently in Tasks 1–3; `update_working_memory` and the `summarizer` subagent are self-contained in Task 4.
