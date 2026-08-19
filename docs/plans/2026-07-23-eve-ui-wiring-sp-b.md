# SP-B — Wire the Eve Runtime into the App UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Behind a `useEveAgent` feature flag, run chat turns on the Eve server via a new Next adapter route that translates Eve NDJSON → AI SDK SSE, so text, tool activity, and the interactive `gap_analysis`/`form_summary` cards render in the existing chat UI — with legacy `/api/chat` untouched.

**Architecture:** A new route `app/(chat)/api/eve-chat/route.ts` authenticates, HTTP-proxies to the Eve server (`EVE_SERVER_URL`), and pipes Eve's NDJSON through a pure translator (`lib/ai/eve/stream-adapter.ts`) into a `createUIMessageStream` writer → `JsonToSseTransformStream` (the exact shape `components/chat.tsx` already consumes). Per-chat Eve session continuity is held in-memory (`lib/ai/eve/session-continuity.ts`). The only client change is a flag-gated transport `api` URL. Legacy `/api/chat`, `route.ts`, Postgres history, and Redis are unchanged (history/persistence is SP-C).

**Tech Stack:** Next.js 16 route · AI SDK v7 (`ai`) UI message stream · Eve `0.27.0` server (NDJSON) · next-auth · zod · vitest (node config) · pnpm.

## Global Constraints

- **Additive.** Do NOT modify `app/(chat)/api/chat/route.ts`, `lib/ai/prompts/*`, `lib/kernel/*`, `lib/ai/tools/*`, or `agent/*`. The intended edits outside new files are: add one flag to `lib/feature-flags.ts`, switch the transport `api` in `components/chat.tsx`, and append `EVE_SERVER_URL` to `.env.example`. Everything else is new files under `app/(chat)/api/eve-chat/` and `lib/ai/eve/`.
- **Flag OFF (default) === today.** `useEveAgent` defaults OFF; with it off, `chat.tsx` points at `/api/chat` and behaves identically. Never change the default-off behavior.
- **Two servers in dev.** The adapter proxies to the Eve server; run `npx eve dev` (Node 24, env loaded via `set -a; . ./.env.local; set +a`) alongside `pnpm dev` (Next). `EVE_SERVER_URL` defaults to `http://127.0.0.1:2000`. The Next route itself needs no Eve import and no Node 24 (it only does HTTP + AI SDK).
- **The translator must never import `eve` or anything `server-only`-guarded beyond what a Next route already uses** — it maps plain JS objects to AI SDK chunks. This keeps it unit-testable and avoids the SP-A `server-only` bundler issue.
- **Confirm Eve event field paths against a real capture** (Task 2 Step 1) using `scripts/eve-stream-capture.sh` before finalizing the translator — the shapes below come from the spike's Q3 catalog (`docs/eve-spike-findings.md`) and must be verified.
- pnpm only; secrets only in `.env.local` (never committed); Node 24 for any `eve dev` / node-config vitest run.
- Deferred (do NOT build): Postgres message save, Redis resumable stream, `data-compacting`/`data-checkpoint` indicators, server-side Eve abort, removing legacy route (all SP-C); Vercel topology (SP-D).

## Eve event shapes (from the Q3 capture — verify in Task 2 Step 1)

```
session.started   { }
turn.started      { }
message.received  { }                                  (echo of inbound user msg — ignore)
step.started      { }
actions.requested { data: { actions: [ { kind: "tool-call", toolName, input, callId } ] } }
action.result     { data: { result: { kind: "tool-result", callId, toolName, output }, status } }
step.completed    { data: { finishReason, usage: { inputTokens, outputTokens, cacheReadTokens, cacheWriteTokens, costUsd } } }
message.appended  { data: { messageDelta, messageSoFar } }
message.completed { data: { finishReason, ... final text ... } }
turn.completed    { }
session.waiting   { data: { continuationToken, wait } }
error/abort       (SHAPE UNVERIFIED — capture in Task 2 Step 1)
```

---

### Task 1: In-memory session-continuity store

**Files:**
- Create: `lib/ai/eve/session-continuity.ts`
- Test: `tests/agent/eve-session-continuity.test.ts`

**Interfaces:**
- Consumes: nothing.
- Produces: `getContinuity(userId: string, chatId: string): { eveSessionId: string; continuationToken: string } | undefined`, `setContinuity(userId, chatId, value): void`, `clearContinuity(userId, chatId): void`.

- [ ] **Step 1: Write the failing test**

Create `tests/agent/eve-session-continuity.test.ts`:
```ts
import { describe, it, expect } from 'vitest';
import { getContinuity, setContinuity, clearContinuity } from '@/lib/ai/eve/session-continuity';

describe('session-continuity', () => {
  it('returns undefined for an unknown chat', () => {
    expect(getContinuity('u1', 'c-unknown')).toBeUndefined();
  });
  it('stores and retrieves per (user, chat)', () => {
    setContinuity('u1', 'c1', { eveSessionId: 's1', continuationToken: 't1' });
    expect(getContinuity('u1', 'c1')).toEqual({ eveSessionId: 's1', continuationToken: 't1' });
    // isolation: same chatId, different user is separate
    expect(getContinuity('u2', 'c1')).toBeUndefined();
  });
  it('overwrites on repeated set (new continuation token)', () => {
    setContinuity('u1', 'c2', { eveSessionId: 's2', continuationToken: 't2' });
    setContinuity('u1', 'c2', { eveSessionId: 's2', continuationToken: 't2b' });
    expect(getContinuity('u1', 'c2')?.continuationToken).toBe('t2b');
  });
  it('clears an entry', () => {
    setContinuity('u1', 'c3', { eveSessionId: 's3', continuationToken: 't3' });
    clearContinuity('u1', 'c3');
    expect(getContinuity('u1', 'c3')).toBeUndefined();
  });
});
```

- [ ] **Step 2: Run it — expect FAIL** (module missing)

```bash
export PATH="$HOME/.nvm/versions/node/v24.18.0/bin:$PATH"
pnpm exec vitest run -c vitest.config.node.mjs tests/agent/eve-session-continuity.test.ts
```
Expected: FAIL, cannot resolve `@/lib/ai/eve/session-continuity`.

- [ ] **Step 3: Implement**

Create `lib/ai/eve/session-continuity.ts`:
```ts
// In-memory per-(user, chat) Eve session continuity. SINGLE-PROCESS and lost
// on restart — SP-C replaces this with a Postgres-backed mapping. A restart
// mid-conversation simply starts a fresh Eve session on the next message.
export interface EveContinuity {
  eveSessionId: string;
  continuationToken: string;
}

const store = new Map<string, EveContinuity>();
const key = (userId: string, chatId: string) => `${userId}:${chatId}`;

export function getContinuity(userId: string, chatId: string): EveContinuity | undefined {
  return store.get(key(userId, chatId));
}
export function setContinuity(userId: string, chatId: string, value: EveContinuity): void {
  store.set(key(userId, chatId), value);
}
export function clearContinuity(userId: string, chatId: string): void {
  store.delete(key(userId, chatId));
}
```

- [ ] **Step 4: Run it — expect PASS (4/4)**

```bash
pnpm exec vitest run -c vitest.config.node.mjs tests/agent/eve-session-continuity.test.ts
```

- [ ] **Step 5: Commit**

```bash
git add lib/ai/eve/session-continuity.ts tests/agent/eve-session-continuity.test.ts
git commit -m "feat(eve-ui): in-memory Eve session-continuity store

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Eve NDJSON → AI SDK UIMessage translator (the core; TDD)

**Files:**
- Create: `lib/ai/eve/stream-adapter.ts`
- Test: `tests/agent/eve-stream-adapter.test.ts`

**Interfaces:**
- Consumes: AI SDK `UIMessageStreamWriter` (`write(chunk)`), a `generateId` fn.
- Produces:
  - `EVE_TOOL_NAME_MAP: Record<string, string>` (`{ gap_analysis: 'gapAnalysis', form_summary: 'formSummary' }`).
  - `mapToolName(eveName: string): string` — mapped name or the original.
  - `extractLatestUserText(message: unknown): string` — pull the text from the AI SDK UIMessage the client sends as `body.message`.
  - `translateEveEvent(event: any, writer: Writer, ctx: { textId: string | null; generateId: () => string }): { textId: string | null; done: boolean; continuationToken?: string }` — apply ONE Eve event to the writer; returns updated text-block id, whether the turn is done (`turn.completed`/`session.waiting`), and any `continuationToken` seen. `Writer` is `{ write(chunk: any): void }`.

- [ ] **Step 1: Capture + confirm real Eve event shapes**

Under Node 24 with a running `npx eve dev` (env loaded), capture a tool-calling turn and confirm the field paths in the "Eve event shapes" table above:
```bash
export PATH="$HOME/.nvm/versions/node/v24.18.0/bin:$PATH"; set -a; . ./.env.local; set +a
# start eve dev on :2010 in another shell, then:
BASE=http://127.0.0.1:2010 OUT=/tmp/eve-capture.ndjson ./scripts/eve-stream-capture.sh
```
Read `/tmp/eve-capture.ndjson`; adjust the test fixtures + translator field paths in the following steps to match exactly what Eve emits (e.g. confirm `actions.requested` nests under `data.actions[]`, `message.appended` under `data.messageDelta`). Also trigger an error turn if feasible and record the error event shape. Do NOT commit the capture. Note any field-path corrections in the report.

- [ ] **Step 2: Write the failing test**

Create `tests/agent/eve-stream-adapter.test.ts` (adjust fixtures to the Step 1 capture):
```ts
import { describe, it, expect } from 'vitest';
import { mapToolName, extractLatestUserText, translateEveEvent } from '@/lib/ai/eve/stream-adapter';

function collect() {
  const chunks: any[] = [];
  return { writer: { write: (c: any) => chunks.push(c) }, chunks };
}
const gen = () => 'txt-1';

describe('mapToolName', () => {
  it('maps snake_case card tools to camelCase for message.tsx renderers', () => {
    expect(mapToolName('gap_analysis')).toBe('gapAnalysis');
    expect(mapToolName('form_summary')).toBe('formSummary');
  });
  it('passes through unmapped tools', () => {
    expect(mapToolName('browser')).toBe('browser');
    expect(mapToolName('check_submit_gate')).toBe('check_submit_gate');
  });
});

describe('extractLatestUserText', () => {
  it('pulls text from an AI SDK UIMessage parts array', () => {
    const msg = { role: 'user', parts: [{ type: 'text', text: 'apply for WIC' }] };
    expect(extractLatestUserText(msg)).toBe('apply for WIC');
  });
  it('falls back to empty string when no text part', () => {
    expect(extractLatestUserText({ role: 'user', parts: [] })).toBe('');
  });
});

describe('translateEveEvent', () => {
  it('streams text: message.appended -> text-start + text-delta, message.completed -> text-end', () => {
    const { writer, chunks } = collect();
    let ctx = { textId: null as string | null, generateId: gen };
    let r = translateEveEvent({ type: 'message.appended', data: { messageDelta: 'Hel' } }, writer, ctx);
    ctx = { ...ctx, textId: r.textId };
    r = translateEveEvent({ type: 'message.appended', data: { messageDelta: 'lo' } }, writer, ctx);
    ctx = { ...ctx, textId: r.textId };
    translateEveEvent({ type: 'message.completed', data: {} }, writer, ctx);
    expect(chunks).toEqual([
      { type: 'text-start', id: 'txt-1' },
      { type: 'text-delta', id: 'txt-1', delta: 'Hel' },
      { type: 'text-delta', id: 'txt-1', delta: 'lo' },
      { type: 'text-end', id: 'txt-1' },
    ]);
  });
  it('maps a tool call + result to AI SDK tool chunks with the camelCase name', () => {
    const { writer, chunks } = collect();
    const ctx = { textId: null, generateId: gen };
    translateEveEvent(
      { type: 'actions.requested', data: { actions: [{ kind: 'tool-call', toolName: 'gap_analysis', input: { formName: 'WIC' }, callId: 'call-1' }] } },
      writer, ctx,
    );
    translateEveEvent(
      { type: 'action.result', data: { result: { kind: 'tool-result', callId: 'call-1', toolName: 'gap_analysis', output: { rendered: true } } } },
      writer, ctx,
    );
    expect(chunks).toEqual([
      { type: 'tool-input-available', toolCallId: 'call-1', toolName: 'gapAnalysis', input: { formName: 'WIC' } },
      { type: 'tool-output-available', toolCallId: 'call-1', output: { rendered: true } },
    ]);
  });
  it('maps step.completed.usage to a transient data-token-usage event', () => {
    const { writer, chunks } = collect();
    translateEveEvent(
      { type: 'step.completed', data: { usage: { inputTokens: 100, outputTokens: 20, cacheReadTokens: 40 } } },
      writer, { textId: null, generateId: gen },
    );
    expect(chunks).toEqual([
      { type: 'data-token-usage', data: { inputTokens: 100, outputTokens: 20, cachedInputTokens: 40 }, transient: true },
    ]);
  });
  it('signals done + captures continuationToken on session.waiting', () => {
    const { writer } = collect();
    const r = translateEveEvent({ type: 'session.waiting', data: { continuationToken: 'tok-9' } }, writer, { textId: null, generateId: gen });
    expect(r.done).toBe(true);
    expect(r.continuationToken).toBe('tok-9');
  });
  it('ignores lifecycle/echo events without writing', () => {
    const { writer, chunks } = collect();
    for (const t of ['session.started', 'turn.started', 'message.received', 'step.started']) {
      translateEveEvent({ type: t, data: {} }, writer, { textId: null, generateId: gen });
    }
    expect(chunks).toEqual([]);
  });
});
```

- [ ] **Step 3: Run it — expect FAIL** (module missing)

```bash
pnpm exec vitest run -c vitest.config.node.mjs tests/agent/eve-stream-adapter.test.ts
```

- [ ] **Step 4: Implement the translator**

Create `lib/ai/eve/stream-adapter.ts`:
```ts
// Pure translator: Eve NDJSON events -> AI SDK v7 UIMessage stream chunks.
// No `eve` import, no server-only — maps plain objects, so it is unit-testable
// and safe in a Next route. Chunk shapes match AI SDK v7 (text-start/delta/end,
// tool-input-available, tool-output-available, data-* transient).

export const EVE_TOOL_NAME_MAP: Record<string, string> = {
  gap_analysis: 'gapAnalysis',
  form_summary: 'formSummary',
};

export function mapToolName(eveName: string): string {
  return EVE_TOOL_NAME_MAP[eveName] ?? eveName;
}

interface Writer { write(chunk: any): void }
interface Ctx { textId: string | null; generateId: () => string }

// Pull the user's text out of the AI SDK UIMessage the client sends as body.message.
export function extractLatestUserText(message: unknown): string {
  const m = message as { parts?: Array<{ type?: string; text?: string }> } | null;
  if (!m?.parts) return '';
  return m.parts
    .filter((p) => p.type === 'text' && typeof p.text === 'string')
    .map((p) => p.text as string)
    .join('')
    .trim();
}

// Apply ONE Eve event to the writer. Returns updated text-block id, whether the
// turn is finished, and any continuation token seen.
export function translateEveEvent(
  event: any,
  writer: Writer,
  ctx: Ctx,
): { textId: string | null; done: boolean; continuationToken?: string } {
  let textId = ctx.textId;
  switch (event?.type) {
    case 'message.appended': {
      const delta = event.data?.messageDelta ?? '';
      if (!delta) break;
      if (textId === null) {
        textId = ctx.generateId();
        writer.write({ type: 'text-start', id: textId });
      }
      writer.write({ type: 'text-delta', id: textId, delta });
      break;
    }
    case 'message.completed': {
      if (textId !== null) {
        writer.write({ type: 'text-end', id: textId });
        textId = null;
      }
      break;
    }
    case 'actions.requested': {
      for (const a of event.data?.actions ?? []) {
        if (a?.kind !== 'tool-call') continue;
        writer.write({
          type: 'tool-input-available',
          toolCallId: a.callId,
          toolName: mapToolName(a.toolName),
          input: a.input ?? {},
        });
      }
      break;
    }
    case 'action.result': {
      const r = event.data?.result;
      if (r?.kind === 'tool-result') {
        writer.write({ type: 'tool-output-available', toolCallId: r.callId, output: r.output });
      }
      break;
    }
    case 'step.completed': {
      const u = event.data?.usage;
      if (u) {
        writer.write({
          type: 'data-token-usage',
          data: {
            inputTokens: u.inputTokens ?? 0,
            outputTokens: u.outputTokens ?? 0,
            cachedInputTokens: u.cacheReadTokens ?? 0,
          },
          transient: true,
        });
      }
      break;
    }
    case 'session.waiting':
      return { textId, done: true, continuationToken: event.data?.continuationToken };
    case 'turn.completed':
      return { textId, done: true };
    // session.started / turn.started / message.received / step.started: ignored.
    default:
      break;
  }
  return { textId, done: false };
}
```
If Step 1's capture showed different field paths (e.g. text under `data.text` not `data.messageDelta`), correct them here and in the fixtures so the tests reflect reality.

- [ ] **Step 5: Run it — expect PASS**

```bash
pnpm exec vitest run -c vitest.config.node.mjs tests/agent/eve-stream-adapter.test.ts
```
Expected: all pass. If the Step 1 capture forced field-path changes, both fixtures and impl reflect them and still pass.

- [ ] **Step 6: Commit**

```bash
git add lib/ai/eve/stream-adapter.ts tests/agent/eve-stream-adapter.test.ts
git commit -m "feat(eve-ui): Eve NDJSON -> AI SDK UIMessage translator (TDD)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Eve HTTP client

**Files:**
- Create: `lib/ai/eve/eve-client.ts`

**Interfaces:**
- Consumes: `EVE_SERVER_URL` env.
- Produces:
  - `createEveSession(message: string): Promise<{ sessionId: string; continuationToken: string }>` — POST `/eve/v1/session`.
  - `continueEveSession(sessionId: string, continuationToken: string, message: string): Promise<{ continuationToken: string }>` — POST `/eve/v1/session/:id`.
  - `openEveStream(sessionId: string, signal?: AbortSignal): Promise<Response>` — GET `/eve/v1/session/:id/stream` (NDJSON body).
  - `parseNdjson(stream: ReadableStream<Uint8Array>): AsyncGenerator<any>` — yield parsed JSON per line.

- [ ] **Step 1: Implement (thin HTTP wrapper)**

Create `lib/ai/eve/eve-client.ts`:
```ts
const EVE_URL = process.env.EVE_SERVER_URL ?? 'http://127.0.0.1:2000';

async function postJson(path: string, body: unknown) {
  const res = await fetch(`${EVE_URL}${path}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(`Eve server ${path} responded ${res.status}: ${await res.text().catch(() => '')}`);
  }
  const sessionId = res.headers.get('x-eve-session-id') ?? '';
  const json = (await res.json().catch(() => ({}))) as { continuationToken?: string };
  return { sessionId, continuationToken: json.continuationToken ?? '' };
}

export async function createEveSession(message: string) {
  const { sessionId, continuationToken } = await postJson('/eve/v1/session', { message });
  return { sessionId, continuationToken };
}

export async function continueEveSession(sessionId: string, continuationToken: string, message: string) {
  const { continuationToken: next } = await postJson(`/eve/v1/session/${sessionId}`, { continuationToken, message });
  return { continuationToken: next };
}

export async function openEveStream(sessionId: string, signal?: AbortSignal): Promise<Response> {
  const res = await fetch(`${EVE_URL}/eve/v1/session/${sessionId}/stream`, { signal });
  if (!res.ok || !res.body) {
    throw new Error(`Eve stream ${sessionId} responded ${res.status}`);
  }
  return res;
}

export async function* parseNdjson(stream: ReadableStream<Uint8Array>): AsyncGenerator<any> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let nl: number;
      while ((nl = buf.indexOf('\n')) !== -1) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (line) {
          try { yield JSON.parse(line); } catch { /* skip malformed line */ }
        }
      }
    }
    const tail = buf.trim();
    if (tail) { try { yield JSON.parse(tail); } catch { /* ignore */ } }
  } finally {
    reader.releaseLock();
  }
}
```
Note: confirm the create/continue request/response shapes against the Task 2 Step 1 capture (esp. the POST body key `message` and the `x-eve-session-id` header + `continuationToken` in the body — established in SP-A but re-verify).

- [ ] **Step 2: Type/compile check**

```bash
export PATH="$HOME/.nvm/versions/node/v24.18.0/bin:$PATH"
pnpm exec tsc --noEmit 2>&1 | grep "lib/ai/eve/eve-client" || echo "eve-client typechecks clean"
```
Expected: no errors referencing `eve-client.ts` (the pre-existing 7-error baseline is unrelated).

- [ ] **Step 3: Commit**

```bash
git add lib/ai/eve/eve-client.ts
git commit -m "feat(eve-ui): Eve HTTP client (create/continue/stream + NDJSON parser)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: The adapter route

**Files:**
- Create: `app/(chat)/api/eve-chat/route.ts`

**Interfaces:**
- Consumes: `auth` (`@/app/(auth)/auth`), `createEveSession`/`continueEveSession`/`openEveStream`/`parseNdjson` (Task 3), `getContinuity`/`setContinuity` (Task 1), `translateEveEvent`/`extractLatestUserText` (Task 2), `createUIMessageStream` + `JsonToSseTransformStream` + `generateId` (`ai` / `@/lib/utils`).
- Produces: `POST` handler returning an AI-SDK-SSE `Response`.

- [ ] **Step 1: Implement the route**

Create `app/(chat)/api/eve-chat/route.ts`:
```ts
import { createUIMessageStream, JsonToSseTransformStream } from 'ai';
import { auth } from '@/app/(auth)/auth';
import { generateUUID } from '@/lib/utils';
import { ChatSDKError } from '@/lib/errors';
import { getContinuity, setContinuity } from '@/lib/ai/eve/session-continuity';
import { createEveSession, continueEveSession, openEveStream, parseNdjson } from '@/lib/ai/eve/eve-client';
import { translateEveEvent, extractLatestUserText } from '@/lib/ai/eve/stream-adapter';

export const maxDuration = 300; // 5 min for long web-automation turns

export async function POST(request: Request) {
  const session = await auth();
  if (!session?.user?.id) {
    return new ChatSDKError('unauthorized:chat').toResponse();
  }
  const userId = session.user.id;

  let body: { id?: string; message?: unknown };
  try {
    body = await request.json();
  } catch {
    return new ChatSDKError('bad_request:api').toResponse();
  }
  const chatId = body.id;
  const text = extractLatestUserText(body.message);
  if (!chatId || !text) {
    return new ChatSDKError('bad_request:api').toResponse();
  }

  // Resolve or create the Eve session for this chat.
  let sessionId: string;
  try {
    const existing = getContinuity(userId, chatId);
    if (existing) {
      const { continuationToken } = await continueEveSession(existing.eveSessionId, existing.continuationToken, text);
      sessionId = existing.eveSessionId;
      setContinuity(userId, chatId, { eveSessionId: sessionId, continuationToken });
    } else {
      const created = await createEveSession(text);
      sessionId = created.sessionId;
      setContinuity(userId, chatId, { eveSessionId: sessionId, continuationToken: created.continuationToken });
    }
  } catch (err) {
    // Eve server unreachable or errored on session create/continue.
    console.error('[eve-chat] session error:', err);
    return new ChatSDKError('offline:chat').toResponse();
  }

  const stream = createUIMessageStream({
    execute: async ({ writer }) => {
      const res = await openEveStream(sessionId);
      let ctx = { textId: null as string | null, generateId: generateUUID };
      for await (const event of parseNdjson(res.body!)) {
        const r = translateEveEvent(event, writer, ctx);
        ctx = { ...ctx, textId: r.textId };
        if (r.continuationToken) {
          setContinuity(userId, chatId, { eveSessionId: sessionId, continuationToken: r.continuationToken });
        }
        if (r.done) break; // stop at turn.completed / session.waiting — do not hang on Eve's open stream
      }
    },
    generateId: generateUUID,
    onError: () => 'Oops, an error occurred running the agent.',
  });

  return new Response(stream.pipeThrough(new JsonToSseTransformStream()));
}
```
Note: confirm `ChatSDKError` codes (`unauthorized:chat`, `bad_request:api`, `offline:chat`) exist in `lib/errors.ts`; if a code differs, use the closest existing one (do NOT add new error infrastructure). Confirm the auth import path matches `route.ts`'s (`@/app/(auth)/auth`).

- [ ] **Step 2: Compile check + route registration**

```bash
export PATH="$HOME/.nvm/versions/node/v24.18.0/bin:$PATH"
pnpm exec tsc --noEmit 2>&1 | grep "api/eve-chat" || echo "eve-chat route typechecks clean"
```
Expected: no errors referencing the new route.

- [ ] **Step 3: Live smoke test (requires eve dev + next dev)**

In three shells (Node 24 + env for eve dev): (a) `npx eve dev --no-ui --port 2000`; (b) `EVE_SERVER_URL=http://127.0.0.1:2000 pnpm dev`; (c) curl the route with a signed-in session cookie is awkward — instead assert the route exists and rejects unauthenticated:
```bash
curl -i -X POST http://localhost:3000/api/eve-chat -H 'content-type: application/json' -d '{"id":"c1","message":{"role":"user","parts":[{"type":"text","text":"hi"}]}}'
```
Expected: a `401`/unauthorized `ChatSDKError` JSON (auth rejects the cookieless request) — proving the route is wired and auth-guarded. Full authenticated end-to-end is Task 6 via the browser UI. Record the response.

- [ ] **Step 4: Commit**

```bash
git add "app/(chat)/api/eve-chat/route.ts"
git commit -m "feat(eve-ui): adapter route proxying chat turns to the Eve server

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Feature flag + client transport switch + env

**Files:**
- Modify: `lib/feature-flags.ts`, `components/chat.tsx`, `.env.example`

**Interfaces:**
- Consumes: `isFeatureEnabled` (existing), the transport in `chat.tsx`.
- Produces: a `useEveAgent` flag; `chat.tsx` points at `/api/eve-chat` when it's on.

- [ ] **Step 1: Add the flag**

In `lib/feature-flags.ts`, extend the union and record (leave `declutterToolCalls` intact):
```ts
export type FeatureFlagKey = 'declutterToolCalls' | 'useEveAgent';
```
Add to `FEATURE_FLAGS`:
```ts
  useEveAgent: {
    key: 'useEveAgent',
    label: 'Use Eve agent',
    description: 'Route chat turns through the Eve agent (adapter route) instead of the legacy loop.',
    defaultValue: false,
  },
```

- [ ] **Step 2: Switch the transport in `chat.tsx`**

In `components/chat.tsx`, import the flag helper and select the api. Add near the top of the component:
```ts
import { isFeatureEnabled } from '@/lib/feature-flags';
// inside the component, before useChat:
const eveApi = isFeatureEnabled('useEveAgent') ? '/api/eve-chat' : '/api/chat';
```
Change the transport's `api: '/api/chat',` to `api: eveApi,`. Change nothing else (the `prepareSendMessagesRequest` body, `onData`, and rendering stay — the adapter emits the shape they already consume). Note: `isFeatureEnabled` reads `localStorage`, so it evaluates client-side on mount; a flag flip takes effect on the next chat load, which is acceptable.

- [ ] **Step 3: Add the env var**

Append to `.env.example`:
```
# Base URL of the Eve agent server (npx eve dev). Used by the /api/eve-chat adapter.
EVE_SERVER_URL=http://127.0.0.1:2000
```

- [ ] **Step 4: Verify flag OFF = legacy unchanged**

```bash
export PATH="$HOME/.nvm/versions/node/v24.18.0/bin:$PATH"
pnpm exec tsc --noEmit 2>&1 | grep -E "feature-flags|chat.tsx" || echo "flag + chat.tsx typecheck clean"
```
Confirm by reading the diff that with `useEveAgent` default `false`, `eveApi` is `/api/chat` — i.e. no behavior change when the flag is off.

- [ ] **Step 5: Commit**

```bash
git add lib/feature-flags.ts components/chat.tsx .env.example
git commit -m "feat(eve-ui): useEveAgent flag switches the chat transport to the adapter route

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: End-to-end verification + docs

**Files:**
- Modify: `agent/README.md` (add a short "Using Eve from the app UI (SP-B)" section)

**Interfaces:**
- Consumes: the whole SP-B stack.
- Produces: a verified end-to-end run + docs.

- [ ] **Step 1: Authenticated end-to-end in the browser**

Run `npx eve dev --no-ui --port 2000` (Node 24, env loaded) and `EVE_SERVER_URL=http://127.0.0.1:2000 pnpm dev`. In the browser, enable the `useEveAgent` flag (dev flag menu / set `localStorage['ff:useEveAgent']='true'`), open a chat, and send a form-filling task (e.g. "Go to example.com and read the heading", then a real gap-analysis-triggering task). Confirm: text streams; the browser tool runs (real Kernel); if the agent calls `gap_analysis`, the interactive gap card renders (`tool-gapAnalysis`); filling it and submitting sends a follow-up that continues the same Eve session; `form_summary` renders as a card. Record the observations (redact secrets).

- [ ] **Step 2: Confirm flag OFF path + additive-ness**

- Toggle the flag OFF, reload, send a message → legacy `/api/chat` handles it (network tab shows `/api/chat`). 
- `git diff --stat 85da775..HEAD -- "app/(chat)/api/chat/route.ts" lib/ai/prompts lib/kernel lib/ai/tools agent/tools agent/skills` → expect EMPTY (SP-B did not touch legacy agent code).
- Run `pnpm exec vitest run -c vitest.config.node.mjs tests/agent/` → all pass (continuity + adapter + the SP-A read-reference test).

- [ ] **Step 2b: Fallback when Eve is down**

With `pnpm dev` running but `npx eve dev` STOPPED, flag ON, send a message. Confirm the UI shows a clean error (from the `offline:chat` `ChatSDKError`), not a hang or a crash. Record it.

- [ ] **Step 3: Document**

Add a short section to `agent/README.md`: "Using Eve from the app UI (SP-B)" — run `npx eve dev` + `pnpm dev`, set `EVE_SERVER_URL`, enable the `useEveAgent` flag; the `/api/eve-chat` adapter translates Eve NDJSON → AI SDK SSE; continuity is in-memory (SP-C adds Postgres); history/persistence and legacy-route removal are SP-C. No marketing language.

- [ ] **Step 4: Commit**

```bash
git add agent/README.md
git commit -m "docs(eve-ui): document running Eve from the app UI (SP-B)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review notes

- **Spec coverage:** adapter route → Task 4; stream translator + tool-name map + latest-message extraction → Task 2; in-memory continuity → Task 1; Eve HTTP client → Task 3; `useEveAgent` flag + transport switch + `EVE_SERVER_URL` → Task 5; interactive card rendering → Tasks 2 (name map to `tool-gapAnalysis`/`tool-formSummary`) + 6 (verified in UI incl. round-trip); flag-OFF-unchanged + additive → Task 5 Step 4 + Task 6 Step 2; error/unreachable handling → Task 4 + Task 6 Step 2b; capture-verify Eve shapes → Task 2 Step 1. Non-goals (Postgres/Redis, compaction indicators, server-side abort, legacy removal, Vercel) untouched.
- **Placeholder scan:** every code step has complete code; test fixtures are concrete; the one genuine unknown (exact Eve event field paths + error shape) is resolved by Task 2 Step 1's real capture before the translator is finalized — not a placeholder.
- **Type/name consistency:** `getContinuity/setContinuity/clearContinuity` + `EveContinuity` (Task 1) used verbatim in Task 4; `translateEveEvent/extractLatestUserText/mapToolName/EVE_TOOL_NAME_MAP` (Task 2) used in Task 4; `createEveSession/continueEveSession/openEveStream/parseNdjson` (Task 3) used in Task 4; `useEveAgent` flag key consistent (Task 5); tool-name map yields `gapAnalysis`/`formSummary` so client parts are `tool-gapAnalysis`/`tool-formSummary` matching `message.tsx:495,513`.
- **AI SDK chunk shapes** (`text-start`/`text-delta`/`text-end`, `tool-input-available`, `tool-output-available`, `data-token-usage` transient) confirmed against `node_modules/ai/dist/index.d.ts`; `data-token-usage` payload matches what `chat.tsx:161-176` accumulates (`inputTokens`/`outputTokens`/`cachedInputTokens`).
