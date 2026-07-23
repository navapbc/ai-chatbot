# Model Selector Drives Eve's Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the dev model picker (non-prod `modelOverride`) drive the Eve agent's model — the adapter maps the picked id to an AI Gateway slug and passes it to Eve via an `x-eve-model` header; a custom `AuthFn` surfaces it as an auth attribute; and a `defineDynamic` model resolver on `agent.ts` applies it per session (falling back to `sonnet-4.6`).

**Architecture:** Additive. New pure `lib/ai/eve/model-map.ts` (`toGatewaySlug`). `eve-client.ts` gains an optional `model` that rides as the `x-eve-model` header on session-create. `agent/channels/eve.ts` gets a loopback-gated custom `AuthFn` that reads that header into `attributes.eveModel`. `agent/agent.ts` switches `model` to `defineDynamic({ fallback, events })` reading `ctx.session.auth.initiator?.attributes?.eveModel`. Legacy `/api/chat`/`route.ts`/providers untouched; dev/eval only (rides the existing non-prod `modelOverride` gate).

**Tech Stack:** Eve `0.27.0` (`defineDynamic` model + custom `AuthFn`) · AI Gateway model slugs · Next adapter route · zod · vitest (node config) · pnpm · Node 24.

## Global Constraints

- **Node 24 for `eve dev` / node-config vitest**: prefix commands with `export PATH="$HOME/.nvm/versions/node/v24.18.0/bin:$PATH"`; verify `node -v`=v24. State does NOT persist between Bash calls.
- **Secrets in `.env.local`** (never print/commit): `AI_GATEWAY_API_KEY`, `KERNEL_API_KEY`, `DATABASE_URL`. `eve dev` needs them loaded: `set -a; . ./.env.local; set +a`.
- **pnpm only**; never commit node_modules/.eve/secrets.
- **Additive; dev/eval only.** Do NOT modify legacy `app/(chat)/api/chat/route.ts`, `lib/ai/providers.ts`, `components/chat.tsx` (it already sends `modelOverride` only in non-prod — leave it), `lib/kernel`, `lib/ai/tools`, `agent/tools`, `agent/skills`. The changed files are exactly: NEW `lib/ai/eve/model-map.ts` (+ test), `lib/ai/eve/eve-client.ts`, `app/(chat)/api/eve-chat/route.ts`, `agent/channels/eve.ts`, `agent/agent.ts`, and `agent/README.md`.
- **Auth must not weaken.** The custom `AuthFn` accepts ONLY loopback requests (mirrors `localDev`) and returns `null` otherwise (falling through to the existing entries). The `x-eve-model` header is untrusted input: it is only ever used to look up a known gateway slug, never executed or trusted as a credential.
- **Fallback model stays `anthropic/claude-sonnet-4.6`** (the SP-A default); `compaction: { thresholdPercent: 0.75 }` on `agent.ts` is preserved.
- **Confirmed Eve API:** `defineAgent`/`defineDynamic` from `eve`; model resolver returns a gateway model-id string or `null`; `SessionAuthContext = { attributes: Record<string,string|string[]>; subject?: string; ... }`; a custom `AuthFn<Request>` (from `eve/channels/auth`) receives the `Request`, may read headers, and returns a `SessionAuthContext` (accept) or `null` (skip). Resolver reads `ctx.session.auth.initiator?.attributes`. Re-verify any detail against `node_modules/eve/docs/agent-config.md` + `guides/auth-and-route-protection.md` at author time.

---

### Task 1: Id → gateway-slug map (pure, TDD)

**Files:**
- Create: `lib/ai/eve/model-map.ts`
- Test: `tests/agent/eve-model-map.test.ts`

**Interfaces:**
- Consumes: nothing.
- Produces: `toGatewaySlug(modelOverrideId?: string | null): string | undefined`.

- [ ] **Step 1: Write the failing test**

Create `tests/agent/eve-model-map.test.ts`:
```ts
import { describe, it, expect } from 'vitest';
import { toGatewaySlug } from '@/lib/ai/eve/model-map';

describe('toGatewaySlug', () => {
  it('maps Claude picker ids to gateway slugs', () => {
    expect(toGatewaySlug('claude-opus-4-8')).toBe('anthropic/claude-opus-4.8');
    expect(toGatewaySlug('claude-opus-4-7')).toBe('anthropic/claude-opus-4.7');
    expect(toGatewaySlug('claude-sonnet-4-6')).toBe('anthropic/claude-sonnet-4.6');
    expect(toGatewaySlug('claude-haiku-4-5')).toBe('anthropic/claude-haiku-4.5');
  });
  it('maps the gpt-5.4 family to gateway slugs', () => {
    expect(toGatewaySlug('gpt-5.4')).toBe('openai/gpt-5.4');
    expect(toGatewaySlug('gpt-5.4-pro')).toBe('openai/gpt-5.4-pro');
    expect(toGatewaySlug('gpt-5.4-mini')).toBe('openai/gpt-5.4-mini');
    expect(toGatewaySlug('gpt-5.4-nano')).toBe('openai/gpt-5.4-nano');
  });
  it('returns undefined for unmapped / base / empty ids', () => {
    expect(toGatewaySlug('chat-model')).toBeUndefined();
    expect(toGatewaySlug('chat-model-reasoning')).toBeUndefined();
    expect(toGatewaySlug('')).toBeUndefined();
    expect(toGatewaySlug(undefined)).toBeUndefined();
    expect(toGatewaySlug(null)).toBeUndefined();
    expect(toGatewaySlug('something-unknown')).toBeUndefined();
  });
});
```

- [ ] **Step 2: Run it — expect FAIL** (module missing)

```bash
export PATH="$HOME/.nvm/versions/node/v24.18.0/bin:$PATH"
pnpm exec vitest run -c vitest.config.node.mjs tests/agent/eve-model-map.test.ts
```

- [ ] **Step 3: Implement**

Create `lib/ai/eve/model-map.ts`:
```ts
// Maps the dev model-override picker ids (lib/ai/providers.ts customProvider +
// lib/ai/models.ts) to AI Gateway model slugs Eve routes through. Unmapped ids
// return undefined so the caller sends no model header and Eve uses its
// fallback (anthropic/claude-sonnet-4.6). Slugs are dot-versioned gateway ids
// (verify against the AI Gateway model catalog).
const MODEL_MAP: Record<string, string> = {
  'claude-opus-4-8': 'anthropic/claude-opus-4.8',
  'claude-opus-4-7': 'anthropic/claude-opus-4.7',
  'claude-sonnet-4-6': 'anthropic/claude-sonnet-4.6',
  'claude-haiku-4-5': 'anthropic/claude-haiku-4.5',
  'gpt-5.4': 'openai/gpt-5.4',
  'gpt-5.4-pro': 'openai/gpt-5.4-pro',
  'gpt-5.4-mini': 'openai/gpt-5.4-mini',
  'gpt-5.4-nano': 'openai/gpt-5.4-nano',
};

export function toGatewaySlug(modelOverrideId?: string | null): string | undefined {
  if (!modelOverrideId) return undefined;
  return MODEL_MAP[modelOverrideId];
}
```
Note: confirm each slug exists in the AI Gateway catalog at author time. If a slug is not offered by the gateway (e.g. `claude-opus-4.7`), REMOVE that entry from the map (it falls through to `undefined` → fallback) rather than ship an id the gateway rejects — and drop its test assertion.

- [ ] **Step 4: Run it — expect PASS**

```bash
pnpm exec vitest run -c vitest.config.node.mjs tests/agent/eve-model-map.test.ts
```

- [ ] **Step 5: Commit**

```bash
git add lib/ai/eve/model-map.ts tests/agent/eve-model-map.test.ts
git commit -m "feat(eve-model): map dev picker ids to AI Gateway slugs (toGatewaySlug)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Eve side — header→attribute AuthFn + dynamic model resolver (the crux; verify live)

**Files:**
- Modify: `agent/channels/eve.ts`
- Modify: `agent/agent.ts`

**Interfaces:**
- Consumes: the `x-eve-model` request header (sent by Task 3).
- Produces: an Eve agent whose model is chosen per session from `ctx.session.auth.initiator?.attributes?.eveModel`, else `anthropic/claude-sonnet-4.6`.

- [ ] **Step 1: Confirm the AuthFn + SessionAuthContext + defineDynamic shapes**

```bash
grep -rEn "interface SessionAuthContext" node_modules/eve/dist/src/channel/types.d.ts
sed -n '58,70p' node_modules/eve/dist/src/channel/types.d.ts
grep -rEn "AuthFn|withAuthChallenges|localDev" node_modules/eve/dist/src/public/channels/auth.d.ts | head
```
Confirm: `SessionAuthContext` requires `attributes` (a `Record<string,string|string[]>`), `subject` optional; `AuthFn<Request>` is `(request: Request) => SessionAuthContext | null | Promise<...>`. Adjust the code below to the exact required fields if more than `attributes`/`subject` are mandatory.

- [ ] **Step 2: Add the loopback-gated model-attribute AuthFn to `agent/channels/eve.ts`**

Prepend a custom `AuthFn` BEFORE the existing entries; it accepts ONLY loopback requests carrying the header and attaches `attributes.eveModel`, else returns `null` (existing `localDev`/`vercelOidc`/`placeholderAuth` handle everything else unchanged):
```ts
import { eveChannel } from 'eve/channels/eve';
import { type AuthFn, localDev, placeholderAuth, vercelOidc } from 'eve/channels/auth';

// Dev/eval only: on a loopback request (same trust boundary as localDev), read
// the adapter's `x-eve-model` header and expose it as an auth attribute the
// dynamic model resolver reads. Returns null for anything non-loopback or
// header-less, so the existing auth walk is unchanged for all other traffic.
// The header value is untrusted — it is only ever looked up as a known gateway
// slug in the adapter, never used as a credential here.
const modelAttributeAuth: AuthFn<Request> = (request) => {
  const host = new URL(request.url).hostname;
  const isLoopback =
    host === 'localhost' ||
    host.endsWith('.localhost') ||
    host === '127.0.0.1' ||
    host === '::1' ||
    host.startsWith('127.');
  const model = request.headers.get('x-eve-model');
  if (!isLoopback || !model) return null;
  return { subject: 'local-dev', attributes: { eveModel: model } };
};

export default eveChannel({
  auth: [
    modelAttributeAuth,
    vercelOidc(),
    localDev(),
    placeholderAuth(),
  ],
});
```

- [ ] **Step 3: Switch `agent/agent.ts` to a dynamic model**

```ts
import { defineAgent, defineDynamic } from 'eve';

// Model resolves through Vercel AI Gateway. Default is sonnet-4.6; the dev
// model picker can override it per session via the x-eve-model header, which
// agent/channels/eve.ts surfaces as auth attribute `eveModel` (dev/eval only).
export default defineAgent({
  model: defineDynamic({
    fallback: 'anthropic/claude-sonnet-4.6',
    events: {
      'session.started': (_event, ctx) =>
        ctx.session.auth.initiator?.attributes?.eveModel ??
        ctx.session.auth.current?.attributes?.eveModel ??
        null,
    },
  }),
  // Eve manages context compaction internally (no prepareStep hook) — see
  // docs/eve-spike-findings.md Q2.
  compaction: {
    thresholdPercent: 0.75,
  },
});
```
Note: `attributes.eveModel` is typed `string | string[]`; if TS complains about the union where a `string` is required, coerce with `Array.isArray(x) ? x[0] : x`. Confirm `defineDynamic` import is from `eve` (agent-config doc) — if the installed package exports it elsewhere, adjust.

- [ ] **Step 4: VERIFY LIVE (the crux)**

Start the agent (Node 24, env loaded):
```bash
export PATH="$HOME/.nvm/versions/node/v24.18.0/bin:$PATH"; set -a; . ./.env.local; set +a
npx eve dev --no-ui --port 2010
```
In another shell, create a session WITH the model header and confirm the turn runs that model:
```bash
curl -sD - -X POST http://127.0.0.1:2010/eve/v1/session \
  -H 'content-type: application/json' -H 'x-eve-model: anthropic/claude-opus-4.8' \
  -d '{"message":"In one short sentence, what are you?"}' -o /dev/null | tr -d '\r' | grep -i x-eve-session-id
# then read /eve/v1/session/<id>/stream and inspect step.completed / model metadata
```
Expected: the turn's model is `anthropic/claude-opus-4.8` (visible in `step.completed` usage/model metadata or the dev-server log line for the model call). Then repeat WITHOUT the header and confirm the model is the `anthropic/claude-sonnet-4.6` fallback. **If the header does not reach `ctx.session.auth.*.attributes.eveModel`** (resolver sees `null` even with the header), STOP and report BLOCKED with what `ctx.session.auth` actually contained — the plumbing needs a different mechanism (channel `metadata(state)` projection) before Task 3 builds on it. Redact secrets; kill the server. Record both observations.

- [ ] **Step 5: Commit**

```bash
git add agent/channels/eve.ts agent/agent.ts
git commit -m "feat(eve-model): per-session dynamic model from x-eve-model header (dev/eval)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Adapter conveys the slug

**Files:**
- Modify: `lib/ai/eve/eve-client.ts`
- Modify: `app/(chat)/api/eve-chat/route.ts`

**Interfaces:**
- Consumes: `toGatewaySlug` (Task 1); the `x-eve-model` header path (Task 2).
- Produces: `createEveSession(message: string, model?: string)` sending `x-eve-model` when `model` is set; the route mapping `body.modelOverride` → slug → `createEveSession`.

- [ ] **Step 1: Add an optional model to `createEveSession`**

In `lib/ai/eve/eve-client.ts`, thread an optional model header through the create path. Update `postJson` to accept optional extra headers, and `createEveSession`:
```ts
async function postJson(path: string, body: unknown, extraHeaders?: Record<string, string>) {
  const res = await fetch(`${EVE_URL}${path}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', ...(extraHeaders ?? {}) },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(`Eve server ${path} responded ${res.status}: ${await res.text().catch(() => '')}`);
  }
  const sessionId = res.headers.get('x-eve-session-id') ?? '';
  const json = (await res.json().catch(() => ({}))) as { continuationToken?: string };
  return { sessionId, continuationToken: json.continuationToken ?? '' };
}

export async function createEveSession(message: string, model?: string) {
  const { sessionId, continuationToken } = await postJson(
    '/eve/v1/session',
    { message },
    model ? { 'x-eve-model': model } : undefined,
  );
  return { sessionId, continuationToken };
}
```
Leave `continueEveSession` unchanged (model is fixed at session creation — session-scoped).

- [ ] **Step 2: Map + pass the override in the route**

In `app/(chat)/api/eve-chat/route.ts`: import `toGatewaySlug`; widen the body type to include `modelOverride`; compute the slug and pass it to `createEveSession`. Only the NEW-session branch takes the model (continuation is unchanged):
```ts
import { toGatewaySlug } from '@/lib/ai/eve/model-map';
// widen body type:
//   let body: { id?: string; message?: { role?: string }; modelOverride?: string };
// in the create branch:
const model = toGatewaySlug(body.modelOverride);
const created = await createEveSession(text, model);
```
Change only the create branch + the body type + the import; leave continuity, auth, validation, and the stream loop exactly as they are.

- [ ] **Step 3: Typecheck**

```bash
export PATH="$HOME/.nvm/versions/node/v24.18.0/bin:$PATH"
pnpm exec tsc --noEmit 2>&1 | grep -E "eve-client|eve-chat|model-map" || echo "SP model-selector files typecheck clean"
```
Expected: no errors referencing these files (7 pre-existing baseline errors unrelated).

- [ ] **Step 4: Commit**

```bash
git add lib/ai/eve/eve-client.ts "app/(chat)/api/eve-chat/route.ts"
git commit -m "feat(eve-model): adapter maps modelOverride -> gateway slug -> x-eve-model header

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: End-to-end verification + docs

**Files:**
- Modify: `agent/README.md`

**Interfaces:**
- Consumes: the whole feature.
- Produces: a verified path + docs.

- [ ] **Step 1: Full path verification**

Run `npx eve dev --no-ui --port 2000` (Node 24, env loaded) + `EVE_SERVER_URL=http://127.0.0.1:2000 pnpm dev`. In the browser (non-prod): enable `useEveAgent`, set the dev model picker to a mapped model (e.g. Claude Opus 4.8), start a NEW chat, send a message. Confirm the Eve turn ran that model (Eve dev-server log / model metadata). Then set the picker to a model and confirm a DIFFERENT new chat uses it; with the picker unset/base, confirm the sonnet-4.6 fallback. Record observations (redact secrets). If browser auth is awkward, at minimum re-run Task 2 Step 4's curl (header → model) against :2000 and confirm via the adapter that `toGatewaySlug` + header wiring produce the same result.

- [ ] **Step 2: Regression + additive**

```bash
export PATH="$HOME/.nvm/versions/node/v24.18.0/bin:$PATH"
pnpm exec vitest run -c vitest.config.node.mjs tests/agent/    # all pass incl. new model-map
git diff --stat 3d9d9e6..HEAD -- "app/(chat)/api/chat/route.ts" lib/ai/providers.ts components/chat.tsx lib/kernel lib/ai/tools agent/tools agent/skills
```
Expected: all tests pass; the diff is EMPTY (none of the legacy/forbidden files changed). Confirm `components/chat.tsx` is untouched (it already sends `modelOverride` non-prod).

- [ ] **Step 3: Document**

Add a short "Model selection (dev/eval)" note to `agent/README.md`: the dev picker's model drives Eve non-prod via `modelOverride` → `toGatewaySlug` → `x-eve-model` header → `agent/channels/eve.ts` auth attribute → `agent/agent.ts` `defineDynamic` resolver; it's session-scoped (change applies on the next new chat); unmapped/prod → `sonnet-4.6` fallback. No marketing language.

- [ ] **Step 4: Commit**

```bash
git add agent/README.md
git commit -m "docs(eve-model): document dev/eval model selection for the Eve agent

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review notes

- **Spec coverage:** id→slug map → Task 1; adapter conveyance (`modelOverride`→slug→header) → Task 3; Eve dynamic-model resolver + header→attribute plumbing → Task 2 (with the live crux verification the spec required as the "first step" de-risk); session-scoped (create-only header, `session.started`) → Tasks 2/3; fallback `sonnet-4.6` → Task 2; dev/eval-only (rides non-prod `modelOverride`, loopback-gated AuthFn) → Global Constraints + Task 2; validation + additive → Task 4.
- **Plumbing grounded, not open:** the header→resolver path is the documented custom-`AuthFn` → `attributes` → `ctx.session.auth.*.attributes` pattern (Eve's own dynamic-model example), so Task 2 is a grounded build-and-verify, with an explicit BLOCKED path if the header doesn't surface.
- **Placeholder scan:** all code steps are complete; the only "verify at author time" items (exact gateway slugs, `SessionAuthContext` required fields, `defineDynamic` import path) are genuine beta/catalog confirmations with concrete fallback instructions, not placeholders.
- **Type/name consistency:** `toGatewaySlug` (Task 1) consumed in Task 3; the attribute key `eveModel` is written in `agent/channels/eve.ts` and read in `agent/agent.ts` (Task 2) identically; the header name `x-eve-model` is identical in `eve-client.ts` (Task 3) and `agent/channels/eve.ts` (Task 2); fallback slug `anthropic/claude-sonnet-4.6` matches SP-A.
- **Additive/safety:** no legacy file (`route.ts`, providers, chat.tsx) changes; the AuthFn only accepts loopback and treats the header as untrusted; prod stays on fallback because `modelOverride` is sent only non-prod.
