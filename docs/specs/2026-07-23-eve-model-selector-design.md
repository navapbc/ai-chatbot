# Model Selector Drives Eve's Model (Design)

Status: approved for implementation planning
Date: 2026-07-23
Branch: `feat/eve-integration`

## Context

SP-B wired the Eve agent into the chat UI behind the `useEveAgent` flag via the
`/api/eve-chat` adapter route. The dev model-override picker (`ModelSelectorButton`
→ `selected-chat-model-id` in localStorage → sent as `modelOverride` in the request
body, **non-production only**) drives model choice on the legacy `/api/chat` path
(`route.ts` uses `myProvider.languageModel(modelOverride)`). The Eve adapter currently
**ignores** `modelOverride`; Eve always runs the hardcoded `anthropic/claude-sonnet-4.6`
from `agent/agent.ts`.

This feature makes the picker drive Eve's model too. It is a follow-on enhancement to
SP-B (not a new sub-project in the SP-A→SP-D sequence).

Eve routes models through the AI Gateway and its model resolver is `defineDynamic({ fallback, events })`
on `agent.ts`'s `model` field: resolvers run at `session.started`/`turn.started`/`step.started`,
receive `ctx.session`/`ctx.channel`/`ctx.messages`, and return a gateway model-id string
(or `null` to leave the scope unset). The picker's ids (`claude-sonnet-4-6`, `gpt-5.4`, …)
are NOT gateway slugs, so a mapping is required.

## Purpose

When the dev model picker selects a model (non-prod), the Eve agent uses the matching
AI Gateway model for that chat, instead of the hardcoded `sonnet-4.6`.

## Goals (exit criteria)

1. With `useEveAgent` on (non-prod) and the picker set to a supported model, an Eve turn
   runs THAT model — verified via Eve's runtime model identity / `step.completed`.
2. With no override (or an unmapped id), Eve uses its `sonnet-4.6` fallback.
3. Flag-OFF (legacy) path and the default Eve model are unchanged.

## Non-goals (deferred)

- Production / user-facing model selection (this rides the existing non-prod `modelOverride`
  gate; no prod curation, no per-chat DB persistence).
- Per-turn / mid-chat model switching. Model is fixed per Eve **session** (`session.started`);
  changing the picker mid-chat takes effect on the **next new chat**. (Per-turn switching
  re-ingests context at uncached prices per Eve docs — intentionally avoided.)
- Changing the legacy `/api/chat` model path in any way.
- Adding new picker models — only the existing set is mapped.

## Decisions (confirmed with stakeholder)

- **Scope:** dev/eval only — mirror the existing non-prod `modelOverride`.
- **Model set:** map the existing picker ids to gateway slugs (Claude opus-4.8 / sonnet-4.6 /
  haiku-4.5 + the gpt-5.4 family).
- **Mechanism:** adapter maps `modelOverride` → gateway slug and conveys it to Eve; Eve applies
  it via a `defineDynamic` model resolver. The exact header→resolver plumbing is resolved by a
  spike as the plan's first step.
- **Session-scoped** model (`session.started`), not per-turn.

## Components

### 1. Id → gateway-slug map — `lib/ai/eve/model-map.ts` (new, pure, TDD)
`toGatewaySlug(modelOverrideId?: string): string | undefined`:
- `claude-opus-4-8` → `anthropic/claude-opus-4.8`
- `claude-opus-4-7` → `anthropic/claude-opus-4.7`
- `claude-sonnet-4-6` → `anthropic/claude-sonnet-4.6`
- `claude-haiku-4-5` → `anthropic/claude-haiku-4.5`
- `gpt-5.4` → `openai/gpt-5.4`; `gpt-5.4-pro` → `openai/gpt-5.4-pro`; `gpt-5.4-mini` →
  `openai/gpt-5.4-mini`; `gpt-5.4-nano` → `openai/gpt-5.4-nano`
- anything else (`chat-model`, `chat-model-reasoning`, empty/undefined) → `undefined`.

The gateway slugs are confirmed against the AI Gateway catalog at implementation; any slug
Eve/gateway does not recognize is omitted from the map (falls through to `undefined` →
Eve's fallback) rather than shipped broken.

### 2. Adapter conveys the slug — `app/(chat)/api/eve-chat/route.ts` + `lib/ai/eve/eve-client.ts`
- The route reads `modelOverride` from the request body (already sent by `chat.tsx` only in
  non-prod), calls `toGatewaySlug`, and passes the result (if defined) to `createEveSession`.
- `eve-client.ts`'s `createEveSession` gains an optional `model?: string` param; when present
  it sends the slug on the session-create request as the `x-eve-model` header (exact carrier
  confirmed by the spike; header is the default plan).
- Continuation (`continueEveSession`) does NOT re-send the model — model is fixed at session
  creation (session-scoped decision).

### 3. Eve applies it — `agent/agent.ts` (+ possibly `agent/channels/eve.ts`)
- `agent/agent.ts` `model` becomes `defineDynamic({ fallback: 'anthropic/claude-sonnet-4.6',
  events: { 'session.started': (_event, ctx) => <read the slug from ctx> ?? null } })`, keeping
  the existing `compaction` config.
- **Plumbing spike (plan step 1):** determine exactly how the `x-eve-model` header reaches the
  resolver's `ctx` under the built-in `eve` channel. Candidate mechanisms, in preference order:
  1. A custom `AuthFn` on `agent/channels/eve.ts` that reads the header and returns an auth
     context whose `attributes` carry the model → resolver reads `ctx.session.auth.current?.attributes?.model`.
  2. A channel `metadata(state)` projection exposing the header → resolver reads `ctx.channel.metadata`.
  3. A documented fallback if neither surfaces a request header.
  The spike confirms which works against the installed `eve@0.27.0` before the resolver is
  finalized; the resolver reads from whichever the spike establishes.

## Data flow

```
dev picker → localStorage 'selected-chat-model-id' → chat.tsx body.modelOverride (non-prod)
  → POST /api/eve-chat
     toGatewaySlug(modelOverride)  ->  slug | undefined
     createEveSession(text, slug?)  ->  header x-eve-model: <slug>  (only on session create)
  → Eve session.started: defineDynamic resolver reads slug from ctx (spike-confirmed path)
     -> Eve runs that gateway model for the session (else fallback sonnet-4.6)
```

## Error handling

- Unmapped / absent model id → `toGatewaySlug` returns `undefined` → no header sent → Eve
  fallback. Never an error.
- A mapped slug the gateway rejects at request time → Eve's dynamic-model contract "fails
  degrade, never fail the turn" (logs, leaves scope unset → fallback). The adapter does not
  need special handling.
- Malformed header / resolver throw → Eve leaves the scope unset (fallback), per the same
  contract.

## Validation

- Unit: `toGatewaySlug` maps each supported id to its slug and returns `undefined` for
  unmapped/empty ids (node-config vitest).
- Manual/live: with `useEveAgent` on (non-prod), set the picker to `claude-opus-4-8`, start a
  NEW chat, send a message, and confirm the Eve turn ran opus-4.8 (Eve dev logs / runtime
  identity `dynamic:anthropic/claude-sonnet-4.6` fallback with the selected model in
  `step.completed`/model metadata). Unset override → sonnet-4.6. Flag-off → legacy unchanged.
- Additive: legacy `/api/chat`, `route.ts`, providers unchanged; the only `agent/` change is
  `agent.ts` (dynamic model) and possibly `agent/channels/eve.ts` (auth/metadata for the header).

## Risks

- **Header→resolver plumbing (primary).** Whether the built-in `eve` channel surfaces a
  request header to the dynamic resolver (via auth attributes or channel metadata) is
  unconfirmed. Mitigation: the plan's first step is a spike; if no clean path exists, fall
  back to a documented mechanism (e.g. a minimal custom auth verifier) or report that
  per-request model selection isn't expressible and revisit scope.
- **Gateway slug accuracy.** A wrong/unavailable slug fails at request time (degrades to
  fallback, not a crash) — but the picked model silently wouldn't apply. Mitigation: confirm
  slugs against the AI Gateway catalog; the live validation catches a mis-mapping.
- **Auth interaction.** Injecting model into auth attributes must not weaken the channel's
  actual authentication (the header is a hint, not a credential). The spike keeps auth
  verification intact and treats the model header as untrusted input mapped only to a known
  slug set.
- **Prod safety.** The whole path rides `modelOverride`, which `chat.tsx` sends only in
  non-prod; confirm no prod code path forwards it, so production Eve stays on the fallback.
