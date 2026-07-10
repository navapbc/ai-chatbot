# AI SDK v7 Upgrade Analysis

Analysis of upgrading the `ai-chatbot` client from Vercel AI SDK **v6 → v7**.
Branch: `analysis/ai-sdk-v7-upgrade`. Date: 2026-07-09.

All version and API claims below were verified against the published npm manifests
and the actual `ai@7.0.18` / `@ai-sdk/react@4.0.19` `.d.ts` type definitions — not
from documentation summaries alone.

## Implementation status (2026-07-09)

The upgrade was executed on this branch and **`next build` passes** under Node 24.

- Packages bumped to the v7 matrix below, plus `@onkernel/sdk` 0.62 → **0.76.0**
  (the Kernel API client; `agent-browser` is a Vercel Labs package, left untouched).
- Ran `npx @ai-sdk/codemod@4 v7` for the symbol renames, then hand-fixed the token
  fields (the codemod's token transforms were reverted/redone as predicted below).
- **`pnpm exec tsc --noEmit`**: all application/library code compiles clean under v7.
- **`pnpm exec next build`**: ✓ compiled, TypeScript check passed, 24/24 pages generated.

Manual fixes applied beyond the codemod's renames:
- `app/(chat)/api/chat/route.ts` — map the v7 SDK usage to the flat `data-token-usage`
  wire shape at the emit point (`usage.inputTokenDetails?.cacheReadTokens`), keeping the
  client contract stable. (`onStepFinish`→`onStepEnd` and the `createUIMessageStream`
  `onFinish`→`onEnd` renames verified valid in v7.)
- Reverted the codemod's token edits on the app's own token types (`chat.tsx` accumulator,
  `ai-elements/context.tsx` local `getUsage`, `context-usage.tsx`, `side-chat-header.tsx`,
  `evals/pricing.ts`) — these keep flat `cachedInputTokens`/`reasoningTokens` names.
- `chat.tsx` — reverted `useChat`'s `onFinish` (codemod wrongly renamed it to `onEnd`;
  `useChat` in `@ai-sdk/react@4` still uses `onFinish`).
- `evals/helpers.ts` — updated the `HasTotalUsage` interface to the v7 nested shape and
  mapped `result.totalUsage.inputTokenDetails?.cacheReadTokens` into the flat accumulator.
- `lib/ai/tools/browser.ts`, `check-submit-gate.ts` — `ToolCallOptions` → `ToolExecutionOptions`
  is now generic; annotated the options param structurally (`{ abortSignal?: AbortSignal }`)
  to avoid poisoning `tool()`'s input inference with a `never` type argument.

Not addressed by this task (flagged, not blocking the build):
- Two **pre-existing, unrelated** type errors surface under a full `tsc --noEmit` /
  `pnpm test`: `tests/client/consent-page.test.tsx` (stale `onBack` prop) and
  `tests/e2e/session.test.ts` (`Request | null`). They exist unchanged at `HEAD` and do
  not block `next build` (Next 16 type-checks the build graph, not the test files).
- A green build proves **compilation**, not runtime. The `@ai-sdk/google-vertex` 4→5
  path (`vertexAnthropic` auth/model-ids) and the Kernel 0.76 browser calls are not
  exercised by a build — smoke-test the web-automation flow and evals before shipping.

## Verdict

A **feasible, medium-effort** upgrade. The codebase already uses the modern v5/v6
API shape (`streamText`, `generateText`, `tool`, `UIMessage`, `convertToModelMessages`,
`stepCountIs`, `customProvider`), so there is no architectural rewrite. The work is:

1. One genuine **environment requirement** — Node 22+ (currently on Node 20/18).
2. A **mechanical bulk** — relocated usage-token fields (~30 sites, 7 files).
3. A **residual unknown to de-risk** — the provider packages all jump a major version.

Most symbol renames the migration guide lists (`stepCountIs`, `onFinish`, `system`,
`experimental_generateImage`, `experimental_telemetry`, `experimental_throttle`,
`experimental_transform`) **still work as deprecated aliases in v7**, so they do not
block compilation or runtime — a codemod cleans them up.

## Target version matrix

| Package | Current | Target (v7 `latest`) | Bump |
|---|---|---|---|
| `ai` | 6.0.116 | ^7.0.18 | major |
| `@ai-sdk/react` | 3.0.118 | ^4.0.19 | major |
| `@ai-sdk/provider` | 3.0.8 | ^4.0.2 | major |
| `@ai-sdk/anthropic` | ^3.0.58 | ^4.0.10 | major |
| `@ai-sdk/google` | ^3.0.43 | ^4.0.10 | major |
| `@ai-sdk/google-vertex` | ^4.0.80 | ^5.0.13 | major |
| `@ai-sdk/openai` | ^3.0.41 | ^4.0.9 | major |
| `@ai-sdk/gateway` | ^3.0.66 | ^4.0.14 | major |
| `@ai-sdk/xai` | 3.0.67 | ^4.0.9 | major |

Peer dependencies are already satisfied — **no forced React or zod bump**:
- `react` 19.0.1 satisfies `@ai-sdk/react@4`'s peer (`^18 || ~19.0.1 || ~19.1.2 || ^19.2.1`).
- `zod` ^3.25.76 satisfies the providers' peer (`^3.25.76 || ^4.1.8`).

> **Caveat — narrow React peer.** The `@ai-sdk/react@4` peer pins React tightly
> (`~19.0.1`, i.e. 19.0.x only, plus specific 19.1/19.2 patch ranges). It is
> satisfied at the current exact 19.0.1, but any drift to 19.1.x/19.2.x would need a
> matching bump. Treat this as a real pin, not open-ended compatibility.

## Hard breaks (must fix)

### 1. Node 22+ and ESM-only — the one real environment change
`ai@7` sets `engines.node: ">=22"` and ships `"type": "module"` (ESM-only). `ai@6`
was `>=18`. Current state:
- Local dev: Node **v20.15.0**
- CI: `.github/workflows/test.yml` and `lint.yml` pin `node-version: [20]`;
  `evals.yml` uses `20`; the repo-root `code-quality.yml` uses `18`.
- No `.nvmrc`, no `engines` field, no client Dockerfile found.

**Action:** bump all CI workflows and the runtime (Cloud Run / build image) to Node 22;
add `.nvmrc` and an `engines` field. ESM-only is low risk for this app — it is a
Next.js 16 project that already imports `ai` via ESM, and Next bundles dependencies.
The only place to sanity-check is the eval runner (`braintrust eval evals/*.eval.ts`)
and any standalone Node scripts that might `require('ai')`.

### 2. Usage-token fields relocated — the mechanical bulk
These two properties were **removed** from `LanguageModelUsage` and moved under detail
objects. Both removal *and* replacement paths were verified against the v7 `.d.ts`
(`inputTokenDetails.cacheReadTokens` and `outputTokenDetails.reasoningTokens` both
present; `cachedInputTokens`/top-level `reasoningTokens` both absent):

| v6 | v7 |
|---|---|
| `usage.cachedInputTokens` | `usage.inputTokenDetails.cacheReadTokens` |
| `usage.reasoningTokens` | `usage.outputTokenDetails.reasoningTokens` |

These are compile-time breaks (TypeScript errors on the removed properties). Distribution
(27 sites across 7 files), concentrated in token-usage UI + evals:
- `cachedInputTokens` (19): `components/chat.tsx` (5), `evals/helpers.ts` (4),
  `evals/pricing.ts` (3), `hooks/use-token-usage.tsx` (2), `components/context-usage.tsx` (2),
  `components/side-chat-header.tsx` (2), `components/ai-elements/context.tsx` (1)
- `reasoningTokens` (8): `components/ai-elements/context.tsx` (6),
  `components/context-usage.tsx` (1), `components/side-chat-header.tsx` (1)

## Non-issues (verified, no action)

- **`appendResponseMessages` (removed in v7)** — the only two references are in
  **commented-out code** in `lib/db/helpers/01-core-to-parts.ts`. No runtime use.
- **`experimental_transform`** — still the valid `streamText` option name in v7
  (used in `artifacts/text/server.ts`). No change required.
- **`experimental_throttle`** — still accepted by `useChat` in `@ai-sdk/react@4`
  (deprecated alias of `throttle`; used in `components/chat.tsx`).

## Deprecation cleanup (non-blocking; codemod-handled)

All of these still resolve in v7 via deprecated aliases — they compile and run, but
should be renamed for forward-compatibility:

| Used in code | Count | v7 preferred |
|---|---|---|
| `stepCountIs` | 24 | `isStepCount` (exported as `isStepCount as stepCountIs`) |
| `system:` option | ~17 files | `instructions:` |
| `experimental_generateImage` | 3 (`artifacts/image/server.ts`) | `generateImage` |
| `experimental_telemetry` | 1 (`app/(chat)/api/chat/route.ts`) | `telemetry` |
| `onFinish` / `onStepFinish` | 2 / 2 | `onEnd` / `onStepEnd` |
| `result.totalUsage` | 3 (`evals/helpers.ts`) | `result.usage` (verified still present in v7 as deprecated; not a break) |

Note: v7 rejects `system` messages inside `messages`/`prompt` arrays by default
(opt-in via `allowSystemInMessages: true`); top-level `system`/`instructions` are
unaffected. Verify none of our message-construction paths inject a system role.

## Largest residual unknown — provider major bumps

**This is the biggest thing that could turn a clean bump into a debugging session.**
The direct provider call surface is small and centralized — only two files:
- `lib/ai/providers.ts`: `customProvider`, `wrapLanguageModel`,
  `extractReasoningMiddleware` (from `ai`), `openai`, `openai.image('dall-e-3')`,
  and `vertexAnthropic` from `@ai-sdk/google-vertex/anthropic`.
- `evals/helpers.ts`: `anthropic`, `google`, `openai` factories.

But "small surface" ≠ "low risk": `@ai-sdk/anthropic`, `google`, `openai` go 3→4 and
`google-vertex` goes 4→5. Provider majors are exactly where model-id defaults, auth
handling, and option shapes change. **These provider changelogs are unreviewed** and
must be read before the bump is trusted. Google Vertex (the primary web-automation
path, `vertexAnthropic('claude-opus-4-7')`) crossing two majors deserves the closest look.

## Codemod

`@ai-sdk/codemod@4.0.0` exists. Run: `npx @ai-sdk/codemod@4 v7 <dirs>`.

**Measured, not assumed:** the codemod was run against this branch's source
(`app lib components hooks artifacts evals`) and the resulting diff inspected
(25 files, 63 insertions / 63 deletions), then reverted. Findings:

- **Symbol renames — accept these.** Clean, correct, pure renames:
  `system:` → `instructions:`, `stepCountIs` → `isStepCount`,
  `experimental_generateImage` → `generateImage`,
  `experimental_telemetry` → `telemetry`, `onFinish` → `onEnd` (and `onStepFinish`
  → `onStepEnd`). Touched the chat route, all four artifact servers,
  `context-compression.ts`, `request-suggestions.ts`, `actions.ts`, and all eval files.

- **Token-field transform — DO NOT accept wholesale; do it manually.** This is the
  headline risk. The app maintains its **own** token-usage domain model whose field
  names intentionally mirror the old SDK names — the `useTokenUsage` hook
  (`hooks/use-token-usage.tsx`), the accumulator in `components/chat.tsx`, the pricing
  type in `evals/pricing.ts`, and the props in `components/ai-elements/context.tsx` all
  declare `cachedInputTokens` / `reasoningTokens` fields of their own. The codemod
  cannot tell an SDK `usage` object from one of these look-alikes, and rewrote them
  inconsistently, producing broken code:
  - `components/chat.tsx`: rewrote `prev.cachedInputTokens` →
    `prev.inputTokenDetails.cacheReadTokens`, but `prev` is the app's own accumulator
    (still typed `cachedInputTokens: number`) — a type error, not a fix.
  - `components/ai-elements/context.tsx`: rewrote `usage.reasoningTokens` →
    `usage.outputTokenDetails.reasoningTokens`, but the local `usage` prop type declares
    top-level `reasoningTokens?` — won't type-check.
  - `side-chat-header.tsx` / `context-usage.tsx`: emitted `reasoningTokens: undefined`
    — **silent data loss** (reasoning tokens no longer read from the SDK usage object).
  - It left `use-token-usage.tsx`'s type definitions on the old names while rewriting
    their consumers, so the hook and its callers are now out of sync.

  **Manual approach:** the token accounting is a self-contained subsystem (~7 files).
  Identify the one/few points where the raw SDK `usage` object is actually ingested
  (the correct transforms the codemod *did* get right are in `context-usage.tsx` and
  `side-chat-header.tsx`, reading `tokenUsage.inputTokenDetails.cacheReadTokens`), map
  the two nested fields there, and decide whether to keep the app's internal field
  names as-is. Do not let the codemod rewrite the app's own token types.

- **Not transformed** (still deprecated aliases, harmless to leave or rename by hand):
  `experimental_transform`, `experimental_throttle`, `totalUsage`.

## Suggested execution order

1. Bump Node to 22 across CI workflows + runtime; add `.nvmrc` / `engines`.
2. `pnpm up` the `ai` + `@ai-sdk/*` packages to the v7 matrix above.
3. Run the codemod; **accept the symbol renames, revert its token-field edits**
   (see "Codemod" — its token transforms are unsafe against the app's own token model).
4. Migrate the token-usage subsystem by hand at the true SDK ingestion points.
5. Read the provider (esp. `google-vertex` 4→5) changelogs; adjust `lib/ai/providers.ts`.
6. `pnpm build`, `pnpm lint`, `pnpm test`, and a smoke run of the chat agent + evals.
