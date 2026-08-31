import {
  VERTEX_CONTEXT_WINDOW_TOKENS,
  isVertexModelId,
  toVertexModelId,
} from '@/lib/ai/eve/model-map';
import { defineAgent, defineDynamic } from 'eve';

import { vertexAnthropic } from '@ai-sdk/google-vertex/anthropic';

// Models are called directly on Vertex AI, not routed through the Vercel AI
// Gateway: `vertexAnthropic(...)` is an AI SDK `LanguageModel`, which Eve
// classifies as `external` routing and hands straight to the provider. This is
// the same credential path the legacy chat route already uses in production
// (lib/ai/providers.ts), and it avoids the gateway tier that 403s opus/haiku on
// this account. Auth comes from GOOGLE_APPLICATION_CREDENTIALS +
// GOOGLE_VERTEX_PROJECT/GOOGLE_VERTEX_LOCATION — start the server with
// `pnpm eve:dev`, which loads .env.local (bare `eve dev` does not).
// GOOGLE_VERTEX_LOCATION must be `global` for the opus models; see
// agent/README.md "Vertex region". AI_GATEWAY_API_KEY is no longer needed.
//
// Eve manages context compaction internally (there is no prepareStep hook) —
// configure it here rather than porting lib/ai/context-compression.ts. See
// docs/eve-spike-findings.md Q2.
const DEFAULT_MODEL_ID = 'claude-opus-4.8';

// The dev model picker can override the model per session via the x-eve-model
// header, which agent/channels/eve.ts surfaces as auth attribute `eveModel`
// (dev/eval only, loopback-gated). The header carries a Vertex model id already
// mapped by lib/ai/eve/model-map.ts; it is untrusted input, so it is validated
// against that module's allowlist before being turned into a provider instance.
//
// This resolves on `step.started` rather than `session.started` because Eve only
// accepts live `LanguageModel` objects at step scope — session- and turn-scoped
// selections must be serializable model id strings, which would mean gateway
// routing. The selection is re-derived every step from session-scoped auth
// attributes, so it stays stable for the life of the session.
const resolveModelId = (value: unknown): string | null => {
  const requested = Array.isArray(value) ? (value[0] ?? null) : value;
  if (isVertexModelId(requested)) return requested;
  // Accept a raw picker id too, so a hand-rolled curl against `eve dev` does
  // not have to know the app-side mapping. Still allowlist-gated: an id with no
  // MODEL_MAP entry returns undefined and falls through to the fallback.
  if (typeof requested !== 'string') return null;
  return toVertexModelId(requested) ?? null;
};

export default defineAgent({
  model: defineDynamic({
    fallback: vertexAnthropic(DEFAULT_MODEL_ID),
    events: {
      'step.started': (_event, ctx) => {
        const modelId = resolveModelId(
          ctx.session.auth.initiator?.attributes?.eveModel ??
            ctx.session.auth.current?.attributes?.eveModel ??
            null,
        );
        if (modelId === null || modelId === DEFAULT_MODEL_ID) return null;
        return {
          model: vertexAnthropic(modelId),
          modelContextWindowTokens: VERTEX_CONTEXT_WINDOW_TOKENS,
        };
      },
    },
  }),
  // Required: Eve cannot resolve a direct Vertex model's context window from the
  // AI Gateway catalog and refuses to compile without this. See
  // VERTEX_CONTEXT_WINDOW_TOKENS for why it is 200K and not 1M.
  modelContextWindowTokens: VERTEX_CONTEXT_WINDOW_TOKENS,
  compaction: {
    // Compact when context passes this fraction of the window (default 0.9).
    thresholdPercent: 0.75,
  },
  experimental: {
    workflow: {
      // Durable session state lives in Postgres, not on container disk.
      //
      // Eve's default Workflow world persists runs to `.eve/.workflow-data`.
      // That is per-instance and, on Cloud Run, in-memory tmpfs — so a session
      // dies with the instance that served it. This service runs at
      // min_instance_count = 2 / max 20 with best-effort session affinity
      // (terraform/cloud_run.tf), which means instance churn is routine and
      // durable sessions would silently vanish.
      //
      // world-postgres reads WORKFLOW_POSTGRES_URL, falling back to
      // DATABASE_URL — already wired from Secret Manager — so it lands on the
      // same Cloud SQL instance the app uses. The schema is created by
      // scripts/bootstrap-workflow-db.ts at container start.
      //
      // PINNED to @workflow/world-postgres@5.0.0-beta.33 — do not bump without
      // re-checking. Compatibility is by World *spec version*, not by npm
      // version, and the two have diverged:
      //
      //   eve 0.27.13 vendors world-local at spec 5 and hard-fails a World
      //   declaring anything else ("requires a World with matching spec
      //   version 5, but the configured World declares spec version 7").
      //
      //   world-postgres takes its spec from its @workflow/world dep:
      //     beta.33 -> world beta.26 -> SPEC_VERSION_CURRENT = 5   ✅
      //     beta.34 -> world beta.27 -> SLOT_IDENTITY        = 6   ❌
      //     beta.38 -> world beta.31 -> mintedSpecVersion()  = 7   ❌
      //
      // So beta.33 is the newest usable release, and "latest" is broken. The
      // env kill switch on the newer packages only drops 7 to 6, never to 5.
      // Revisit when eve itself moves off spec 5.
      world: '@workflow/world-postgres',
    },
  },
});
