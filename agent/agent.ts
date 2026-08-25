import { defineAgent, defineDynamic } from 'eve';

// Model resolves through Vercel AI Gateway (AI_GATEWAY_API_KEY locally; OIDC on
// Vercel). Eve manages context compaction internally (there is no prepareStep
// hook) — configure it here rather than porting lib/ai/context-compression.ts.
// See docs/eve-spike-findings.md Q2.
//
// Default is sonnet-4.6; the dev model picker can override it per session via
// the x-eve-model header, which agent/channels/eve.ts surfaces as auth
// attribute `eveModel` (dev/eval only, loopback-gated).
export default defineAgent({
  model: defineDynamic({
    fallback: 'openai/gpt-5.4',
    events: {
      'session.started': (_event, ctx) => {
        const value =
          ctx.session.auth.initiator?.attributes?.eveModel ??
          ctx.session.auth.current?.attributes?.eveModel ??
          null;
        return Array.isArray(value) ? (value[0] ?? null) : value;
      },
    },
  }),
  compaction: {
    // Compact when context passes this fraction of the window (default 0.9).
    thresholdPercent: 0.75,
  },
});
