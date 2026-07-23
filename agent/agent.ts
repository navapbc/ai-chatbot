import { defineAgent } from 'eve';

// Model resolves through Vercel AI Gateway (AI_GATEWAY_API_KEY locally; OIDC on
// Vercel). Eve manages context compaction internally (there is no prepareStep
// hook) — configure it here rather than porting lib/ai/context-compression.ts.
// See docs/eve-spike-findings.md Q2.
export default defineAgent({
  model: 'anthropic/claude-sonnet-4.6',
  compaction: {
    // Compact when context passes this fraction of the window (default 0.9).
    thresholdPercent: 0.75,
  },
});
