import { vertexAnthropic } from '@ai-sdk/google-vertex/anthropic';
import { defineAgent } from 'eve';
import { VERTEX_CONTEXT_WINDOW_TOKENS } from '@/lib/ai/eve/model-map';

export default defineAgent({
  description:
    "Walk the application's review/summary screen at the end of filling and produce the structured, source-tagged formSummary field list for the caseworker to review before submission.",
  // Called directly on Vertex AI rather than through the Vercel AI Gateway,
  // matching agent/agent.ts. The explicit context window is required: Eve
  // cannot look a direct provider instance up in the gateway model catalog.
  model: vertexAnthropic('claude-haiku-4-5'),
  modelContextWindowTokens: VERTEX_CONTEXT_WINDOW_TOKENS,
});
