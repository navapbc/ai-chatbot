import { vertexAnthropic } from '@ai-sdk/google-vertex/anthropic';
import { defineAgent } from 'eve';
import { VERTEX_CONTEXT_WINDOW_TOKENS } from '@/lib/ai/eve/model-map';

export default defineAgent({
  description:
    "Research a benefits program's application up front and enumerate ALL fields it will require across every page, so gap analysis is complete before form-filling starts. Returns a field checklist.",
  // Called directly on Vertex AI rather than through the Vercel AI Gateway,
  // matching agent/agent.ts. The explicit context window is required: Eve
  // cannot look a direct provider instance up in the gateway model catalog.
  model: vertexAnthropic('claude-sonnet-4-6'),
  modelContextWindowTokens: VERTEX_CONTEXT_WINDOW_TOKENS,
});
