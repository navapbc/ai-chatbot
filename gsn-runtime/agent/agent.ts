import { defineAgent } from 'eve';
import { vertexAnthropic } from '@ai-sdk/google-vertex/anthropic';
export default defineAgent({
  model: vertexAnthropic('claude-sonnet-5-5'),
  modelContextWindowTokens: 200_000,
  limits: {
    maxInputTokensPerSession: 60_000,
    maxOutputTokensPerSession: 4_000,
  },
  experimental: {
    workflow: {
      world: process.env.WORKFLOW_POSTGRES_URL
        ? '@workflow/world-postgres'
        : undefined,
    },
  },
});
