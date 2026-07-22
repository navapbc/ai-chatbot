import { defineAgent } from 'eve';

// Model resolves through Vercel AI Gateway. Locally this uses
// AI_GATEWAY_API_KEY; on Vercel it uses OIDC (see Task 5).
export default defineAgent({
  model: 'anthropic/claude-sonnet-4.6',
});
