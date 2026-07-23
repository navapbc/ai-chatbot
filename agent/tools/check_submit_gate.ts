import { defineTool } from 'eve/tools';
import { z } from 'zod';

// Example tool. Production logic: lib/ai/tools/check-submit-gate.ts. Probes a
// Turnstile page and force-enables a stuck-disabled submit button so the
// caseworker can take over. It does NOT click submit.
export default defineTool({
  description:
    'On a page with a Cloudflare Turnstile widget where the submit button is stuck disabled, probe and force-enable it. Does not click submit. Do not call on pages without Turnstile.',
  inputSchema: z.object({
    reason: z.string().describe('Why the submit button appears stuck-disabled'),
  }),
  async execute({ reason }) {
    return { enabled: true, reason };
  },
});
