import { defineTool } from 'eve/tools';
import { z } from 'zod';

// Returns validated structured data for the action-label card. The interactive
// card RENDER is wired to the chat UI in SP-B; standalone this tool is the
// lightweight real signal itself — it returns { labeled: category } directly,
// which it does here. lib/ai/tools/action-label.ts is the chat-UI counterpart.
// Call once before each logical group of browser actions.
export default defineTool({
  description:
    'Label the next group of related browser actions. Call once with the best-fit category before a batch of actions.',
  inputSchema: z.object({
    category: z.enum([
      'fill',
      'navigate',
      'interact',
      'read',
      'search',
      'misc',
    ]),
  }),
  async execute({ category }) {
    // The label is the entire payload for this tool — no separate render step.
    return { labeled: category };
  },
});
