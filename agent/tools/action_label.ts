import { defineTool } from 'eve/tools';
import { z } from 'zod';

// Example tool. Production logic: lib/ai/tools/action-label.ts.
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
    // Demonstrative stub: production emits a UI action label.
    return { labeled: category };
  },
});
