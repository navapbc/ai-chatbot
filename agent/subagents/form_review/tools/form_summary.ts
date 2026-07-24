import { defineTool } from 'eve/tools';
import { z } from 'zod';

// Example tool. Production logic: lib/ai/tools/form-summary.ts (interactive
// review card). Called instead of writing a text summary.
export default defineTool({
  description:
    'Render the form-completion summary card. Call instead of writing a text summary of filled fields.',
  inputSchema: z.object({
    clientName: z.string().optional(),
    fields: z.array(
      z.object({
        field: z.string(),
        value: z.string().optional(),
        source: z.enum(['caseworker', 'inferred', 'missing']),
        inputType: z.enum(['select', 'radio', 'checkbox', 'text']).optional(),
        options: z.array(z.string()).optional(),
        required: z.boolean().optional(),
      }),
    ),
  }),
  async execute({ fields }) {
    return { rendered: true, fieldCount: fields.length };
  },
});
