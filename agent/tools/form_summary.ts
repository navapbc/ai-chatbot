import { defineTool } from 'eve/tools';
import { z } from 'zod';

// Returns validated structured data for the form-summary card. The interactive
// card RENDER is wired to the chat UI in SP-B; standalone this tool's job is to
// validate + surface the data, which it does here. Called instead of writing a
// text summary of filled fields. lib/ai/tools/form-summary.ts is the chat-UI
// counterpart.
export default defineTool({
  description:
    'Render the form-completion summary card. Call instead of writing a text summary of filled fields.',
  inputSchema: z.object({
    clientName: z.string().optional(),
    fields: z.array(
      z.object({
        field: z.string(),
        value: z.string().optional(),
        source: z.enum(['database', 'caseworker', 'inferred', 'missing']),
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
