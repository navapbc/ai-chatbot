import { defineTool } from 'eve/tools';
import { z } from 'zod';

// Returns validated structured data for the gap-analysis card. The interactive
// card RENDER is wired to the chat UI in SP-B; standalone this tool's job is to
// validate + surface the data, which it does here. Per the benefits-application
// skill, calling this ENDS the turn — the agent must stop and wait for the
// caseworker. lib/ai/tools/gap-analysis.ts is the chat-UI counterpart.
export default defineTool({
  description:
    'Render the gap-analysis card listing required form fields with no traceable data. Calling this ends your turn.',
  inputSchema: z.object({
    formName: z.string(),
    clientName: z.string().optional(),
    missingFields: z.array(
      z.object({
        field: z.string(),
        options: z.array(z.string()).optional(),
        inputType: z.enum(['select', 'radio', 'checkbox', 'text']).optional(),
        multiSelect: z.boolean().optional(),
        required: z.boolean().optional(),
        note: z.string().optional(),
      }),
    ),
  }),
  async execute({ formName, missingFields }) {
    // Validates and surfaces the missing-field data; card render is SP-B.
    return { rendered: true, formName, missingCount: missingFields.length };
  },
});
