import { defineTool } from 'eve/tools';
import { z } from 'zod';

// Example tool. Production logic: lib/ai/tools/gap-analysis.ts (renders an
// interactive card client-side). Per the benefits-application skill, calling
// this ENDS the turn — the agent must stop and wait for the caseworker.
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
    // Demonstrative stub: production renders the interactive gap card.
    return { rendered: true, formName, missingCount: missingFields.length };
  },
});
