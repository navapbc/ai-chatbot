import { defineTool } from 'eve/tools';
import { z } from 'zod';
import { annotateSummaryFields } from '@/lib/jev/enrich';
import { currentParticipant } from '../../../lib/jev';

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
        source: z
          .enum(['record', 'caseworker', 'inferred', 'missing'])
          .describe(
            '"record" = a labelled field from the participant JSON in the caseworker\'s opening message; "caseworker" = a value the caseworker typed later in this conversation, outside that record; "inferred" = you reasoned it from one of those; "missing" = could not be filled.',
          ),
        inputType: z.enum(['select', 'radio', 'checkbox', 'text']).optional(),
        options: z.array(z.string()).optional(),
        required: z.boolean().optional(),
      }),
    ),
  }),
  async execute({ fields }, ctx) {
    // `jev` checks each filled value against the participant record, and
    // checks the `source` label the agent attached to it. Annotate-only: the
    // agent's fields are never rewritten. The participant comes from session
    // state (agent/hooks/participant.ts); absent it, nothing is annotated.
    const annotated = await annotateSummaryFields({
      participant: currentParticipant(),
      fields,
      signal: ctx?.abortSignal,
    });
    // Rows, not just a count: the annotation lands under `fields[].jev`,
    // matching the legacy route so trace queries work across both trees.
    return { rendered: true, fieldCount: fields.length, fields: annotated };
  },
});
