import { defineTool } from 'eve/tools';
import { z } from 'zod';
import { annotateSummaryFields } from '@/lib/jev/enrich';
import { currentParticipant } from '../lib/jev';

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
        source: z.enum(['caseworker', 'inferred', 'missing']),
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
