import { defineTool } from 'eve/tools';
import { z } from 'zod';
import { annotateGapFields } from '@/lib/jev/enrich';
import { relayCredentialState } from '@/lib/relay/eve-relay-browser';
import { currentParticipant } from '../lib/jev';
import { gapRowsForRelay } from '../lib/relay-rows';

// Returns validated structured data for the gap-analysis card. The interactive
// card RENDER is wired to the chat UI in SP-B; standalone this tool's job is to
// validate + surface the data, which it does here. Per the benefits-application
// skill, calling this ENDS the turn — the agent must stop and wait for the
// caseworker. lib/ai/tools/gap-analysis.ts is the chat-UI counterpart.
export default defineTool({
  description:
    'Render the gap-analysis card listing required form fields with no traceable data. In web chats calling this ends your turn. On extension runs it returns the rows with a check against the record and does not end your turn.',
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
  async execute({ formName, missingFields }, ctx) {
    // On an extension run the rows are the field labels only (no note), and
    // the turn is not ended: the agent goes on to ask for what is missing.
    const onRelay = relayCredentialState(ctx).kind !== 'none';
    // Validates and surfaces the missing-field data; card render is SP-B.
    //
    // `jev` annotates each row with whether the record already answers it.
    // Rows are never dropped — see lib/jev/enrich.ts for why. The participant
    // comes from session state, populated by agent/hooks/participant.ts; when
    // it is absent (no record in the conversation, or a subagent, which
    // inherits no state) the rows come back unannotated and this behaves
    // exactly as before.
    const annotated = await annotateGapFields({
      participant: currentParticipant(),
      fields: onRelay ? gapRowsForRelay(missingFields) : missingFields,
      signal: ctx?.abortSignal,
    });
    // Returns the rows, not just a count, so the annotation lands under
    // `missingFields[].jev` — the same path the legacy route emits. Trace
    // queries over verdict distributions then work across both trees, and an
    // un-annotated row stays visible (no `jev` key) rather than vanishing
    // from a filtered list.
    return {
      rendered: true,
      formName,
      missingCount: missingFields.length,
      missingFields: annotated,
      ...(onRelay
        ? {
            next: 'Ask for what is still missing with ask_question, then fill.',
          }
        : {}),
    };
  },
});
