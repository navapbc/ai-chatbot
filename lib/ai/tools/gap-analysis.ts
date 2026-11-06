import { tool } from 'ai';
import { z } from 'zod';
import type { Participant } from '@/lib/data/participants';
import { annotateGapFields } from '@/lib/jev/enrich';

const gapFieldSchema = z.object({
  field: z.string().describe('Field label'),
  options: z
    .array(z.string())
    .optional()
    .describe('Possible answer options, if applicable'),
  inputType: z
    .enum(['text', 'select', 'date', 'boolean', 'textarea'])
    .optional()
    .describe('Expected input type'),
  multiSelect: z
    .boolean()
    .optional()
    .describe('Whether multiple options can be selected'),
  condition: z
    .string()
    .optional()
    .describe(
      'Condition under which this field is required, e.g. "if pregnant"',
    ),
  required: z
    .boolean()
    .optional()
    .describe('Whether this field is required to submit the form'),
  placeholder: z
    .string()
    .optional()
    .describe('Placeholder hint shown inside the input'),
  note: z
    .string()
    .optional()
    .describe('Short helper text shown under the field label'),
});

/**
 * `participant` is only used to let Jev annotate each row with whether the
 * record already answers it (lib/jev/enrich.ts). Pass null — or leave the
 * `gap-triage` feature off — and this behaves exactly as it did before: the
 * agent's `missingFields` are returned untouched.
 */
export const createGapAnalysisTool = (participant: Participant | null) =>
  tool({
    description:
      'Shows the caseworker a card listing ONLY the missing fields, in the order they appear on the original form. Calling this tool ends your turn — do not call any other tools after it; wait for the caseworker\'s reply. Include only missing fields, no fields you already have. After calling, write one short sentence like "Please provide the missing info above." and stop. If nothing is missing, do not call this tool.',
    inputSchema: z.object({
      formName: z
        .string()
        .optional()
        .describe('Name of the form being filled, e.g. "WIC Application"'),
      clientName: z
        .string()
        .optional()
        .describe('Full name of the participant the form is being filled for'),
      missingFields: z
        .array(gapFieldSchema)
        .describe(
          'Missing fields in the order they appear on the original form.',
        ),
    }),
    execute: async (input, { abortSignal }) => ({
      ...input,
      missingFields: await annotateGapFields({
        participant,
        fields: input.missingFields,
        signal: abortSignal,
      }),
    }),
  });
