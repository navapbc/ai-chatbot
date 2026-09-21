import { defineTool } from 'eve/tools';
import { z } from 'zod';
import { askJev, isJevEnabled } from '@/lib/jev/client';
import { fieldValueQuestions } from '@/lib/jev/questions';
import { flattenRecord } from '@/lib/jev/flatten';
import { currentParticipant } from '../lib/jev';

// Eve counterpart of lib/ai/tools/resolve-field-value.ts. The candidate
// generation, the question, and the `none` escape hatch are shared code —
// only the participant lookup differs, because Eve tools read it from session
// state rather than receiving it at construction (see agent/lib/jev.ts).
export default defineTool({
  description:
    "Resolve which value from the participant record belongs in a form field, before typing it. Pass the field's label exactly as the form shows it. Returns the selected value with a confidence, or no value when the record does not contain one — in which case ask the caseworker via gap_analysis rather than guessing. Use this for fields where several record values could plausibly fit (income amounts, ID numbers, dates, addresses).",
  inputSchema: z
    .object({
      field: z
        .string()
        .describe('The form field label, exactly as it appears on the page'),
    })
    .strict(),
  async execute({ field }: { field: string }, ctx) {
    const participant = currentParticipant();
    if (!participant || !isJevEnabled('field-value')) {
      return {
        resolved: false as const,
        reason: 'field-value resolution is not enabled',
      };
    }

    const leaves = flattenRecord(participant);
    const candidates = [...new Set(Object.values(leaves))];
    if (candidates.length === 0) {
      return { resolved: false as const, reason: 'record has no values' };
    }

    const { state, questions } = fieldValueQuestions(
      participant,
      field,
      candidates,
    );
    const result = await askJev({
      state,
      questions,
      signal: ctx?.abortSignal,
    });
    if (!result.ok) return { resolved: false as const, reason: result.reason };

    const { selection } = result.answers;
    if (selection.choice === 'none') {
      return {
        resolved: false as const,
        reason: 'no value in the record fits this field',
        confidence: selection.confidence,
      };
    }
    const path = Object.keys(leaves).find(
      (k) => leaves[k] === selection.choice,
    );
    return {
      resolved: true as const,
      value: selection.choice,
      recordPath: path ?? null,
      confidence: selection.confidence,
    };
  },
});
