import { tool } from 'ai';
import { z } from 'zod';
import type { Participant } from '@/lib/data/participants';
import { askJev, isJevEnabled } from '@/lib/jev/client';
import { fieldValueQuestions } from '@/lib/jev/questions';
import { annotateVerdict } from '@/lib/jev/telemetry';
import { flattenRecord } from '@/lib/jev/flatten';

/**
 * Pick which value from the participant record belongs in a form field.
 *
 * The agent supplies only the field label. Candidates come from the record in
 * code, and the question is written in code (lib/jev/questions.ts), so the
 * answer is always a verbatim copy of a value that was already in the record
 * rather than a value composed on the spot. `none` is always offered, because
 * Jev cannot select a candidate that was not given to it.
 *
 * Scope, honestly: this reduces invention, it does not eliminate it. The
 * resolved string is returned to the agent, which then types it via `browser
 * fill` — so the model can still alter it in between. Removing that gap means
 * resolving and filling in a single call inside lib/ai/tools/browser.ts, which
 * is a larger change to that tool's mutex and session handling.
 */

export const createResolveFieldValueTool = (participant: Participant | null) =>
  tool({
    description:
      "Resolve which value from the participant record belongs in a form field, before typing it. Pass the field's label exactly as the form shows it. Returns the selected value with a confidence, or no value when the record does not contain one — in which case ask the caseworker via gapAnalysis rather than guessing. Use this for fields where several record values could plausibly fit (income amounts, ID numbers, dates, addresses).",
    inputSchema: z.object({
      field: z
        .string()
        .describe('The form field label, exactly as it appears on the page'),
    }),
    execute: async ({ field }, { abortSignal }) => {
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
        feature: 'field-value',
        field,
        onAnswers: (a) =>
          annotateVerdict(a.selection.choice, a.selection.confidence),
        signal: abortSignal,
      });
      if (!result.ok) {
        return { resolved: false as const, reason: result.reason };
      }

      const { selection } = result.answers;
      if (selection.choice === 'none') {
        return {
          resolved: false as const,
          reason: 'no value in the record fits this field',
          confidence: selection.confidence,
        };
      }

      // Report where in the record the value came from, so the agent can set
      // formSummary's `source` correctly instead of inferring it.
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
