// Jev enrichment for the two card tools.
//
// Both functions are annotate-only and total: given a flag that is off, no
// participant record, or an unreachable API, they return their input
// unchanged. Callers do not branch — they await and pass the result on.
//
// Questions about one row are batched into a single request. Rows themselves
// are separate requests over different state, run concurrently but capped:
// a benefits-application summary can carry 30-60 fields, and firing that many
// at once earns rate-limit responses, which this code turns into missing
// annotations rather than errors — the feature would look like it was working
// while mostly not. Cost and latency here are per row, not per call, and both
// land inline in the agent's turn.
//
// Where the annotations surface: the gap-analysis and form-summary cards
// render from the tool *input* (components/message.tsx), so nothing here
// changes what the caseworker sees. The annotations ride on the tool output,
// which means they reach the model and — the point of this phase — land in
// the Braintrust trace as `execute_tool <name>` output, where the verdict
// distribution can be read before anyone picks a threshold to act on.

import type { Participant } from '@/lib/data/participants';
import { askJev, isJevEnabled } from './client';
import { gapFieldQuestions, summaryFieldQuestions } from './questions';
import { annotateVerdict, withJevBatchSpan } from './telemetry';

/** Attached to a row when Jev answered; absent when it did not run. */
export interface JevAnnotation {
  verdict: string;
  confidence: number;
  /** Full distribution, kept so a threshold can be chosen from real data later. */
  probabilities: Record<string, number>;
}

export interface GapFieldAnnotation extends JevAnnotation {
  /** P(the field is sensitive for a caseworker to supply). */
  sensitive: number;
}

export interface SummaryFieldAnnotation extends JevAnnotation {
  /** P(the agent's own `source` label is accurate). */
  sourceAccurate: number;
}

/** Max simultaneous Jev requests per tool call. */
const CONCURRENCY = 6;

/** `Promise.all` over `items`, at most `CONCURRENCY` in flight. */
const mapCapped = async <In, Out>(
  items: In[],
  fn: (item: In) => Promise<Out>,
): Promise<Out[]> => {
  const out: Out[] = new Array(items.length);
  let next = 0;
  const worker = async () => {
    while (next < items.length) {
      const i = next++;
      out[i] = await fn(items[i]);
    }
  };
  await Promise.all(
    Array.from({ length: Math.min(CONCURRENCY, items.length) }, worker),
  );
  return out;
};

/**
 * Annotate each missing field with whether the record already answers it.
 *
 * Rows are annotated, never dropped. A false positive from Jev would
 * otherwise silently remove a question the caseworker needed to answer, and
 * `gapAnalysis` ends the turn, so there is no later chance to recover it.
 * Filtering can come once the distribution here has been looked at.
 */
export const annotateGapFields = async <T extends { field: string }>(args: {
  participant: Participant | null;
  fields: T[];
  signal?: AbortSignal;
}): Promise<(T & { jev?: GapFieldAnnotation })[]> => {
  const { participant, fields, signal } = args;
  const enabled = Boolean(participant) && isJevEnabled('gap-triage');

  // The batch span covers the disabled case too: "off" and "never reached
  // this code" are otherwise identical in a trace.
  return withJevBatchSpan(
    {
      feature: 'gap-triage',
      enabled,
      skipReason: !participant
        ? 'no_participant'
        : enabled
          ? undefined
          : 'flag_off',
      rowsAttempted: fields.length,
    },
    async () => {
      if (!enabled || !participant) return fields;

      return mapCapped(fields, async (row) => {
        const { state, questions } = gapFieldQuestions(participant, row.field);
        const result = await askJev({
          state,
          questions,
          feature: 'gap-triage',
          field: row.field,
          onAnswers: (a) =>
            annotateVerdict(a.verdict.choice, a.verdict.confidence),
          signal,
        });
        if (!result.ok) return row;
        const { verdict, sensitive } = result.answers;
        return {
          ...row,
          jev: {
            verdict: verdict.choice,
            confidence: verdict.confidence,
            probabilities: { ...verdict.probabilities },
            sensitive: sensitive.noul,
          },
        };
      });
    },
    (row) => 'jev' in row && row.jev !== undefined,
  );
};

/**
 * Annotate each summary row with whether its value is grounded in the record
 * and whether the agent's own `source` label holds.
 *
 * Fields the agent marked `missing` are skipped: there is no value to check,
 * and asking would spend a request to be told so.
 */
export const annotateSummaryFields = async <
  T extends { field: string; value?: string; source: string },
>(args: {
  participant: Participant | null;
  fields: T[];
  /**
   * The caseworker's own turns, record JSON stripped. Without these Jev sees
   * only the record, so it cannot tell an inference from the caseworker's
   * instructions apart from an invention, and has to guess whether a
   * `caseworker` label is true. Optional: a caller that cannot reach the
   * conversation gets the record-only judgment, which is what the
   * form_review subagent does (it inherits no session state).
   */
  caseworkerMessages?: readonly string[];
  signal?: AbortSignal;
}): Promise<(T & { jev?: SummaryFieldAnnotation })[]> => {
  const { participant, fields, caseworkerMessages, signal } = args;
  const enabled = Boolean(participant) && isJevEnabled('summary-check');

  // rowsAttempted counts only the rows this function would actually ask
  // about: a `missing` row is skipped by design, and counting it would make
  // every healthy batch look partially failed.
  const askable = fields.filter(
    (row) => row.source !== 'missing' && Boolean(row.value),
  ).length;

  return withJevBatchSpan(
    {
      feature: 'summary-check',
      enabled,
      skipReason: !participant
        ? 'no_participant'
        : enabled
          ? undefined
          : 'flag_off',
      rowsAttempted: askable,
    },
    async () => {
      if (!enabled || !participant) return fields;

      return mapCapped(fields, async (row) => {
        if (row.source === 'missing' || !row.value) return row;
        const { state, questions } = summaryFieldQuestions(
          participant,
          row,
          caseworkerMessages,
        );
        const result = await askJev({
          state,
          questions,
          feature: 'summary-check',
          field: row.field,
          onAnswers: (a) =>
            annotateVerdict(a.verdict.choice, a.verdict.confidence),
          signal,
        });
        if (!result.ok) return row;
        const { verdict, source_accurate } = result.answers;
        return {
          ...row,
          jev: {
            verdict: verdict.choice,
            confidence: verdict.confidence,
            probabilities: { ...verdict.probabilities },
            sourceAccurate: source_accurate.noul,
          },
        };
      });
    },
    (row) => 'jev' in row && row.jev !== undefined,
  );
};
