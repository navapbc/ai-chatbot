// The questions Jev is asked, written once here so they can be reviewed,
// diffed, and regression-tested. Nothing in the app builds a Jev question at
// runtime.
//
// Every judgment below is annotate-only: it adds a field to something the
// agent already produced and never removes or rewrites it. A caseworker sees
// the same rows either way. That is deliberate for a first rollout — dropping
// a row from a benefits application on a model's say-so needs a threshold
// chosen from observed data, and there is no such data yet.

import { choice, noul } from '@typesafe-ai/sdk';
import type { Participant } from '@/lib/data/participants';

/** Verdicts for one row of the gap-analysis card. */
export const GAP_VERDICTS = {
  genuinely_missing:
    'The participant record contains nothing that answers this field, so the caseworker does have to supply it.',
  present_in_record:
    'The record already contains this value directly, under this name or an obvious synonym.',
  derivable_from_record:
    'The record does not state it, but it follows from what is there (age from date of birth; mailing address from residential when they match).',
} as const;

/**
 * Was this field worth asking the caseworker for?
 *
 * The agent assembles `missingFields` itself, and its false positives are the
 * expensive kind: asking for something the record already answers wastes the
 * caseworker's turn and erodes trust in the card.
 */
export const gapFieldQuestions = (participant: Participant, field: string) => ({
  state: {
    field_being_requested: field,
    participant_record: participant as unknown,
  },
  questions: {
    verdict: choice(
      'The agent says `field_being_requested` cannot be answered from `participant_record` and wants to ask the caseworker for it. Is that right?',
      GAP_VERDICTS,
    ),
    sensitive: noul(
      "Is `field_being_requested` sensitive information a caseworker may be unable or unwilling to supply on the participant's behalf — a Social Security number, immigration or citizenship status, disability, or medical detail?",
    ),
  },
});

/** Verdicts for one row of the form-summary card. */
export const SUMMARY_VERDICTS = {
  grounded:
    'The value appears in the participant record, or follows directly from it.',
  caseworker_supplied:
    'The value is not in the record and reads as something the caseworker provided during the conversation.',
  contradicts_record: 'The record contains a different value for this field.',
  unsupported:
    'The value is not in the record and does not follow from it — it appears to have been invented.',
} as const;

/**
 * Check one filled field against the record, and check the provenance label
 * the agent attached to it.
 *
 * This merges what were two separate ideas — verifying values before submit,
 * and citation-checking the summary's `source` labels. They read the same
 * state and the questions are independent, so per the TypeSafe guidance on
 * batching they belong in one request rather than two passes over the same row.
 */
export const summaryFieldQuestions = (
  participant: Participant,
  field: { field: string; value?: string; source: string },
) => ({
  state: {
    form_field: field.field,
    value_entered: field.value ?? '',
    source_claimed_by_agent: field.source,
    participant_record: participant as unknown,
  },
  questions: {
    verdict: choice(
      'The agent entered `value_entered` into `form_field`. Judge it against `participant_record`.',
      SUMMARY_VERDICTS,
    ),
    source_accurate: noul(
      // The two trees spell the record-derived label differently — the Eve
      // tools emit "record", the legacy tool still emits "database" (a name
      // left over from the removed Apricot integration; there is no database,
      // the record arrives inline in the caseworker's opening message). Both
      // must be described here, or Jev judges one of them against the wrong
      // definition.
      //
      // The "outside that record" clause is load-bearing. `participant_record`
      // reaches the agent INSIDE the caseworker's first message, so without it
      // Jev reads every record-derived value as "the caseworker supplied it"
      // and rates an accurate label inaccurate. That was measured: across 25
      // annotated fields, source_accurate never rose above 0.76, which is not
      // a threshold anything can act on.
      'The agent labelled this value\'s origin as `source_claimed_by_agent`, where "record" (also spelled "database") means it came from `participant_record` — the participant data handed to the agent in the caseworker\'s opening message — "caseworker" means the caseworker supplied it later in the conversation, outside that record, and "inferred" means the agent derived it from one of those. Is that label accurate?',
    ),
  },
});

/**
 * Pick which of `candidates` belongs in `field`.
 *
 * Candidates are produced in code from the record, so the answer is always a
 * verbatim copy of an existing value rather than a newly written one, and
 * `none` is always available because the model cannot pick a value that was
 * not offered.
 */
export const fieldValueQuestions = (
  participant: Participant,
  field: string,
  candidates: readonly string[],
) => ({
  state: {
    form_field: field,
    candidates,
    participant_record: participant as unknown,
  },
  questions: {
    selection: choice(
      'Which entry in `candidates` is the correct value for `form_field`?',
      {
        ...Object.fromEntries(candidates.map((c) => [c, null])),
        none: 'None of these candidates is the right value for this field.',
      },
    ),
  },
});
