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
// NOTE: these wordings changed when `caseworker_messages` was added to the
// state. The keys are unchanged, but a distribution measured before that
// change is NOT comparable with one measured after — `grounded` and
// `caseworker_supplied` both became checkable rather than guessable.
export const SUMMARY_VERDICTS = {
  grounded:
    "The value appears in the participant record, or follows BY INFERENCE from the record or from the caseworker's instructions (for example the applicant's age from their date of birth, or who the application is for when the caseworker said whom to apply for). A value the caseworker stated outright is `caseworker_supplied`, not this.",
  caseworker_supplied:
    'The value is not in the record, and it appears in `caseworker_messages` — the caseworker actually supplied it.',
  contradicts_record: 'The record contains a different value for this field.',
  unsupported:
    'The value is in neither the record nor `caseworker_messages`, and does not follow from either — it appears to have been invented. A default answer still counts: an unticked box reported as "None", or a "No" nobody stated or implied, is invented unless something supports it.',
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
  // The caseworker's own turns, record JSON already stripped
  // (lib/jev/participant.ts). Optional so a caller that cannot reach the
  // conversation — the form_review subagent, which inherits no state — still
  // gets the record-only judgment rather than an error.
  caseworkerMessages: readonly string[] = [],
) => ({
  state: {
    form_field: field.field,
    value_entered: field.value ?? '',
    source_claimed_by_agent: field.source,
    participant_record: participant as unknown,
    caseworker_messages: caseworkerMessages,
  },
  questions: {
    verdict: choice(
      'The agent entered `value_entered` into `form_field`. Judge it against `participant_record` and `caseworker_messages` together — the record is the data the caseworker handed over, and `caseworker_messages` is everything they said, including the task they set and any answers they gave later.',
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
      'The agent labelled this value\'s origin as `source_claimed_by_agent`, where "record" (also spelled "database") means it came from `participant_record` — the participant data handed to the agent in the caseworker\'s opening message — "caseworker" means the caseworker stated it in `caseworker_messages` rather than in that record, and "inferred" means the agent derived it from one of those. Is that label accurate?',
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
