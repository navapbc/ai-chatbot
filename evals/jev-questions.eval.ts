import { Eval } from 'braintrust';
import { askJev, isJevEnabled } from '@/lib/jev/client';
import { gapFieldQuestions, summaryFieldQuestions } from '@/lib/jev/questions';
import { getParticipantById } from '@/lib/data/participants';
import cases from './datasets/jev-questions.json';
import { evalExperimentName } from './helpers';

/**
 * Regression set for the Jev questions in `lib/jev/questions.ts`.
 *
 * Unlike every other suite here, the system under test is NOT the agent — it
 * is the fixed set of questions themselves. That is the point of the repo's
 * rule that questions are written in code and never composed at runtime: a
 * fixed question can be versioned and regression-scored, and this is the
 * scoring half of it. Reword a verdict and this suite tells you which
 * judgments moved.
 *
 * Every expectation is hand-adjudicated against the record and the
 * caseworker's messages, NOT copied from what Jev answered — including three
 * cases (`both-sources-telephone`, `implied-not-living-alone`,
 * `age-derived-adult`) that Jev currently gets wrong. Those are meant to fail
 * until the underlying gap is fixed; see the `why` on each.
 *
 * Why this cannot be a unit test: it calls TypeSafe over the network and the
 * answers are probabilistic, so it belongs with the evals, where a score
 * distribution over repeats is the unit of comparison rather than a pass/fail
 * assertion.
 */

/** Repeats per case — Jev is probabilistic and its variance is itself unknown. */
const REPEATS = Number.parseInt(process.env.JEV_EVAL_REPEATS ?? '3', 10);

/** Where a `high`/`low` expectation sits. A noul is a probability in [0,1]. */
const NOUL_THRESHOLD = 0.5;

const participant = getParticipantById(cases.recordId);
if (!participant) {
  throw new Error(
    `evals/datasets/jev-questions.json references record_id ${cases.recordId}, which is not in lib/data/participants.ts.`,
  );
}

// `askJev` never throws: with no key or the feature off it returns
// { ok: false }, which would score every row 0 and read as a total regression
// rather than "this did not run". Detect it once here and return null scores
// instead — the same non-applicable convention regression-scenarios uses.
const ENABLED = isJevEnabled('summary-check') && isJevEnabled('gap-triage');
if (!ENABLED) {
  console.warn(
    '::warning::jev-questions eval skipped — needs TYPESAFE_API_KEY and JEV_FEATURES to include summary-check and gap-triage. Scores will be null, NOT zero.',
  );
}

type SummaryCase = (typeof cases.summary)[number];
type GapCase = (typeof cases.gap)[number];
type AnyCase = (SummaryCase | GapCase) & { kind: 'summary' | 'gap' };

interface Outcome {
  ran: boolean;
  verdict?: string;
  /** Probability Jev put on the verdict we expected — the calibration signal. */
  expectedVerdictP?: number;
  /** `sourceAccurate` for summary rows, `sensitive` for gap rows. */
  noul?: number;
}

const isSummary = (c: AnyCase): c is SummaryCase & { kind: 'summary' } =>
  c.kind === 'summary';

const rows: AnyCase[] = [
  ...cases.summary.map((c) => ({ ...c, kind: 'summary' as const })),
  ...cases.gap.map((c) => ({ ...c, kind: 'gap' as const })),
];

Eval('labs-asp', {
  experimentName: evalExperimentName('Jev Questions'),

  // Repeats are rows, not retries: Braintrust averages them, so a case that
  // flips between runs shows up as a mid-range score rather than as noise
  // attributed to whichever run happened to execute.
  data: () =>
    rows.flatMap((c) =>
      Array.from({ length: REPEATS }, (_, i) => ({
        input: { ...c, repeat: i },
        expected: c.expectVerdict,
        metadata: { id: c.id, kind: c.kind, why: c.why, repeat: i },
      })),
    ),

  task: async (input: AnyCase & { repeat: number }): Promise<Outcome> => {
    if (!ENABLED) return { ran: false };

    const { state, questions } = isSummary(input)
      ? summaryFieldQuestions(
          participant,
          { field: input.field, value: input.value, source: input.source },
          cases.caseworkerMessages,
        )
      : gapFieldQuestions(participant, input.field);

    const result = await askJev({
      state,
      questions,
      feature: isSummary(input) ? 'summary-check' : 'gap-triage',
      field: input.field,
    });
    if (!result.ok) return { ran: false };

    const answers = result.answers as {
      verdict: { choice: string; probabilities: Record<string, number> };
      source_accurate?: { noul: number };
      sensitive?: { noul: number };
    };
    return {
      ran: true,
      verdict: answers.verdict.choice,
      expectedVerdictP: answers.verdict.probabilities[input.expectVerdict] ?? 0,
      noul: (answers.source_accurate ?? answers.sensitive)?.noul,
    };
  },

  scores: [
    // Did Jev pick the adjudicated verdict?
    ({ output, expected }) => {
      if (!output?.ran) return { name: 'verdict_correct', score: null };
      return {
        name: 'verdict_correct',
        score: output.verdict === expected ? 1 : 0,
      };
    },

    // How much probability mass landed on the adjudicated verdict. Strictly
    // more informative than the binary above: a 0.58/0.40 near-miss and a
    // 0.95/0.02 confident miss both score 0 there, and 0.40 vs 0.02 here.
    // This is the number to watch when judging whether a reword helped.
    ({ output }) => {
      if (!output?.ran)
        return { name: 'expected_verdict_probability', score: null };
      return {
        name: 'expected_verdict_probability',
        score: output.expectedVerdictP ?? 0,
      };
    },

    // The second axis: sourceAccurate on summary rows, sensitive on gap rows.
    // Scored as a direction, not a value — the dataset asserts which side of
    // the midpoint a case belongs on, which is all that can be adjudicated by
    // hand. `mislabel-record-as-caseworker` and `gap-help-at-home-missing`
    // are the cases that stop a constant-output noul from scoring well.
    ({ output, metadata }) => {
      if (!output?.ran || output.noul === undefined) {
        return { name: 'noul_direction_correct', score: null };
      }
      const row = rows.find((r) => r.id === metadata?.id);
      const want =
        (row as SummaryCase | undefined)?.expectSourceAccurate ??
        (row as GapCase | undefined)?.expectSensitive;
      if (want !== 'high' && want !== 'low') {
        return { name: 'noul_direction_correct', score: null };
      }
      const isHigh = output.noul > NOUL_THRESHOLD;
      return {
        name: 'noul_direction_correct',
        score: (want === 'high') === isHigh ? 1 : 0,
      };
    },
  ],
});
