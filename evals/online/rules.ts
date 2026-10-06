// Online-scoring rules. One rule per (scope, filter) — NOT one per judge.
// Braintrust runs every scorer in a rule's `scorers` array against the same
// trace fetch and idle timer, so adding a judge that shares a filter means
// appending to `scorers`, not creating another automation.
//
// Each scorer still writes its own `scores.<name>` key, which is what makes
// scores filterable and chartable per dimension.

import { EVALUATOR_DEFINITION as GAP_ANALYSIS_ASKING } from './scorers/gap-analysis-asking';
import { EVALUATOR_DEFINITION as HALLUCINATION } from './scorers/hallucination';
import { EVALUATOR_DEFINITION as HALLUCINATION_JEV } from './scorers/hallucination-jev';
import { EVALUATOR_DEFINITION as SUMMARY_ATTRIBUTION } from './scorers/summary-attribution';
import { EVALUATOR_DEFINITION as VERBOSITY } from './scorers/verbosity';

/** Shape POSTed to /v1/function; structural so every judge module fits. */
export interface OnlineScorer {
  name: string;
  slug: string;
  description: string;
  prompt_data: Record<string, unknown>;
  metadata?: Record<string, unknown>;
  /**
   * Feature flag. `false` means apply.ts skips the scorer entirely, so it is
   * never uploaded and never added to its rule — which matters because
   * Braintrust runs every scorer in a rule's array, so activating a rule
   * would otherwise activate a new judge alongside the established ones.
   * Defaults to enabled when omitted.
   */
  enabled?: boolean;
}

export interface OnlineRule {
  name: string;
  description: string;
  /** Judges sharing this rule's scope and filter. */
  scorers: OnlineScorer[];
  btqlFilter: string;
  scope: { type: 'trace'; idle_seconds: number };
  samplingRate: number;
}

// Trace scope, not span: filterAISpans drops the OTEL root.
// 120s, not the 30s default: late spans restart the timer and re-score.
const TRACE_SCOPE = { type: 'trace' as const, idle_seconds: 120 };

// Jev scorers are opt-in. `lib/feature-flags.ts` is the wrong gate for them:
// it is a browser/localStorage mechanism, and this config is pushed to
// Braintrust by a CLI where there is no `window`. An env var is the
// equivalent for a build-time tool.
//
// Off by default, so `pnpm eval:online:apply` keeps producing exactly the
// judges it produced before. Turn on with:
//   JEV_ONLINE_SCORING=true pnpm eval:online:apply dev --dry-run
const JEV_ONLINE_SCORING = process.env.JEV_ONLINE_SCORING === 'true';

export const RULES: OnlineRule[] = [
  {
    name: 'Form run quality',
    description:
      'Judges that apply to any trace where the agent drove a form. Filtering on the browser tool also keeps out orphan spans, which a judge will otherwise score with an invented rationale.',
    scorers: [
      HALLUCINATION,
      VERBOSITY,
      // Additive: runs beside HALLUCINATION rather than replacing it, and
      // writes its own scores key. Disagreement between the two is the point.
      { ...HALLUCINATION_JEV, enabled: JEV_ONLINE_SCORING },
    ],
    btqlFilter: "span_attributes.name = 'execute_tool browser'",
    scope: TRACE_SCOPE,
    samplingRate: 1,
  },
  {
    name: 'Gap analysis asking quality',
    description:
      'Judges that only apply once the agent has asked the caseworker for missing fields.',
    scorers: [GAP_ANALYSIS_ASKING],
    btqlFilter: "span_attributes.name = 'execute_tool gapAnalysis'",
    scope: TRACE_SCOPE,
    samplingRate: 1,
  },
  {
    name: 'Form summary quality',
    description:
      'Judges that grade the summary the agent shows the caseworker at the end of a run.',
    scorers: [SUMMARY_ATTRIBUTION],
    btqlFilter: "span_attributes.name = 'execute_tool formSummary'",
    scope: TRACE_SCOPE,
    samplingRate: 1,
  },
];
