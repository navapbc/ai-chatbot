// Online-scoring rules. One rule per (scope, filter) — NOT one per judge.
// Braintrust runs every scorer in a rule's `scorers` array against the same
// trace fetch and idle timer, so adding a judge that shares a filter means
// appending to `scorers`, not creating another automation.
//
// Each scorer still writes its own `scores.<name>` key, which is what makes
// scores filterable and chartable per dimension.

import { EVALUATOR_DEFINITION as GAP_ANALYSIS_ASKING } from './scorers/gap-analysis-asking';

export type OnlineScorer = typeof GAP_ANALYSIS_ASKING;

export interface OnlineRule {
  name: string;
  description: string;
  /** Judges sharing this rule's scope and filter. */
  scorers: OnlineScorer[];
  btqlFilter: string;
  scope: { type: 'trace'; idle_seconds: number };
  samplingRate: number;
}

export const RULES: OnlineRule[] = [
  {
    name: 'Gap analysis asking quality',
    description:
      'Judges that read a form-filling trace in which the agent asked the caseworker for missing fields.',
    scorers: [GAP_ANALYSIS_ASKING],
    // Without this, orphan `agent-browser close` spans form empty 1-span
    // traces that a judge will score anyway.
    btqlFilter: "span_attributes.name = 'execute_tool gapAnalysis'",
    // Trace scope, not span: filterAISpans drops the OTEL root.
    // 120s, not 60: late spans restarted the timer and re-scored the trace.
    scope: { type: 'trace', idle_seconds: 120 },
    samplingRate: 1,
  },
];
