// Online-scoring rules. One rule per (scope, filter) — NOT one per judge.
// Braintrust runs every scorer in a rule's `scorers` array against the same
// trace fetch and idle timer, so adding a judge that shares a filter means
// appending to `scorers`, not creating another automation.
//
// Each scorer still writes its own `scores.<name>` key, which is what makes
// scores filterable and chartable per dimension.

import { EVALUATOR_DEFINITION as GAP_ANALYSIS_ASKING } from './scorers/gap-analysis-asking';
import { EVALUATOR_DEFINITION as HALLUCINATION } from './scorers/hallucination';
import { EVALUATOR_DEFINITION as SUMMARY_ATTRIBUTION } from './scorers/summary-attribution';
import { EVALUATOR_DEFINITION as VERBOSITY } from './scorers/verbosity';

/** Shape POSTed to /v1/function; structural so every judge module fits. */
export interface OnlineScorer {
  name: string;
  slug: string;
  description: string;
  prompt_data: Record<string, unknown>;
  metadata?: Record<string, unknown>;
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

export const RULES: OnlineRule[] = [
  {
    name: 'Form run quality',
    description:
      'Judges that apply to any trace where the agent drove a form. Filtering on the browser tool also keeps out orphan spans, which a judge will otherwise score with an invented rationale.',
    scorers: [HALLUCINATION, VERBOSITY],
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
