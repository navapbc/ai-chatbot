// Online variant of evals/scorers/ask-questions.ts. Ground truth comes from the
// participant JSON in the caseworker's opening message, not an eval serializer.
// Uses REST, not projects.scorers.create: ScorerPromptOpts has no preprocessor
// or allow_skip.

import { JUDGE_MODEL, PASS_THRESHOLD } from './shared';

export const CHOICE_SCORES = { A: 1, B: 0.5, C: 0 } as const;

export const EVALUATOR_SLUG = 'gap-analysis-asking-online';

// Runs per span, merged in trace order. Dropping browser/reference spans cuts
// a small trace from ~52KB to ~6KB.
export const PREPROCESSOR_CODE = `const NOISY = new Set(['browser', 'readReference', 'actionLabel']);
const CAP = 1500;

function clip(v) {
  const s = typeof v === 'string' ? v : JSON.stringify(v);
  if (s == null) return '';
  return s.length > CAP ? s.slice(0, CAP) + ' …[truncated]' : s;
}

function handler({ input, output, span_attributes }) {
  const name = span_attributes?.name || '';
  const items = [];

  if (name.startsWith('invoke_agent') || name.startsWith('chat ')) {
    const msgs = Array.isArray(input) ? input : input?.messages;
    if (Array.isArray(msgs)) {
      for (const m of msgs) {
        if (m?.role === 'user' || m?.role === 'system') {
          items.push({ role: m.role, content: clip(m.content) });
        }
      }
    }
    const outMsgs = Array.isArray(output) ? output : [output];
    for (const o of outMsgs) {
      if (o?.content) items.push({ role: 'assistant', content: clip(o.content) });
    }
  }

  if (name.startsWith('execute_tool ')) {
    const tool = name.slice('execute_tool '.length);
    if (!NOISY.has(tool)) {
      items.push({ role: 'tool', content: \`\${tool} args=\${clip(input)} result=\${clip(output)}\` });
    }
  }

  return items;
}
`;

export const SYSTEM_PROMPT = `You are evaluating a production trace of an AI form-filling agent that helps caseworkers complete government benefits forms.

You are grading ONE thing: the quality of the agent's \`gapAnalysis\` calls — the questions it asks the caseworker for information it could not obtain itself.

The trace gives you:
- The caseworker's opening message, which usually embeds the participant record as JSON (this is the ground truth for what the agent already knows)
- Any \`getApricotRecord\` tool result (additional ground truth; note it sometimes fails, in which case the JSON in the prompt is the only source)
- The agent's reasoning and its \`gapAnalysis\` calls listing \`missingFields\`

## What good asking looks like
- Asks only for fields genuinely absent from, or unusable in, the participant record
- Uses plain-English field labels, not internal names or selectors
- Does not ask about sensitive fields (SSN, disability, veteran status) unless the form requires them
- Correctly declines to reuse a value that looks similar but means something different (e.g. treating a CalWorks ID as a MediCal case number)
- Explains ambiguity with a short note when a field is genuinely unclear

## What bad asking looks like
- Asks for name, DOB, address, phone, or email when those are plainly in the record
- Asks for sensitive fields the form does not require
- Uses technical labels ("value for field income_monthly_gross")
- Re-asks the same field multiple times within one turn
- Misses a required field that the form needs and the record does not supply

## Scope
Judge only the asking behavior. Ignore browser mechanics, navigation problems, tool errors, and infrastructure failures — those are covered by other scorers.

If the trace contains no \`gapAnalysis\` call at all — including when it is empty or has no readable content — you MUST choose Skip. Never infer a gapAnalysis call that is not present. Never fall back to (C) for a trace you could not read: (C) means the agent asked badly, not that there was nothing to grade.`;

export const USER_PROMPT = `## Production trace
{{preprocessed}}

## Evaluation
Choose exactly one:

(A) Asked only for fields genuinely missing from the participant record, in plain English, without asking about unrequired sensitive fields, and without re-asking known values.

(B) Largely on target, with minor issues — a slightly technical label, one redundant ask, or a single stray question about a field the record arguably covers.

(C) Major issues: asked for fields the record already supplies, asked for unrequired sensitive fields, fabricated questions about fields not on the form, or missed a required missing field.

Skip: the trace contains no gapAnalysis call, or has no readable content to grade.`;

export const EVALUATOR_DEFINITION = {
  name: 'Gap Analysis Asking (Online)',
  slug: EVALUATOR_SLUG,
  description:
    "Trace-scoped LLM judge for production traffic. Grades whether the agent's gapAnalysis calls asked the caseworker for the right missing fields, in plain English, without re-asking values already in the participant record. Skips traces with no gapAnalysis call.",
  prompt_data: {
    prompt: {
      type: 'chat' as const,
      messages: [
        { role: 'system' as const, content: SYSTEM_PROMPT },
        { role: 'user' as const, content: USER_PROMPT },
      ],
    },
    options: { model: JUDGE_MODEL, params: {} },
    parser: {
      type: 'llm_classifier' as const,
      use_cot: true,
      choice_scores: CHOICE_SCORES,
    },
    preprocessor: { type: 'inline' as const, code: PREPROCESSOR_CODE },
    // Unreliable when set via REST: the judge picks (C) instead of Skip on
    // empty traces. BTQL_FILTER is what actually keeps those out.
    allow_skip: true,
  },
  // B (0.5) is "minor issues" and should not count as passing.
  metadata: { __pass_threshold: PASS_THRESHOLD },
};
