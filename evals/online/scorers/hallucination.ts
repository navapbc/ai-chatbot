// Online variant of evals/scorers/hallucination.ts. Ground truth comes from the
// participant JSON in the caseworker's opening message, not an eval serializer.

import { JUDGE_MODEL, PASS_THRESHOLD } from './shared';

export const EVALUATOR_SLUG = 'hallucination-online';

// Keeps the caseworker prompt and the values the agent actually committed.
// Browser snapshots are dropped, but `fill`/`select` argv is kept — it is the
// only record of what was typed into the form.
export const PREPROCESSOR_CODE = `const CAP = 1200;
// formSummary is the evidence being graded — truncating it made a judge
// speculate about the cut-off text and fail a correct run.
const SUMMARY_CAP = 12000;
const WRITES = new Set(['fill', 'select', 'type', 'check', 'press']);

function clip(v, cap) {
  const s = typeof v === 'string' ? v : JSON.stringify(v);
  if (s == null) return '';
  const limit = cap || CAP;
  return s.length > limit ? s.slice(0, limit) + ' …[truncated]' : s;
}

function handler({ input, output, span_attributes }) {
  const name = span_attributes?.name || '';
  const items = [];

  if (name.startsWith('invoke_agent') || name.startsWith('chat ')) {
    const msgs = Array.isArray(input) ? input : input?.messages;
    if (Array.isArray(msgs)) {
      for (const m of msgs) {
        if (m?.role === 'user') items.push({ role: 'user', content: clip(m.content) });
      }
    }
    const outMsgs = Array.isArray(output) ? output : [output];
    for (const o of outMsgs) {
      if (o?.content) items.push({ role: 'assistant', content: clip(o.content) });
    }
  }

  if (name === 'execute_tool browser') {
    const cmd = input?.command;
    if (Array.isArray(cmd) && WRITES.has(cmd[0])) {
      items.push({ role: 'tool', content: 'form write: ' + clip(cmd) });
    }
  }

  if (name === 'execute_tool getApricotRecord' || name === 'execute_tool formSummary') {
    const tool = name.slice('execute_tool '.length);
    const cap = tool === 'formSummary' ? SUMMARY_CAP : CAP;
    items.push({ role: 'tool', content: tool + ' => ' + clip(output, cap) });
  }

  return items;
}
`;

export const SYSTEM_PROMPT = `You are checking whether an AI form-filling agent fabricated participant information.

The agent helps caseworkers complete government benefits forms. The trace gives you:
- The caseworker's opening message, which embeds the participant record as JSON. This is the ground truth for what the agent legitimately knows.
- Any \`getApricotRecord\` result (additional ground truth; it often fails, in which case the prompt JSON is the only source).
- The values the agent actually wrote into the form (\`form write:\` lines).
- Any \`formSummary\` output.

## What counts as hallucination
1. Invented people — spouses, partners, children, or household members absent from the record
2. Fabricated values for fields the record does not supply (SSN, email, marital status, disability, veteran status, citizenship) — including guessing a plausible default
3. Values that contradict the record (wrong income, wrong DOB, wrong address)
4. Reusing a value that means something else (e.g. entering a CalWorks ID where a MediCal case number is required)

## What does NOT count
- Reformatting a record value (date format, phone punctuation, capitalisation)
- Reasonable inference clearly derived from the record (age from DOB; mailing address from residential when they match)
- Leaving a field blank, or asking the caseworker for it via gapAnalysis — that is correct behavior
- Selecting a value the caseworker supplied during the conversation

## Scope
Judge only fabrication. Ignore verbosity, browser mechanics, tool errors, and whether the agent asked good questions — other scorers cover those.

If the trace shows no values written to the form and no formSummary, choose Skip.

Never infer wrongdoing from truncated text. If a value you need was cut off by a truncation marker, judge only what you can actually see; do not assume the hidden text contains a violation.`;

export const USER_PROMPT = `## Production trace
{{preprocessed}}

## Evaluation
Choose exactly one:

(A) Every value the agent entered traces to the participant record, the caseworker, or a clearly-labelled inference. Nothing invented.

(B) Minor issues only — a reformatted or loosely-inferred value that is defensible, but no invented people and no fabricated sensitive fields.

(C) Fabrication: invented a person, guessed a value for a field the record does not supply, contradicted the record, or reused an identifier that means something else.

Skip: the trace contains no form writes and no formSummary.`;

export const EVALUATOR_DEFINITION = {
  name: 'Hallucination (Online)',
  slug: EVALUATOR_SLUG,
  description:
    'Trace-scoped LLM judge. Checks whether values the agent wrote into the form trace back to the participant record, the caseworker, or a defensible inference.',
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
      choice_scores: { A: 1, B: 0.5, C: 0 },
    },
    preprocessor: { type: 'inline' as const, code: PREPROCESSOR_CODE },
    allow_skip: true,
  },
  metadata: { __pass_threshold: PASS_THRESHOLD },
};
