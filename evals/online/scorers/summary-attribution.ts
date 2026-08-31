// Online variant of evals/scorers/summary-attribution.ts. Grades whether each
// formSummary source label matches where the value actually came from.

import { JUDGE_MODEL, PASS_THRESHOLD } from './shared';

export const EVALUATOR_SLUG = 'summary-attribution-online';

// Needs the record (prompt), what was written, and the summary's claims.
export const PREPROCESSOR_CODE = `const CAP = 1200;
// formSummary is the evidence being graded — truncating it made a judge
// speculate about the cut-off text and fail a correct run.
const SUMMARY_CAP = 12000;
const WRITES = new Set(['fill', 'select', 'type', 'check']);

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
  }

  if (name === 'execute_tool browser') {
    const cmd = input?.command;
    if (Array.isArray(cmd) && WRITES.has(cmd[0])) {
      items.push({ role: 'tool', content: 'form write: ' + clip(cmd) });
    }
  }

  if (name === 'execute_tool formSummary') {
    items.push({ role: 'tool', content: 'formSummary: ' + clip(input, SUMMARY_CAP) });
  }

  return items;
}
`;

export const SYSTEM_PROMPT = `You are checking the source labels in an AI form-filling agent's summary.

After filling a form, the agent emits a \`formSummary\` listing each field with a value and a source label:

- **database** — taken from the participant record (may be reformatted, but the value must be present there)
- **caseworker** — supplied by the caseworker during the conversation
- **inferred** — logically derived from record data (age from DOB; mailing address from residential when identical)
- **missing** — not available; left blank or flagged

The trace gives you the participant record (in the caseworker's opening message), the values actually written to the form, and the formSummary itself.

## Good attribution
- Every "database" value genuinely appears in the record
- Every "inferred" value follows from something in the record
- Every "caseworker" value matches something the caseworker actually said
- "missing" is used for blanks, not for values that were in fact entered
- Summary values match what was written to the form

## Bad attribution
- A fabricated or guessed value labelled "database"
- A known record value labelled "missing" or "caseworker"
- An "inferred" value with no derivation from the record
- Summary values that contradict the actual form writes

## Scope
Judge only attribution accuracy. Ignore verbosity, question quality, and browser mechanics.

If the trace contains no formSummary call, choose Skip.

Never infer wrongdoing from truncated text. If a value you need was cut off by a truncation marker, judge only what you can actually see; do not assume the hidden text contains a violation.`;

export const USER_PROMPT = `## Production trace
{{preprocessed}}

## Evaluation
Choose exactly one:

(A) Every source label matches the real origin, and summary values match what was entered.

(B) One or two labels are arguably wrong — a record value marked "inferred", or a direct value marked "database" when it was reformatted — but nothing fabricated is attributed to the database.

(C) Misleading: a fabricated value attributed to "database", known record values marked "missing", or summary values that contradict what was actually entered.

Skip: the trace contains no formSummary call.`;

export const EVALUATOR_DEFINITION = {
  name: 'Summary Attribution (Online)',
  slug: EVALUATOR_SLUG,
  description:
    "Trace-scoped LLM judge. Grades whether each formSummary field's source label (database/caseworker/inferred/missing) reflects where the value actually came from.",
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
