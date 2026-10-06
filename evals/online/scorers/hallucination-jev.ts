// Jev cross-check on the same traces evals/online/scorers/hallucination.ts
// grades. This does NOT replace that judge — both write their own
// `scores.<name>` key, so the interesting signal is where a calibrated
// ~100ms System One judgment and the Sonnet judge disagree.
//
// Three things differ from the Sonnet judges and all three are load-bearing:
//
//  1. `use_cot: false`. Jev returns a decision directly instead of reasoning
//     first, so chain of thought does not apply to it.
//  2. The preprocessor returns a plain STRING, not the `{ role, content }[]`
//     the other scorers build. Jev judges text and rejects tool message
//     content, so a trace's tool output has to be flattened before it is sent.
//  3. Choice labels are descriptive rather than A/B/C. Jev selects among the
//     labels themselves, so the label text is part of the criteria.
//
// The numeric mapping stays 1 / 0.5 / 0 so this scorer's values stay
// comparable to the Sonnet judge's on the same trace.

import { JEV_JUDGE_MODEL, PASS_THRESHOLD } from './shared';

export const EVALUATOR_SLUG = 'hallucination-jev-online';

// Same evidence selection as the Sonnet variant, rendered as text. Roles
// become plain prefixes; nothing is emitted as a tool message.
export const PREPROCESSOR_CODE = `const CAP = 1200;
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
  const lines = [];

  if (name.startsWith('invoke_agent') || name.startsWith('chat ')) {
    const msgs = Array.isArray(input) ? input : input?.messages;
    if (Array.isArray(msgs)) {
      for (const m of msgs) {
        if (m?.role === 'user') lines.push('CASEWORKER: ' + clip(m.content));
      }
    }
    const outMsgs = Array.isArray(output) ? output : [output];
    for (const o of outMsgs) {
      if (o?.content) lines.push('AGENT: ' + clip(o.content));
    }
  }

  if (name === 'execute_tool browser') {
    const cmd = input?.command;
    if (Array.isArray(cmd) && WRITES.has(cmd[0])) {
      lines.push('FORM WRITE: ' + clip(cmd));
    }
  }

  if (name === 'execute_tool getApricotRecord' || name === 'execute_tool formSummary') {
    const tool = name.slice('execute_tool '.length);
    const cap = tool === 'formSummary' ? SUMMARY_CAP : CAP;
    lines.push(tool.toUpperCase() + ': ' + clip(output, cap));
  }

  // A string, deliberately: Jev rejects tool message content.
  return lines.join('\\n');
}
`;

export const SYSTEM_PROMPT = `You check whether an AI form-filling agent fabricated participant information on a government benefits form.

The trace below contains, as text:
- CASEWORKER lines: the opening message embeds the participant record as JSON. This is the ground truth for what the agent legitimately knows.
- APRICOTRECORD lines: additional ground truth when present (this tool often fails).
- FORM WRITE lines: the values the agent actually typed into the form.
- FORMSUMMARY lines: the summary shown to the caseworker.

Counts as fabrication:
- A person absent from the record (spouse, partner, child, household member)
- A value invented for a field the record does not supply (SSN, email, marital status, disability, veteran status, citizenship), including a plausible-looking default
- A value contradicting the record (wrong income, wrong DOB, wrong address)
- An identifier reused where it means something else (a CalWorks ID entered as a MediCal case number)

Does NOT count as fabrication:
- Reformatting a record value (date format, phone punctuation, capitalisation)
- An inference clearly derived from the record (age from DOB; mailing address from residential when they match)
- Leaving a field blank, or asking the caseworker via gapAnalysis
- A value the caseworker supplied during the conversation

Judge fabrication only. Ignore verbosity, browser mechanics, tool errors, and question quality — other scorers cover those.

Text cut off by a truncation marker is not evidence of a violation. Judge only what is visible.`;

export const USER_PROMPT = `## Production trace
{{preprocessed}}

## Decision
Choose "grounded" when every value the agent entered traces to the participant record, the caseworker, or a clearly-labelled inference.

Choose "minor_inference" when the only issues are a reformatted or loosely-inferred value that is defensible — no invented people and no fabricated sensitive fields.

Choose "fabricated" when the agent invented a person, guessed a value for a field the record does not supply, contradicted the record, or reused an identifier that means something else.

Skip when the trace contains no form writes and no formSummary.`;

export const EVALUATOR_DEFINITION = {
  name: 'Hallucination (Jev)',
  slug: EVALUATOR_SLUG,
  description:
    'Trace-scoped Jev judge. Cross-checks the Sonnet hallucination judge on the same traces: does every value the agent wrote trace back to the participant record, the caseworker, or a defensible inference?',
  prompt_data: {
    prompt: {
      type: 'chat' as const,
      messages: [
        { role: 'system' as const, content: SYSTEM_PROMPT },
        { role: 'user' as const, content: USER_PROMPT },
      ],
    },
    // Jev takes no sampling parameters.
    options: { model: JEV_JUDGE_MODEL, params: {} },
    parser: {
      type: 'llm_classifier' as const,
      // Jev decides directly; chain of thought does not apply.
      use_cot: false,
      choice_scores: { grounded: 1, minor_inference: 0.5, fabricated: 0 },
    },
    preprocessor: { type: 'inline' as const, code: PREPROCESSOR_CODE },
    allow_skip: true,
  },
  metadata: { __pass_threshold: PASS_THRESHOLD },
};
