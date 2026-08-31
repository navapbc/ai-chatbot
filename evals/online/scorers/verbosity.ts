// Online variant of evals/scorers/verbosity.ts. Grades the agent's
// caseworker-facing prose for conciseness.

import { JUDGE_MODEL, PASS_THRESHOLD } from './shared';

export const EVALUATOR_SLUG = 'verbosity-online';

// Assistant text only. Tool calls and browser output are what the agent DID,
// not what the caseworker had to read.
export const PREPROCESSOR_CODE = `const CAP = 1200;

function clip(v) {
  const s = typeof v === 'string' ? v : JSON.stringify(v);
  if (s == null) return '';
  return s.length > CAP ? s.slice(0, CAP) + ' …[truncated]' : s;
}

function handler({ output, span_attributes }) {
  const name = span_attributes?.name || '';
  if (!name.startsWith('invoke_agent') && !name.startsWith('chat ')) return [];

  const items = [];
  const outMsgs = Array.isArray(output) ? output : [output];
  for (const o of outMsgs) {
    const c = o?.content;
    if (typeof c === 'string' && c.trim()) {
      items.push({ role: 'assistant', content: clip(c) });
    }
  }
  return items;
}
`;

export const SYSTEM_PROMPT = `You are grading an AI form-filling agent's text responses for conciseness.

The agent fills benefits applications for caseworkers, who are busy and want short scannable updates — not commentary on every click. You are shown only the agent's own messages, in order.

## Good
- 1–3 short sentences per message
- States intent ("Filling in personal information now") without narrating each action
- No technical leakage: no CSS selectors, no \`@e9\` refs, no raw field IDs, no internal tool names
- Speaks up when something meaningful happens — a section done, a blocker, a question — not after every field

## Bad
- Play-by-play: "Now I'm clicking next. Now I'm typing M-a-r-i-a..."
- Walls of text describing what the caseworker can already see
- Leaking selectors or refs ("filling @e9 with 'Garcia'")
- Several paragraphs where one sentence would do

## Scope
Judge only the prose the caseworker reads. Ignore correctness of the data, question quality, and whether the form was completed — other scorers cover those.

If the trace contains no assistant text, choose Skip.`;

export const USER_PROMPT = `## Agent messages, in order
{{preprocessed}}

## Evaluation
Choose exactly one:

(A) Concise and action-oriented throughout. No play-by-play, no technical leakage, nothing a caseworker would skim past.

(B) Slightly chatty — occasional unnecessary narration, one message that could be tighter, or a stray technical reference. Still usable.

(C) Verbose, play-by-play, or technical. Narrates individual actions, leaks selectors or refs, or buries the status in walls of text.

Skip: the trace contains no assistant text.`;

export const EVALUATOR_DEFINITION = {
  name: 'Verbosity (Online)',
  slug: EVALUATOR_SLUG,
  description:
    "Trace-scoped LLM judge. Grades the agent's caseworker-facing messages for conciseness, penalising play-by-play narration and leaked selectors.",
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
