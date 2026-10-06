// Recovers the participant record from the conversation.
//
// There is no participant database and no request field carrying the record:
// `buildApplicationPrompt` (lib/data/participants.ts) serialises it into the
// caseworker's opening message, so reading it back out means parsing that
// message. This is the inverse of that writer, and the only reader.
//
// Scans every user message rather than just the latest, because a session
// carries across forms — the caseworker goes WIC → IHSS → BenefitsCal in one
// chat (tests/../session-carryover.eval.ts), and the record appears only in
// the message that started the session.

import type { Participant } from '@/lib/data/participants';

/** A message with enough shape to scan; matches AI SDK and Eve message forms. */
interface ScannableMessage {
  role?: string;
  content?: unknown;
}

const textOf = (content: unknown): string => {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .map((part) =>
      part && typeof part === 'object' && 'text' in part
        ? String((part as { text: unknown }).text ?? '')
        : '',
    )
    .join('\n');
};

/**
 * A participant is identified structurally, not by matching the whole
 * interface: the agent is given whatever `lib/data/participants.ts` holds, and
 * that shape has changed before. `record_id` plus a nested participant name is
 * specific enough to not collide with other JSON a caseworker might paste.
 */
const looksLikeParticipant = (value: unknown): value is Participant => {
  if (!value || typeof value !== 'object') return false;
  const v = value as Record<string, unknown>;
  if (typeof v.record_id !== 'string') return false;
  const p = v.participant;
  if (!p || typeof p !== 'object') return false;
  const name = (p as Record<string, unknown>).name;
  return Boolean(name && typeof name === 'object');
};

/**
 * Pull the first participant-shaped JSON object out of a message's text.
 *
 * Brace-matching rather than a regex: the record is pretty-printed and nested,
 * so a non-greedy match stops at the first inner `}` and a greedy one swallows
 * trailing prose. Quote and escape state are tracked so a brace inside a
 * string value (an address, a note) does not end the scan early.
 */
export const extractParticipantFromText = (text: string): Participant | null =>
  findParticipantSpan(text)?.participant ?? null;

/**
 * Same scan, but also reporting where the record sat in the text.
 *
 * The span is what lets `extractCaseworkerMessages` hand Jev the caseworker's
 * own words WITHOUT a second copy of the record: the record already travels
 * as `participant_record`, and repeating it inside the message text would
 * both waste the request and let Jev double-count it as corroboration.
 */
export const findParticipantSpan = (
  text: string,
): { participant: Participant; start: number; end: number } | null => {
  for (
    let start = text.indexOf('{');
    start !== -1;
    start = text.indexOf('{', start + 1)
  ) {
    let depth = 0;
    let inString = false;
    let escaped = false;
    for (let i = start; i < text.length; i++) {
      const ch = text[i];
      if (escaped) {
        escaped = false;
      } else if (ch === '\\') {
        escaped = true;
      } else if (ch === '"') {
        inString = !inString;
      } else if (!inString && ch === '{') {
        depth++;
      } else if (!inString && ch === '}') {
        depth--;
        if (depth === 0) {
          try {
            const parsed = JSON.parse(text.slice(start, i + 1));
            if (looksLikeParticipant(parsed)) {
              return { participant: parsed, start, end: i + 1 };
            }
          } catch {
            // Not JSON, or not complete — try the next `{`.
          }
          break;
        }
      }
    }
  }
  return null;
};

/**
 * Find the participant record in a conversation, or null when there isn't one.
 *
 * Null is a normal outcome, not an error: a caseworker can open a chat and
 * type freely. Every Jev feature treats it the same as the flag being off.
 */
export const extractParticipant = (
  messages: readonly ScannableMessage[] | undefined,
): Participant | null => {
  if (!messages) return null;
  for (const message of messages) {
    if (message?.role !== 'user') continue;
    const found = extractParticipantFromText(textOf(message.content));
    if (found) return found;
  }
  return null;
};

/**
 * Total characters of caseworker text handed to Jev, newest-first.
 *
 * These messages ride on EVERY summary row's request (~25 per card), so an
 * unbounded transcript multiplies cost and latency by the row count. The cap
 * keeps the recent turns — the answers to a gap analysis, which is exactly
 * what `caseworker_supplied` has to be checked against — and drops the oldest
 * first.
 */
export const CASEWORKER_CONTEXT_CHAR_BUDGET = 8000;

/** One caseworker turn, with the participant record stripped out. */
export const stripParticipantRecord = (text: string): string => {
  const span = findParticipantSpan(text);
  if (!span) return text.trim();
  return (text.slice(0, span.start) + text.slice(span.end))
    .replace(/\n{3,}/g, '\n\n')
    .trim();
};

/**
 * The caseworker's own words, in order, for Jev to check a value against.
 *
 * Why this exists: `summaryFieldQuestions` used to receive only
 * `participant_record`, so Jev could neither validate an inference drawn from
 * the caseworker's instructions nor verify that a `caseworker` label matched
 * something actually said. Measured on run wrun_01M37CZX: of six values Jev
 * called `unsupported`, two were sound inferences it had no way to see —
 * "applying for yourself" (the task names the participant as the applicant)
 * and "minor adopted child = No" (age 26, derivable from the record's DOB but
 * only once you know the question is about the applicant).
 *
 * Assistant turns are deliberately excluded: the agent's own account of what
 * it did is not evidence that the caseworker said it.
 */
export const extractCaseworkerMessages = (
  messages: readonly ScannableMessage[] | undefined,
  charBudget: number = CASEWORKER_CONTEXT_CHAR_BUDGET,
): string[] => {
  if (!messages) return [];
  const all = messages
    .filter((m) => m?.role === 'user')
    .map((m) => stripParticipantRecord(textOf(m.content)))
    .filter((t) => t.length > 0);

  // Walk backwards so the cap drops the OLDEST turns, then restore order.
  const kept: string[] = [];
  let used = 0;
  for (let i = all.length - 1; i >= 0; i--) {
    const next = used + all[i].length;
    if (kept.length > 0 && next > charBudget) break;
    kept.push(all[i]);
    used = next;
  }
  return kept.reverse();
};
