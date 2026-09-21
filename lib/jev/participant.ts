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
export const extractParticipantFromText = (
  text: string,
): Participant | null => {
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
            if (looksLikeParticipant(parsed)) return parsed;
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
