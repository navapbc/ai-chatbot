import type {
  InputAnswer,
  PendingInputRequest,
} from '../../../lib/ai/eve/input-requests';
import { isSensitiveField } from '../../../lib/relay/sensitive-fields';

// How the Needs You card renders one pending question. Requests reach the
// panel through a validator that checks only that `pending` is an array, so
// every field here is read defensively and the functions never throw.
//
// - sensitive: the skill's convention (options exactly `provided` and `skip`,
//   no free text), whatever the prompt; or a free-text question (no options)
//   whose prompt the field classifier flags. The card never offers a text
//   input for it: the convention gets "I entered it myself" and "Skip", a
//   flagged free-text question gets a fixed notice and no answer.
// - choice: any other question with options (a prompt that merely mentions a
//   sensitive topic stays answerable: an option id carries no value), or one
//   that does not allow free text. With no options it cannot be answered here.
// - text: free text is allowed, there are no options, the prompt is not
//   flagged.

export type QuestionKind = 'sensitive' | 'choice' | 'text';

const isRecord = (v: unknown): v is Record<string, unknown> =>
  typeof v === 'object' && v !== null && !Array.isArray(v);

const optionIds = (r: Record<string, unknown>): (string | null)[] =>
  Array.isArray(r.options)
    ? r.options.map((o) => (isRecord(o) && typeof o.id === 'string' ? o.id : null))
    : [];

const hasSensitiveShape = (r: Record<string, unknown>): boolean => {
  if (r.allowFreeform === true) return false;
  const ids = optionIds(r);
  if (ids.length !== 2) return false;
  return (
    (ids[0] === 'provided' && ids[1] === 'skip') ||
    (ids[0] === 'skip' && ids[1] === 'provided')
  );
};

const promptFlagged = (r: Record<string, unknown>): boolean => {
  if (typeof r.prompt !== 'string' || r.prompt === '') return false;
  try {
    return isSensitiveField({ label: r.prompt, name: r.prompt });
  } catch {
    // Fail toward sensitive: the card then offers no text input.
    return true;
  }
};

export const classifyQuestion = (req: PendingInputRequest): QuestionKind => {
  const r: Record<string, unknown> = isRecord(req) ? req : {};
  if (hasSensitiveShape(r)) return 'sensitive';
  const options = Array.isArray(r.options) ? r.options : [];
  if (options.length > 0 || r.allowFreeform !== true) return 'choice';
  return promptFlagged(r) ? 'sensitive' : 'text';
};

/**
 * The answer for a sensitive question's button: by option id, and only when
 * the request carries that option. A free-text question flagged by its prompt
 * has no such option and gets null (no answer from the panel): sending the word as
 * free text is not done, because the skill tells the agent that free text is
 * the field's value, so it could type "provided" over what the caseworker
 * entered.
 */
export const sensitiveAnswer = (
  req: PendingInputRequest,
  choice: 'provided' | 'skip',
): InputAnswer | null => {
  if (!isRecord(req)) return null;
  const r: Record<string, unknown> = req;
  if (typeof r.requestId !== 'string' || r.requestId === '') return null;
  return optionIds(r).includes(choice)
    ? { requestId: r.requestId, optionId: choice }
    : null;
};
