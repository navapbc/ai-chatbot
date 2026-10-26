export interface InputOption {
  id: string;
  label: string;
}

export interface PendingInputRequest {
  requestId: string;
  prompt: string;
  display: 'text' | 'select';
  options: InputOption[];
  allowFreeform: boolean;
}

export interface InputAnswer {
  requestId: string;
  optionId?: string;
  text?: string;
}

export type AnswerError =
  | 'unknown_request'
  | 'missing_answer'
  | 'duplicate_answer'
  | 'unknown_option'
  | 'text_not_allowed'
  | 'empty_answer';

const MAX_TEXT_LENGTH = 2000;

const isRecord = (v: unknown): v is Record<string, unknown> =>
  typeof v === 'object' && v !== null && !Array.isArray(v);

/** Parses `event.data.requests` from Eve's `input.requested`; drops malformed entries. */
export function parseInputRequests(raw: unknown): PendingInputRequest[] {
  if (!Array.isArray(raw)) return [];
  const out: PendingInputRequest[] = [];
  for (const item of raw) {
    if (!isRecord(item)) continue;
    const { requestId, prompt, display } = item;
    if (typeof requestId !== 'string' || requestId === '') continue;
    if (typeof prompt !== 'string') continue;
    if (display !== 'text' && display !== 'select') continue;
    const options: InputOption[] = [];
    if (Array.isArray(item.options)) {
      for (const o of item.options) {
        if (
          isRecord(o) &&
          typeof o.id === 'string' &&
          typeof o.label === 'string'
        ) {
          options.push({ id: o.id, label: o.label });
        }
      }
    }
    out.push({
      requestId,
      prompt,
      display,
      options,
      // With options, fail closed: only an explicit true permits free text.
      // With none, text is the only possible answer: Eve's ask_question omits
      // allowFreeform by default, and an option-only request with no options
      // could never be answered, deadlocking the run.
      allowFreeform: options.length === 0 || item.allowFreeform === true,
    });
  }
  return out;
}

const fail = (error: AnswerError) => ({ ok: false, error }) as const;

/**
 * Checks caseworker answers against the pending requests. Every pending
 * request needs exactly one answer. Free text is accepted only where
 * `allowFreeform` is true; on such a request an `optionId` that is also sent
 * with text is used and the text is dropped.
 */
export function validateAnswers(
  pending: readonly PendingInputRequest[],
  answers: unknown,
): { ok: true; responses: InputAnswer[] } | { ok: false; error: AnswerError } {
  if (!Array.isArray(answers)) return fail('missing_answer');
  const byId = new Map(pending.map((p) => [p.requestId, p]));
  const seen = new Set<string>();
  const responses: InputAnswer[] = [];

  for (const answer of answers) {
    if (!isRecord(answer)) return fail('unknown_request');
    const { requestId, optionId, text } = answer;
    const request =
      typeof requestId === 'string' ? byId.get(requestId) : undefined;
    if (!request) return fail('unknown_request');
    if (seen.has(request.requestId)) return fail('duplicate_answer');
    seen.add(request.requestId);

    if (text !== undefined && text !== null && !request.allowFreeform) {
      return fail('text_not_allowed');
    }

    if (optionId !== undefined && optionId !== null) {
      if (
        typeof optionId !== 'string' ||
        !request.options.some((o) => o.id === optionId)
      ) {
        return fail('unknown_option');
      }
      responses.push({ requestId: request.requestId, optionId });
      continue;
    }

    if (typeof text !== 'string') return fail('empty_answer');
    const trimmed = text.trim();
    if (trimmed === '' || trimmed.length > MAX_TEXT_LENGTH) {
      return fail('empty_answer');
    }
    responses.push({ requestId: request.requestId, text: trimmed });
  }

  if (pending.some((p) => !seen.has(p.requestId)))
    return fail('missing_answer');
  return { ok: true, responses };
}
