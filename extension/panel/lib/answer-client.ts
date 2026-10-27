import type { InputAnswer } from '../../../lib/ai/eve/input-requests';
import {
  AnswerAlreadyGivenError,
  AnswerInFlightError,
  AnswerInvalidError,
  AnswerRefusedError,
  NotConnectedError,
  RunEndedError,
} from './copy';
import {
  type EveChatPostDeps,
  FIRST_CHUNK_TIMEOUT_MS,
  postEveChatTurn,
  type RefusalReader,
} from './eve-chat-post';

// Sends the caseworker's answers to the agent's open questions: an answer
// turn on the relay run, `{ id: runId, inputResponses }` and nothing else (no
// `message`, no `relay`). The server validates the answers against the
// pending set and runs its sensitive-value guard on any text; the panel does
// not reimplement that guard, it only maps the refusal to fixed copy.
//
// Nothing about an answer is logged, stored or kept: the answers are copied
// into the request body and dropped when the call returns. A 400's body is
// read (bounded) only to pick a reason from a fixed allowlist; its text never
// leaves this module.

export type AnswerDeps = EveChatPostDeps;
export type AnswerResult = { ok: true } | { ok: false; error: unknown };

export const ANSWER_TIMEOUT_MS = FIRST_CHUNK_TIMEOUT_MS;
export const MAX_ANSWER_CHARS = 2000;
/** A refusal body larger than this is not read for a reason. */
const MAX_REFUSAL_BYTES = 4096;

// Request ids with an answer on its way. Module-level: the screen builds the
// sender once, but a second panel press must still see the first.
const inFlight = new Set<string>();

const isRecord = (v: unknown): v is Record<string, unknown> =>
  typeof v === 'object' && v !== null && !Array.isArray(v);

/** Fresh, minimal answers, or null when any answer is invalid. */
const normalize = (answers: unknown): InputAnswer[] | null => {
  if (!Array.isArray(answers) || answers.length === 0) return null;
  const seen = new Set<string>();
  const out: InputAnswer[] = [];
  for (const answer of answers) {
    if (!isRecord(answer)) return null;
    const { requestId, optionId, text } = answer;
    if (typeof requestId !== 'string' || requestId === '') return null;
    if (seen.has(requestId)) return null;
    seen.add(requestId);
    const hasOption = optionId !== undefined;
    const hasText = text !== undefined;
    if (hasOption === hasText) return null;
    if (hasOption) {
      if (typeof optionId !== 'string' || optionId === '') return null;
      out.push({ requestId, optionId });
      continue;
    }
    if (typeof text !== 'string') return null;
    const trimmed = text.trim();
    if (trimmed === '' || trimmed.length > MAX_ANSWER_CHARS) return null;
    out.push({ requestId, text: trimmed });
  }
  return out;
};

const REASONS: Record<string, () => Error> = {
  unknown_request: () => new AnswerAlreadyGivenError(),
  answers_pending: () => new AnswerAlreadyGivenError(),
  sensitive_value_in_message: () => new AnswerRefusedError(),
  message_too_large: () => new AnswerRefusedError(),
  run_ended: () => new RunEndedError(),
};

/** Reads at most MAX_REFUSAL_BYTES; null when larger, unreadable or aborted. */
const readSmall = async (
  response: Response,
  signal: AbortSignal,
): Promise<string | null> => {
  if (!response.body) return null;
  const reader = response.body.getReader();
  const handleAbort = () => {
    reader.cancel().catch(() => {});
  };
  if (signal.aborted) handleAbort();
  signal.addEventListener('abort', handleAbort);
  const chunks: Uint8Array[] = [];
  let size = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (signal.aborted) return null;
      if (done) break;
      size += value.byteLength;
      if (size > MAX_REFUSAL_BYTES) return null;
      chunks.push(value);
    }
  } catch {
    return null;
  } finally {
    signal.removeEventListener('abort', handleAbort);
    reader.cancel().catch(() => {});
  }
  const bytes = new Uint8Array(size);
  let at = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, at);
    at += chunk.byteLength;
  }
  return new TextDecoder().decode(bytes);
};

/** A 400's reason as a fixed error; undefined for the status mapping. */
const readReason: RefusalReader = async (response, signal) => {
  if (response.status !== 400) return undefined;
  const text = await readSmall(response, signal);
  if (text === null) return undefined;
  let cause: unknown;
  try {
    const parsed: unknown = JSON.parse(text);
    cause = isRecord(parsed) ? parsed.cause : undefined;
  } catch {
    return undefined;
  }
  if (typeof cause !== 'string' || !Object.hasOwn(REASONS, cause)) {
    return undefined;
  }
  return REASONS[cause]();
};

export const answerRequests = async (
  deps: AnswerDeps,
  run: { runId: string },
  answers: InputAnswer[],
): Promise<AnswerResult> => {
  const responses = normalize(answers);
  if (!responses) return { ok: false, error: new AnswerInvalidError() };
  const ids = responses.map((r) => r.requestId);
  if (ids.some((id) => inFlight.has(id))) {
    return { ok: false, error: new AnswerInFlightError() };
  }
  for (const id of ids) inFlight.add(id);
  try {
    await postEveChatTurn(
      deps,
      JSON.stringify({ id: run.runId, inputResponses: responses }),
      readReason,
    );
    return { ok: true };
  } catch (error) {
    return { ok: false, error };
  } finally {
    for (const id of ids) inFlight.delete(id);
  }
};

/**
 * The real sender for the Needs You card: the browser's fetch against the
 * build's app origin. An empty origin (an unconnected build) never fetches.
 */
export const createAnswerSender =
  (appOrigin: string) =>
  (runId: string, answers: InputAnswer[]): Promise<AnswerResult> => {
    if (!appOrigin) {
      return Promise.resolve({ ok: false, error: new NotConnectedError() });
    }
    return answerRequests(
      { fetch: (input, init) => globalThis.fetch(input, init), appOrigin },
      { runId },
      answers,
    );
  };
