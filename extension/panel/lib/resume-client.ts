import { AnswerInFlightError, NotConnectedError, RunEndedError } from './copy';
import {
  type EveChatPostDeps,
  FIRST_CHUNK_TIMEOUT_MS,
  postEveChatTurn,
  type RefusalReader,
} from './eve-chat-post';

// Tells the agent the caseworker is done with a pause: a follow-up user turn
// on the relay run, `{ id: runId, message }`. The text is fixed and carries
// no page or applicant content (the server's sensitive-value guard passes
// it). The agent's snapshot refs died with the pause, so the text tells it
// to read the page again before it goes on. A 400 `run_ended` maps to the
// fixed ended copy; everything else is the status mapping.

export const RESUME_TIMEOUT_MS = FIRST_CHUNK_TIMEOUT_MS;

export const RESUME_MESSAGE =
  'I paused the run, changed the page myself, and I am resuming it now. ' +
  'Nothing you remember about the page is reliable. Take a fresh snapshot ' +
  'first, compare it with your progress report, and treat any field that is ' +
  'now filled as done: do not overwrite it. Then continue with what is left. ' +
  'Call report_progress again right away so the counts match the page.';

export type ResumeResult = { ok: true } | { ok: false; error: unknown };

const inFlight = new Set<string>();

const readReason: RefusalReader = async (response, signal) => {
  if (response.status !== 400) return undefined;
  const reader = response.body?.getReader();
  if (!reader) return undefined;
  const handleAbort = () => {
    reader.cancel().catch(() => {});
  };
  if (signal.aborted) handleAbort();
  signal.addEventListener('abort', handleAbort);
  let text = '';
  try {
    for (let size = 0; size <= 4096; ) {
      const { done, value } = await reader.read();
      if (done) break;
      size += value.byteLength;
      text += new TextDecoder().decode(value, { stream: true });
    }
  } catch {
    return undefined;
  } finally {
    signal.removeEventListener('abort', handleAbort);
    reader.cancel().catch(() => {});
  }
  try {
    const parsed: unknown = JSON.parse(text);
    const cause =
      typeof parsed === 'object' && parsed !== null
        ? (parsed as Record<string, unknown>).cause
        : undefined;
    return cause === 'run_ended' ? new RunEndedError() : undefined;
  } catch {
    return undefined;
  }
};

export const resumeAgentTurn = async (
  deps: EveChatPostDeps,
  run: { runId: string },
  newId: () => string,
): Promise<ResumeResult> => {
  if (inFlight.has(run.runId)) {
    return { ok: false, error: new AnswerInFlightError() };
  }
  inFlight.add(run.runId);
  try {
    await postEveChatTurn(
      deps,
      JSON.stringify({
        id: run.runId,
        message: {
          id: newId(),
          role: 'user',
          parts: [{ type: 'text', text: RESUME_MESSAGE }],
        },
      }),
      readReason,
    );
    return { ok: true };
  } catch (error) {
    return { ok: false, error };
  } finally {
    inFlight.delete(run.runId);
  }
};

/** The real sender: the browser's fetch against the build's app origin. */
export const createResumeSender =
  (appOrigin: string) =>
  (runId: string): Promise<ResumeResult> => {
    if (!appOrigin) {
      return Promise.resolve({ ok: false, error: new NotConnectedError() });
    }
    return resumeAgentTurn(
      { fetch: (input, init) => globalThis.fetch(input, init), appOrigin },
      { runId },
      () => crypto.randomUUID(),
    );
  };
