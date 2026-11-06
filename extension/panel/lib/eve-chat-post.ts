import { AppApiError } from '../../src/app-api';

// One POST to the app's eve-chat route, shared by the run starter and the
// answer client. The route answers with the turn's SSE stream and keeps
// reading the turn after the client leaves, so this reads only until the
// first chunk, then cancels the reader (not awaited: the panel never waits on
// the turn). Headers, the first chunk and any refusal body read share one
// timeout. Errors are typed; nothing here logs.

/** Bounds the eve-chat request: headers and the first chunk. */
export const FIRST_CHUNK_TIMEOUT_MS = 15_000;

type Timer = ReturnType<typeof setTimeout>;

export type EveChatPostDeps = {
  fetch: typeof fetch;
  appOrigin: string;
  setTimeout?: (fn: () => void, ms: number) => Timer;
  clearTimeout?: (id: Timer) => void;
};

export const codeForStatus = (status: number): AppApiError['code'] => {
  if (status === 401) return 'signed_out';
  if (status === 403) return 'forbidden';
  if (status === 429) return 'rate_limited';
  if (status === 400) return 'bad_request';
  return 'server_error';
};

/**
 * Turns a non-2xx response into the error to throw, or undefined for the
 * status mapping. Given the request's abort signal so a body read is bounded
 * by the same timeout. Absent, the body is never read.
 */
export type RefusalReader = (
  response: Response,
  signal: AbortSignal,
) => Promise<unknown>;

/**
 * Posts `body` (already JSON) to `${appOrigin}/api/eve-chat`. Resolves once
 * the first chunk of the turn's stream arrives; throws a typed error
 * otherwise.
 */
export const postEveChatTurn = async (
  deps: EveChatPostDeps,
  body: string,
  readRefusal?: RefusalReader,
): Promise<void> => {
  const set = deps.setTimeout ?? ((fn, ms) => setTimeout(fn, ms));
  const clear = deps.clearTimeout ?? ((id) => clearTimeout(id));
  const controller = new AbortController();
  const timer = set(() => controller.abort(), FIRST_CHUNK_TIMEOUT_MS);
  try {
    let response: Response;
    try {
      response = await deps.fetch(`${deps.appOrigin}/api/eve-chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body,
        credentials: 'include',
        cache: 'no-store',
        redirect: 'error',
        signal: controller.signal,
      });
    } catch {
      throw new AppApiError('network', 0);
    }
    if (!response.ok) {
      const refused = readRefusal
        ? await readRefusal(response, controller.signal)
        : undefined;
      if (refused !== undefined) throw refused;
      // Not read (or already read by the hook): its text must not reach the
      // panel.
      response.body?.cancel().catch(() => {});
      throw new AppApiError(codeForStatus(response.status), response.status);
    }
    if (!response.body) throw new AppApiError('network', 0);
    const reader = response.body.getReader();
    let first: ReadableStreamReadResult<Uint8Array>;
    try {
      first = await reader.read();
    } catch {
      throw new AppApiError('network', 0);
    }
    reader.cancel().catch(() => {});
    if (first.done) throw new AppApiError('network', 0);
  } finally {
    clear(timer);
  }
};
