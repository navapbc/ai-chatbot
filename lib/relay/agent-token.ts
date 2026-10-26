// Eve-side fetch of the run's short-lived relay agent token. The token is
// never stored in Eve session state (Eve's Postgres world persists session
// attributes and run inputs): only the non-secret run id is, and this fetches
// a fresh token from the app at call time.
//
// This file is free of `server-only` and Eve imports so Eve's bundler can take
// it. Nothing here may put the token, the internal secret or a caught error's
// message into a returned value, a log line or a thrown error.

export type AgentTokenResult =
  | { ok: true; token: string; expiresAt: number }
  | { ok: false; error: string };

export interface AgentTokenOptions {
  fetchImpl?: typeof fetch;
  baseUrl?: string;
  /** Epoch seconds, the unit of `expiresAt`. */
  now?: () => number;
}

const REQUEST_TIMEOUT_MS = 5_000;
const REFRESH_MARGIN_SECONDS = 60;

const ERR_UNAVAILABLE = 'The relay token service is unavailable.';
const ERR_REFUSED = 'The relay token request was refused.';
const ERR_INVALID = 'The relay token response was invalid.';

// Per run id, in this process only.
const cache = new Map<string, { token: string; expiresAt: number }>();

/** Test hook. */
export function _resetAgentTokenCache(): void {
  cache.clear();
}

/** Never throws. Errors are fixed strings, free of any credential. */
export async function fetchAgentToken(
  runId: string,
  opts: AgentTokenOptions = {},
): Promise<AgentTokenResult> {
  try {
    const now = opts.now ?? (() => Math.floor(Date.now() / 1000));
    const cached = cache.get(runId);
    if (cached && cached.expiresAt - REFRESH_MARGIN_SECONDS > now()) {
      return { ok: true, ...cached };
    }

    const secret = process.env.EVE_INTERNAL_SECRET;
    if (!secret) return { ok: false, error: ERR_UNAVAILABLE };

    const baseUrl =
      opts.baseUrl ?? process.env.APP_INTERNAL_URL ?? 'http://127.0.0.1:3000';
    const fetchImpl = opts.fetchImpl ?? fetch;

    let response: Response;
    try {
      response = await fetchImpl(`${baseUrl}/api/relay/agent-token`, {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          'x-eve-internal-secret': secret,
        },
        body: JSON.stringify({ runId }),
        signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
      });
    } catch {
      return { ok: false, error: ERR_UNAVAILABLE };
    }
    if (!response.ok) return { ok: false, error: ERR_REFUSED };

    let body: unknown;
    try {
      body = await response.json();
    } catch {
      return { ok: false, error: ERR_INVALID };
    }
    const { token, expiresAt } = (body ?? {}) as Record<string, unknown>;
    if (
      typeof token !== 'string' ||
      token.length === 0 ||
      typeof expiresAt !== 'number' ||
      !Number.isFinite(expiresAt)
    ) {
      return { ok: false, error: ERR_INVALID };
    }
    cache.set(runId, { token, expiresAt });
    return { ok: true, token, expiresAt };
  } catch {
    return { ok: false, error: ERR_UNAVAILABLE };
  }
}
