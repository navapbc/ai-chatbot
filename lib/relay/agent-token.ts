// Eve-side fetch of the run's short-lived relay agent token. The token is
// never stored in Eve session state (Eve's Postgres world persists session
// attributes and run inputs): only the non-secret run id is, and this fetches
// a fresh token from the app at call time.
//
// This file is free of `server-only` and Eve imports so Eve's bundler can take
// it. Nothing here may put the token, the internal secret or a caught error's
// message into a returned value, a log line or a thrown error.

import { internalSecret } from './internal-secret';

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
// Epoch seconds stay below this until the year 5138; milliseconds do not.
const MAX_PLAUSIBLE_EXPIRES_AT = 1e11;

const ERR_UNAVAILABLE = 'The relay token service is unavailable.';
const ERR_REFUSED = 'The relay token request was refused.';
const ERR_INVALID = 'The relay token response was invalid.';

// Per (run id, Eve session id), in this process only: the route grants a
// token to the session the run was created with, so a token cached for one
// session must never be handed to another.
const cache = new Map<string, { token: string; expiresAt: number }>();
const cacheKey = (runId: string, sessionId: string) =>
  JSON.stringify([runId, sessionId]);

/** Test hook. */
export function _resetAgentTokenCache(): void {
  cache.clear();
}

/**
 * Never throws. Errors are fixed strings, free of any credential.
 * `sessionId` is the calling Eve session (`ctx.session.id`); the route only
 * mints for the session the run was created with.
 */
export async function fetchAgentToken(
  runId: string,
  sessionId: string,
  opts: AgentTokenOptions = {},
): Promise<AgentTokenResult> {
  try {
    if (!sessionId) return { ok: false, error: ERR_REFUSED };
    const key = cacheKey(runId, sessionId);
    const now = opts.now ?? (() => Math.floor(Date.now() / 1000));
    const cached = cache.get(key);
    if (cached && cached.expiresAt - REFRESH_MARGIN_SECONDS > now()) {
      return { ok: true, ...cached };
    }

    const secret = internalSecret();
    if (!secret) return { ok: false, error: ERR_UNAVAILABLE };

    const baseUrl =
      opts.baseUrl ?? process.env.APP_INTERNAL_URL ?? 'http://127.0.0.1:3000';
    const fetchImpl = opts.fetchImpl ?? fetch;

    let response: Response;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    try {
      response = await fetchImpl(`${baseUrl}/api/relay/agent-token`, {
        method: 'POST',
        // A redirect would replay the secret header to another location.
        redirect: 'error',
        headers: {
          'content-type': 'application/json',
          'x-eve-internal-secret': secret,
        },
        body: JSON.stringify({ runId, sessionId }),
        signal: controller.signal,
      });
    } catch {
      return { ok: false, error: ERR_UNAVAILABLE };
    } finally {
      clearTimeout(timer);
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
      !Number.isFinite(expiresAt) ||
      expiresAt > MAX_PLAUSIBLE_EXPIRES_AT
    ) {
      return { ok: false, error: ERR_INVALID };
    }
    cache.set(key, { token, expiresAt });
    return { ok: true, token, expiresAt };
  } catch {
    return { ok: false, error: ERR_UNAVAILABLE };
  }
}
