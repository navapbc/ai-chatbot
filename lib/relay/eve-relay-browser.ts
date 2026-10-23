import type { ToolContext } from 'eve/tools';
import { runCommand } from '@/lib/kernel/cli';
import { browserIdentity } from '@/lib/kernel/session-identity';
import { cliSessionName } from '@/lib/kernel/session-store';

// Drives the caseworker's own browser tab through the relay. This module
// deliberately does not import `lib/kernel/eve-browser.ts`: that file builds a
// Kernel client at load, and Eve's bundler cannot take anything that pulls in
// `server-only`. Everything imported here is free of both.
//
// The relay token is a bearer credential for the run. It arrives only through
// Eve session auth attributes, and nothing in this file may put it in a log,
// an error, a tool result or a trace attribute.

export interface RelayCredentials {
  url: string;
  token: string;
}

export interface RelayCommandResult {
  success: boolean;
  data?: unknown;
  error?: string | null;
  liveViewUrl: null;
}

const COMMAND_TIMEOUT_MS = 120_000;
const LOCAL_HOSTS = new Set(['127.0.0.1', 'localhost']);

const asString = (value: unknown): string | null =>
  typeof value === 'string' && value.length > 0 ? value : null;

/**
 * Reads the relay credentials off the session's auth attributes, initiator
 * first (it is pinned to the session creator, see `browserIdentity`), then the
 * current caller. A principal counts only when it carries both values.
 */
export function relayCredentialsFrom(
  ctx: ToolContext,
): RelayCredentials | null {
  const { initiator, current } = ctx.session.auth;
  for (const principal of [initiator, current]) {
    const url = asString(principal?.attributes?.relayUrl);
    const token = asString(principal?.attributes?.relayToken);
    if (url && token) return { url, token };
  }
  return null;
}

type CdpUrl = { ok: true; value: string } | { ok: false; error: string };

/** `<origin>/cdp?token=<token>`; wss:, or ws: on a local host only. */
function buildCdpUrl({ url, token }: RelayCredentials): CdpUrl {
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    return { ok: false, error: 'The relay URL is not a valid URL.' };
  }
  const secure = parsed.protocol === 'wss:';
  const local = parsed.protocol === 'ws:' && LOCAL_HOSTS.has(parsed.hostname);
  if (!secure && !local) {
    return {
      ok: false,
      error: 'The relay URL must use wss:// (ws:// is allowed for localhost).',
    };
  }
  return {
    ok: true,
    value: `${parsed.origin}/cdp?token=${encodeURIComponent(token)}`,
  };
}

/** Removes the token (raw and URL-encoded) from text we are about to return. */
function scrub(text: string, token: string): string {
  return text
    .split(token)
    .join('[redacted]')
    .split(encodeURIComponent(token))
    .join('[redacted]');
}

// Per-session serialization, same discipline as the Kernel path: a snapshot's
// `@eN` refs must still describe the page when the next command runs.
const sessionQueues = new Map<string, Promise<unknown>>();
function withSessionQueue<T>(key: string, fn: () => Promise<T>): Promise<T> {
  const prev = sessionQueues.get(key) ?? Promise.resolve();
  const next = prev.then(fn, fn);
  sessionQueues.set(
    key,
    next.then(
      () => {},
      () => {},
    ),
  );
  return next;
}

/** Never throws. Failures come back as `{ success: false, error }`. */
export async function runRelayCommand(
  ctx: ToolContext,
  command: readonly string[],
): Promise<RelayCommandResult> {
  const credentials = relayCredentialsFrom(ctx);
  if (!credentials) {
    return {
      success: false,
      error: 'No relay credentials on this session.',
      liveViewUrl: null,
    };
  }
  const cdp = buildCdpUrl(credentials);
  if (!cdp.ok) {
    return { success: false, error: cdp.error, liveViewUrl: null };
  }

  const { sessionId, userId } = browserIdentity(ctx);
  return withSessionQueue(sessionId, async () => {
    try {
      const response = await runCommand(command, {
        session: cliSessionName(userId, sessionId),
        cdpUrl: cdp.value,
        // The URL carries the token; keep it out of the process list.
        cdpViaEnv: true,
        timeoutMs: COMMAND_TIMEOUT_MS,
      });
      return {
        success: response.success,
        data: response.data,
        error:
          response.error == null
            ? response.error
            : scrub(response.error, credentials.token),
        liveViewUrl: null,
      };
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : String(error);
      return {
        success: false,
        error: scrub(message, credentials.token),
        liveViewUrl: null,
      };
    }
  });
}
