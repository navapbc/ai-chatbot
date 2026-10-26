import type { ToolContext } from 'eve/tools';
import { runCommand } from '@/lib/kernel/cli';
import { redactText, redactionTargets } from '@/lib/kernel/redact';
import { browserIdentity } from '@/lib/kernel/session-identity';
import { cliSessionName } from '@/lib/kernel/session-store';
import { fetchAgentToken } from '@/lib/relay/agent-token';
import { LOOPBACK_HOSTS, toRelayWsUrl } from '@/lib/relay/url';

// Drives the caseworker's own browser tab through the relay. This module
// deliberately does not import `lib/kernel/eve-browser.ts`: that file builds a
// Kernel client at load, and Eve's bundler cannot take anything that pulls in
// `server-only`. Everything imported here is free of both.
//
// The relay token is a bearer credential for the run. It is never in Eve
// session state (Eve persists auth attributes): the session carries only a
// non-secret run reference, and the token is fetched at call time
// (lib/relay/agent-token.ts). Nothing in this file may put it in a log, an
// error, a tool result or a trace attribute.

export interface RelayRunRef {
  url: string;
  runId: string;
}

export interface RelayCommandResult {
  success: boolean;
  data?: unknown;
  error?: string | null;
  liveViewUrl: null;
}

const COMMAND_TIMEOUT_MS = 120_000;

const asString = (value: unknown): string | null =>
  typeof value === 'string' && value.length > 0 ? value : null;

export type RelayCredentialState =
  | { kind: 'none' }
  | { kind: 'complete'; ref: RelayRunRef }
  | { kind: 'partial' };

/**
 * Classifies the session's relay run reference. The first principal carrying
 * either attribute decides (initiator first, it is pinned to the session
 * creator, see `browserIdentity`; then the current caller). Exactly one of the
 * two is `partial`, which callers must treat as a failure, never as "no relay".
 */
export function relayCredentialState(ctx: ToolContext): RelayCredentialState {
  const { initiator, current } = ctx.session.auth;
  for (const principal of [initiator, current]) {
    const url = asString(principal?.attributes?.relayUrl);
    const runId = asString(principal?.attributes?.relayRunId);
    if (url && runId) return { kind: 'complete', ref: { url, runId } };
    if (url || runId) return { kind: 'partial' };
  }
  return { kind: 'none' };
}

/** The run reference when complete, otherwise null. */
export function relayRunRefFrom(ctx: ToolContext): RelayRunRef | null {
  const state = relayCredentialState(ctx);
  return state.kind === 'complete' ? state.ref : null;
}

type CdpUrl = { ok: true; value: string } | { ok: false; error: string };

/** `<origin>/cdp?token=<token>`; wss:, or ws: on a local host only. */
function buildCdpUrl(url: string, token: string): CdpUrl {
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    return { ok: false, error: 'The relay URL is not a valid URL.' };
  }
  const secure = parsed.protocol === 'wss:';
  const local =
    parsed.protocol === 'ws:' && LOOPBACK_HOSTS.has(parsed.hostname);
  // When the deployment names its relay, that is the only one a session may
  // point at: the URL arrives as session data, so it is not trusted on its own.
  const publicUrl = process.env.RELAY_PUBLIC_URL;
  if (publicUrl) {
    const allowed = toRelayWsUrl(publicUrl);
    if (!allowed) {
      return { ok: false, error: 'The relay is not configured correctly.' };
    }
    if (toRelayWsUrl(url) !== allowed) {
      return { ok: false, error: 'The relay URL is not the configured relay.' };
    }
  }
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
  const ref = relayRunRefFrom(ctx);
  if (!ref) {
    return {
      success: false,
      error: 'No relay credentials on this session.',
      liveViewUrl: null,
    };
  }
  const agentToken = await fetchAgentToken(ref.runId);
  if (!agentToken.ok) {
    return { success: false, error: agentToken.error, liveViewUrl: null };
  }
  const { token } = agentToken;
  const cdp = buildCdpUrl(ref.url, token);
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
        redact: [token, cdp.value],
        timeoutMs: COMMAND_TIMEOUT_MS,
      });
      return {
        success: response.success,
        data: response.data,
        error: response.error,
        liveViewUrl: null,
      };
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : String(error);
      return {
        success: false,
        error: redactText(message, redactionTargets([token, cdp.value])),
        liveViewUrl: null,
      };
    }
  });
}
