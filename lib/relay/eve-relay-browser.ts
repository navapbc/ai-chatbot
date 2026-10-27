import type { ToolContext } from 'eve/tools';
import { runCommand } from '@/lib/kernel/cli';
import { redactText, redactionTargets } from '@/lib/kernel/redact';
import { browserIdentity } from '@/lib/kernel/session-identity';
import { cliSessionName } from '@/lib/kernel/session-store';
import { fetchAgentToken } from '@/lib/relay/agent-token';
import {
  COMMAND_REFUSED,
  relayCommandRefusal,
} from '@/lib/relay/command-policy';
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
type RelayOrigin = { ok: true; origin: string } | { ok: false; error: string };

/**
 * Policy check on the relay URL: wss:, or ws: on a local host only, and the
 * configured relay when the deployment names one. Runs before any token is
 * fetched, so a refused URL never causes a mint.
 */
function checkRelayUrl(url: string): RelayOrigin {
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
  return { ok: true, origin: parsed.origin };
}

/** `<origin>/cdp?token=<token>`. Never throws, never echoes the token. */
function buildCdpUrl(origin: string, token: string): CdpUrl {
  try {
    return {
      ok: true,
      value: `${origin}/cdp?token=${encodeURIComponent(token)}`,
    };
  } catch {
    return { ok: false, error: 'The relay token could not be used.' };
  }
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

// --- recovery after an extension restart ---------------------------------
//
// When the extension's service worker restarts, the relay drops the agent
// connection (reason `extension restarted`) and the extension loses its
// debugger session, so every `@eN` ref the agent-browser daemon holds is void.
// A command that fails on a lost connection moves the run to a fresh daemon
// session name. Only a read-only command is run again; anything else, and any
// command naming a ref until a new snapshot succeeds, gets CONNECTION_RESTORED
// without running, so nothing with effects on the page is ever replayed.

/**
 * Transport-level failures only, anchored at the start of the error text so
 * page-originated text cannot trigger a retry. Shapes, as agent-browser
 * 0.33.2 reports them:
 * - `CDP response channel closed`: agent-browser's own, when the relay
 *   socket closed under a pending command.
 * - `CDP error (<Domain.method>): <message>` where the whole message is a
 *   relay or extension transport error: `not attached` (extension,
 *   bridge.ts), `tab or extension disconnected` and `extension restarted`
 *   (relay/hub.ts).
 * - `ECONNRESET` / `EPIPE` leading the text (socket-level errors).
 */
const CONNECTION_LOST =
  /^(?:CDP response channel closed$|CDP error \([\w.]+\): (?:not attached|tab or extension disconnected|extension restarted)$|(?:ECONNRESET|EPIPE)\b)/;
const ELEMENT_REF = /@e\d+/;
const CLOSE_TIMEOUT_MS = 5_000;
export const CONNECTION_RESTORED =
  'The browser connection was restored. Take a new snapshot before continuing.';

/**
 * The only commands retried after a lost connection: reads that cannot change
 * page state. Key is the command, value the allowed first arguments (null:
 * any arguments). Everything else, unknown commands included, is not retried.
 */
const RETRY_SAFE: Readonly<Record<string, readonly string[] | null>> = {
  snapshot: null,
  screenshot: null,
  get: ['url', 'title'],
  tab: ['list'],
};

/**
 * Restarts seen per run id. Like the session queue above, this and
 * `needsSnapshot` assume one Eve process drives a run (single instance), and
 * neither is ever pruned: one small entry per run for the process lifetime.
 */
const restarts = new Map<string, number>();
/** Runs whose refs are void: set on rotation, cleared by a snapshot. */
const needsSnapshot = new Set<string>();

const sessionName = (base: string, generation: number) =>
  generation === 0 ? base : `${base}-r${generation}`;

const namesRef = (command: readonly string[]) =>
  command.some((arg) => ELEMENT_REF.test(arg));

const retrySafe = (command: readonly string[]) => {
  const [name, sub, ...rest] = command;
  if (!Object.hasOwn(RETRY_SAFE, name) || namesRef(command)) return false;
  const subs = RETRY_SAFE[name];
  if (subs === null) return true;
  // `tab` alone lists tabs, like `tab list`.
  if (name === 'tab')
    return sub === undefined || (sub === 'list' && !rest.length);
  return sub !== undefined && subs.includes(sub);
};

/** Never throws. Failures come back as `{ success: false, error }`. */
export async function runRelayCommand(
  ctx: ToolContext,
  command: readonly string[],
): Promise<RelayCommandResult> {
  // First, before any credential, URL or token step: a refused command never
  // mints a token or starts the CLI. The error does not echo the argv.
  if (relayCommandRefusal(command)) {
    return { success: false, error: COMMAND_REFUSED, liveViewUrl: null };
  }
  const ref = relayRunRefFrom(ctx);
  if (!ref) {
    return {
      success: false,
      error: 'No relay credentials on this session.',
      liveViewUrl: null,
    };
  }
  const checked = checkRelayUrl(ref.url);
  if (!checked.ok) {
    return { success: false, error: checked.error, liveViewUrl: null };
  }
  // Bound to this Eve session: the route refuses any session other than the
  // one the run was created with (a fixed, credential-free error here).
  const agentToken = await fetchAgentToken(ref.runId, ctx.session.id);
  if (!agentToken.ok) {
    return { success: false, error: agentToken.error, liveViewUrl: null };
  }
  const { token } = agentToken;
  const cdp = buildCdpUrl(checked.origin, token);
  if (!cdp.ok) {
    return { success: false, error: cdp.error, liveViewUrl: null };
  }

  const { sessionId, userId } = browserIdentity(ctx);
  const base = cliSessionName(userId, sessionId);
  const secrets = [token, cdp.value];
  const scrub = (text: string) => redactText(text, redactionTargets(secrets));
  const generationAtCall = restarts.get(ref.runId) ?? 0;

  const execute = async (
    argv: readonly string[],
    session: string,
    timeoutMs = COMMAND_TIMEOUT_MS,
  ): Promise<RelayCommandResult> => {
    try {
      const response = await runCommand(argv, {
        session,
        cdpUrl: cdp.value,
        // The URL carries the token; keep it out of the process list.
        cdpViaEnv: true,
        redact: secrets,
        timeoutMs,
      });
      return {
        success: response.success,
        data: response.data,
        error:
          typeof response.error === 'string'
            ? scrub(response.error)
            : response.error,
        liveViewUrl: null,
      };
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : String(error);
      return { success: false, error: scrub(message), liveViewUrl: null };
    }
  };
  const restored = (): RelayCommandResult => ({
    success: false,
    error: CONNECTION_RESTORED,
    liveViewUrl: null,
  });

  // Inside the queue: the commands of one session see and advance the
  // generation one at a time, so one restart rotates the name once.
  return withSessionQueue(sessionId, async () => {
    const generation = restarts.get(ref.runId) ?? 0;
    // A restart happened while this command waited, or no snapshot has run
    // since one: its refs describe a page the new daemon never saw.
    if (
      namesRef(command) &&
      (generation !== generationAtCall || needsSnapshot.has(ref.runId))
    ) {
      return restored();
    }
    const settle = (result: RelayCommandResult) => {
      if (result.success && command[0] === 'snapshot') {
        needsSnapshot.delete(ref.runId);
      }
      return result;
    };
    const result = await execute(command, sessionName(base, generation));
    if (result.success || !CONNECTION_LOST.test(result.error ?? '')) {
      return settle(result);
    }

    // Another session of the same run may have moved on already; never go back.
    const next = Math.max(generation + 1, restarts.get(ref.runId) ?? 0);
    restarts.set(ref.runId, next);
    needsSnapshot.add(ref.runId);
    // Shut the old daemon down so it cannot re-dial the run and hold the
    // relay's single agent slot. Best effort, bounded, result ignored.
    await execute(['close'], sessionName(base, generation), CLOSE_TIMEOUT_MS);
    if (!retrySafe(command)) return restored();
    return settle(await execute(command, sessionName(base, next)));
  });
}
