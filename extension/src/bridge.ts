/// <reference types="chrome" />
import {
  commandAddresses,
  contextIdFromObjectId,
  contextInScope,
  createFrameScope,
  documentOrigin,
  type FrameScope,
  frameInScope,
  framesOfTree,
  NO_FRAME_CONTENT_METHODS,
  pruneAXTree,
  pruneDescribedNode,
  pruneFrameTree,
  REFUSED_ORIGIN,
  scopeFrameEvent,
} from '../../lib/relay/frames';
import {
  checkCommand,
  filterEvent,
  inScope,
  isJavascriptUrl,
  isMethodAllowed,
  normalizeOrigin,
} from '../../lib/relay/policy';
import type {
  ExtensionToRelay,
  RelayToExtension,
  StopReason,
} from '../../lib/relay/protocol';
import {
  createRedactor,
  descriptorFromAttributes,
  hasSensitiveShape,
  type Redactor,
} from '../../lib/relay/redactor';
import {
  createMaskSession,
  createScreenshotQueue,
  MASK_WORLD,
  ScreenshotMaskError,
  withScreenshotMask,
} from '../../lib/relay/screenshot-mask';
import { createAppApi } from './app-api';
import { APP_ORIGIN } from './config';
import { createJevClassifier, type JevClassifier } from './jev-classifier';
import {
  type Describe,
  redactErrorMessage,
  redactEventParams,
  redactReply,
} from './redaction';

// Forwards CDP between the relay and chrome.debugger for exactly one tab.
// Ported from spike/cdp-relay/extension/sw.js. Nothing here may log or echo
// the token or a URL that carries it.

export type BridgeMessage =
  | {
      type: 'startRun';
      relayUrl: string;
      token: string;
      tabId: number;
      allowedFrameOrigins?: string[];
    }
  | { type: 'stopRun' }
  | { type: 'addKnownValue'; value: string }
  | { type: 'getRunStatus' }
  /** The panel re-entered the run's known values (count only, no values). */
  | { type: 'registrationComplete'; count?: number };

/**
 * The bridge's run state for the panel. Ids and counts only: never the
 * token, the relay URL or a known value.
 */
export interface RunStatus {
  ok: true;
  active: boolean;
  runId?: string;
  origin?: string;
  attached?: boolean;
  /**
   * True on a run rebuilt after a worker restart until the panel sends
   * `registrationComplete`: the known values died with the old worker, so
   * screenshots are refused meanwhile.
   */
  needsRegistration?: boolean;
  /** Jev classify() calls that failed (P4b-e); absent without Jev. */
  jevFailures?: number;
  /** Requests made to the app's classify-field route; absent without Jev. */
  jevRequests?: number;
}

export type BridgeResponse =
  | { ok: true }
  | { ok: false; error: string }
  | RunStatus;

interface RunConfig {
  relayUrl: string;
  token: string;
  tabId: number;
  allowedFrameOrigins: string[];
}

interface Child {
  targetId?: string;
  /** CDP target type (iframe, worker, ...); decides screenshot masking. */
  type?: string;
  url: string;
  /** Rejects this child's in-flight commands; see dropChild. */
  inFlight: Set<(error: Error) => void>;
  /**
   * Settles once the bridge has enabled the child's Page domain (its own
   * navigation events keep `url` current): true on success, false when it
   * failed, which leaves the child unverifiable and so out of scope.
   */
  pageReady: Promise<boolean>;
  pageEnabled?: boolean;
  /**
   * The frames and contexts of a frame child's own session (its document
   * and the same-site frames inside it), seeded from Page.getFrameTree once
   * its Page domain is on. Workers have none.
   */
  frameScope?: FrameScope;
}

interface Run extends RunConfig {
  runId: string;
  origin: string;
  /** Token expiry, epoch seconds. */
  exp: number;
  socketUrl: string;
  ws?: WebSocket;
  stopped: boolean;
  /** True while the extension holds chrome.debugger on the tab. */
  attached: boolean;
  /**
   * Set synchronously the moment the tab leaves scope, before any await, so
   * no command or event gets through while the run is being torn down.
   */
  blocked: boolean;
  /** The tab's main frame id (its target id), known once attached. */
  mainFrameId?: string;
  /** Child (iframe, worker) sessions of the tab, keyed by CDP sessionId. */
  children: Map<string, Child>;
  /**
   * Frames and execution contexts of the tab session itself (same-origin,
   * srcdoc and same-site cross-origin iframes share it), from raw events.
   */
  frameScope: FrameScope;
  reconnectTimer?: ReturnType<typeof setTimeout>;
  expiryTimer?: ReturnType<typeof setTimeout>;
  reconnectDelayMs: number;
  /**
   * Set on a run rebuilt from storage after a service worker restart. Cleared
   * on the first socket open, right after `register` and `resumed` are sent,
   * so a later reconnect never repeats it. If that `resumed` is lost, the
   * agent's commands fail with `not attached` and the Eve tool rotates its
   * agent-browser session on that error instead.
   */
  announceResume: boolean;
  /**
   * Masks sensitive values in everything sent up the socket. Its known
   * values live in worker memory only: never in chrome.storage, cleared in
   * stopRun, and empty on a run rebuilt after a worker restart.
   */
  redactor: Redactor;
  /**
   * The redactor's classifyExtra hook (the app's classify-field route), when
   * the build has an APP_ORIGIN. Its failures() feeds the run status.
   */
  jev?: JevClassifier;
  /** True once a known value was registered; see addKnownValue. */
  hasKnownValues: boolean;
  /**
   * Set on a run rebuilt after a worker restart (its known values were lost
   * with the old worker) until the panel sends `registrationComplete`.
   * Screenshots are refused while it is set.
   */
  needsRegistration: boolean;
  /** The socket `register` was sent on; see send/stop delivery. */
  registeredWs?: WebSocket;
  /** The relay closed the run (a final close code): no stop is owed. */
  relayClosed: boolean;
  /** Ends the run unless the tab is re-attached in time (P4b-d). */
  detachTimer?: ReturnType<typeof setTimeout>;
  /** The bridge's focus-reading isolated world in the main frame, if made. */
  focusWorld?: number;
  /**
   * The dialog showing now, from the raw opening event: whether a frame out
   * of scope (or a disallowed child) raised it. Cleared when it closes.
   */
  dialog?: { outOfScope: boolean };
}

const STORAGE_KEY = 'run';
const LOOPBACK_HOSTS = new Set(['127.0.0.1', 'localhost']);
const OUT_OF_SCOPE = 'tab is outside the allowed origin';
const FRAME_OUT_OF_SCOPE = 'frame is outside the allowed origins';
const FIRST_RECONNECT_MS = 1000;
const MAX_RECONNECT_MS = 30_000;
// A stop the socket could not carry is retried this long at most (P4b).
const PENDING_STOP_MS = 30_000;
// A non-user detach ends the run unless the tab is re-attached in time.
const DETACH_GRACE_MS = 5_000;
const REGISTRATION_REFUSED = 'registration required';
/**
 * While a run needs registration (its redactor lost the known values in a
 * worker restart), the agent may send only these: methods whose result is
 * empty (no page text) and which keep the run's session alive. Everything
 * else, including any method not listed here, is refused with
 * REGISTRATION_REFUSED: DOM.*, Accessibility.*, Runtime.evaluate and
 * callFunctionOn, Page.captureScreenshot, getFrameTree and navigate, ...
 */
const UNREGISTERED_METHODS: ReadonlySet<string> = new Set([
  'Runtime.enable',
  'Runtime.runIfWaitingForDebugger',
  'Page.enable',
  'Page.bringToFront',
  'Page.handleJavaScriptDialog',
  'Network.enable',
  'DOM.enable',
  'Accessibility.enable',
  'Target.setAutoAttach',
  'Input.insertText',
  'Input.dispatchKeyEvent',
  'Input.dispatchMouseEvent',
]);
/**
 * Events forwarded while a run needs registration: the allowed events
 * (ALLOWED_EVENT_METHODS) that carry no URL, title, origin, stack or other
 * page text. Dropped meanwhile: Page.frameNavigated, navigatedWithinDocument,
 * frameRequestedNavigation, frameStartedNavigating and frameAttached (URL or
 * stack), Network.requestWillBeSent and responseReceived (URL),
 * Runtime.executionContextCreated (origin, name). The two dialog events are
 * rebuilt with empty text by filterEvent; the Target events are forwarded
 * with their URL and title blanked (see withoutTargetUrl).
 */
const UNREGISTERED_EVENTS: ReadonlySet<string> = new Set([
  'DOM.documentUpdated',
  'Network.loadingFailed',
  'Network.loadingFinished',
  'Network.requestServedFromCache',
  'Page.domContentEventFired',
  'Page.frameDetached',
  'Page.frameStartedLoading',
  'Page.frameStoppedLoading',
  'Page.javascriptDialogClosed',
  'Page.javascriptDialogOpening',
  'Page.loadEventFired',
  'Runtime.executionContextDestroyed',
  'Runtime.executionContextsCleared',
  'Target.attachedToTarget',
  'Target.detachedFromTarget',
  'Target.targetInfoChanged',
]);
// Relay close codes that end the run: normal (run closed, replaced) and
// policy (refused). Anything else is a dropped connection.
const FINAL_CLOSE_CODES = new Set([1000, 1008]);
// Target lifecycle events forwarded even for a disallowed child, so the agent
// can keep its session bookkeeping; their target URL and title are blanked.
const LIFECYCLE_EVENTS = new Set([
  'Target.attachedToTarget',
  'Target.detachedFromTarget',
  'Target.targetInfoChanged',
]);
// Navigation signals sent before anything commits.
const PRE_COMMIT_EVENTS = new Set([
  'Page.frameRequestedNavigation',
  'Page.frameStartedNavigating',
  'Page.frameScheduledNavigation',
  'Network.requestWillBeSent',
]);
// Main-frame navigation signals, from the first request to the commit.
const NAVIGATION_EVENTS = new Set([
  'Page.frameRequestedNavigation',
  'Page.frameStartedNavigating',
  'Page.frameScheduledNavigation',
  'Page.navigatedWithinDocument',
  'Page.frameNavigated',
  'Network.requestWillBeSent',
]);

let run: Run | undefined;

const nowSeconds = () => Math.floor(Date.now() / 1000);

const tabSessionId = (tabId: number) => `tab-${tabId}`;

const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

/** The relay's `/extension` socket URL, or an error for a disallowed scheme. */
export function relaySocketUrl(
  relayUrl: string,
): { ok: true; url: string } | { ok: false; error: string } {
  let parsed: URL;
  try {
    parsed = new URL(relayUrl);
  } catch {
    return { ok: false, error: 'relay URL is invalid' };
  }
  const secure = parsed.protocol === 'wss:';
  const loopback =
    parsed.protocol === 'ws:' && LOOPBACK_HOSTS.has(parsed.hostname);
  if (!secure && !loopback) {
    return {
      ok: false,
      error: 'relay URL must use wss: (ws: only on 127.0.0.1 or localhost)',
    };
  }
  if (parsed.username || parsed.password || parsed.search || parsed.hash) {
    return { ok: false, error: 'relay URL must not carry credentials' };
  }
  const base = parsed.pathname.replace(/\/+$/, '');
  return { ok: true, url: `${parsed.origin}${base}/extension` };
}

/**
 * Reads the run claims from the token payload. The relay verifies the
 * signature; this only lets the extension fail early and learn the run.
 */
export function readClaims(
  token: string,
): { runId: string; origin: string; exp: number } | null {
  try {
    const parts = token.split('.');
    if (parts.length !== 2) return null;
    const b64 = parts[0].replace(/-/g, '+').replace(/_/g, '/');
    const json = atob(b64.padEnd(Math.ceil(b64.length / 4) * 4, '='));
    const claims: unknown = JSON.parse(json);
    if (!isObject(claims)) return null;
    const { runId, role, origin, exp } = claims;
    if (role !== 'extension') return null;
    if (typeof runId !== 'string' || runId === '') return null;
    if (typeof origin !== 'string' || normalizeOrigin(origin) !== origin) {
      return null;
    }
    if (typeof exp !== 'number' || exp <= nowSeconds()) return null;
    return { runId, origin, exp };
  } catch {
    return null;
  }
}

const validFrameOrigins = (value: unknown): string[] | null => {
  if (value === undefined) return [];
  if (!Array.isArray(value)) return null;
  for (const entry of value) {
    if (typeof entry !== 'string' || normalizeOrigin(entry) !== entry) {
      return null;
    }
  }
  return [...value];
};

async function tabUrl(tabId: number): Promise<string | null> {
  try {
    const tab = await chrome.tabs.get(tabId);
    return tab.url ?? null;
  } catch {
    return null;
  }
}

async function tabTarget(tabId: number) {
  const targets = await chrome.debugger.getTargets();
  return targets.find((t) => t.tabId === tabId && t.type === 'page');
}

/** A target as the agent sees it; URL and title blank while unregistered. */
const agentTargetInfo = (current: Run, t: chrome.debugger.TargetInfo) => {
  const info = targetInfo(t);
  return current.needsRegistration ? { ...info, url: '', title: '' } : info;
};

const targetInfo = (t: chrome.debugger.TargetInfo) => ({
  targetId: t.id,
  type: t.type,
  title: t.title,
  url: t.url,
  attached: t.attached,
  canAccessOpener: false,
});

function send(current: Run, message: ExtensionToRelay) {
  const ws = current.ws;
  if (ws?.readyState !== WebSocket.OPEN) return;
  try {
    ws.send(JSON.stringify(message));
  } catch {
    // The close handler deals with a dead socket.
  }
}

// --- run lifecycle -----------------------------------------------------------

/**
 * startRun, stopRun and resumeRun run one at a time, in call order, behind
 * this chain, so two of them can never interleave their awaits and leave two
 * runs (and two sockets). The internal stops (left origin, tab closed, token
 * expiry, relay close, detach) do not queue: each ends the run it observed,
 * synchronously, through endRun, and a queued one would end a newer run.
 */
let lifecycle: Promise<unknown> = Promise.resolve();

function serialized<T>(task: () => Promise<T>): Promise<T> {
  const result = lifecycle.then(task);
  // A failed call must not poison the calls queued behind it.
  lifecycle = result.catch(() => {});
  return result;
}

export function startRun(
  msg: Extract<BridgeMessage, { type: 'startRun' }>,
): Promise<BridgeResponse> {
  return serialized(() => startRunNow(msg));
}

async function startRunNow(
  msg: Extract<BridgeMessage, { type: 'startRun' }>,
): Promise<BridgeResponse> {
  if (typeof msg.relayUrl !== 'string' || typeof msg.token !== 'string') {
    return { ok: false, error: 'relayUrl and token are required' };
  }
  if (!Number.isInteger(msg.tabId)) {
    return { ok: false, error: 'tabId must be an integer' };
  }
  const socket = relaySocketUrl(msg.relayUrl);
  if (!socket.ok) return socket;
  const claims = readClaims(msg.token);
  if (!claims) return { ok: false, error: 'token is invalid or expired' };
  const allowedFrameOrigins = validFrameOrigins(msg.allowedFrameOrigins);
  if (!allowedFrameOrigins) {
    return {
      ok: false,
      error: 'allowedFrameOrigins must be exact https origins',
    };
  }
  const url = await tabUrl(msg.tabId);
  if (url === null) return { ok: false, error: 'tab not found' };
  if (!inScope(url, claims.origin)) return { ok: false, error: OUT_OF_SCOPE };

  // A run restarted under the same run id (e.g. a refreshed token) is not
  // ended on the relay: its new socket's `register` replaces the old one
  // there, while a `stop` would close the run for good (and refuse the new
  // socket). Its known values are cleared with the old run, so the panel
  // must register them again before screenshots are allowed.
  const previous = run;
  const sameRun = previous !== undefined && previous.runId === claims.runId;
  const valuesLost =
    sameRun && (previous.hasKnownValues || previous.needsRegistration);
  await endRun('replaced', { notifyRelay: !sameRun });
  const config: RunConfig = {
    relayUrl: msg.relayUrl,
    token: msg.token,
    tabId: msg.tabId,
    allowedFrameOrigins,
  };
  await chrome.storage.session.set({ [STORAGE_KEY]: config });
  begin(config, socket.url, claims, false, valuesLost);
  return { ok: true };
}

/**
 * Ends the run (queued behind any startRun/resumeRun in flight): tells the
 * relay to close it (so the agent is cut off at once rather than held for
 * the resume window), closes the socket, detaches.
 */
export function stopRun(reason: StopReason = 'user'): Promise<BridgeResponse> {
  return serialized(() => endRun(reason));
}

/**
 * Ends whatever run is current, now. The synchronous part (forgetting the
 * run, blocking it, clearing its redactor and timers, stopping its Jev
 * classifier, telling the relay) completes before the first await, so a
 * caller that checked `run === current` ends exactly that run.
 *
 * The relay is told with a `stop` frame on the run's socket when that socket
 * is open; otherwise the stop is queued for delivery on a short-lived socket
 * (see queueStop). Not when the relay itself closed the run, or when
 * `notifyRelay` is false.
 */
async function endRun(
  reason: StopReason,
  opts: { notifyRelay?: boolean } = {},
): Promise<BridgeResponse> {
  const current = run;
  run = undefined;
  // Before any detach await, so a slow stop cannot remove a newer run's config.
  const forgotten = chrome.storage.session.remove(STORAGE_KEY);
  if (current) {
    current.stopped = true;
    current.blocked = true;
    current.redactor.clear();
    current.hasKnownValues = false;
    current.jev?.stop();
    clearTimeout(current.reconnectTimer);
    clearTimeout(current.expiryTimer);
    clearTimeout(current.detachTimer);
    if (opts.notifyRelay !== false && !current.relayClosed) {
      if (current.ws?.readyState === WebSocket.OPEN) {
        // The hub ignores everything before `register`.
        if (current.registeredWs !== current.ws) sendRegister(current);
        send(current, { type: 'stop', reason });
      } else {
        queueStop(current, reason);
      }
    }
    try {
      current.ws?.close(1000, 'run stopped');
    } catch {
      // Already closed.
    }
    // Let a screenshot in flight restore the page while the debugger is
    // still attached (it sees `stopped` and skips its capture); bounded, so
    // a hung page cannot hold the stop.
    let waited: ReturnType<typeof setTimeout> | undefined;
    await Promise.race([
      screenshotsIdle(current.tabId),
      new Promise((resolve) => {
        waited = setTimeout(resolve, STOP_WAIT_MS);
      }),
    ]);
    clearTimeout(waited);
    dropChildren(current);
    // The mask's worlds die with the run's debugger session; a newer run (if
    // one started meanwhile) tracks its own.
    if (run === undefined) maskContexts.clear();
    current.attached = false;
    // Always let go: a blocked run has already cleared `attached`.
    try {
      await chrome.debugger.detach({ tabId: current.tabId });
    } catch {
      // Not attached.
    }
  }
  await forgotten;
  return { ok: true };
}

/**
 * The tab left the run origin. Terminal in plan 1: block synchronously, then
 * stop the run; a later relay `attach` finds no run and the tab is never
 * re-attached, even if it comes back.
 */
function leaveOrigin(current: Run) {
  if (current.blocked) return;
  current.blocked = true;
  if (run === current) void endRun('left-origin');
}

/** Picks up a run persisted before a service worker restart. */
export function resumeRun(): Promise<void> {
  return serialized(resumeRunNow);
}

async function resumeRunNow(): Promise<void> {
  if (run) return;
  const stored = await chrome.storage.session.get(STORAGE_KEY);
  const config = stored[STORAGE_KEY] as RunConfig | undefined;
  if (!config) return;
  const socket = relaySocketUrl(config.relayUrl);
  const claims = readClaims(config.token);
  const frames = validFrameOrigins(config.allowedFrameOrigins);
  if (!socket.ok || !claims || !frames || run) {
    if (!run) await chrome.storage.session.remove(STORAGE_KEY);
    return;
  }
  // The restart dropped chrome.debugger and every child session, so the
  // agent's element refs and sessions are void: tell the relay once.
  begin({ ...config, allowedFrameOrigins: frames }, socket.url, claims, true);
}

/**
 * A per-run Jev classifier, only when the build has an APP_ORIGIN; undefined
 * (no hook, deterministic verdicts only) otherwise or if the origin is bad.
 * A run resumed after a worker restart gets a fresh one (and budget).
 */
function jevFor(runId: string): JevClassifier | undefined {
  if (!APP_ORIGIN) return undefined;
  try {
    return createJevClassifier({
      api: createAppApi({ origin: APP_ORIGIN }),
      runId,
    });
  } catch {
    return undefined;
  }
}

function begin(
  config: RunConfig,
  socketUrl: string,
  claims: { runId: string; origin: string; exp: number },
  resumed = false,
  // The known values lived in a dead worker's memory, or in a replaced run.
  needsRegistration = resumed,
) {
  const jev = jevFor(claims.runId);
  const current: Run = {
    ...config,
    ...claims,
    socketUrl,
    stopped: false,
    attached: false,
    blocked: false,
    children: new Map(),
    frameScope: createFrameScope(),
    reconnectDelayMs: FIRST_RECONNECT_MS,
    announceResume: resumed,
    redactor: jev
      ? createRedactor({ classifyExtra: jev.classify })
      : createRedactor(),
    jev,
    hasKnownValues: false,
    needsRegistration,
    relayClosed: false,
  };
  // A stop still queued for this run id would close the run just begun.
  dropPendingStops(current.runId);
  run = current;
  current.expiryTimer = setTimeout(
    () => {
      if (run === current) void endRun('token-expired');
    },
    Math.max(0, current.exp * 1000 - Date.now()),
  );
  connect(current);
}

function connect(current: Run) {
  if (current.stopped || run !== current) return;
  if (current.exp <= nowSeconds()) {
    void endRun('token-expired');
    return;
  }
  const url = new URL(current.socketUrl);
  url.searchParams.set('token', current.token);
  let ws: WebSocket;
  try {
    ws = new WebSocket(url.toString());
  } catch {
    scheduleReconnect(current);
    return;
  }
  current.ws = ws;

  ws.onopen = async () => {
    const tab = await tabUrl(current.tabId);
    if (current.ws !== ws || current.stopped) return;
    if (tab === null) {
      void endRun('user');
      return;
    }
    // The tab left the origin while the socket was down: the run is over.
    if (!inScope(tab, current.origin)) {
      leaveOrigin(current);
      return;
    }
    current.reconnectDelayMs = FIRST_RECONNECT_MS;
    sendRegister(current);
    if (current.announceResume) {
      current.announceResume = false;
      send(current, { type: 'resumed' });
    }
  };

  ws.onmessage = async (event) => {
    if (current.ws !== ws) return;
    let msg: RelayToExtension;
    try {
      msg = JSON.parse(String(event.data));
    } catch {
      return;
    }
    if (!isObject(msg) || typeof msg.id !== 'string') return;
    try {
      const result = await handleCommand(current, msg);
      send(current, { type: 'reply', id: msg.id, result });
    } catch (error) {
      send(current, {
        type: 'reply',
        id: msg.id,
        // A CDP error message can quote page text: scrub known values.
        error: redactErrorMessage(
          current.redactor,
          error instanceof Error ? error.message : 'command failed',
        ),
      });
    }
  };

  ws.onclose = (event) => {
    if (current.ws !== ws) return;
    current.ws = undefined;
    if (current.stopped || run !== current) return;
    if (FINAL_CLOSE_CODES.has(event.code)) {
      // The relay ended (or refused) the run: no stop is owed to it.
      current.relayClosed = true;
      void endRun('user');
      return;
    }
    scheduleReconnect(current);
  };
}

function sendRegister(current: Run) {
  current.registeredWs = current.ws;
  send(current, {
    type: 'register',
    runId: current.runId,
    tabId: current.tabId,
    origin: current.origin,
  });
}

// --- stop delivery -------------------------------------------------------------

/**
 * A `stop` the run's socket could not carry (it was connecting, or down
 * between reconnects) when the run ended. The run itself is already gone
 * (torn down locally at once); this keeps only what one short-lived socket
 * needs to tell the relay: it opens with the run's token, sends `register`
 * (the hub ignores anything before it) and then `stop` as its first frame
 * after that, and closes. Failed attempts retry with the reconnect backoff.
 * Dropped once the token expires, after PENDING_STOP_MS, when the relay
 * refuses the socket with a final close code (the run is already closed or
 * the token is refused), or when a run with the same id begins. Memory only:
 * a worker restart loses it. Never logged: its URL carries the token.
 */
interface PendingStop {
  runId: string;
  tabId: number;
  origin: string;
  token: string;
  socketUrl: string;
  reason: StopReason;
  /** Epoch ms after which the stop is dropped: 30 s or token expiry. */
  dropAt: number;
  delayMs: number;
  ws?: WebSocket;
  timer?: ReturnType<typeof setTimeout>;
}

const pendingStops = new Set<PendingStop>();

function queueStop(current: Run, reason: StopReason) {
  const dropAt = Math.min(Date.now() + PENDING_STOP_MS, current.exp * 1000);
  if (dropAt <= Date.now()) return;
  const pending: PendingStop = {
    runId: current.runId,
    tabId: current.tabId,
    origin: current.origin,
    token: current.token,
    socketUrl: current.socketUrl,
    reason,
    dropAt,
    delayMs: FIRST_RECONNECT_MS,
  };
  pendingStops.add(pending);
  deliverStop(pending);
}

function dropStop(pending: PendingStop) {
  pendingStops.delete(pending);
  clearTimeout(pending.timer);
  const ws = pending.ws;
  pending.ws = undefined;
  try {
    ws?.close(1000, 'run stopped');
  } catch {
    // Already closed.
  }
}

function dropPendingStops(runId: string) {
  for (const pending of [...pendingStops]) {
    if (pending.runId === runId) dropStop(pending);
  }
}

function deliverStop(pending: PendingStop) {
  if (!pendingStops.has(pending)) return;
  if (Date.now() >= pending.dropAt) {
    dropStop(pending);
    return;
  }
  const url = new URL(pending.socketUrl);
  url.searchParams.set('token', pending.token);
  let ws: WebSocket;
  try {
    ws = new WebSocket(url.toString());
  } catch {
    retryStop(pending);
    return;
  }
  pending.ws = ws;
  ws.onopen = () => {
    if (pending.ws !== ws) return;
    try {
      ws.send(
        JSON.stringify({
          type: 'register',
          runId: pending.runId,
          tabId: pending.tabId,
          origin: pending.origin,
        } satisfies ExtensionToRelay),
      );
      ws.send(
        JSON.stringify({
          type: 'stop',
          reason: pending.reason,
        } satisfies ExtensionToRelay),
      );
    } catch {
      // The close handler retries.
      return;
    }
    // Delivered: sends queued before close() go out ahead of the close.
    dropStop(pending);
  };
  ws.onclose = (event) => {
    if (pending.ws !== ws) return;
    pending.ws = undefined;
    if (FINAL_CLOSE_CODES.has(event.code)) {
      dropStop(pending);
      return;
    }
    retryStop(pending);
  };
}

function retryStop(pending: PendingStop) {
  if (!pendingStops.has(pending)) return;
  const delay = Math.min(pending.delayMs, pending.dropAt - Date.now());
  if (delay <= 0) {
    dropStop(pending);
    return;
  }
  pending.delayMs = Math.min(pending.delayMs * 2, MAX_RECONNECT_MS);
  pending.timer = setTimeout(() => deliverStop(pending), delay);
}

function scheduleReconnect(current: Run) {
  if (current.stopped || run !== current) return;
  clearTimeout(current.reconnectTimer);
  const delay = current.reconnectDelayMs;
  current.reconnectDelayMs = Math.min(delay * 2, MAX_RECONNECT_MS);
  current.reconnectTimer = setTimeout(() => connect(current), delay);
}

// --- commands ----------------------------------------------------------------

/** Fails (and ends the run) unless the tab is still in scope. */
async function requireTabInScope(current: Run) {
  if (current.blocked) throw new Error(OUT_OF_SCOPE);
  const url = await tabUrl(current.tabId);
  if (url === null) throw new Error('tab closed');
  if (current.blocked || !inScope(url, current.origin)) {
    leaveOrigin(current);
    throw new Error(OUT_OF_SCOPE);
  }
}

// Out-of-process frames have their own child session, judged by its URL
// here. Frames that share the tab's session (same-origin, srcdoc and
// same-site cross-origin iframes) are judged per command and per event
// through `frameScope` (a frame child has its own too); see resolveFrames.
const frameAllowed = (current: Run, url: string) =>
  inScope(url, current.origin) ||
  current.allowedFrameOrigins.some((origin) => inScope(url, origin));

/** A child is in scope by its URL, once its Page domain is enabled. */
const childAllowed = (current: Run, child: Child) =>
  child.pageEnabled !== false && frameAllowed(current, child.url);

/**
 * True when an out-of-process frame (its child's target id is its frame id)
 * is in scope by its child session.
 */
const childFrameAllowed = (current: Run, frameId: string) => {
  for (const child of current.children.values()) {
    if (child.targetId === frameId) return childAllowed(current, child);
  }
  return false;
};

/**
 * True when a frame a session names is in scope: one of the session's own
 * frames by URL (about:blank and srcdoc by their parent's), or an
 * out-of-process frame by its child session. Unknown: false.
 */
const frameAllowedIn = (current: Run, scope: FrameScope, frameId: string) =>
  frameInScope(
    scope.frames,
    frameId,
    current.origin,
    current.allowedFrameOrigins,
  ) || childFrameAllowed(current, frameId);

/** True when a context is known, of an allowed origin, in an allowed frame. */
const contextAllowedIn = (current: Run, scope: FrameScope, contextId: number) =>
  contextInScope(scope, current, contextId, (frameId) =>
    childFrameAllowed(current, frameId),
  );

async function handleCommand(
  current: Run,
  msg: RelayToExtension,
): Promise<unknown> {
  if (current.stopped) throw new Error('run stopped');
  switch (msg.type) {
    case 'listTargets': {
      await requireTabInScope(current);
      const target = await tabTarget(current.tabId);
      return redact(current, 'Target.getTargets', {
        targetInfos: target ? [agentTargetInfo(current, target)] : [],
      });
    }
    case 'attach': {
      const target = await tabTarget(current.tabId);
      if (!target || target.id !== msg.targetId) {
        throw new Error('unknown target');
      }
      await requireTabInScope(current);
      // Start clean so a new relay client sees every child session announced
      // afresh (the spike's re-attach fix).
      current.attached = false;
      dropChildren(current);
      current.frameScope = createFrameScope();
      current.focusWorld = undefined;
      current.dialog = undefined;
      try {
        await chrome.debugger.detach({ tabId: current.tabId });
      } catch {
        // Not attached.
      }
      if (current.blocked || current.stopped) throw new Error(OUT_OF_SCOPE);
      await chrome.debugger.attach({ tabId: current.tabId }, '1.3');
      if (current.blocked || current.stopped) {
        // The run ended while attaching; let go again.
        await chrome.debugger.detach({ tabId: current.tabId }).catch(() => {});
        throw new Error(OUT_OF_SCOPE);
      }
      // The bridge owns the Page domain: leave-origin detection reads Page
      // navigation events, so it never depends on the agent enabling Page
      // (and the agent cannot disable it: Page.disable is not allowed). This
      // does not go through the agent allowlist and the agent gets no reply.
      // Fail closed: without it the run is not attached.
      try {
        await chrome.debugger.sendCommand(
          { tabId: current.tabId },
          'Page.enable',
        );
      } catch {
        console.warn('relay: Page.enable failed; refusing attach');
        await chrome.debugger.detach({ tabId: current.tabId }).catch(() => {});
        throw new Error('could not enable page events');
      }
      // Frame scope starts from the current tree; Page events keep it
      // current from here (events that raced the reply were applied first,
      // and the tree supersedes them). Fail closed like Page.enable.
      try {
        const tree = await chrome.debugger.sendCommand(
          { tabId: current.tabId },
          'Page.getFrameTree',
        );
        const seeded = framesOfTree(tree);
        if (seeded.length === 0) throw new Error('empty frame tree');
        current.frameScope.frames = new Map(seeded);
      } catch {
        console.warn('relay: Page.getFrameTree failed; refusing attach');
        await chrome.debugger.detach({ tabId: current.tabId }).catch(() => {});
        throw new Error('could not read the frame tree');
      }
      if (current.blocked || current.stopped) {
        await chrome.debugger.detach({ tabId: current.tabId }).catch(() => {});
        throw new Error(OUT_OF_SCOPE);
      }
      current.attached = true;
      // Re-attached in time: the detach no longer ends the run (P4b-d).
      clearTimeout(current.detachTimer);
      current.detachTimer = undefined;
      current.mainFrameId = target.id;
      return redact(current, 'Target.attachToTarget', {
        sessionId: tabSessionId(current.tabId),
        targetInfo: agentTargetInfo(current, { ...target, attached: true }),
      });
    }
    case 'createTarget': {
      const check = checkCommand(
        { method: 'Target.createTarget', params: { url: msg.url } },
        { origin: current.origin },
      );
      if (!check.ok) throw new Error(check.reason);
      throw new Error('only the registered tab is exposed');
    }
    case 'closeTarget':
      throw new Error('closing the registered tab is not allowed');
    case 'cdp':
      return forwardCdp(current, msg);
    default:
      throw new Error('unknown command');
  }
}

async function forwardCdp(
  current: Run,
  msg: Extract<RelayToExtension, { type: 'cdp' }>,
): Promise<unknown> {
  const { sessionId, method, params } = msg;
  if (typeof method !== 'string' || !isMethodAllowed(method)) {
    throw new Error(`method not allowed: ${String(method)}`);
  }
  const check = checkCommand({ method, params }, { origin: current.origin });
  if (!check.ok) throw new Error(check.reason);
  if (params !== undefined && !isObject(params)) {
    throw new Error('invalid params');
  }

  // After a worker restart the redactor has lost the known values, so
  // nothing that can carry page text may go up until the panel re-registers
  // them (fail closed: unlisted methods are refused too).
  if (current.needsRegistration && !UNREGISTERED_METHODS.has(method)) {
    throw new Error(REGISTRATION_REFUSED);
  }
  const screenshot = method === 'Page.captureScreenshot';
  const lock = lockFor(current.tabId);

  const isTab = sessionId === tabSessionId(current.tabId);
  const child = isTab ? undefined : current.children.get(sessionId);
  if (!isTab && !child) throw new Error('unknown session');

  // Frame scope: a child must have its Page domain on and its frame tree
  // read. The command's frame, context, object and node handles are
  // resolved against its session's scope now (this may ask Chrome) and
  // re-checked synchronously below.
  if (child && !(await withTimeout(child.pageReady, MASK_STEP_MS, 'frame'))) {
    throw new Error(FRAME_OUT_OF_SCOPE);
  }
  const session = frameSession(current, sessionId, child);
  const addresses = commandAddresses(params);
  // A frame child without a scope (never expected once pageReady is true)
  // takes no handle at all; a worker has no frames and is judged by URL.
  if (!session && child && !WORKER_TYPES.has(child.type ?? '')) {
    throw new Error(FRAME_OUT_OF_SCOPE);
  }
  // Deny by default: the agent may accept or type into a dialog only while
  // one is known to be open and was raised by a frame in scope. A plain
  // dismissal is always allowed, so the tab never stays frozen. (One raised
  // out of scope was already answered by the bridge.)
  if (
    method === 'Page.handleJavaScriptDialog' &&
    !(current.dialog && !current.dialog.outOfScope) &&
    (!isObject(params) || params.accept !== false || 'promptText' in params)
  ) {
    throw new Error(
      current.dialog?.outOfScope ? FRAME_OUT_OF_SCOPE : 'no dialog is open',
    );
  }
  // The lookups run under the tab's shared hold, so no screenshot's mask
  // window opens while they talk to the page.
  let handles: { frameIds: string[]; contextIds: number[] } = {
    frameIds: [],
    contextIds: [],
  };
  if (!screenshot) {
    await untilNoScreenshot(lock);
    const releaseLookups = holdShared(lock);
    try {
      if (session) handles = await resolveFrames(current, session, addresses);
    } finally {
      releaseLookups();
    }
  }
  const framesInScope = () =>
    !(screenshot && outOfScopeFramePresent(current)) &&
    (!child || childAllowed(current, child)) &&
    (!session ||
      (handles.frameIds.every((frameId) =>
        frameAllowedIn(current, session.scope, frameId),
      ) &&
        handles.contextIds.every((contextId) =>
          contextAllowedIn(current, session.scope, contextId),
        )));

  // The tab's current URL is checked on every command, child or not. An
  // ordinary command waits out a screenshot's mask window first, and takes
  // its shared hold right after these checks (see the send below).
  for (;;) {
    if (!screenshot) await untilNoScreenshot(lock);
    await requireTabInScope(current);
    if (!framesInScope()) throw new Error(FRAME_OUT_OF_SCOPE);
    if (!current.attached) throw new Error('not attached');
    if (screenshot || !lock.writer) break;
  }

  // describe() below is answered by the same session that produced the
  // result: backend node ids are only meaningful within their own renderer.
  const debuggee: chrome.debugger.DebuggerSession = child
    ? { tabId: current.tabId, sessionId }
    : { tabId: current.tabId };
  const sendIt = () =>
    tracked(child, () =>
      chrome.debugger.sendCommand(
        debuggee,
        method,
        params as Record<string, unknown> | undefined,
      ),
    );
  // A screenshot paints every in-process frame: it is refused while any
  // frame out of scope is present, checked right before each capture (no
  // await in between) and again after it, when the image is discarded.
  const capture = async () => {
    if (outOfScopeFramePresent(current)) throw new Error(FRAME_OUT_OF_SCOPE);
    const shot = await sendIt();
    if (outOfScopeFramePresent(current)) throw new Error(FRAME_OUT_OF_SCOPE);
    return shot;
  };
  let raw: unknown;
  if (screenshot) {
    raw = await exclusive(current, lock, (deadline) =>
      captureWithRetries(current, capture, deadline),
    );
  } else {
    // Shared hold: no await between taking it and chrome.debugger receiving
    // the command (tracked() calls sendCommand synchronously), released when
    // the command settles. Redaction below runs outside the hold.
    const release = holdShared(lock);
    let pending: Promise<unknown>;
    try {
      // Input is judged under the same hold as its send, and refused if the
      // set of frames out of scope changed while it was being judged.
      if (isInputAction(method, params)) {
        const before = inputGuardSignature(current);
        if (before !== '') {
          await checkInputTarget(current, method, params);
          if (
            inputGuardSignature(current) !== before ||
            !framesInScope() ||
            current.blocked ||
            current.stopped ||
            !current.attached
          ) {
            throw new Error(FRAME_OUT_OF_SCOPE);
          }
        }
      }
      pending = sendIt();
    } catch (error) {
      // A throw before the send must not leak the hold.
      release();
      throw error;
    }
    pending.then(release, release);
    raw = await pending;
  }
  // A frame that left scope while the command ran gets nothing back.
  if (!framesInScope()) throw new Error(FRAME_OUT_OF_SCOPE);
  const result = await redact(
    current,
    method,
    session ? pruneResult(current, session.scope, method, raw) : raw,
    describeVia(debuggee, child),
  );
  // The redaction may have waited on describe(); a run that ended meanwhile
  // sends nothing more.
  if (current.blocked || current.stopped) throw new Error(OUT_OF_SCOPE);
  if (!framesInScope()) throw new Error(FRAME_OUT_OF_SCOPE);
  return result;
}

// --- frame scope of a session ---------------------------------------------------

const FRAME_CHECK_GROUP = 'relay-frame-check';
// Each lookup step is bounded, and one discovery pass as a whole; a lookup
// that fails or times out refuses.
const FRAME_STEP_MS = 3_000;
const DISCOVERY_MS = 8_000;
/** The bridge's isolated world for reading where keyboard focus is. */
const FOCUS_WORLD = 'relay-focus';
const FRAME_OWNER_TAGS = /^(IFRAME|FRAME|OBJECT|EMBED|FENCEDFRAME)$/;
// Mouse actions that act on the page (a move only hovers).
const MOUSE_ACTIONS = new Set(['mousePressed', 'mouseReleased', 'mouseWheel']);
const KEY_INPUT_METHODS = new Set([
  'Input.dispatchKeyEvent',
  'Input.insertText',
]);

/** A fresh object group, so concurrent lookups never release each other's. */
const checkGroup = () => `${FRAME_CHECK_GROUP}-${crypto.randomUUID()}`;

/** A CDP session whose frames the bridge tracks: the tab or a frame child. */
interface FrameSession {
  scope: FrameScope;
  debuggee: chrome.debugger.DebuggerSession;
  /** The session's own top frame: the tab's main frame or the child's. */
  rootFrameId?: string;
  child?: Child;
}

function frameSession(
  current: Run,
  sessionId: string,
  child: Child | undefined,
): FrameSession | undefined {
  if (!child) {
    return {
      scope: current.frameScope,
      debuggee: { tabId: current.tabId },
      rootFrameId: current.mainFrameId,
    };
  }
  if (!child.frameScope) return undefined;
  return {
    scope: child.frameScope,
    debuggee: { tabId: current.tabId, sessionId },
    rootFrameId: child.targetId,
    child,
  };
}

const sessionSend = (
  session: FrameSession,
  method: string,
  params?: object,
  deadline = Number.POSITIVE_INFINITY,
) =>
  withTimeout(
    tracked(session.child, () =>
      chrome.debugger.sendCommand(
        session.debuggee,
        method,
        params as Record<string, unknown> | undefined,
      ),
    ),
    Math.max(0, Math.min(FRAME_STEP_MS, deadline - Date.now())),
    'frame check',
  );

const releaseGroup = (session: FrameSession, objectGroup: string) =>
  sessionSend(session, 'Runtime.releaseObjectGroup', { objectGroup }).catch(
    () => {},
  );

const objectIdOf = (result: unknown, key: 'object' | 'result') => {
  const object = isObject(result) ? result[key] : undefined;
  return isObject(object) && typeof object.objectId === 'string'
    ? object.objectId
    : undefined;
};

const describedNode = (described: unknown) => {
  const node = isObject(described) ? described.node : undefined;
  return isObject(node) ? node : undefined;
};

/**
 * Learns the main-world context of every frame of a session, straight
 * through chrome.debugger: for an agent that never enabled Runtime (so no
 * context events were seen). The top frame's context is where `document`
 * evaluates (`document` is unforgeable, so the page cannot redirect it); a
 * subframe's is where its owner's content document resolves. Each context
 * also records its document's origin, read from the DOM (not from script),
 * so a frame map that lags a navigation cannot vouch for it.
 */
async function discoverContexts(session: FrameSession) {
  const scope = session.scope;
  const deadline = Date.now() + DISCOVERY_MS;
  const objectGroup = checkGroup();
  const send = (method: string, params: object) =>
    sessionSend(session, method, params, deadline);
  try {
    for (const [frameId, frame] of [...scope.frames]) {
      let objectId: string | undefined;
      let documentUrl: unknown;
      if (frameId === session.rootFrameId || frame.parentId === undefined) {
        const evaluated = await send('Runtime.evaluate', {
          expression: 'document',
          objectGroup,
        });
        objectId = objectIdOf(evaluated, 'result');
        if (!objectId) continue;
        const described = await send('DOM.describeNode', {
          objectId,
        });
        documentUrl = describedNode(described)?.documentURL;
      } else {
        const owner = await send('DOM.getFrameOwner', {
          frameId,
        }).catch(() => undefined);
        const ownerId = isObject(owner) ? owner.backendNodeId : undefined;
        if (typeof ownerId !== 'number') continue;
        const node = describedNode(
          await send('DOM.describeNode', {
            backendNodeId: ownerId,
          }),
        );
        if (!node || node.frameId !== frameId) continue;
        const document = node.contentDocument;
        if (!isObject(document)) continue;
        documentUrl = document.documentURL;
        const resolved = await send('DOM.resolveNode', {
          backendNodeId: document.backendNodeId,
          objectGroup,
        }).catch(() => undefined);
        objectId = objectIdOf(resolved, 'object');
      }
      const contextId = objectId ? contextIdFromObjectId(objectId) : undefined;
      if (contextId === undefined || scope.contexts.has(contextId)) continue;
      scope.contexts.set(contextId, frameId);
      // No readable URL: the context is not trusted to its frame.
      const origin =
        typeof documentUrl === 'string'
          ? documentOrigin(documentUrl)
          : REFUSED_ORIGIN;
      if (origin !== undefined) scope.contextOrigins.set(contextId, origin);
    }
  } finally {
    void sessionSend(session, 'Runtime.releaseObjectGroup', {
      objectGroup,
    }).catch(() => {});
  }
}

/** The context a DOM node lives in: its own frame's main world. */
async function nodeContext(
  session: FrameSession,
  node: { backendNodeId?: unknown; nodeId?: unknown },
): Promise<number | undefined> {
  // Resolved without a context param, so the node lands in its own frame's
  // main world. A CDP error (a stale node id) is passed on as the reply.
  const objectGroup = checkGroup();
  let resolved: unknown;
  try {
    resolved = await sessionSend(session, 'DOM.resolveNode', {
      ...node,
      objectGroup,
    });
  } finally {
    await releaseGroup(session, objectGroup);
  }
  const objectId = objectIdOf(resolved, 'object');
  return objectId ? contextIdFromObjectId(objectId) : undefined;
}

/**
 * The frames and contexts a command addresses, through every frameId,
 * context id, unique context id, object id and DOM node id it carries (all
 * of them, so one in-scope handle cannot cover an out-of-scope one). Throws
 * FRAME_OUT_OF_SCOPE for a malformed or unknown handle (unknown contexts are
 * looked up once through the DOM first); the caller checks each one's
 * scope, again right before sending.
 */
async function resolveFrames(
  current: Run,
  session: FrameSession,
  addresses: ReturnType<typeof commandAddresses>,
): Promise<{ frameIds: string[]; contextIds: number[] }> {
  const scope = session.scope;
  const frameIds: string[] = [];
  for (const frameId of addresses.frameIds) {
    if (typeof frameId !== 'string') throw new Error(FRAME_OUT_OF_SCOPE);
    frameIds.push(frameId);
  }
  const candidates: unknown[] = [...addresses.contextIds];
  for (const uniqueId of addresses.uniqueContextIds) {
    candidates.push(
      typeof uniqueId === 'string' ? scope.uniqueContexts.get(uniqueId) : null,
    );
  }
  for (const objectId of addresses.objectIds) {
    candidates.push(
      typeof objectId === 'string' ? contextIdFromObjectId(objectId) : null,
    );
  }
  for (const node of addresses.nodes) {
    candidates.push(await nodeContext(session, node));
  }
  const contextIds: number[] = [];
  let discovered = false;
  for (const contextId of candidates) {
    if (typeof contextId !== 'number') throw new Error(FRAME_OUT_OF_SCOPE);
    if (!scope.contexts.has(contextId) && !discovered) {
      discovered = true;
      await discoverContexts(session);
    }
    if (!scope.contexts.has(contextId)) throw new Error(FRAME_OUT_OF_SCOPE);
    contextIds.push(contextId);
  }
  if (current.blocked || current.stopped) throw new Error(OUT_OF_SCOPE);
  return { frameIds, contextIds };
}

/**
 * Removes out-of-scope frames from a session's result before it is
 * redacted: their AX nodes, their frame-tree subtrees and their documents
 * inside a described node.
 */
function pruneResult(
  current: Run,
  scope: FrameScope,
  method: string,
  raw: unknown,
): unknown {
  const keep = (frameId: string) => frameAllowedIn(current, scope, frameId);
  switch (method) {
    case 'Accessibility.getFullAXTree':
      return pruneAXTree(raw, keep);
    case 'DOM.describeNode':
      return pruneDescribedNode(raw, keep);
    case 'Page.getFrameTree': {
      // Judged by the scope the events built: a frame it does not know is
      // left out.
      const pruned = pruneFrameTree(raw, keep);
      if (pruned === null) throw new Error(FRAME_OUT_OF_SCOPE);
      return pruned;
    }
    default:
      // Fail closed: a method not classified in frames.ts sends nothing.
      if (NO_FRAME_CONTENT_METHODS.has(method)) return raw;
      throw new Error(FRAME_OUT_OF_SCOPE);
  }
}

/**
 * True while a frame out of scope is present in the tab's renderer, or in
 * the renderer of an allowed out-of-process frame. Frames that are disallowed
 * out-of-process children are not counted (they are not in these scopes):
 * plan 1 lets them paint (ads, reCAPTCHA).
 */
function outOfScopeFramePresent(current: Run): boolean {
  const any = (scope: FrameScope) => {
    for (const frameId of scope.frames.keys()) {
      if (!frameAllowedIn(current, scope, frameId)) return true;
    }
    return false;
  };
  if (any(current.frameScope)) return true;
  for (const child of current.children.values()) {
    if (child.frameScope && childAllowed(current, child)) {
      if (any(child.frameScope)) return true;
    }
  }
  return false;
}

/**
 * The frames input must not reach, as a comparable string: every frame out
 * of scope in the tab's renderer or an allowed child's, and every frame
 * child (not a worker) that is itself disallowed. Empty when there is none,
 * so input on ordinary pages pays no lookup.
 */
function inputGuardSignature(current: Run): string {
  const ids: string[] = [];
  const collect = (key: string, scope: FrameScope) => {
    for (const frameId of scope.frames.keys()) {
      if (!frameAllowedIn(current, scope, frameId))
        ids.push(`${key}:${frameId}`);
    }
  };
  collect('tab', current.frameScope);
  for (const [sessionId, child] of current.children) {
    if (WORKER_TYPES.has(child.type ?? '')) continue;
    if (!childAllowed(current, child)) {
      ids.push(`child:${sessionId}`);
    } else if (child.frameScope) {
      collect(sessionId, child.frameScope);
    }
  }
  return ids.sort().join(',');
}

const isInputAction = (method: string, params: unknown) =>
  KEY_INPUT_METHODS.has(method) ||
  (method === 'Input.dispatchMouseEvent' &&
    isObject(params) &&
    MOUSE_ACTIONS.has(String(params.type)));

/**
 * Judges a described node as an input target: the frame it shows, when it
 * is a frame owner (undefined when that cannot be read), undefined for a
 * host with a closed shadow root (its focused element cannot be seen), else
 * null (an ordinary element of an in-scope document).
 */
function inputTargetFrame(node: Record<string, unknown> | undefined) {
  if (!node) return undefined;
  if (typeof node.frameId === 'string') return node.frameId;
  if (FRAME_OWNER_TAGS.test(String(node.localName ?? '').toUpperCase())) {
    return undefined;
  }
  const roots = Array.isArray(node.shadowRoots) ? node.shadowRoots : [];
  if (roots.some((r) => isObject(r) && r.shadowRootType === 'closed')) {
    return undefined;
  }
  return null;
}

/**
 * The frame that keyboard focus is in, read in the bridge's own isolated
 * world of the main frame (page script cannot tamper with it): focus is
 * followed through open shadow roots and into reachable same-origin frames,
 * and stops at a frame whose document is not reachable (cross-origin or out
 * of process). Null when focus is on an ordinary element of a reachable
 * document; undefined when it cannot be classified (a frame owner without a
 * frame, a closed shadow root).
 */
async function focusedFrameId(
  current: Run,
  tab: FrameSession,
): Promise<string | null | undefined> {
  const expression = `(() => {
    let el = document.activeElement;
    for (let i = 0; i < 64 && el; i++) {
      while (el.shadowRoot && el.shadowRoot.activeElement) {
        el = el.shadowRoot.activeElement;
      }
      if (!${FRAME_OWNER_TAGS}.test(el.tagName)) return el;
      let doc = null;
      try { doc = el.contentDocument; } catch {}
      if (!doc || !doc.activeElement) return el;
      el = doc.activeElement;
    }
    return el;
  })()`;
  const objectGroup = checkGroup();
  const evaluate = async (contextId: number) =>
    sessionSend(tab, 'Runtime.evaluate', {
      expression,
      contextId,
      objectGroup,
    });
  try {
    let evaluated: unknown;
    try {
      if (current.focusWorld === undefined) throw new Error('no world');
      evaluated = await evaluate(current.focusWorld);
    } catch {
      // No world yet, or its document is gone: make one (its context
      // events are dropped like the mask's, see isBridgeWorldCreated).
      const world = await sessionSend(tab, 'Page.createIsolatedWorld', {
        frameId: current.mainFrameId,
        worldName: FOCUS_WORLD,
      });
      const contextId = isObject(world) ? world.executionContextId : undefined;
      if (typeof contextId !== 'number') return undefined;
      current.focusWorld = contextId;
      evaluated = await evaluate(contextId);
    }
    const objectId = objectIdOf(evaluated, 'result');
    if (!objectId) return null;
    // Pierced one level: a closed shadow root shows up here.
    return inputTargetFrame(
      describedNode(
        await sessionSend(tab, 'DOM.describeNode', {
          objectId,
          depth: 1,
          pierce: true,
        }),
      ),
    );
  } finally {
    await releaseGroup(tab, objectGroup);
  }
}

/**
 * Refuses input that would land outside scope: a key or text while focus is
 * in a frame out of scope or a disallowed out-of-process frame (or cannot be
 * placed), a mouse press, release or wheel at a point in one. The caller
 * runs it only while such a frame exists (see inputGuardSignature), so
 * ordinary pages pay nothing. Input goes to the tab whatever session sends
 * it.
 */
async function checkInputTarget(current: Run, method: string, params: unknown) {
  const tab: FrameSession = {
    scope: current.frameScope,
    debuggee: { tabId: current.tabId },
    rootFrameId: current.mainFrameId,
  };
  const judge = (frameId: string | null | undefined) => {
    if (frameId === null) return;
    // Unknown or unreadable: refuse. A frame id an out-of-process child
    // owns is judged by that child (frameAllowedIn -> childFrameAllowed).
    if (frameId === undefined || !frameAllowedIn(current, tab.scope, frameId)) {
      throw new Error(FRAME_OUT_OF_SCOPE);
    }
  };
  if (method !== 'Input.dispatchMouseEvent') {
    judge(await focusedFrameId(current, tab));
    return;
  }
  const { x, y } = params as Record<string, unknown>;
  if (typeof x !== 'number' || typeof y !== 'number') {
    throw new Error(FRAME_OUT_OF_SCOPE);
  }
  // Hit-tests into in-process frames and names the frame of the node; over
  // an out-of-process frame it stops at the owner element.
  const hit = await sessionSend(tab, 'DOM.getNodeForLocation', {
    x: Math.round(x),
    y: Math.round(y),
  }).catch(() => undefined);
  if (!isObject(hit) || typeof hit.frameId !== 'string') {
    throw new Error(FRAME_OUT_OF_SCOPE);
  }
  judge(hit.frameId);
  // The node hit may itself be a frame owner: judge the frame it shows.
  const node = describedNode(
    await sessionSend(tab, 'DOM.describeNode', {
      backendNodeId: hit.backendNodeId,
    }).catch(() => undefined),
  );
  judge(node ? inputTargetFrame(node) : undefined);
}

/**
 * Runs a command for a child session with its reject in the child's
 * `inFlight`, so a frame removed meanwhile fails it at once (see dropChild);
 * a tab-session command runs as is.
 */
function tracked<T>(child: Child | undefined, run: () => Promise<T>) {
  if (!child) return run();
  return new Promise<T>((resolve, reject) => {
    child.inFlight.add(reject);
    run()
      .then(resolve, reject)
      .finally(() => child.inFlight.delete(reject));
  });
}

// --- screenshots -------------------------------------------------------------

// Every mask step (one CDP call) and the capture are bounded, and all the
// applies together share one budget, so a hung page or a child that never
// answers ends in a refusal or a capture error inside the relay's 30s
// command timeout: at most 10s of applies, 3s to settle, 8s to capture, 3s
// to verify and 3s to restore (sessions settle, verify and restore in
// parallel). The page restores itself after SELF_RESTORE_MS (25s) if the
// worker never gets to; verify refuses a capture that outlived it.
const MASK_STEP_MS = 3_000;
const MASK_APPLY_BUDGET_MS = 10_000;
const CAPTURE_MS = 8_000;
const STOP_WAIT_MS = 5_000;
// A screenshot waits this long for the commands already in flight to
// settle, then gives up (refused) so a hung command cannot wedge the tab.
const DRAIN_MS = 5_000;
// An ordinary command waits this long for a screenshot's window to end.
const COMMAND_WAIT_MS = 30_000;
// A screenshot whose verify saw the page change is retried this often, this
// far apart, while more than MIN_RETRY_REMAINING_MS of its deadline is left.
const MAX_RETRIES = 2;
const RETRY_DELAY_MS = 250;
const MIN_RETRY_REMAINING_MS = 5_000;
// One deadline per screenshot, covering the drain, every attempt's applies
// and every capture. Verify and restore (one 3s step each) run after it, so
// the reply still lands inside the relay's 30s command timeout.
const SCREENSHOT_DEADLINE_MS = 22_000;

const sleep = (ms: number) =>
  new Promise<void>((resolve) => setTimeout(resolve, ms));

/**
 * A per-tab reader/writer lock, at module level rather than per run (a run
 * replaced mid-screenshot still owns masks on the page). Ordinary commands
 * hold it shared while chrome.debugger runs them; a screenshot holds it
 * exclusively for its whole mask window, after the commands in flight have
 * drained. `queue` orders screenshots one at a time.
 */
interface TabLock {
  queue: ReturnType<typeof createScreenshotQueue>;
  readers: number;
  drained: Array<() => void>;
  /** Set while a screenshot holds the tab; resolves when it lets go. */
  writer?: Promise<void>;
}
const tabLocks = new Map<number, TabLock>();

function lockFor(tabId: number): TabLock {
  let lock = tabLocks.get(tabId);
  if (!lock) {
    lock = { queue: createScreenshotQueue(), readers: 0, drained: [] };
    tabLocks.set(tabId, lock);
  }
  return lock;
}

/** Waits (bounded) until no screenshot holds the tab. */
async function untilNoScreenshot(lock: TabLock) {
  const deadline = Date.now() + COMMAND_WAIT_MS;
  while (lock.writer) {
    const left = deadline - Date.now();
    if (left <= 0) throw new Error('tab busy with a screenshot');
    await Promise.race([lock.writer, sleep(left)]);
  }
}

function holdShared(lock: TabLock): () => void {
  lock.readers += 1;
  let released = false;
  return () => {
    if (released) return;
    released = true;
    lock.readers -= 1;
    if (lock.readers === 0) {
      for (const notify of lock.drained.splice(0)) notify();
    }
  };
}

/** True once no command holds the tab, false after `ms`. */
function drained(lock: TabLock, ms: number): Promise<boolean> {
  if (lock.readers === 0) return Promise.resolve(true);
  return new Promise((resolve) => {
    const timer = setTimeout(() => resolve(false), ms);
    lock.drained.push(() => {
      clearTimeout(timer);
      resolve(true);
    });
  });
}

/**
 * Runs a screenshot with the tab held exclusively: queued behind other
 * screenshots of the tab, new commands held from the moment it starts,
 * commands already in flight drained first (bounded). A run stopped before
 * or while it waits never applies a mask.
 */
function exclusive<T>(
  current: Run,
  lock: TabLock,
  task: (deadline: number) => Promise<T>,
) {
  return lock.queue(async () => {
    if (current.stopped || current.blocked) throw new ScreenshotMaskError();
    // One deadline for the drain and every attempt.
    const deadline = Date.now() + SCREENSHOT_DEADLINE_MS;
    let release = () => {};
    lock.writer = new Promise<void>((resolve) => {
      release = resolve;
    });
    try {
      if (!(await drained(lock, DRAIN_MS))) throw new ScreenshotMaskError();
      if (current.stopped || current.blocked) throw new ScreenshotMaskError();
      return await task(deadline);
    } finally {
      lock.writer = undefined;
      release();
    }
  });
}

/** Resolves once every queued screenshot of the tab has finished. */
async function screenshotsIdle(tabId: number) {
  await tabLocks.get(tabId)?.queue(async () => {});
}

/**
 * captureMasked, tried again (up to MAX_RETRIES, RETRY_DELAY_MS apart, while
 * over MIN_RETRY_REMAINING_MS of the deadline is left) only when verify
 * refused because the page changed
 * during the window, as a re-render or text churn does; a structural
 * refusal (closed shadow root, uncovered frame, stale mask) is final. The
 * refusal message is the same either way.
 */
async function captureWithRetries(
  current: Run,
  capture: () => Promise<unknown>,
  deadline: number,
): Promise<unknown> {
  for (let attempt = 0; ; attempt++) {
    try {
      return await captureMasked(current, capture, deadline);
    } catch (error) {
      const retry =
        error instanceof ScreenshotMaskError &&
        error.retryable &&
        attempt < MAX_RETRIES &&
        deadline - Date.now() > MIN_RETRY_REMAINING_MS &&
        !current.stopped &&
        !current.blocked;
      if (!retry) {
        throw error instanceof ScreenshotMaskError
          ? new ScreenshotMaskError()
          : error;
      }
      await sleep(RETRY_DELAY_MS);
    }
  }
}

/**
 * Isolated-world execution contexts the masking created, per CDP session.
 * Their Runtime.executionContextCreated / Destroyed events are dropped, so
 * the agent never sees (or tries to use) the mask's world.
 */
const maskContexts = new Set<string>();
const contextKey = (sessionId: string | undefined, id: unknown) =>
  `${sessionId ?? ''}:${String(id)}`;

/** True when the event is about the mask's isolated world: drop it. */
function isMaskWorldEvent(
  sessionId: string | undefined,
  method: string,
  params: unknown,
): boolean {
  if (!isObject(params)) return false;
  if (method === 'Runtime.executionContextCreated') {
    const context = isObject(params.context) ? params.context : undefined;
    if (!context || !BRIDGE_WORLDS.has(String(context.name))) return false;
    maskContexts.add(contextKey(sessionId, context.id));
    return true;
  }
  if (method === 'Runtime.executionContextDestroyed') {
    return maskContexts.delete(
      contextKey(sessionId, params.executionContextId),
    );
  }
  return false;
}
/**
 * Tracks the dialog showing now. One raised by a frame out of scope (or a
 * disallowed child, or a frame it cannot place) is answered by the bridge
 * at once, so the renderer it freezes (the tab's, for a same-site frame) is
 * released and no agent-chosen text reaches it: an alert is accepted (its
 * only answer), anything else dismissed (a beforeunload dismissal keeps the
 * page). Its opening and closing are then not forwarded. Returns whether the
 * event may go on.
 */
function dialogEventForwarded(
  current: Run,
  source: chrome.debugger.DebuggerSession,
  method: string,
  params: unknown,
): boolean {
  if (method === 'Page.javascriptDialogClosed') {
    const shown = current.dialog;
    current.dialog = undefined;
    return !shown?.outOfScope;
  }
  if (method !== 'Page.javascriptDialogOpening') return true;
  const frameId = isObject(params) ? params.frameId : undefined;
  let inScope: boolean;
  if (source.sessionId === undefined) {
    inScope =
      typeof frameId === 'string' &&
      frameAllowedIn(current, current.frameScope, frameId);
  } else {
    const child = current.children.get(source.sessionId);
    inScope =
      child !== undefined &&
      childAllowed(current, child) &&
      (child.frameScope === undefined ||
        (typeof frameId === 'string' &&
          frameAllowedIn(current, child.frameScope, frameId)));
  }
  current.dialog = { outOfScope: !inScope };
  if (inScope) return true;
  const type = isObject(params) ? params.type : undefined;
  chrome.debugger
    .sendCommand(source, 'Page.handleJavaScriptDialog', {
      accept: type === 'alert',
    })
    .catch(() => {
      // Already answered, or the page went away.
    });
  return false;
}

/** Isolated worlds the bridge makes: never shown to, or usable by, the agent. */
const BRIDGE_WORLDS = new Set([MASK_WORLD, FOCUS_WORLD]);

const isMaskWorldCreated = (method: string, params: unknown) =>
  method === 'Runtime.executionContextCreated' &&
  isObject(params) &&
  isObject(params.context) &&
  BRIDGE_WORLDS.has(String(params.context.name));

// Child targets with no document to mask (and nothing painted on screen).
const WORKER_TYPES = new Set([
  'worker',
  'shared_worker',
  'service_worker',
  'worklet',
  'shared_storage_worklet',
  'auction_worklet',
]);

const withTimeout = <T>(promise: Promise<T>, ms: number, what: string) =>
  new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`${what} timed out`)), ms);
    promise.then(resolve, reject).finally(() => clearTimeout(timer));
  });

/**
 * Captures a screenshot with every sensitive field hidden. The bridge masks
 * the tab session and every attached frame session (an out-of-process
 * iframe paints into the tab's screenshot) straight through chrome.debugger,
 * never through the agent's allowlist, waits for each to render a frame,
 * captures, checks that every mask is still in place in the same documents,
 * then restores. Any failure is a
 * refusal with a fixed message; the raw image is only returned when every
 * session was masked and verified.
 *
 * Ordering: forwardCdp runs every screenshot under the tab's lock held
 * exclusively (see exclusive): a second screenshot (from the tab, any child
 * session, or another run on the same tab) waits until the first has
 * restored the page, and no other command of the tab runs inside the mask
 * window. Page script can still change the page during the window; verify
 * refuses the capture when it touched a field, a frame, a mask or showed a
 * known value.
 */
async function captureMasked(
  current: Run,
  capture: () => Promise<unknown>,
  deadline: number,
): Promise<unknown> {
  const objectGroup = `mask-${crypto.randomUUID()}`;
  const frames = [...current.children].filter(
    ([, child]) => !WORKER_TYPES.has(child.type ?? ''),
  );
  // An out-of-process iframe's target id is its frame id.
  const covered = new Set(
    frames.flatMap(([, child]) => (child.targetId ? [child.targetId] : [])),
  );
  const isCoveredFrame = (frameId: string) => covered.has(frameId);
  const maskSend =
    (debuggee: chrome.debugger.DebuggerSession, child?: Child) =>
    (method: string, params?: Record<string, unknown>) =>
      withTimeout(
        tracked(child, () =>
          chrome.debugger.sendCommand(debuggee, method, params),
        ),
        MASK_STEP_MS,
        'screenshot mask',
      );
  // A field is masked by its value too: a value-shaped number, or a value
  // the caseworker registered. Known values stay in the worker.
  const isSensitiveValue = (value: string) =>
    hasSensitiveShape(value) || current.redactor.hasKnownValue(value);
  const hasKnownValue = (text: string) => current.redactor.hasKnownValue(text);
  const common = {
    objectGroup,
    isCoveredFrame,
    isSensitiveValue,
    hasKnownValue,
    knownValuesRegistered: () =>
      current.hasKnownValues || current.needsRegistration,
  };
  const masks = [
    createMaskSession({
      ...common,
      send: maskSend({ tabId: current.tabId }),
      onWorld: (id) => maskContexts.add(contextKey(undefined, id)),
    }),
    ...frames.map(([sessionId, child]) =>
      createMaskSession({
        ...common,
        send: maskSend({ tabId: current.tabId, sessionId }, child),
        outOfProcess: true,
        onWorld: (id) => maskContexts.add(contextKey(sessionId, id)),
      }),
    ),
  ];
  // An apply that outlives the shared budget is refused; its session's
  // restore then stops it before its next step.
  const applyBy = Math.min(Date.now() + MASK_APPLY_BUDGET_MS, deadline);
  const sessions = masks.map((mask) => ({
    apply: () => {
      // A run stopped between applies masks nothing more.
      if (current.stopped || current.blocked) {
        return Promise.reject(new Error('run stopped'));
      }
      return withTimeout(
        mask.apply(),
        Math.max(0, applyBy - Date.now()),
        'masking',
      );
    },
    restore: () => mask.restore(),
  }));
  const known = new Set(current.children.keys());
  return withScreenshotMask(sessions, async () => {
    await Promise.all(masks.map((mask) => mask.settle()));
    // A run stopped meanwhile only restores (stopRun waits for that).
    if (current.stopped || current.blocked) throw new ScreenshotMaskError();
    const shot = await withTimeout(
      capture(),
      Math.max(0, Math.min(CAPTURE_MS, deadline - Date.now())),
      'screenshot',
    );
    await Promise.all(masks.map((mask) => mask.verify()));
    // A frame that attached during the window was never masked.
    for (const [sessionId, child] of current.children) {
      if (!known.has(sessionId) && !WORKER_TYPES.has(child.type ?? '')) {
        throw new ScreenshotMaskError();
      }
    }
    return shot;
  });
}

/**
 * Answers the redactor's DOM lookups straight through chrome.debugger, not
 * through the agent's allowlist, so the agent never sees them. Returns
 * undefined (classified as sensitive) for anything but an element node.
 * A child's lookups are tracked in its `inFlight`, so a frame removed
 * mid-lookup fails them at once instead of waiting out the redactor's
 * per-call timeout.
 */
const describeVia =
  (debuggee: chrome.debugger.DebuggerSession, child?: Child): Describe =>
  async (backendNodeId) => {
    const send = () =>
      chrome.debugger.sendCommand(debuggee, 'DOM.describeNode', {
        backendNodeId,
      });
    const described = child
      ? await new Promise<unknown>((resolve, reject) => {
          child.inFlight.add(reject);
          send()
            .then(resolve, reject)
            .finally(() => child.inFlight.delete(reject));
        })
      : await send();
    const node = isObject(described) ? described.node : undefined;
    if (!isObject(node) || !Array.isArray(node.attributes)) return undefined;
    return descriptorFromAttributes(node.attributes);
  };

/** The redacted result, or an error reply: never the raw result. */
const redact = (
  current: Run,
  method: string,
  raw: unknown,
  describe?: Describe,
): Promise<unknown> => redactReply(current.redactor, method, raw, describe);

/**
 * Forgets a child session and fails its in-flight commands. chrome.debugger
 * never settles a command whose child session detaches first (a page removing
 * an iframe while the agent sets it up), so without this the relay only
 * answers at its command timeout and the agent stalls behind it.
 */
function dropChild(current: Run, sessionId: string) {
  const child = current.children.get(sessionId);
  if (!child) return;
  current.children.delete(sessionId);
  for (const reject of child.inFlight) reject(new Error('session detached'));
  child.inFlight.clear();
}

function dropChildren(current: Run) {
  for (const sessionId of [...current.children.keys()]) {
    dropChild(current, sessionId);
  }
}

// --- browser events ----------------------------------------------------------

/**
 * Enables the Page domain on a frame child, straight through chrome.debugger,
 * so the child's own navigation events keep its URL and frame scope current,
 * then seeds its frame scope from its frame tree. Workers have no Page
 * domain. Fails closed: a child whose Page domain or tree cannot be read is
 * out of scope.
 */
function enableChildPage(current: Run, sessionId: string, child: Child) {
  if (WORKER_TYPES.has(child.type ?? '')) {
    child.pageEnabled = true;
    return Promise.resolve(true);
  }
  const debuggee = { tabId: current.tabId, sessionId };
  const scope = createFrameScope();
  // Until the tree is read, only the child's own frame is known.
  if (child.targetId) scope.frames.set(child.targetId, { url: child.url });
  child.frameScope = scope;
  return chrome.debugger
    .sendCommand(debuggee, 'Page.enable')
    .then(() => chrome.debugger.sendCommand(debuggee, 'Page.getFrameTree'))
    .then(
      (tree) => {
        const seeded = framesOfTree(tree);
        if (seeded.length === 0) return false;
        scope.frames = new Map(seeded);
        return true;
      },
      () => false,
    )
    .then((ok) => {
      child.pageEnabled = ok;
      return ok;
    });
}

/**
 * Keeps a child's URL current from its own main-frame navigation (an
 * out-of-process frame's target id is its frame id), in the same turn as
 * the event, so the event that moves it out of scope is itself dropped and
 * the next command is refused. Target.targetInfoChanged also updates it.
 */
function trackChildNavigation(
  child: Child,
  method: string,
  params: Record<string, unknown>,
) {
  if (method === 'Page.frameNavigated') {
    const frame = isObject(params.frame) ? params.frame : undefined;
    if (frame && frame.id === child.targetId) {
      child.url = typeof frame.url === 'string' ? frame.url : '';
    }
  } else if (method === 'Page.navigatedWithinDocument') {
    if (params.frameId === child.targetId) {
      child.url = typeof params.url === 'string' ? params.url : '';
    }
  }
}

function trackChildren(
  current: Run,
  method: string,
  params: unknown,
  sessionId?: string,
) {
  if (!isObject(params)) return;
  if (sessionId !== undefined) {
    const from = current.children.get(sessionId);
    if (from) trackChildNavigation(from, method, params);
  }
  const info = isObject(params.targetInfo) ? params.targetInfo : undefined;
  if (method === 'Target.attachedToTarget') {
    if (typeof params.sessionId !== 'string') return;
    const child: Child = {
      targetId: typeof info?.targetId === 'string' ? info.targetId : undefined,
      type: typeof info?.type === 'string' ? info.type : undefined,
      // An unknown URL is treated as out of scope.
      url: typeof info?.url === 'string' ? info.url : '',
      inFlight: current.children.get(params.sessionId)?.inFlight ?? new Set(),
      pageReady: Promise.resolve(false),
    };
    current.children.set(params.sessionId, child);
    child.pageReady = enableChildPage(current, params.sessionId, child);
  } else if (method === 'Target.detachedFromTarget') {
    if (typeof params.sessionId === 'string') {
      dropChild(current, params.sessionId);
    }
  } else if (method === 'Target.targetInfoChanged') {
    if (!info || typeof info.targetId !== 'string') return;
    for (const child of current.children.values()) {
      if (child.targetId === info.targetId) {
        child.url = typeof info.url === 'string' ? info.url : '';
      }
    }
  }
}

/**
 * Resumes a child target that auto-attach paused (waitForDebuggerOnStart).
 * agent-browser only resumes the tab session, and cannot resume a child the
 * bridge refuses, so without this a frame added mid-run stays blank in the
 * caseworker's tab.
 */
function resumeIfPaused(current: Run, method: string, params: unknown) {
  if (method !== 'Target.attachedToTarget' || !isObject(params)) return;
  if (params.waitingForDebugger !== true) return;
  if (typeof params.sessionId !== 'string') return;
  chrome.debugger
    .sendCommand(
      { tabId: current.tabId, sessionId: params.sessionId },
      'Runtime.runIfWaitingForDebugger',
    )
    .catch(() => {
      // The child went away first.
    });
}

/** The URL a main-frame navigation event is heading to, if it is one. */
function mainFrameNavigationUrl(
  current: Run,
  method: string,
  params: unknown,
): string | undefined {
  if (!NAVIGATION_EVENTS.has(method) || !isObject(params)) return undefined;
  if (method === 'Page.frameNavigated') {
    const frame = isObject(params.frame) ? params.frame : undefined;
    if (!frame || frame.parentId !== undefined) return undefined;
    return typeof frame.url === 'string' ? frame.url : '';
  }
  if (params.frameId !== current.mainFrameId) return undefined;
  let url: string;
  if (method === 'Network.requestWillBeSent') {
    if (params.type !== 'Document') return undefined;
    const request = isObject(params.request) ? params.request : undefined;
    url = typeof request?.url === 'string' ? request.url : '';
  } else {
    url = typeof params.url === 'string' ? params.url : '';
  }
  // A pre-commit signal for `javascript:` (an ASP.NET __doPostBack link, a
  // void(0) button) runs script in this document and never commits a new
  // one, so it is not leaving. Committed navigations are checked above.
  if (PRE_COMMIT_EVENTS.has(method) && isJavascriptUrl(url)) return undefined;
  return url;
}

/** The child session a lifecycle event is about, if any. */
function lifecycleSubject(current: Run, params: unknown): Child | undefined {
  if (!isObject(params)) return undefined;
  if (typeof params.sessionId === 'string') {
    const child = current.children.get(params.sessionId);
    if (child) return child;
  }
  const info = isObject(params.targetInfo) ? params.targetInfo : undefined;
  if (typeof info?.targetId !== 'string') return undefined;
  for (const child of current.children.values()) {
    if (child.targetId === info.targetId) return child;
  }
  return undefined;
}

/** Blanks the URL and title (which carries the URL for frames). */
function withoutTargetUrl(params: unknown): unknown {
  if (!isObject(params) || !isObject(params.targetInfo)) return params;
  return {
    ...params,
    targetInfo: { ...params.targetInfo, url: '', title: '' },
  };
}

export function onDebuggerEvent(
  source: chrome.debugger.DebuggerSession,
  method: string,
  params?: unknown,
) {
  const current = run;
  if (!current || source.tabId !== current.tabId) return;
  if (current.blocked) return;
  // Frame scope is kept from every raw tab event, sent on or not, even
  // between chrome.debugger.attach and the end of the attach (the frame
  // tree read there then supersedes it).
  // The mask's isolated worlds are never recorded, so a command naming one
  // is refused as an unknown context. A frame child's events keep its own
  // scope the same way.
  const scope =
    source.sessionId === undefined
      ? current.frameScope
      : current.children.get(source.sessionId)?.frameScope;
  const frameEventAllowed =
    scope === undefined ||
    (!isMaskWorldCreated(method, params) &&
      scopeFrameEvent(scope, current, method, params, (frameId) =>
        childFrameAllowed(current, frameId),
      ));
  // Dialogs are tracked (and an out-of-scope one answered) from raw events
  // too, attached or not, so a dialog opened during an attach is known.
  const dialogForwarded = dialogEventForwarded(current, source, method, params);
  // Nothing leaves the browser unless we hold the tab and it is in scope.
  if (!current.attached || !dialogForwarded) return;

  if (source.sessionId === undefined) {
    const navUrl = mainFrameNavigationUrl(current, method, params);
    if (navUrl !== undefined && !inScope(navUrl, current.origin)) {
      // Drop the event that would reveal the new document, then stop.
      leaveOrigin(current);
      return;
    }
  }

  trackChildren(current, method, params, source.sessionId);
  resumeIfPaused(current, method, params);

  const lifecycle = LIFECYCLE_EVENTS.has(method);
  let forwarded: unknown = params;
  if (source.sessionId !== undefined) {
    // An event from a child session: only allowed frames speak.
    const from = current.children.get(source.sessionId);
    const fromAllowed = from !== undefined && childAllowed(current, from);
    if (!fromAllowed && !lifecycle) return;
  }
  if (lifecycle) {
    const subject = lifecycleSubject(current, params);
    if (subject && !childAllowed(current, subject)) {
      forwarded = withoutTargetUrl(params);
    }
  }

  if (isMaskWorldEvent(source.sessionId, method, params)) return;
  // An event about a frame (or context) out of scope, or a navigation
  // heading out of scope (it names its destination before committing), is
  // dropped, on the tab or a frame child.
  if (!frameEventAllowed) return;

  const filtered = filterEvent(method, forwarded);
  if (!filtered) return;
  // Unregistered (see UNREGISTERED_EVENTS): only events with no page text.
  if (current.needsRegistration) {
    if (!UNREGISTERED_EVENTS.has(filtered.method)) return;
    if (LIFECYCLE_EVENTS.has(filtered.method)) {
      filtered.params = withoutTargetUrl(filtered.params);
    }
  }
  // Routing and scope above used the raw params; only what is sent is
  // redacted. `null` means the event cannot be redacted: drop it.
  const redacted = redactEventParams(
    current.redactor,
    filtered.method,
    filtered.params,
  );
  if (redacted === null) return;
  send(current, {
    type: 'event',
    sessionId: source.sessionId ?? tabSessionId(current.tabId),
    method: filtered.method,
    params: redacted,
  });
}

export function onDebuggerDetach(
  source: chrome.debugger.Debuggee,
  reason: string,
) {
  const current = run;
  if (!current || source.tabId !== current.tabId) return;
  // The caseworker dismissed Chrome's debugging infobar: the run is over.
  if (reason === 'canceled_by_user') {
    current.blocked = true;
    void endRun('debugger-canceled');
    return;
  }
  const announced = current.attached;
  current.attached = false;
  dropChildren(current);
  // P4b-d: the tab must be attached again (the agent re-attaches through the
  // relay) within the grace period, or the run ends.
  clearTimeout(current.detachTimer);
  current.detachTimer = setTimeout(() => {
    current.detachTimer = undefined;
    if (run === current && !current.attached) {
      void endRun('debugger-detached');
    }
  }, DETACH_GRACE_MS);
  // The agent was never given this session (a failed attach): say nothing.
  if (!announced) return;
  send(current, {
    type: 'event',
    method: 'Target.detachedFromTarget',
    params: { sessionId: tabSessionId(current.tabId), reason },
  });
}

/** Ends the run the moment the tab commits a URL outside scope. */
export function onTabUpdated(tabId: number, info: { url?: string }) {
  const current = run;
  if (!current || tabId !== current.tabId || info.url === undefined) return;
  if (!inScope(info.url, current.origin)) leaveOrigin(current);
}

export function onTabRemoved(tabId: number) {
  const current = run;
  if (!current || tabId !== current.tabId) return;
  send(current, { type: 'tabClosed' });
  void endRun('user');
}

// --- messages ----------------------------------------------------------------

/**
 * Registers a value the caseworker typed into a sensitive field, so it is
 * scrubbed wherever it is echoed. The service worker only passes on messages
 * from this extension. A value that cannot be registered is refused, and the
 * caller must not type it. Never logs or echoes the value.
 */
function addKnownValue(value: unknown): BridgeResponse {
  const current = run;
  if (!current || current.stopped) return { ok: false, error: 'no active run' };
  if (typeof value !== 'string' || !current.redactor.addKnownValue(value)) {
    return { ok: false, error: 'known value not registered' };
  }
  current.hasKnownValues = true;
  return { ok: true };
}

/** The run's state for the panel: ids and counts, never a token or URL. */
function getRunStatus(): RunStatus {
  const current = run;
  if (!current || current.stopped) return { ok: true, active: false };
  const status: RunStatus = {
    ok: true,
    active: true,
    runId: current.runId,
    origin: current.origin,
    attached: current.attached,
    needsRegistration: current.needsRegistration,
  };
  if (current.jev) {
    status.jevFailures = current.jev.failures();
    status.jevRequests = current.jev.requests();
  }
  return status;
}

/**
 * The panel re-entered the run's known values after a worker restart (it
 * sends only how many, never the values): screenshots may be taken again.
 */
function registrationComplete(count: unknown): BridgeResponse {
  const current = run;
  if (!current || current.stopped) return { ok: false, error: 'no active run' };
  if (
    count !== undefined &&
    (typeof count !== 'number' || !Number.isInteger(count) || count < 0)
  ) {
    return { ok: false, error: 'count must be a non-negative integer' };
  }
  current.needsRegistration = false;
  return { ok: true };
}

export async function handleMessage(message: unknown): Promise<BridgeResponse> {
  if (!isObject(message)) return { ok: false, error: 'invalid message' };
  if (message.type === 'startRun') {
    return startRun(message as Extract<BridgeMessage, { type: 'startRun' }>);
  }
  if (message.type === 'stopRun') return stopRun('user');
  if (message.type === 'addKnownValue') return addKnownValue(message.value);
  if (message.type === 'getRunStatus') return getRunStatus();
  if (message.type === 'registrationComplete') {
    return registrationComplete(message.count);
  }
  return { ok: false, error: 'unknown message type' };
}
