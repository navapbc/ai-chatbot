/// <reference types="chrome" />
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
  type Redactor,
} from '../../lib/relay/redactor';
import {
  createMaskSession,
  createScreenshotQueue,
  ScreenshotMaskError,
  withScreenshotMask,
} from '../../lib/relay/screenshot-mask';
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
  | { type: 'addKnownValue'; value: string };

export type BridgeResponse = { ok: true } | { ok: false; error: string };

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
  /** Runs this run's screenshots one at a time (see captureMasked). */
  screenshots: ReturnType<typeof createScreenshotQueue>;
}

const STORAGE_KEY = 'run';
const LOOPBACK_HOSTS = new Set(['127.0.0.1', 'localhost']);
const OUT_OF_SCOPE = 'tab is outside the allowed origin';
const FRAME_OUT_OF_SCOPE = 'frame is outside the allowed origins';
const FIRST_RECONNECT_MS = 1000;
const MAX_RECONNECT_MS = 30_000;
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

export async function startRun(
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

  await stopRun('replaced');
  const config: RunConfig = {
    relayUrl: msg.relayUrl,
    token: msg.token,
    tabId: msg.tabId,
    allowedFrameOrigins,
  };
  await chrome.storage.session.set({ [STORAGE_KEY]: config });
  begin(config, socket.url, claims);
  return { ok: true };
}

/**
 * Ends the run: tells the relay to close it (so the agent is cut off at once
 * rather than held for the resume window), closes the socket, detaches.
 */
export async function stopRun(
  reason: StopReason = 'user',
): Promise<BridgeResponse> {
  const current = run;
  run = undefined;
  // Before any detach await, so a slow stop cannot remove a newer run's config.
  const forgotten = chrome.storage.session.remove(STORAGE_KEY);
  if (current) {
    current.stopped = true;
    current.blocked = true;
    current.redactor.clear();
    clearTimeout(current.reconnectTimer);
    clearTimeout(current.expiryTimer);
    send(current, { type: 'stop', reason });
    try {
      current.ws?.close(1000, 'run stopped');
    } catch {
      // Already closed.
    }
    dropChildren(current);
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
  if (run === current) void stopRun('left-origin');
}

/** Picks up a run persisted before a service worker restart. */
export async function resumeRun(): Promise<void> {
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

function begin(
  config: RunConfig,
  socketUrl: string,
  claims: { runId: string; origin: string; exp: number },
  resumed = false,
) {
  const current: Run = {
    ...config,
    ...claims,
    socketUrl,
    stopped: false,
    attached: false,
    blocked: false,
    children: new Map(),
    reconnectDelayMs: FIRST_RECONNECT_MS,
    announceResume: resumed,
    redactor: createRedactor(),
    screenshots: createScreenshotQueue(),
  };
  run = current;
  current.expiryTimer = setTimeout(
    () => {
      if (run === current) void stopRun('token-expired');
    },
    Math.max(0, current.exp * 1000 - Date.now()),
  );
  connect(current);
}

function connect(current: Run) {
  if (current.stopped || run !== current) return;
  if (current.exp <= nowSeconds()) {
    void stopRun('token-expired');
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
      void stopRun();
      return;
    }
    // The tab left the origin while the socket was down: the run is over.
    if (!inScope(tab, current.origin)) {
      leaveOrigin(current);
      return;
    }
    current.reconnectDelayMs = FIRST_RECONNECT_MS;
    send(current, {
      type: 'register',
      runId: current.runId,
      tabId: current.tabId,
      origin: current.origin,
    });
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
      void stopRun();
      return;
    }
    scheduleReconnect(current);
  };
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

// Known gap (R17): this checks out-of-process child sessions only. A
// cross-origin but same-site iframe shares the tab's renderer and session, so
// commands reaching it through the tab session are not frame-checked.
const frameAllowed = (current: Run, url: string) =>
  inScope(url, current.origin) ||
  current.allowedFrameOrigins.some((origin) => inScope(url, origin));

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
        targetInfos: target ? [targetInfo(target)] : [],
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
      current.attached = true;
      current.mainFrameId = target.id;
      return redact(current, 'Target.attachToTarget', {
        sessionId: tabSessionId(current.tabId),
        targetInfo: targetInfo({ ...target, attached: true }),
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

  const isTab = sessionId === tabSessionId(current.tabId);
  const child = isTab ? undefined : current.children.get(sessionId);
  if (!isTab && !child) throw new Error('unknown session');

  // The tab's current URL is checked on every command, child or not.
  await requireTabInScope(current);
  if (child && !frameAllowed(current, child.url)) {
    throw new Error(FRAME_OUT_OF_SCOPE);
  }
  if (!current.attached) throw new Error('not attached');

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
  const raw =
    method === 'Page.captureScreenshot'
      ? await current.screenshots(() => captureMasked(current, sendIt))
      : await sendIt();
  const result = await redact(
    current,
    method,
    raw,
    describeVia(debuggee, child),
  );
  // The redaction may have waited on describe(); a run that ended meanwhile
  // sends nothing more.
  if (current.blocked || current.stopped) throw new Error(OUT_OF_SCOPE);
  return result;
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

// Every mask step (one CDP call) and the capture itself are bounded, so a
// hung page or a child that never answers ends in a refusal or a capture
// error well inside the relay's command timeout. The page restores itself
// after SELF_RESTORE_MS (25s) if the worker never gets to.
const MASK_STEP_MS = 3_000;
const CAPTURE_MS = 10_000;
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
 * never through the agent's allowlist, captures, checks that every mask is
 * still in place in the same documents, then restores. Any failure is a
 * refusal with a fixed message; the raw image is only returned when every
 * session was masked and verified.
 *
 * Ordering: forwardCdp runs every screenshot through the run's queue, so a
 * second screenshot (from the tab or any child session) waits until the
 * first has restored the page; overlapping masks would otherwise record each
 * other's style as the page's own. Other agent commands are not held: none
 * can unmask a field (eval is refused at the tool layer), and a navigation
 * or re-render during the window fails the verify step instead.
 */
async function captureMasked(
  current: Run,
  capture: () => Promise<unknown>,
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
  const sessions = [
    createMaskSession({
      send: maskSend({ tabId: current.tabId }),
      objectGroup,
      isCoveredFrame,
    }),
    ...frames.map(([sessionId, child]) =>
      createMaskSession({
        send: maskSend({ tabId: current.tabId, sessionId }, child),
        objectGroup,
        isCoveredFrame,
      }),
    ),
  ];
  const known = new Set(current.children.keys());
  return withScreenshotMask(sessions, async () => {
    const shot = await withTimeout(capture(), CAPTURE_MS, 'screenshot');
    for (const session of sessions) await session.verify();
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

function trackChildren(current: Run, method: string, params: unknown) {
  if (!isObject(params)) return;
  const info = isObject(params.targetInfo) ? params.targetInfo : undefined;
  if (method === 'Target.attachedToTarget') {
    if (typeof params.sessionId !== 'string') return;
    current.children.set(params.sessionId, {
      targetId: typeof info?.targetId === 'string' ? info.targetId : undefined,
      type: typeof info?.type === 'string' ? info.type : undefined,
      // An unknown URL is treated as out of scope.
      url: typeof info?.url === 'string' ? info.url : '',
      inFlight: current.children.get(params.sessionId)?.inFlight ?? new Set(),
    });
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
  // Nothing leaves the browser unless we hold the tab and it is in scope.
  if (!current.attached || current.blocked) return;

  if (source.sessionId === undefined) {
    const navUrl = mainFrameNavigationUrl(current, method, params);
    if (navUrl !== undefined && !inScope(navUrl, current.origin)) {
      // Drop the event that would reveal the new document, then stop.
      leaveOrigin(current);
      return;
    }
  }

  trackChildren(current, method, params);
  resumeIfPaused(current, method, params);

  const lifecycle = LIFECYCLE_EVENTS.has(method);
  let forwarded: unknown = params;
  if (source.sessionId !== undefined) {
    // An event from a child session: only allowed frames speak.
    const from = current.children.get(source.sessionId);
    const fromAllowed = from !== undefined && frameAllowed(current, from.url);
    if (!fromAllowed && !lifecycle) return;
  }
  if (lifecycle) {
    const subject = lifecycleSubject(current, params);
    if (subject && !frameAllowed(current, subject.url)) {
      forwarded = withoutTargetUrl(params);
    }
  }

  const filtered = filterEvent(method, forwarded);
  if (!filtered) return;
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
    void stopRun('debugger-canceled');
    return;
  }
  current.attached = false;
  dropChildren(current);
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
  void stopRun();
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
  return { ok: true };
}

export async function handleMessage(message: unknown): Promise<BridgeResponse> {
  if (!isObject(message)) return { ok: false, error: 'invalid message' };
  if (message.type === 'startRun') {
    return startRun(message as Extract<BridgeMessage, { type: 'startRun' }>);
  }
  if (message.type === 'stopRun') return stopRun('user');
  if (message.type === 'addKnownValue') return addKnownValue(message.value);
  return { ok: false, error: 'unknown message type' };
}
