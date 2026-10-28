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
  inRunOrigins,
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
import { isSensitiveField } from '../../lib/relay/sensitive-fields';
import {
  DEEP_ACTIVE,
  FIELD_KIND,
  FOCUS_FIELD,
  FRAME_OWNER_TAGS,
  SCAN_UNFILLED,
  TYPE_INTO,
  UNFILLED_AT,
  UNFILLED_SUMMARY,
} from './typing-scripts';

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
  /**
   * Ends the run. With `runId`, only that run: another run (or none) is left
   * alone and the answer is still `{ ok: true }`. Without it, whatever run is
   * current (older callers and tests).
   */
  | { type: 'stopRun'; runId?: string }
  | { type: 'addKnownValue'; runId: string; value: string }
  | { type: 'getRunStatus' }
  /** The panel re-entered the run's known values (count only, no values). */
  | { type: 'registrationComplete'; runId: string; count?: number }
  /**
   * The caseworker allowed the origin the run is held on (see holdOrigin)
   * for the rest of this run: exactly `pendingOrigin`, an https origin.
   */
  | { type: 'allowOrigin'; origin: string }
  /**
   * Registers `value` for scrubbing, then types it into the field focused in
   * the run's tab if the redactor masks that field (see typeSensitive).
   * Answers `{ ok: true }` or a fixed refusal, never the value.
   */
  | { type: 'typeSensitive'; runId: string; value: string }
  /**
   * Finds the required fields still empty in the run's tab and highlights
   * the first (see showUnfilledField). Answers counts and flags only
   * (UnfilledFieldResult) or a fixed refusal, never a value, name, label or
   * page text.
   */
  | { type: 'showUnfilledField'; runId: string }
  /**
   * Watches the run's tab for a form submission while the panel is in
   * Review (see armSubmitWatch); `getRunStatus` reports it as `submitted`.
   * Idempotent. With `reset`, first forgets what the watch saw (keeping its
   * attach). Answers `{ ok: true }` or a fixed refusal.
   */
  | { type: 'armSubmitWatch'; runId: string; reset?: boolean }
  /** Stops the watch and forgets what it saw (see disarmSubmitWatch). */
  | { type: 'disarmSubmitWatch'; runId: string };

/**
 * showUnfilledField's answer: whether a field was highlighted, how many
 * required fields are still empty (`remaining`, every one the scan found),
 * whether the highlighted one is sensitive (any doubt counts as sensitive),
 * and whether more than UNFILLED_SCAN_CAP were found.
 */
export interface UnfilledFieldResult {
  ok: true;
  found: boolean;
  remaining: number;
  sensitive?: boolean;
  capped?: boolean;
}

/**
 * The bridge's run state for the panel. Ids and counts only: never the
 * token, the relay URL or a known value.
 */
export interface RunStatus {
  ok: true;
  active: boolean;
  runId?: string;
  origin?: string;
  /**
   * The tab the run drives, so a reopened panel can refresh the run's token
   * (a same-run-id startRun needs the tab).
   */
  tabId?: number;
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
  /**
   * The https origin (origin only: never a path or query) the tab went to
   * outside the run's origins, while the run is held waiting for
   * `allowOrigin`; absent when not held.
   */
  pendingOrigin?: string;
  /**
   * The token's own expiry (Unix seconds, from its claims; never the token),
   * so a reopened panel schedules its refresh from the real value.
   */
  exp?: number;
  /**
   * True with `active: false` while a same-run startRun is replacing the run
   * or a service-worker resume is in flight: the run is not gone, the answer
   * is just momentarily empty. `runId` is set for a same-run startRun.
   */
  restarting?: boolean;
  /**
   * Set only while the submit watch is armed, once it saw a form submission
   * request followed by a main-frame navigation in scope (see
   * watchForSubmission): when, in epoch ms. Never a URL, a query or page
   * text. A "Next" in a multi-step form looks the same.
   */
  submitted?: { at: number };
  /**
   * True while the submit watch is armed and the tab is held for it (the
   * agent's attach or the watch's own). False once a detach, a failed agent
   * attach or a token refresh lost it, so the panel knows to arm again.
   */
  armed?: boolean;
}

export type BridgeResponse =
  | { ok: true }
  | { ok: false; error: string }
  | RunStatus
  | UnfilledFieldResult;

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

/** The submit watch's state (see watchForSubmission). */
interface SubmitWatch {
  /**
   * A main-frame form submission the page requested, not yet committed:
   * when, whether it was a POST, the main frame's URL at the request (memory
   * only, never reported) and the scope epoch then.
   */
  pending?: { at: number; post: boolean; fromUrl?: string; epoch: number };
  /** The first submission seen while armed. */
  submitted?: { at: number };
}

interface Run extends RunConfig {
  runId: string;
  origin: string;
  /** Token expiry, epoch seconds. */
  exp: number;
  socketUrl: string;
  ws?: WebSocket;
  stopped: boolean;
  /**
   * True while the extension holds chrome.debugger on the tab for the
   * agent. Set only by the agent's relay `attach` command: forwardCdp and
   * event forwarding require it, so after a refresh or detach the agent gets
   * `not attached` and the Eve tool rotates its stale daemon.
   */
  attached: boolean;
  /**
   * True while typeSensitive holds chrome.debugger on the tab for itself
   * (the agent never attached, or not since a refresh). Never set with
   * `attached`: it lets only typeSensitive's own calls through, forwards no
   * event and tracks no child session, and is let go when the typing ends.
   */
  typingAttached: boolean;
  /**
   * True while the submit watch holds chrome.debugger on the tab for itself
   * (armed in Review with the agent not attached, say after a token
   * refresh). Like `typingAttached` it never sets `attached`, forwards no
   * event and tracks no child session. Typing and the unfilled-field scan
   * use it as an existing attach (no attach of their own, no let-go). The
   * agent's `attach` takes it over (attachTab clears it); a debugger detach,
   * disarming and the end of the run clear it. At most one of `attached`,
   * `typingAttached` and `reviewAttached` is set.
   */
  reviewAttached: boolean;
  /**
   * The submit watch, while armed (see armSubmitWatch); undefined when not.
   * Memory only. A token refresh is an endRun plus begin, so the watch is
   * lost with the old run and the panel arms it again.
   */
  submitWatch?: SubmitWatch;
  /** Bumped by every disarm, so an arm queued before it does not land. */
  watchGeneration: number;
  /**
   * The arm attach in flight and the watch generation it started in, so
   * concurrent arms (the panel re-arms on every poll) share one queued
   * attach step instead of each queueing their own.
   */
  armInFlight?: { generation: number; done: Promise<void> };
  /**
   * This run's attach chain (see queuedAttach): created in begin, so a run
   * never inherits a stuck chain from an earlier one.
   */
  attachChain: Promise<unknown>;
  /** Set while a typeSensitive call is between its checks and its reply. */
  typingNow: boolean;
  /** Set while a showUnfilledField call is between its checks and reply. */
  scanNow: boolean;
  /**
   * Hides showUnfilledField's Overlay highlight when it runs out; cleared
   * (and the highlight hidden or gone) on a main-frame navigation, a detach,
   * an attach, stop and the next call.
   */
  highlightTimer?: ReturnType<typeof setTimeout>;
  /**
   * Set synchronously the moment the tab leaves scope, before any await, so
   * no command or event gets through while the run is held (see
   * holdOrigin) or being torn down. Not terminal: a hold that ends with the
   * tab back in scope, or an allowed origin, clears it, so a check after an
   * await must compare `scopeEpoch` too (see requireSameScope).
   */
  blocked: boolean;
  /** Bumped every time a hold starts or moves to another origin. */
  scopeEpoch: number;
  /**
   * Run origins allowed by `allowOrigin` for this run only, beyond the
   * token's `origin` (at most MAX_EXTRA_ORIGINS). Memory only, never in
   * chrome.storage: a run rebuilt after a worker restart has none, so if its
   * tab is on one it ends with `left-origin` (see connect). Not a frame
   * origin list: allowedFrameOrigins is never widened.
   */
  extraOrigins: string[];
  /** The https origin the tab is held on, waiting for `allowOrigin`. */
  pendingOrigin: string | null;
  /** Ends a hold that no `allowOrigin` or return to scope answered. */
  holdTimer?: ReturnType<typeof setTimeout>;
  /** Epoch ms the current hold began, across origins, for the total cap. */
  holdStartedAt?: number;
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
   * The values addKnownValue accepted, in worker memory only, because the
   * redactor has no getter: a same-run startRun (token refresh) registers
   * them on the new redactor. Cleared by every other end; never logged,
   * stored or sent.
   */
  knownValues: string[];
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
/**
 * Run ids this worker has begun (ids only: never a token or a value), most
 * recent last, in chrome.storage.session so a worker restart keeps them.
 * A startRun for one of them, with no live run of that id to ask, needs
 * re-registration: whatever ended the earlier run (token expiry, a detach,
 * a stop) also dropped its known values.
 */
const BEGUN_KEY = 'begunRuns';
const BEGUN_CAP = 20;
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
const ORIGIN_NOT_PENDING = 'origin not pending';
// esbuild defines this only in a test-hooks build given
// EXTENSION_TEST_ORIGIN_HOLD_MS (scripts/extension-build-config.ts), so the
// e2e suites see a hold end in seconds. Undeclared everywhere else: 60 s.
declare const __ORIGIN_HOLD_MS__: number | undefined;
/**
 * How long a run is held after the tab leaves for another https origin,
 * waiting for `allowOrigin` (or the tab to come back), before it ends with
 * `left-origin`. Each move to yet another origin restarts it, up to
 * ORIGIN_HOLD_CAP_MS in total.
 */
const ORIGIN_HOLD_MS =
  typeof __ORIGIN_HOLD_MS__ === 'number' ? __ORIGIN_HOLD_MS__ : 60_000;
const ORIGIN_HOLD_CAP_MS = 2 * ORIGIN_HOLD_MS;
const MAX_EXTRA_ORIGINS = 3;
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

/**
 * A target as the agent sees it; URL and title blank while unregistered.
 * `attached` is the agent's own attach, not Chrome's (which also counts a
 * typing attach, see typeSensitive, and any other debugger client).
 */
const agentTargetInfo = (current: Run, t: chrome.debugger.TargetInfo) => {
  const info = { ...targetInfo(t), attached: current.attached };
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

/** Bounds the carried list; the redactor's own cap is far lower. */
const MAX_CARRIED_VALUES = 256;

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
  // The tab may also sit on an origin the live run of this same id was
  // allowed (see carriedOrigins); rechecked below against the snapshot.
  if (!inScope(url, claims.origin) && !onCarriedOrigin(url, claims)) {
    return { ok: false, error: OUT_OF_SCOPE };
  }

  // A run restarted under the same run id (e.g. a refreshed token) is not
  // ended on the relay: its new socket's `register` replaces the old one
  // there, while a `stop` would close the run for good (and refuse the new
  // socket). Its known values are carried to the new run (see below); only
  // a restart, which loses them, makes the panel register them again before
  // screenshots are allowed.
  // A live run of the same id says exactly whether values were lost; with
  // none (it ended on its own, or the worker restarted) the record of begun
  // run ids decides, failing closed.
  const begun = await readBegunRuns();
  await chrome.storage.session.set({
    [BEGUN_KEY]: [
      ...begun.filter((id) => id !== claims.runId),
      claims.runId,
    ].slice(-BEGUN_CAP),
  });
  // Snapshot after the last await, with endRun called in the same tick: its
  // synchronous start clears `run`, so no value can register on the old run
  // (addKnownValue is not serialized) after the list is copied.
  const previous = run;
  const sameRun = previous !== undefined && previous.runId === claims.runId;
  const carried = sameRun ? previous.knownValues.slice() : [];
  const valuesLost = sameRun
    ? previous.needsRegistration
    : begun.includes(claims.runId);
  // Allowed origins carry only for the same run id and token origin, copied
  // here (endRun clears them). The tab check is repeated against this copy:
  // a run that ended or changed during the awaits above carries nothing, so
  // a tab on an origin that is no longer allowed is refused, with the old
  // run left as it was.
  const carriedOrigins =
    sameRun && previous.origin === claims.origin
      ? previous.extraOrigins.slice(0, MAX_EXTRA_ORIGINS)
      : [];
  if (
    !inScope(url, claims.origin) &&
    !carriedOrigins.some((origin) => inScope(url, origin))
  ) {
    return { ok: false, error: OUT_OF_SCOPE };
  }
  // From here until begin finishes `run` is briefly undefined; a same-run
  // refresh is not a gone run, so getRunStatus says `restarting` (cleared in
  // finally, so a throw cannot leave it set).
  if (sameRun) restartingRunIds.push(claims.runId);
  try {
    await endRun('replaced', { notifyRelay: !sameRun });
    const config: RunConfig = {
      relayUrl: msg.relayUrl,
      token: msg.token,
      tabId: msg.tabId,
      allowedFrameOrigins,
    };
    await chrome.storage.session.set({ [STORAGE_KEY]: config });
    begin(config, socket.url, claims, false, valuesLost);
    carryKnownValues(carried);
    carryExtraOrigins(carriedOrigins);
    return { ok: true };
  } finally {
    if (sameRun) {
      restartingRunIds.splice(restartingRunIds.indexOf(claims.runId), 1);
    }
  }
}

/**
 * Run ids of same-run startRuns between their snapshot and `begin`, and the
 * count of resumeRun calls queued or in flight. Module state, not on `Run`,
 * because `run` is undefined in exactly the window they describe.
 */
const restartingRunIds: string[] = [];
let resumesInFlight = 0;

/**
 * True when `url` is on an origin the live run was allowed, and that run has
 * the incoming token's run id and origin.
 */
function onCarriedOrigin(
  url: string,
  claims: { runId: string; origin: string },
): boolean {
  const current = run;
  return (
    current !== undefined &&
    !current.stopped &&
    current.runId === claims.runId &&
    current.origin === claims.origin &&
    current.extraOrigins.some((origin) => inScope(url, origin))
  );
}

/**
 * Sets the origins a same-run refresh carried on the run just begun (memory
 * only, deduped, at most MAX_EXTRA_ORIGINS). Synchronous with begin. The
 * hold state is not carried: the new run starts unheld.
 */
function carryExtraOrigins(origins: string[]) {
  const current = run;
  if (!current || origins.length === 0) return;
  current.extraOrigins = [...new Set(origins)].slice(0, MAX_EXTRA_ORIGINS);
}

/**
 * Registers the values a same-run refresh carried on the run just begun.
 * Synchronous with begin, so no event or reply is redacted without them. A
 * value the new redactor refuses (it accepted it before) counts as lost.
 */
function carryKnownValues(values: string[]) {
  const current = run;
  if (!current || values.length === 0) return;
  for (const value of values) {
    if (!addAccepted(current, value)) current.needsRegistration = true;
  }
}

/** Registers a value on the redactor and remembers it if it was accepted. */
function addAccepted(current: Run, value: string): boolean {
  const known = current.knownValues.includes(value);
  // A full list would drop a value at the next refresh unflagged: refuse.
  if (!known && current.knownValues.length >= MAX_CARRIED_VALUES) return false;
  if (!current.redactor.addKnownValue(value)) return false;
  if (!known) current.knownValues.push(value);
  current.hasKnownValues = true;
  return true;
}

/** The begun run ids; empty when unreadable or malformed. */
async function readBegunRuns(): Promise<string[]> {
  try {
    const stored = await chrome.storage.session.get(BEGUN_KEY);
    const ids: unknown = stored[BEGUN_KEY];
    return Array.isArray(ids)
      ? ids.filter((id): id is string => typeof id === 'string')
      : [];
  } catch {
    return [];
  }
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
 * Ends the run only if it is `runId`, checked in the queue (so a startRun
 * queued before it counts). Any other case changes nothing and succeeds.
 */
function stopRunFor(runId: unknown): Promise<BridgeResponse> {
  return serialized(async () => {
    const current = run;
    if (!current || current.stopped || current.runId !== runId) {
      return { ok: true };
    }
    return endRun('user');
  });
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
    current.knownValues = [];
    current.jev?.stop();
    clearTimeout(current.reconnectTimer);
    clearTimeout(current.expiryTimer);
    clearTimeout(current.detachTimer);
    clearTimeout(current.holdTimer);
    // The detach below takes the highlight with it.
    clearHighlight(current, false);
    // Allowed origins are for this run only.
    current.extraOrigins = [];
    current.pendingOrigin = null;
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
    current.typingAttached = false;
    current.reviewAttached = false;
    current.submitWatch = undefined;
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

/** True when the URL is in one of the run's origins (token or allowed). */
const inRunScope = (current: Run, url: string) => inRunOrigins(url, current);

/**
 * The tab is heading to, or committed, `url` outside the run's origins.
 * Blocks synchronously either way. An https URL holds the run on its origin
 * (see holdOrigin); anything else (http, about:, chrome:, data:, blob:,
 * javascript:, unparsable) ends it at once with `left-origin`.
 */
function leaveOrigin(current: Run, url: string) {
  if (current.stopped || run !== current) return;
  const origin = normalizeOrigin(url);
  if (origin === null) {
    endLeftOrigin(current);
    return;
  }
  holdOrigin(current, origin);
}

/** Ends the run for leaving its origins; a later `attach` finds no run. */
function endLeftOrigin(current: Run) {
  current.blocked = true;
  if (run === current && !current.stopped) void endRun('left-origin');
}

/**
 * Holds the run on `origin` instead of ending it (a login redirect, say):
 * every agent command is refused and nothing is forwarded, while the bridge
 * keeps tracking frames, dialogs and children. The hold ends when the tab
 * commits a URL in the run's origins again, when `allowOrigin` names this
 * origin, or with `left-origin` after ORIGIN_HOLD_MS (restarted by a move to
 * another origin, ORIGIN_HOLD_CAP_MS in total). Another page of the same
 * origin changes nothing.
 */
function holdOrigin(current: Run, origin: string) {
  if (current.pendingOrigin === origin) return;
  const now = Date.now();
  current.blocked = true;
  current.scopeEpoch += 1;
  current.pendingOrigin = origin;
  current.holdStartedAt ??= now;
  clearTimeout(current.holdTimer);
  current.holdTimer = undefined;
  const wait = Math.min(
    ORIGIN_HOLD_MS,
    current.holdStartedAt + ORIGIN_HOLD_CAP_MS - now,
  );
  if (wait <= 0) {
    endLeftOrigin(current);
    return;
  }
  current.holdTimer = setTimeout(() => {
    current.holdTimer = undefined;
    if (current.pendingOrigin !== null) endLeftOrigin(current);
  }, wait);
}

/** Ends a hold: the tab is back in scope, or its origin was allowed. */
function releaseHold(current: Run) {
  if (current.pendingOrigin === null || current.stopped) return;
  clearTimeout(current.holdTimer);
  current.holdTimer = undefined;
  current.holdStartedAt = undefined;
  current.pendingOrigin = null;
  current.blocked = false;
}

/**
 * Throws unless the run is live and its tab has not left scope since
 * `epoch`: a hold may have started and ended while the caller awaited, and
 * whatever it read meanwhile may be the other origin's.
 */
function requireSameScope(current: Run, epoch: number) {
  if (current.blocked || current.stopped || current.scopeEpoch !== epoch) {
    throw new Error(OUT_OF_SCOPE);
  }
}

/** Picks up a run persisted before a service worker restart. */
export function resumeRun(): Promise<void> {
  resumesInFlight += 1;
  return serialized(resumeRunNow).finally(() => {
    resumesInFlight -= 1;
  });
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
    typingAttached: false,
    reviewAttached: false,
    watchGeneration: 0,
    attachChain: Promise.resolve(),
    typingNow: false,
    scanNow: false,
    blocked: false,
    scopeEpoch: 0,
    extraOrigins: [],
    pendingOrigin: null,
    children: new Map(),
    frameScope: createFrameScope(),
    reconnectDelayMs: FIRST_RECONNECT_MS,
    announceResume: resumed,
    redactor: jev
      ? createRedactor({ classifyExtra: jev.classify })
      : createRedactor(),
    jev,
    hasKnownValues: false,
    knownValues: [],
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
    // The tab left the run's origins while the socket was down.
    if (!inRunScope(current, tab)) {
      // A run rebuilt after a worker restart lost its allowed origins with
      // the old worker (they are never persisted): it is not held, it ends.
      if (current.announceResume) {
        endLeftOrigin(current);
        return;
      }
      leaveOrigin(current, tab);
      // Held: still register, so the run can go on once the hold ends.
      if (run !== current || current.stopped) return;
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

/** Fails (and holds or ends the run) unless the tab is still in scope. */
async function requireTabInScope(current: Run) {
  if (current.blocked) throw new Error(OUT_OF_SCOPE);
  const url = await tabUrl(current.tabId);
  if (url === null) throw new Error('tab closed');
  if (current.blocked) throw new Error(OUT_OF_SCOPE);
  if (!inRunScope(current, url)) {
    leaveOrigin(current, url);
    throw new Error(OUT_OF_SCOPE);
  }
}

// Out-of-process frames have their own child session, judged by its URL
// here. Frames that share the tab's session (same-origin, srcdoc and
// same-site cross-origin iframes) are judged per command and per event
// through `frameScope` (a frame child has its own too); see resolveFrames.
const frameAllowed = (current: Run, url: string) =>
  inRunScope(current, url) ||
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
    current.extraOrigins,
  ) || childFrameAllowed(current, frameId);

/**
 * True when a context is known, of an allowed origin, in an allowed frame.
 * `current` is the ScopeOrigins here (and for scopeFrameEvent): its origin,
 * extraOrigins and allowedFrameOrigins.
 */
const contextAllowedIn = (current: Run, scope: FrameScope, contextId: number) =>
  contextInScope(scope, current, contextId, (frameId) =>
    childFrameAllowed(current, frameId),
  );

/**
 * Every attach of a run (the agent's relay `attach` and typeSensitive's)
 * and the typing let-go run one at a time behind the run's chain, so two of
 * them never interleave their detach, attach and frame-tree awaits on the
 * tab. Every await inside a step is bounded (ATTACH_STEP_MS for typing,
 * AGENT_ATTACH_STEP_MS for the agent's attach), so a step
 * always settles and a failed one does not stop the chain. Per run: a new
 * run starts with a fresh chain (its first attach detaches first, so a
 * replaced run's late attach is harmless).
 */
function queuedAttach<T>(current: Run, step: () => Promise<T>): Promise<T> {
  const result = current.attachChain.then(step);
  current.attachChain = result.catch(() => {});
  return result;
}

/**
 * Bounds each chrome.debugger (and tab lookup) call of an attach step. A
 * call that settles after its timeout changes no flag: an attach that lands
 * late leaves the tab held by nobody (no event is forwarded, no command
 * passes), and the next attach detaches first.
 */
// Typing (attach and let-go): the panel is waiting on the reply.
const ATTACH_STEP_MS = 3_000;
// The agent's attach: a heavy page can take seconds to answer Page.enable or
// Page.getFrameTree. An attach step has at most 5 bounded awaits (tab,
// detach, attach, Page.enable, frame tree, plus a detach on failure that
// replaces the rest), so about 50 s, plus at most one typing step (about
// 15 s) ahead in the queue. That exceeds the hub's 30 s command timeout
// (relay/hub.ts; relay/server.ts sets no override): the hub answers
// `extension did not answer`, the Eve tool treats it as a lost connection
// and rotates, and the stale queued step may still finish, which is
// harmless because the next attach detaches first.
const AGENT_ATTACH_STEP_MS = 10_000;
const bounded = <T>(promise: Promise<T>, what: string, ms = ATTACH_STEP_MS) =>
  withTimeout(promise, ms, what);

/** Best-effort, bounded: let go of the tab. */
const detachTab = (tabId: number, ms = ATTACH_STEP_MS) =>
  bounded(chrome.debugger.detach({ tabId }), 'detach', ms).catch(() => {});

/**
 * The attach core (run only through queuedAttach): checks the tab is in
 * scope and unchanged since `epoch`, lets go of any earlier attach (agent or
 * typing) and clears what it left, attaches, enables the bridge-owned Page
 * domain and seeds the frame scope from the frame tree, then checks the
 * scope again. Fails closed, detaching again on every failure after the
 * attach. Sets none of `attached`, `typingAttached` and `reviewAttached`
 * (it clears all three): the caller sets its own.
 */
async function attachTab(
  current: Run,
  epoch: number,
  targetId: string,
  stepMs: number,
) {
  // Stopped, replaced or moved out of scope since the call (throws).
  const live = () => {
    if (run !== current) throw new Error('run replaced');
    requireSameScope(current, epoch);
  };
  await bounded(requireTabInScope(current), 'tab', stepMs);
  live();
  current.attached = false;
  current.typingAttached = false;
  // The agent's attach takes over a review attach: the watch, if armed,
  // goes on under the new attach. A submission requested under the old
  // session is forgotten (its frame ids and URL are re-read below).
  current.reviewAttached = false;
  if (current.submitWatch) current.submitWatch.pending = undefined;
  // The detach below takes the highlight with it.
  clearHighlight(current, false);
  dropChildren(current);
  current.frameScope = createFrameScope();
  current.focusWorld = undefined;
  current.dialog = undefined;
  await detachTab(current.tabId, stepMs);
  live();
  try {
    await bounded(
      chrome.debugger.attach({ tabId: current.tabId }, '1.3'),
      'attach',
      stepMs,
    );
  } catch (error) {
    // A timed-out attach may still land: let go (best effort).
    await detachTab(current.tabId, stepMs);
    throw error;
  }
  try {
    live();
    // The bridge owns the Page domain: leave-origin detection reads Page
    // navigation events, so it never depends on the agent enabling Page
    // (and the agent cannot disable it: Page.disable is not allowed). This
    // does not go through the agent allowlist and the agent gets no reply.
    // Fail closed: without it the run is not attached.
    try {
      await bounded(
        chrome.debugger.sendCommand({ tabId: current.tabId }, 'Page.enable'),
        'page events',
        stepMs,
      );
    } catch {
      console.warn('relay: Page.enable failed; refusing attach');
      throw new Error('could not enable page events');
    }
    // Frame scope starts from the current tree; Page events keep it
    // current from here (events that raced the reply were applied first,
    // and the tree supersedes them). Fail closed like Page.enable.
    try {
      const tree = await bounded(
        chrome.debugger.sendCommand(
          { tabId: current.tabId },
          'Page.getFrameTree',
        ),
        'frame tree',
        stepMs,
      );
      const seeded = framesOfTree(tree);
      if (seeded.length === 0) throw new Error('empty frame tree');
      current.frameScope.frames = new Map(seeded);
    } catch {
      console.warn('relay: Page.getFrameTree failed; refusing attach');
      throw new Error('could not read the frame tree');
    }
    // A hold that came and went meanwhile may have seeded the other
    // origin's frame tree: refuse, the caller attaches again.
    live();
  } catch (error) {
    // Ended, replaced or out of scope while attaching, or a step failed:
    // let go again. (A replaced run's tab is detached first by the new
    // run's own attach anyway.)
    if (run === current || run === undefined) {
      await detachTab(current.tabId, stepMs);
    }
    throw error;
  }
  current.mainFrameId = targetId;
}

async function handleCommand(
  current: Run,
  msg: RelayToExtension,
): Promise<unknown> {
  if (current.stopped) throw new Error('run stopped');
  // Taken before any await: a reply computed across a hold is refused.
  const epoch = current.scopeEpoch;
  switch (msg.type) {
    case 'listTargets': {
      await requireTabInScope(current);
      const target = await tabTarget(current.tabId);
      const result = await redact(current, 'Target.getTargets', {
        targetInfos: target ? [agentTargetInfo(current, target)] : [],
      });
      requireSameScope(current, epoch);
      return result;
    }
    case 'attach': {
      const target = await tabTarget(current.tabId);
      if (!target || target.id !== msg.targetId) {
        throw new Error('unknown target');
      }
      // Start clean so a new relay client sees every child session announced
      // afresh (the spike's re-attach fix); a typing attach is taken over.
      // The flag is set inside the queued step, so no queued typing attach
      // can run between the attach and the flag.
      await queuedAttach(current, async () => {
        await attachTab(current, epoch, target.id, AGENT_ATTACH_STEP_MS);
        current.attached = true;
        // Re-attached in time: the detach no longer ends the run (P4b-d).
        clearTimeout(current.detachTimer);
        current.detachTimer = undefined;
      });
      return redact(current, 'Target.attachToTarget', {
        sessionId: tabSessionId(current.tabId),
        targetInfo: agentTargetInfo(current, { ...target, attached: true }),
      });
    }
    case 'createTarget': {
      // Agent-initiated URLs stay bound to the token origin, as at the hub
      // (relay/hub.ts checkCommand on agent.origin): see forwardCdp.
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
      return forwardCdp(current, msg, epoch);
    default:
      throw new Error('unknown command');
  }
}

async function forwardCdp(
  current: Run,
  msg: Extract<RelayToExtension, { type: 'cdp' }>,
  epoch: number,
): Promise<unknown> {
  const { sessionId, method, params } = msg;
  if (typeof method !== 'string' || !isMethodAllowed(method)) {
    throw new Error(`method not allowed: ${String(method)}`);
  }
  // Page.navigate's URL stays bound to the token origin, not the extra
  // origins: the hub refuses any other (relay/hub.ts, checkCommand on the
  // agent token's origin) and the protocol does not tell it about allowed
  // origins, so an allowed origin is reached only by the page's own
  // navigation (the redirect that caused the hold). See the Task 10 notes.
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
  // Held (or ending): refused before the frame lookups below, which would
  // otherwise talk to the held page (DOM.resolveNode, context discovery).
  if (current.blocked) throw new Error(OUT_OF_SCOPE);
  // Before any lookup below talks to the page: a typing attach (see
  // typeSensitive) holds the debugger without the agent having attached,
  // and none of it is the agent's. Checked again after every await.
  if (!current.attached) throw new Error('not attached');
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
    // In scope now is not enough: a hold that came and went (A -> B -> A)
    // while this command awaited means the page is a new document.
    requireSameScope(current, epoch);
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
    requireSameScope(current, epoch);
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
      // Checked unconditionally, with no await before the send: the input
      // guard above may have awaited, and a scope bounce refuses the command.
      requireSameScope(current, epoch);
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
  // sends nothing more, nor one whose tab left scope at any point since the
  // command arrived (a hold may have come and gone).
  requireSameScope(current, epoch);
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
 * The focused element (see DEEP_ACTIVE) as a remote object of the bridge's
 * own isolated world of the main frame, in `objectGroup` (the caller
 * releases it). Null when nothing is focused; undefined when the world
 * cannot be made. Throws when a lookup fails.
 */
async function focusedElement(
  current: Run,
  tab: FrameSession,
  objectGroup: string,
): Promise<string | null | undefined> {
  const expression = `(${DEEP_ACTIVE})(document)`;
  const evaluate = async (contextId: number) =>
    sessionSend(tab, 'Runtime.evaluate', {
      expression,
      contextId,
      objectGroup,
    });
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
  return objectIdOf(evaluated, 'result') ?? null;
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
  const objectGroup = checkGroup();
  try {
    const objectId = await focusedElement(current, tab, objectGroup);
    if (objectId === undefined) return undefined;
    if (objectId === null) return null;
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

/** A form submission counts only if it commits within this (P4c4-d). */
const SUBMIT_WINDOW_MS = 30_000;
const FORM_SUBMISSION_REASONS: ReadonlySet<unknown> = new Set([
  'formSubmissionGet',
  'formSubmissionPost',
]);

/**
 * The submit watch (P4c4-d), from raw events of the tab's own session while
 * armed. A main-frame `Page.frameRequestedNavigation` whose reason is a form
 * submission, for the current tab, not `javascript:`, with the run not held,
 * records a pending submission (any other main-frame request clears it).
 * The next main-frame `Page.frameNavigated` consumes it, and sets
 * `submitted` only when: within SUBMIT_WINDOW_MS, an ordinary navigation
 * (not a back/forward-cache restore, not an error page), the run never held
 * since the request (same scope epoch, not blocked), the URL in the run's
 * origins, and the URL differs from the one at the request, or the request
 * was a POST (a POST may answer on the same URL). A main-frame
 * `Page.frameStoppedLoading` before any commit (a 204, a cancelled load)
 * clears it. Same-document navigations, iframes and child sessions never
 * count. Out of scope, the existing hold and leave-origin rules apply
 * (elsewhere in onDebuggerEvent) and nothing is set. Reads event params
 * only; keeps no URL beyond the pending request; never throws.
 */
function watchForSubmission(
  current: Run,
  source: chrome.debugger.DebuggerSession,
  method: string,
  params: unknown,
) {
  const watch = current.submitWatch;
  if (!watch || source.sessionId !== undefined || !isObject(params)) return;
  const mainFrameId = current.mainFrameId;
  if (mainFrameId === undefined) return;
  if (method === 'Page.frameRequestedNavigation') {
    if (params.frameId !== mainFrameId) return;
    const url = typeof params.url === 'string' ? params.url : '';
    watch.pending =
      FORM_SUBMISSION_REASONS.has(params.reason) &&
      params.disposition === 'currentTab' &&
      !isJavascriptUrl(url) &&
      !current.blocked
        ? {
            at: Date.now(),
            post: params.reason === 'formSubmissionPost',
            fromUrl: current.frameScope.frames.get(mainFrameId)?.url,
            epoch: current.scopeEpoch,
          }
        : undefined;
    return;
  }
  if (method === 'Page.frameStoppedLoading') {
    if (params.frameId === mainFrameId) watch.pending = undefined;
    return;
  }
  if (method !== 'Page.frameNavigated') return;
  const frame = isObject(params.frame) ? params.frame : undefined;
  if (!frame || frame.parentId !== undefined || frame.id !== mainFrameId) {
    return;
  }
  const pending = watch.pending;
  watch.pending = undefined;
  if (!pending || watch.submitted) return;
  const url = typeof frame.url === 'string' ? frame.url : '';
  if (
    Date.now() - pending.at > SUBMIT_WINDOW_MS ||
    (params.type !== undefined && params.type !== 'Navigation') ||
    frame.unreachableUrl !== undefined ||
    current.blocked ||
    current.pendingOrigin !== null ||
    current.scopeEpoch !== pending.epoch ||
    !inRunScope(current, url) ||
    (!pending.post &&
      (pending.fromUrl === undefined || url === pending.fromUrl))
  ) {
    return;
  }
  watch.submitted = { at: Date.now() };
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
  // A run being torn down tracks nothing more. A held run (blocked, see
  // holdOrigin) keeps tracking frames, dialogs and children below, so its
  // state is current if the hold ends, and forwards nothing.
  if (current.stopped) return;
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
  // A highlight is for the page it was drawn on: any main-frame navigation
  // ends it, attached or not.
  if (
    current.highlightTimer !== undefined &&
    source.sessionId === undefined &&
    (method === 'Page.frameNavigated' ||
      method === 'Page.navigatedWithinDocument') &&
    mainFrameNavigationUrl(current, method, params) !== undefined
  ) {
    clearHighlight(current, true);
  }
  // The submit watch reads raw tab events, attached or not (it reads only
  // event params, never the page, and sends nothing on).
  watchForSubmission(current, source, method, params);
  if (!dialogForwarded) return;
  // Nothing leaves the browser unless the agent attached and the tab is in
  // scope. A typing or review attach (see typeSensitive, armSubmitWatch)
  // forwards nothing and tracks no child session; it only holds (or ends)
  // the run the moment the main frame heads out of scope, as below. Its hold
  // is released by onTabUpdated.
  if (!current.attached) {
    if (
      (current.typingAttached || current.reviewAttached) &&
      source.sessionId === undefined
    ) {
      const navUrl = mainFrameNavigationUrl(current, method, params);
      if (
        navUrl !== undefined &&
        !inRunScope(current, navUrl) &&
        (current.pendingOrigin === null || !PRE_COMMIT_EVENTS.has(method))
      ) {
        leaveOrigin(current, navUrl);
      }
    }
    return;
  }

  if (source.sessionId === undefined) {
    const navUrl = mainFrameNavigationUrl(current, method, params);
    if (navUrl !== undefined && !inRunScope(current, navUrl)) {
      // Drop the event that would reveal the new document, then hold or
      // stop. While held, only a commit moves the hold (a pre-commit signal
      // may never commit).
      if (current.pendingOrigin === null || !PRE_COMMIT_EVENTS.has(method)) {
        leaveOrigin(current, navUrl);
      }
      return;
    }
    // Back in scope: the hold ends on the commit itself, in order with the
    // held document's events (all dropped before it).
    if (navUrl !== undefined && method === 'Page.frameNavigated') {
      releaseHold(current);
    }
  }

  trackChildren(current, method, params, source.sessionId);
  resumeIfPaused(current, method, params);
  // The bridge's own isolated worlds are tracked whether or not it is held.
  const bridgeWorldEvent = isMaskWorldEvent(source.sessionId, method, params);
  // Held (or torn down): tracked above, never forwarded.
  if (current.blocked) return;

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

  if (bridgeWorldEvent) return;
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
  // A typing attach lost this way is not re-made by itself: the typing in
  // flight fails, and the grace below applies as to any detach (the agent
  // must attach within it, or the run ends).
  current.typingAttached = false;
  // The watch's attach is gone and so is the watch: nothing is watched
  // until the panel arms it again (which attaches again, unless the detach
  // grace below is running).
  current.reviewAttached = false;
  current.submitWatch = undefined;
  current.watchGeneration += 1;
  current.focusWorld = undefined;
  // The highlight died with the session.
  clearHighlight(current, false);
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

/** Holds (or ends) the run the moment the tab commits a URL outside scope. */
export function onTabUpdated(tabId: number, info: { url?: string }) {
  const current = run;
  if (!current || tabId !== current.tabId || info.url === undefined) return;
  if (!inRunScope(current, info.url)) {
    leaveOrigin(current, info.url);
    return;
  }
  // Back in scope. An attached tab is released by its Page.frameNavigated
  // instead: tabs.onUpdated is not ordered with the debugger's events, so
  // the held document's late events could otherwise be forwarded.
  if (!current.attached) releaseHold(current);
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
function addKnownValue(runId: unknown, value: unknown): BridgeResponse {
  const current = run;
  if (!current || current.stopped) return { ok: false, error: 'no active run' };
  if (runId !== current.runId) return { ok: false, error: 'wrong run' };
  if (typeof value !== 'string' || !addAccepted(current, value)) {
    return { ok: false, error: 'known value not registered' };
  }
  return { ok: true };
}

/**
 * typeSensitive's refusals, exactly as the panel maps them: fixed strings,
 * never the value, a field's name or label, or page text.
 */
type TypeRefusal =
  | 'no active run'
  | 'wrong run'
  | 'not attached'
  | 'run is held'
  | 'value not accepted'
  | 'no focused field'
  | 'field not sensitive'
  | 'field out of scope'
  | 'type failed';

/** The redactor's own limit; checked here too so the reason is fixed. */
const MAX_TYPED_LENGTH = 512;
// The typing call (TYPE_INTO) settles once the text is in; a page that hangs
// it fails the typing (it is never sent again).
const TYPE_MS = 3_000;

const returnedValue = (reply: unknown) =>
  isObject(reply) && isObject(reply.result) ? reply.result.value : undefined;

/**
 * Types a sensitive value the caseworker gave the panel into the field they
 * focused in the tab, for them: plan 4c-3. The value is registered for
 * scrubbing first (refused: nothing is typed), then the focused element is
 * checked (an enabled, editable text field, in a frame in scope, that the
 * redactor masks once filled), then it is typed into that same element
 * (TYPE_INTO, one call, never retried), not into whatever has focus. A
 * refusal after registering leaves the value registered.
 *
 * When the agent is not attached (a fresh run, or after a token refresh),
 * it attaches for itself after registering (attachForTyping: the same
 * scope checks as the relay `attach`) and lets go before answering; it
 * never sets the agent-facing `attached`, so agent commands still get
 * `not attached`. An attach failure answers `not attached`.
 *
 * This is worker-initiated CDP outside the agent's command policy, so it
 * re-applies what forwardCdp applies to agent input: run live and attached,
 * not held, tab in scope and unchanged since the call (requireSameScope),
 * screenshots waited out and the shared tab hold kept to the send, the
 * focused frame judged like a command's handle, a timeout. Never throws,
 * never logs; the reply is `{ ok: true }` or one fixed TypeRefusal.
 */
async function typeSensitive(
  runId: unknown,
  value: unknown,
): Promise<BridgeResponse> {
  const refuse = (error: TypeRefusal): BridgeResponse => ({ ok: false, error });
  const current = run;
  if (!current || current.stopped) return refuse('no active run');
  if (runId !== current.runId) return refuse('wrong run');
  if (current.blocked) return refuse('run is held');
  // One at a time: a second call while one is typing is refused (nothing
  // registered, nothing typed); the panel sends one per press. Nor while a
  // showUnfilledField scan holds the tab (it may let go of its attach).
  if (current.typingNow || current.scanNow) return refuse('type failed');
  // Registered before any await, in the same turn as the checks above, so a
  // token refresh either carries it or refuses it (see startRunNow).
  if (
    typeof value !== 'string' ||
    value.trim() === '' ||
    value.length > MAX_TYPED_LENGTH ||
    !addAccepted(current, value)
  ) {
    return refuse('value not accepted');
  }
  const epoch = current.scopeEpoch;
  let refusal: TypeRefusal | null;
  let attachedHere = false;
  current.typingNow = true;
  try {
    // Not attached (a fresh run, or a refresh): attach for this typing only.
    if (!holdsTab(current)) {
      attachedHere = await attachForTyping(current, epoch);
    }
    refusal = await typeIntoFocused(current, value, epoch);
  } catch (error) {
    // The error's own text is never passed on: it may name page state.
    if (run !== current || current.stopped) refusal = 'type failed';
    else if (current.blocked) refusal = 'run is held';
    else if (error instanceof Error && error.message === FRAME_OUT_OF_SCOPE) {
      refusal = 'field out of scope';
    } else if (!holdsTab(current)) {
      refusal = 'not attached';
    } else refusal = 'type failed';
  } finally {
    // Let go of the typing attach before answering, so the debugger is held
    // only while typing (unless the agent attached meanwhile: then it is
    // the agent's, and this changes nothing).
    if (attachedHere) await releaseTypingAttach(current);
    current.typingNow = false;
  }
  return refusal === null ? { ok: true } : refuse(refusal);
}

/**
 * typeSensitive's own attach, when the agent has not attached: the attach
 * core (attachTab) under the same scope checks as the relay `attach`, then
 * `typingAttached`. Never sets the agent-facing `attached` (see Run). True
 * when this call attached; false when the agent attached first (its attach
 * is used as is). Throws on any failure; the caller answers `not attached`
 * (or `run is held` when the tab left scope). Refused while a detach grace
 * is running: the run is waiting for the agent, not for typing.
 */
async function attachForTyping(current: Run, epoch: number): Promise<boolean> {
  if (current.detachTimer !== undefined) throw new Error('not attached');
  const target = await bounded(tabTarget(current.tabId), 'targets');
  if (!target) throw new Error('not attached');
  return queuedAttach(current, async () => {
    if (run !== current || current.stopped) throw new Error('run replaced');
    // The agent's attach, or the submit watch's, is used as is (and never
    // let go of here).
    if (current.attached || current.reviewAttached) return false;
    await attachTab(current, epoch, target.id, ATTACH_STEP_MS);
    current.typingAttached = true;
    return true;
  });
}

/**
 * True while the bridge holds chrome.debugger on the tab for anyone: the
 * agent, a typing or scan, or the submit watch. Worker-initiated page work
 * (typeSensitive, showUnfilledField) may run under any of them.
 */
const holdsTab = (current: Run) =>
  current.attached || current.typingAttached || current.reviewAttached;

/**
 * Lets go of the typing attach (queued with every attach), unless the run
 * was replaced or ended (endRun detaches) or the agent attached meanwhile
 * (which took it over and cleared `typingAttached`), or the submit watch
 * took it over (armSubmitWatch, likewise). Chrome sends no
 * onDetach for the extension's own detach, so no detach grace starts.
 */
function releaseTypingAttach(current: Run): Promise<void> {
  return queuedAttach(current, async () => {
    if (run !== current || current.stopped) return;
    if (current.attached || !current.typingAttached) return;
    current.typingAttached = false;
    current.focusWorld = undefined;
    current.dialog = undefined;
    current.frameScope = createFrameScope();
    await detachTab(current.tabId);
  });
}

/** See typeSensitive: null once the value was typed, else why not. */
async function typeIntoFocused(
  current: Run,
  value: string,
  epoch: number,
): Promise<TypeRefusal | null> {
  const lock = lockFor(current.tabId);
  // The tab session as it is now; a re-attach replaces its frame scope.
  const tab: FrameSession = {
    scope: current.frameScope,
    debuggee: { tabId: current.tabId },
    rootFrameId: current.mainFrameId,
  };
  // After every await: the same run, attached to the same session, its tab
  // never out of scope since the call (throws).
  const live = () => {
    requireSameScope(current, epoch);
    if (run !== current) throw new Error('run replaced');
    if (!holdsTab(current) || tab.scope !== current.frameScope) {
      throw new Error('not attached');
    }
  };
  // As forwardCdp: wait out a screenshot's mask window, check the tab, and
  // take the shared hold with no await after the last check.
  for (;;) {
    await untilNoScreenshot(lock);
    await requireTabInScope(current);
    live();
    if (!lock.writer) break;
  }
  const release = holdShared(lock);
  const objectGroup = checkGroup();
  try {
    const signature = inputGuardSignature(current);
    const objectId = await focusedElement(current, tab, objectGroup);
    live();
    if (objectId === null) return 'no focused field';
    if (objectId === undefined) return 'type failed';
    const kind = returnedValue(
      await sessionSend(tab, 'Runtime.callFunctionOn', {
        objectId,
        functionDeclaration: FIELD_KIND,
        returnByValue: true,
      }),
    );
    live();
    if (kind === 'frame') return 'field out of scope';
    if (kind !== 'entry') return 'no focused field';
    const node = describedNode(
      await sessionSend(tab, 'DOM.describeNode', { objectId }),
    );
    live();
    const backendNodeId = node?.backendNodeId;
    if (typeof backendNodeId !== 'number') return 'type failed';
    // The field's frame: the context of its own main world, judged like a
    // command's node handle (unknown or out of scope refuses).
    const { contextIds } = await resolveFrames(
      current,
      tab,
      commandAddresses({ backendNodeId }),
    );
    live();
    const inScope = () =>
      contextIds.length === 1 &&
      contextAllowedIn(current, tab.scope, contextIds[0]);
    if (!inScope()) return 'field out of scope';
    // The verdict the redactor gives this field once it holds a value: the
    // accessibility node (its accessible name included) and describe(),
    // classifyExtra (Jev) included. Unknown is not masked.
    const tree = await sessionSend(tab, 'Accessibility.getPartialAXTree', {
      backendNodeId,
      fetchRelatives: false,
    });
    live();
    const axNode = (
      isObject(tree) && Array.isArray(tree.nodes) ? tree.nodes : []
    ).find((n) => isObject(n) && n.backendDOMNodeId === backendNodeId);
    if (axNode === undefined) return 'type failed';
    const masked = await current.redactor.masksFilledField(
      axNode,
      describeVia(tab.debuggee),
    );
    live();
    if (!masked) return 'field not sensitive';
    // No await from here to the send. The value goes to the element that
    // was checked, never to whatever has focus: TYPE_INTO re-checks focus
    // and types in the same script task (Input.insertText would type
    // wherever focus is by the time the renderer gets it). Not race-free:
    // a page `beforeinput` handler runs inside that task and can move focus
    // or the selection, so the text lands elsewhere; the field's value then
    // did not change and the answer is 'type failed' (the value stays
    // registered, so a 3+ character echo is still scrubbed).
    if (inputGuardSignature(current) !== signature || !inScope()) {
      return 'field out of scope';
    }
    let typed: unknown;
    try {
      typed = returnedValue(
        await withTimeout(
          chrome.debugger.sendCommand(tab.debuggee, 'Runtime.callFunctionOn', {
            objectId,
            functionDeclaration: TYPE_INTO,
            arguments: [{ value }],
            returnByValue: true,
            silent: true,
          }),
          TYPE_MS,
          'typing',
        ),
      );
    } catch {
      return 'type failed';
    }
    if (typed === 'moved') return 'no focused field';
    return typed === 'typed' ? null : 'type failed';
  } finally {
    await releaseGroup(tab, objectGroup);
    release();
  }
}

// --- show the next unfilled field (plan 4c-4) -------------------------------

/**
 * showUnfilledField's refusals: fixed strings, as typeSensitive's, never a
 * value, a field's name or label, or page text.
 */
type ScanRefusal =
  | 'no active run'
  | 'wrong run'
  | 'not attached'
  | 'run is held'
  | 'field out of scope'
  | 'registration pending'
  | 'scan failed';

/** More candidates than this sets `capped` (P4c4-a). */
const UNFILLED_SCAN_CAP = 12;
/** One deadline for a whole call's page work after the attach. */
const SCAN_MS = 8_000;
/** Kept for the highlight: classification stops this long before it. */
const CLASSIFY_RESERVE_MS = 3_000;
/** How long the Overlay highlight stays (P4c4-c). */
const HIGHLIGHT_MS = 6_000;
// The highlight: a box and its padding, no info tooltip (it would paint the
// element's tag, id and classes), no rulers.
const HIGHLIGHT_CONFIG = {
  showInfo: false,
  showRulers: false,
  showExtensionLines: false,
  contentColor: { r: 255, g: 176, b: 0, a: 0.2 },
  paddingColor: { r: 255, g: 176, b: 0, a: 0.2 },
  borderColor: { r: 214, g: 92, b: 0, a: 1 },
};

/**
 * Ends showUnfilledField's highlight: clears its timer and, with `hide`,
 * asks the page to drop it (best effort, only on the agent's or the submit
 * watch's live attach: a detach takes the highlight with it). Sends nothing
 * but Overlay calls, which carry no page content.
 */
function clearHighlight(current: Run, hide: boolean) {
  if (current.highlightTimer === undefined) return;
  clearTimeout(current.highlightTimer);
  current.highlightTimer = undefined;
  if (hide) void hideHighlight(current);
}

// Not under the tab lock or the scope checks of a page read: it sends only
// Overlay calls, which read nothing from the page and change no DOM, so a
// screenshot's mask window or a held tab has nothing to fear from them.
async function hideHighlight(current: Run) {
  if (run !== current || current.stopped) return;
  if (!current.attached && !current.reviewAttached) return;
  const debuggee = { tabId: current.tabId };
  try {
    await withTimeout(
      chrome.debugger.sendCommand(debuggee, 'Overlay.hideHighlight'),
      FRAME_STEP_MS,
      'highlight',
    );
    await withTimeout(
      chrome.debugger.sendCommand(debuggee, 'Overlay.disable'),
      FRAME_STEP_MS,
      'highlight',
    );
  } catch {
    // Detached or gone: nothing is drawn.
  }
}

/**
 * Finds the required fields still empty in the run's tab and shows the
 * caseworker the first one (plan 4c-4, P4c4-a..c). Read-only: types
 * nothing, registers nothing, changes no attribute.
 *
 * Refused, with no CDP call, when there is no run, it is another run, the
 * run needs re-registration (`registration pending`: after a worker restart
 * the scrubbing state is incomplete), the run is held, or a scan or a
 * typing is in flight (`scan failed`). Then, as typeSensitive: attach for
 * itself when the agent is not attached (attachForTyping, never setting
 * `attached`), wait out a screenshot and keep the shared tab hold for its
 * sends, re-check the run and scope after every await (requireSameScope),
 * one deadline (SCAN_MS).
 *
 * One SCAN_UNFILLED evaluation in the bridge's `relay-focus` isolated world
 * (page script cannot change what it reads) answers
 * nothing but a count. The first candidate is checked to be in a frame in
 * scope and is classified: sensitive when its attributes or accessible name
 * are flagged by isSensitiveField or the redactor masks it once filled
 * (masksFilledField), and on any failure or timeout. Only that field is
 * classified: its verdict is the reply's `sensitive`, and classifying the
 * others would add page reads and Jev requests that change nothing.
 *
 * The highlight: on the agent's attach or the submit watch's (both outlive
 * the call), DOM.scrollIntoViewIfNeeded and Overlay.highlightNode (no DOM
 * footprint, so the agent's views and the
 * screenshot mask are unchanged; Overlay events are never forwarded) for
 * HIGHLIGHT_MS. On its own attach, which is let go before answering (and a
 * detach takes an Overlay highlight with it), or when Overlay fails: the
 * field is scrolled into view and focused instead (FOCUS_FIELD).
 *
 * Never throws, never logs; the reply is an UnfilledFieldResult or one
 * fixed ScanRefusal.
 */
async function showUnfilledField(runId: unknown): Promise<BridgeResponse> {
  const refuse = (error: ScanRefusal): BridgeResponse => ({ ok: false, error });
  const current = run;
  if (!current || current.stopped) return refuse('no active run');
  if (runId !== current.runId) return refuse('wrong run');
  if (current.needsRegistration) return refuse('registration pending');
  if (current.blocked) return refuse('run is held');
  if (current.scanNow || current.typingNow) return refuse('scan failed');
  const epoch = current.scopeEpoch;
  let reply: BridgeResponse;
  let attachedHere = false;
  current.scanNow = true;
  try {
    // A new call replaces the last highlight.
    if (current.highlightTimer !== undefined) {
      clearHighlight(current, false);
      await hideHighlight(current);
    }
    if (!holdsTab(current)) {
      attachedHere = await attachForTyping(current, epoch);
    }
    reply = await scanAndShow(current, epoch, Date.now() + SCAN_MS);
  } catch (error) {
    // The error's own text is never passed on: it may name page state.
    if (run !== current || current.stopped) reply = refuse('scan failed');
    else if (current.blocked) reply = refuse('run is held');
    else if (error instanceof Error && error.message === FRAME_OUT_OF_SCOPE) {
      reply = refuse('field out of scope');
    } else if (!holdsTab(current)) {
      reply = refuse('not attached');
    } else reply = refuse('scan failed');
  } finally {
    if (attachedHere) await releaseTypingAttach(current);
    current.scanNow = false;
  }
  return reply;
}

/** See showUnfilledField: the page work, under the tab lock. */
async function scanAndShow(
  current: Run,
  epoch: number,
  deadline: number,
): Promise<UnfilledFieldResult> {
  const lock = lockFor(current.tabId);
  const tab: FrameSession = {
    scope: current.frameScope,
    debuggee: { tabId: current.tabId },
    rootFrameId: current.mainFrameId,
  };
  // After every await: the same run and session, never out of scope since
  // the call, within the deadline (throws).
  const live = () => {
    requireSameScope(current, epoch);
    if (run !== current) throw new Error('run replaced');
    if (!holdsTab(current) || tab.scope !== current.frameScope) {
      throw new Error('not attached');
    }
    if (Date.now() > deadline) throw new Error('scan timed out');
  };
  const send = (method: string, params?: object) =>
    sessionSend(tab, method, params, deadline);
  for (;;) {
    await untilNoScreenshot(lock);
    await requireTabInScope(current);
    live();
    if (!lock.writer) break;
  }
  const release = holdShared(lock);
  const objectGroup = checkGroup();
  try {
    const found = await scanUnfilled(current, tab, objectGroup, deadline);
    live();
    if (found === undefined) throw new Error('scan failed');
    const { listId, count } = found;
    const capped = count > UNFILLED_SCAN_CAP;
    if (count === 0) return { ok: true, found: false, remaining: 0 };
    const objectId = objectIdOf(
      await send('Runtime.callFunctionOn', {
        objectId: listId,
        functionDeclaration: UNFILLED_AT,
        arguments: [{ value: 0 }],
        objectGroup,
      }),
      'result',
    );
    live();
    if (objectId === undefined) throw new Error('scan failed');
    const node = describedNode(await send('DOM.describeNode', { objectId }));
    live();
    const backendNodeId = node?.backendNodeId;
    if (typeof backendNodeId !== 'number') throw new Error('scan failed');
    // The field's frame, judged like a command's node handle.
    const { contextIds } = await resolveFrames(
      current,
      tab,
      commandAddresses({ backendNodeId }),
    );
    live();
    if (
      contextIds.length !== 1 ||
      !contextAllowedIn(current, tab.scope, contextIds[0])
    ) {
      throw new Error(FRAME_OUT_OF_SCOPE);
    }
    // Classification ends CLASSIFY_RESERVE_MS before the deadline, so a slow
    // verdict degrades to sensitive and the highlight still has time.
    const classifyBy = deadline - CLASSIFY_RESERVE_MS;
    const sensitive = await fieldSensitive(
      current,
      tab,
      backendNodeId,
      node?.attributes,
      classifyBy,
    );
    live();
    let highlighted = false;
    if (current.attached || current.reviewAttached) {
      try {
        // Overlay needs the DOM domain on; enabling it again is harmless.
        // DOM is never disabled here: the agent relies on it.
        await send('DOM.enable');
        live();
        await send('Overlay.enable');
        live();
        try {
          await send('DOM.scrollIntoViewIfNeeded', { backendNodeId });
          live();
          await send('Overlay.highlightNode', {
            backendNodeId,
            highlightConfig: HIGHLIGHT_CONFIG,
          });
        } catch (error) {
          // Turn Overlay off again before the fallback (best effort).
          await send('Overlay.disable').catch(() => {});
          throw error;
        }
        highlighted = true;
        // Armed before anything else can throw, so a drawn highlight always
        // ends.
        clearHighlight(current, false);
        current.highlightTimer = setTimeout(() => {
          current.highlightTimer = undefined;
          void hideHighlight(current);
        }, HIGHLIGHT_MS);
      } catch {
        // Overlay unavailable: the fallback below. Ended, replaced or held
        // meanwhile: live() answers that instead.
      }
    }
    live();
    if (!highlighted) {
      await send('Runtime.callFunctionOn', {
        objectId,
        functionDeclaration: FOCUS_FIELD,
        returnByValue: true,
        silent: true,
      });
      live();
    }
    const result: UnfilledFieldResult = {
      ok: true,
      found: true,
      remaining: count,
      sensitive,
    };
    if (capped) result.capped = true;
    return result;
  } finally {
    await releaseGroup(tab, objectGroup);
    release();
  }
}

/**
 * Runs SCAN_UNFILLED in the bridge's `relay-focus` world of the main frame
 * (made as focusedElement makes it) and reads back only the candidate count:
 * the candidates stay in the page, by reference (`listId`, in
 * `objectGroup`). Undefined when the world cannot be made or the answer is
 * malformed. Throws when a lookup fails.
 */
async function scanUnfilled(
  current: Run,
  tab: FrameSession,
  objectGroup: string,
  deadline: number,
): Promise<{ listId: string; count: number } | undefined> {
  const send = (method: string, params?: object) =>
    sessionSend(tab, method, params, deadline);
  const evaluate = (contextId: number) =>
    send('Runtime.evaluate', {
      expression: `(${SCAN_UNFILLED})(document)`,
      contextId,
      objectGroup,
      silent: true,
    });
  let evaluated: unknown;
  try {
    if (current.focusWorld === undefined) throw new Error('no world');
    evaluated = await evaluate(current.focusWorld);
    if (objectIdOf(evaluated, 'result') === undefined) {
      throw new Error('no answer');
    }
  } catch {
    // No world yet, or its document is gone: make one (its context events
    // are dropped like the mask's, see isBridgeWorldCreated).
    const world = await send('Page.createIsolatedWorld', {
      frameId: current.mainFrameId,
      worldName: FOCUS_WORLD,
    });
    const contextId = isObject(world) ? world.executionContextId : undefined;
    if (typeof contextId !== 'number') return undefined;
    current.focusWorld = contextId;
    evaluated = await evaluate(contextId);
  }
  const listId = objectIdOf(evaluated, 'result');
  if (listId === undefined) return undefined;
  const summary = returnedValue(
    await send('Runtime.callFunctionOn', {
      objectId: listId,
      functionDeclaration: UNFILLED_SUMMARY,
      returnByValue: true,
    }),
  );
  const count = isObject(summary) ? summary.count : undefined;
  if (typeof count !== 'number' || !Number.isInteger(count) || count < 0) {
    return undefined;
  }
  return { listId, count };
}

/**
 * Whether a field is sensitive, for the reply's `sensitive` flag: its own
 * attributes (isSensitiveField), its accessible name and role
 * (isSensitiveField; masksFilledField alone answers false for a checkbox,
 * radio or native select by role), or the redactor's verdict once filled
 * (masksFilledField, Jev included). Unlike masksFilledField, every failure
 * (no attributes, no accessibility node, a lookup error, the deadline)
 * answers true. Never throws.
 */
async function fieldSensitive(
  current: Run,
  tab: FrameSession,
  backendNodeId: number,
  attributes: unknown,
  deadline: number,
): Promise<boolean> {
  try {
    if (!Array.isArray(attributes)) return true;
    const fromAttributes = descriptorFromAttributes(attributes);
    if (isSensitiveField(fromAttributes)) return true;
    const tree = await sessionSend(
      tab,
      'Accessibility.getPartialAXTree',
      { backendNodeId, fetchRelatives: false },
      deadline,
    );
    const axNode = (
      isObject(tree) && Array.isArray(tree.nodes) ? tree.nodes : []
    ).find((n) => isObject(n) && n.backendDOMNodeId === backendNodeId);
    if (!isObject(axNode)) return true;
    const axText = (value: unknown) =>
      isObject(value) && typeof value.value === 'string'
        ? value.value
        : undefined;
    const label = axText(axNode.name);
    const role = axText(axNode.role);
    if (
      isSensitiveField({
        ...fromAttributes,
        ...(label === undefined ? {} : { label }),
        ...(role === undefined ? {} : { role }),
      })
    ) {
      return true;
    }
    const left = deadline - Date.now();
    if (left <= 0) return true;
    return await withTimeout(
      current.redactor.masksFilledField(axNode, describeVia(tab.debuggee)),
      left,
      'classify',
    );
  } catch {
    return true;
  }
}

// --- the submit watch (plan 4c-4) ---------------------------------------------

/** armSubmitWatch's refusals: fixed strings, as typeSensitive's. */
type WatchRefusal =
  | 'no active run'
  | 'wrong run'
  | 'not attached'
  | 'run is held';

/**
 * Arms the submit watch (P4c4-d, P4c4-e): from now until disarmed, a form
 * submission in the run's tab followed by a main-frame navigation in scope
 * sets `submitted` in the run status (see watchForSubmission). The panel
 * arms it only in Review, so the agent's own Next clicks cannot count.
 *
 * Reads no page text (only navigation event params), so a run that needs
 * registration may arm. Refused with no CDP call when there is no run, it
 * is another run, or the run is held. Under the agent's attach it only
 * marks the watch armed. Otherwise it attaches for itself (attachForReview:
 * the attach core with the relay `attach`'s scope checks, bounded steps,
 * queued with every attach) as `reviewAttached`, never `attached`, so agent
 * commands still get `not attached`; a typing or scan attach in flight is
 * taken over rather than replaced. A failed attach answers `not attached`
 * (or `run is held` when the tab left scope) and leaves the watch unarmed,
 * so `{ ok: true }` always means the tab is watched.
 *
 * Idempotent: arming again keeps what the watch saw, and re-attaches if
 * the attach was lost (a debugger detach disarms the watch; the status's
 * `armed` then reads false). Arms that arrive while an arm attach of the
 * same watch generation is in flight wait for it rather than queueing
 * another attach step, so a panel re-arming on every poll cannot grow the
 * attach queue behind a hung attach. A token refresh is an endRun plus
 * begin: the watch is lost with the old run and the panel arms it again.
 *
 * With `reset`, the seen submission and any pending request are forgotten
 * first (before the held check: forgetting is always safe), with no CDP
 * call and the attach kept; then it arms as above. Never throws, never logs.
 */
async function armSubmitWatch(
  runId: unknown,
  reset = false,
): Promise<BridgeResponse> {
  const refuse = (error: WatchRefusal): BridgeResponse => ({
    ok: false,
    error,
  });
  const current = run;
  if (!current || current.stopped) return refuse('no active run');
  if (runId !== current.runId) return refuse('wrong run');
  if (reset && current.submitWatch) {
    current.submitWatch.pending = undefined;
    current.submitWatch.submitted = undefined;
  }
  if (current.blocked) return refuse('run is held');
  if (current.attached || current.reviewAttached) {
    current.submitWatch ??= {};
    return { ok: true };
  }
  const generation = current.watchGeneration;
  let flight = current.armInFlight;
  if (flight === undefined || flight.generation !== generation) {
    const done = attachForReview(current, current.scopeEpoch, generation);
    const started = { generation, done };
    flight = started;
    current.armInFlight = started;
    // Cleared once it settles (either way), unless a newer arm replaced it.
    void done
      .catch(() => {})
      .finally(() => {
        if (current.armInFlight === started) current.armInFlight = undefined;
      });
  }
  try {
    await flight.done;
  } catch {
    if (run !== current || current.stopped) return refuse('no active run');
    if (current.blocked) return refuse('run is held');
    return refuse('not attached');
  }
  return { ok: true };
}

/**
 * armSubmitWatch's attach, run through queuedAttach. Arms the watch in the
 * same step as the flag it rests on, unless a disarm came first (its
 * let-go, queued behind this step, then lets go of an attach made here).
 * Throws when the watch is not armed. Refused while a detach grace is
 * running: the run is waiting for the agent.
 */
async function attachForReview(
  current: Run,
  epoch: number,
  generation: number,
): Promise<void> {
  if (current.detachTimer !== undefined) throw new Error('not attached');
  const target = await bounded(tabTarget(current.tabId), 'targets');
  if (!target) throw new Error('not attached');
  await queuedAttach(current, async () => {
    if (run !== current || current.stopped) throw new Error('run replaced');
    if (current.watchGeneration !== generation) throw new Error('disarmed');
    if (current.detachTimer !== undefined) throw new Error('not attached');
    requireSameScope(current, epoch);
    if (current.typingAttached) {
      // A typing or scan holds the tab for itself: the watch takes that
      // attach over (no detach, so the call in flight goes on under it, and
      // its let-go then finds nothing to let go of).
      current.typingAttached = false;
      current.reviewAttached = true;
    } else if (!current.attached && !current.reviewAttached) {
      await attachTab(current, epoch, target.id, ATTACH_STEP_MS);
      current.reviewAttached = true;
    }
    if (current.watchGeneration !== generation) throw new Error('disarmed');
    current.submitWatch ??= {};
  });
}

/**
 * Disarms the submit watch: forgets a pending or seen submission (so
 * `submitted` is no longer reported) and lets go of the watch's own attach
 * (releaseReviewAttach), never the agent's. Answers `{ ok: true }` for the
 * run even when nothing was armed; `no active run` or `wrong run`
 * otherwise. Never throws, never logs.
 */
async function disarmSubmitWatch(runId: unknown): Promise<BridgeResponse> {
  const current = run;
  if (!current || current.stopped) return { ok: false, error: 'no active run' };
  if (runId !== current.runId) return { ok: false, error: 'wrong run' };
  current.submitWatch = undefined;
  current.watchGeneration += 1;
  await releaseReviewAttach(current);
  return { ok: true };
}

/**
 * Lets go of the watch's own attach (queued with every attach, so an arm's
 * attach still in flight is let go of after it lands), unless the run
 * ended or was replaced (endRun detaches), the agent took it over, or the
 * watch was armed again meanwhile. A typing or scan running under it then
 * fails closed (`not attached`). Chrome sends no onDetach for the
 * extension's own detach, so no detach grace starts.
 */
function releaseReviewAttach(current: Run): Promise<void> {
  return queuedAttach(current, async () => {
    if (run !== current || current.stopped) return;
    if (current.attached || !current.reviewAttached) return;
    if (current.submitWatch !== undefined) return;
    current.reviewAttached = false;
    // The detach below takes the highlight with it.
    clearHighlight(current, false);
    current.focusWorld = undefined;
    current.dialog = undefined;
    current.frameScope = createFrameScope();
    await detachTab(current.tabId);
  });
}

/**
 * Lets the run go on at the origin it is held on (plan 4c's panel asks the
 * caseworker). Only an exact https origin equal to `pendingOrigin`, with the
 * tab still on it (or back in scope), at most MAX_EXTRA_ORIGINS per run; it
 * holds for this run only and is never persisted (see Run.extraOrigins).
 * Anything else changes nothing. The relay is not told: agent-initiated
 * navigation stays bound to the token origin (see forwardCdp).
 */
async function allowOrigin(origin: unknown): Promise<BridgeResponse> {
  const current = run;
  if (!current || current.stopped) return { ok: false, error: 'no active run' };
  const pending = () =>
    run === current &&
    !current.stopped &&
    typeof origin === 'string' &&
    normalizeOrigin(origin) === origin &&
    current.pendingOrigin === origin;
  if (!pending()) return { ok: false, error: ORIGIN_NOT_PENDING };
  if (current.extraOrigins.length >= MAX_EXTRA_ORIGINS) {
    return { ok: false, error: 'too many allowed origins' };
  }
  const url = await tabUrl(current.tabId);
  // Re-checked after the await: the hold may have moved or ended.
  if (
    !pending() ||
    typeof origin !== 'string' ||
    url === null ||
    !(inScope(url, origin) || inRunScope(current, url))
  ) {
    return { ok: false, error: ORIGIN_NOT_PENDING };
  }
  current.extraOrigins.push(origin);
  releaseHold(current);
  return { ok: true };
}

/**
 * The run's state for the panel: ids (run, tab), the origin and counts, never
 * a token or URL.
 */
function getRunStatus(): RunStatus {
  const current = run;
  if (!current || current.stopped) {
    const restartingRunId = restartingRunIds[0];
    if (restartingRunId !== undefined) {
      return {
        ok: true,
        active: false,
        restarting: true,
        runId: restartingRunId,
      };
    }
    if (resumesInFlight > 0)
      return { ok: true, active: false, restarting: true };
    return { ok: true, active: false };
  }
  const status: RunStatus = {
    ok: true,
    active: true,
    runId: current.runId,
    origin: current.origin,
    tabId: current.tabId,
    attached: current.attached,
    needsRegistration: current.needsRegistration,
    exp: current.exp,
    armed:
      current.submitWatch !== undefined &&
      (current.attached || current.reviewAttached),
  };
  if (current.jev) {
    status.jevFailures = current.jev.failures();
    status.jevRequests = current.jev.requests();
  }
  if (current.pendingOrigin !== null) {
    status.pendingOrigin = current.pendingOrigin;
  }
  // Only while armed (the watch is undefined otherwise): a time, nothing else.
  const submitted = current.submitWatch?.submitted;
  if (submitted) status.submitted = { at: submitted.at };
  return status;
}

/**
 * The panel re-entered the run's known values after a worker restart (it
 * sends only how many, never the values): screenshots may be taken again.
 */
function registrationComplete(runId: unknown, count: unknown): BridgeResponse {
  const current = run;
  if (!current || current.stopped) return { ok: false, error: 'no active run' };
  if (runId !== current.runId) return { ok: false, error: 'wrong run' };
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
  if (message.type === 'stopRun') {
    return message.runId === undefined
      ? stopRun('user')
      : stopRunFor(message.runId);
  }
  if (message.type === 'addKnownValue')
    return addKnownValue(message.runId, message.value);
  if (message.type === 'getRunStatus') return getRunStatus();
  if (message.type === 'registrationComplete') {
    return registrationComplete(message.runId, message.count);
  }
  if (message.type === 'allowOrigin') return allowOrigin(message.origin);
  if (message.type === 'typeSensitive') {
    return typeSensitive(message.runId, message.value);
  }
  if (message.type === 'showUnfilledField') {
    return showUnfilledField(message.runId);
  }
  if (message.type === 'armSubmitWatch') {
    return armSubmitWatch(message.runId, message.reset === true);
  }
  if (message.type === 'disarmSubmitWatch') {
    return disarmSubmitWatch(message.runId);
  }
  return { ok: false, error: 'unknown message type' };
}
