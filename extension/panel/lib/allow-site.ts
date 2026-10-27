import { ALLOW_SITE_COPY } from './copy';
import type { RunSessionState } from './run-session';
import { describeTabUrl, UNKNOWN_PAGE } from './tab-url';
import { WorkerError } from './worker-api';

// The allow-this-site card's controller, free of React so it can be tested
// with fake timers. The worker holds the run when the tab goes to another
// https origin; only the caseworker can let the agent follow it, for this run.
//
// - allow() is the ONLY path to the worker's allowOrigin, and only the Allow
//   button's click calls allow(): no timer, poll, tab read or update does.
// - The WORKER decides whether the hold is still on. Allow is enabled while
//   the session's current pending origin is the card's origin and no allow is
//   in flight or done; the countdown is informational only (the bridge resets
//   its hold clock when the tab bounces back in scope, which a 2 s poll can
//   miss, so the panel's count may run out first).
// - One allowOrigin per press, at most one in flight (the card is keyed by
//   the pending origin, so a new origin gets a new controller): a press while
//   one is in flight, after success, or once the pending origin moved or
//   cleared sends nothing. A refusal shows a fixed message and disables Allow
//   for good; only an unreachable worker leaves Allow for another press.
// - The tab's URL is read with getTab at start and every 2 s, reduced at once
//   to origin plus path (describeTabUrl), and kept only in the view: never
//   logged, never stored.

export const ALLOW_TAB_POLL_MS = 2000;
export const ALLOW_TICK_MS = 1000;

type Timer = ReturnType<typeof setInterval>;

export type AllowSiteDeps = {
  allowOrigin: (origin: string) => Promise<void>;
  getTab: (tabId: number) => Promise<{ url?: string } | undefined>;
  /** Called once after the worker accepted the allow, even if disposed. */
  onAllowed: () => void;
  /** Milliseconds. */
  now: () => number;
  setInterval: (fn: () => void, ms: number) => Timer;
  clearInterval: (id: Timer) => void;
};

export type AllowSitePhase = 'waiting' | 'allowing' | 'allowed' | 'refused';

/** Which fixed message the card shows after a press that did not allow. */
export type AllowSiteNotice = 'holdEnded' | 'tooMany' | 'unreachable';

export type AllowSiteView = {
  /** The origin this card asks about (from the worker, rendered as text). */
  origin: string;
  /** Origin plus path of the tab, or UNKNOWN_PAGE; null until first read. */
  tabText: string | null;
  phase: AllowSitePhase;
  notice: AllowSiteNotice | null;
  canAllow: boolean;
  /** Seconds left in the hold, floored at 0; null when unknown. */
  secondsLeft: number | null;
};

/** The pending origin to ask about, or null when no card shows. */
export const showAllowSite = (
  state: Pick<RunSessionState, 'status' | 'pendingOrigin'> | null,
): string | null => {
  if (!state || state.status !== 'running') return null;
  if (typeof state.pendingOrigin !== 'string' || state.pendingOrigin === '') {
    return null;
  }
  return state.pendingOrigin;
};

export const createAllowSiteController = (
  input: { origin: string; tabId: number; holdEndsAt?: number },
  deps: AllowSiteDeps,
  onView: (view: AllowSiteView) => void,
) => {
  const { origin, tabId, holdEndsAt } = input;
  let pendingOrigin: string | undefined;
  let stopping = false;
  let phase: AllowSitePhase = 'waiting';
  let notice: AllowSiteNotice | null = null;
  let tabText: string | null = null;
  let disposed = false;
  let reading = false;
  let tabTimer: Timer | undefined;
  let tickTimer: Timer | undefined;

  const secondsLeft = (): number | null =>
    typeof holdEndsAt === 'number' && Number.isFinite(holdEndsAt)
      ? Math.max(0, Math.ceil((holdEndsAt - deps.now()) / 1000))
      : null;

  const canAllow = () =>
    phase === 'waiting' && !stopping && pendingOrigin === origin;

  const view = (): AllowSiteView => ({
    origin,
    tabText,
    phase,
    notice,
    canAllow: canAllow(),
    secondsLeft: secondsLeft(),
  });

  const emit = () => {
    if (!disposed) onView(view());
  };

  const readTab = async () => {
    if (reading || disposed) return;
    reading = true;
    let text = UNKNOWN_PAGE;
    try {
      const tab = await deps.getTab(tabId);
      text = describeTabUrl(tab?.url);
    } catch {
      text = UNKNOWN_PAGE;
    } finally {
      reading = false;
    }
    if (disposed) return;
    tabText = text;
    emit();
  };

  const stopTimers = () => {
    if (tabTimer !== undefined) deps.clearInterval(tabTimer);
    if (tickTimer !== undefined) deps.clearInterval(tickTimer);
    tabTimer = undefined;
    tickTimer = undefined;
  };

  return {
    view,
    /** Starts the tab reads and the countdown. Never touches allowOrigin. */
    start() {
      if (disposed || tabTimer !== undefined) return;
      void readTab();
      tabTimer = deps.setInterval(() => void readTab(), ALLOW_TAB_POLL_MS);
      tickTimer = deps.setInterval(emit, ALLOW_TICK_MS);
    },
    /** The session's latest state. Never touches allowOrigin. */
    update(next: { pendingOrigin: string | undefined; stopping: boolean }) {
      pendingOrigin = next.pendingOrigin;
      stopping = next.stopping;
      emit();
    },
    /** The Allow button's click, and nothing else. */
    async allow() {
      if (disposed || !canAllow()) return;
      // Latched before the await: a second press in the same tick is ignored.
      phase = 'allowing';
      notice = null;
      emit();
      try {
        await deps.allowOrigin(origin);
      } catch (error) {
        const code = error instanceof WorkerError ? error.code : undefined;
        if (code === 'worker_unreachable') {
          // No answer from the worker: another PRESS may try again (had the
          // first one landed, the hold clears and the card goes away).
          phase = 'waiting';
          notice = 'unreachable';
        } else {
          phase = 'refused';
          notice = code === 'too_many_origins' ? 'tooMany' : 'holdEnded';
        }
        emit();
        return;
      }
      phase = 'allowed';
      deps.onAllowed();
      emit();
    },
    dispose() {
      disposed = true;
      stopTimers();
    },
  };
};

export type AllowSiteController = ReturnType<typeof createAllowSiteController>;

const defaultClock = () =>
  new Intl.DateTimeFormat(undefined, { hour: '2-digit', minute: '2-digit' });

/** The expiry (Unix SECONDS) as hour:minute in the given format. */
export const formatClock = (
  expiresAt: number,
  format: Intl.DateTimeFormat = defaultClock(),
): string => format.format(new Date(expiresAt * 1000));

/**
 * The P4c2-e note after an allow: a refresh is refused while the tab is on
 * an allowed extra origin, so the run ends at its current expiry. An
 * estimated or unusable expiry gives no time rather than an invented one.
 */
export const allowedNote = (
  expiresAt: number,
  estimated: boolean,
  format?: Intl.DateTimeFormat,
): string =>
  estimated || !Number.isFinite(expiresAt)
    ? ALLOW_SITE_COPY.allowedEstimated
    : ALLOW_SITE_COPY.allowed(formatClock(expiresAt, format));
