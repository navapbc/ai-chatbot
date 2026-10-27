import type { AppApi } from '../../src/app-api';
import type { RunStatus } from '../../src/bridge';
import type { Target } from './state';
import type { WorkerApi } from './worker-api';

// Watches a run the panel started (or rebuilt after reopening) until it ends.
//
// - Polls the worker's getRunStatus every 2 s and the server's run state every
//   5 s; a poll is skipped while the previous one of its kind is in flight,
//   and a failed poll keeps the last state (errors never end a run).
// - The server row says `ended`: the run ended (reason `stopped`).
// - The worker holds no run, or another run (another window replaced it),
//   while the server row is live: the worker ended it by itself (left the
//   origin, token expired, debugger canceled or detached, tab closed). Those
//   reach only the hub, so the server stop route is called once here, or the
//   Eve turn keeps running and agent tokens keep being minted. For the first
//   5 s an inactive worker is not an end yet, unless the run was already seen.
// - The extension token is refreshed at 5 minutes left (P4c-e): a new token,
//   a same-run-id worker startRun on the run's tab, registrationComplete(runId,
//   0). A refusal shows a notice and is tried again at most once a minute.
// - stop(): the server stop, then the worker stop (P4c-f).
//
// Nothing here ever sees an applicant value: only ids, origins and counts.
// No I/O runs at creation (only timers), so a StrictMode mount, dispose and
// mount sends nothing for the disposed session.

export const WORKER_POLL_MS = 2000;
export const SERVER_POLL_MS = 5000;
export const GRACE_MS = 5000;
export const REFRESH_BEFORE_MS = 5 * 60_000;
export const REFRESH_RETRY_MS = 60_000;
/** A rebuilt run's expiry is unknown: assume this little is left. */
export const ESTIMATED_LEFT_S = 5 * 60;

export type EndedReason = 'user' | 'stopped' | 'worker-ended';

export type SessionRun = {
  runId: string;
  /** Unix time in SECONDS, as the token route returns it. */
  expiresAt: number;
  tabId: number;
  /** True for a run rebuilt after reopening: the expiry is a guess. */
  expiresAtEstimated?: boolean;
};

export type RunSessionState = {
  status: 'running' | 'ended';
  endedReason?: EndedReason;
  pendingOrigin?: string;
  jevFailures?: number;
  refreshFailed?: boolean;
  expiresAtEstimated?: boolean;
  secondsLeft: number;
  stopping: boolean;
  stopFailed?: boolean;
};

type Timer = ReturnType<typeof setTimeout>;

export type RunSessionDeps = {
  api: Pick<AppApi, 'getRunState' | 'stopRun' | 'getExtensionToken'>;
  worker: Pick<
    WorkerApi,
    'getRunStatus' | 'stopRun' | 'startRun' | 'registrationComplete'
  >;
  /** Milliseconds. */
  now: () => number;
  setInterval: (fn: () => void, ms: number) => Timer;
  clearInterval: (id: Timer) => void;
  setTimeout: (fn: () => void, ms: number) => Timer;
  clearTimeout: (id: Timer) => void;
  markRecentEnded: (runId: string, reason: string) => Promise<void>;
};

export type RunSession = {
  subscribe(cb: (state: RunSessionState) => void): () => void;
  stop(): Promise<void>;
  dispose(): void;
};

const ignore = async (task: () => Promise<unknown>): Promise<boolean> => {
  try {
    await task();
    return true;
  } catch {
    return false;
  }
};

export const createRunSession = (
  deps: RunSessionDeps,
  run: SessionRun,
): RunSession => {
  const { runId, tabId } = run;
  const createdAt = deps.now();
  let expiresAt = run.expiresAt;
  let estimated = run.expiresAtEstimated === true;

  const listeners = new Set<(state: RunSessionState) => void>();
  let disposed = false;
  let ended = false;
  let endedReason: EndedReason | undefined;
  let stopping = false;
  let stopFailed = false;
  let stopPromise: Promise<void> | undefined;
  let seenActive = false;
  let lastWorker: RunStatus | undefined;

  let workerInFlight = false;
  let serverInFlight = false;
  let registering = false;

  let refreshPromise: Promise<void> | undefined;
  let refreshFailed = false;
  let refreshFailedAt: number | undefined;
  // Bumped when a refresh starts and when it ends: a worker poll answered
  // under another generation saw the run mid-restart and is discarded.
  let generation = 0;

  const secondsLeft = () =>
    Math.max(0, Math.floor((expiresAt * 1000 - deps.now()) / 1000));

  const snapshot = (): RunSessionState => {
    const state: RunSessionState = {
      status: ended ? 'ended' : 'running',
      secondsLeft: secondsLeft(),
      stopping: stopping && !ended,
    };
    if (endedReason) state.endedReason = endedReason;
    if (!ended && lastWorker?.active && lastWorker.runId === runId) {
      if (lastWorker.pendingOrigin) {
        state.pendingOrigin = lastWorker.pendingOrigin;
      }
      if (typeof lastWorker.jevFailures === 'number') {
        state.jevFailures = lastWorker.jevFailures;
      }
    }
    if (refreshFailed && !ended) state.refreshFailed = true;
    if (estimated && !ended) state.expiresAtEstimated = true;
    if (stopFailed) state.stopFailed = true;
    return state;
  };

  const emit = () => {
    if (disposed) return;
    const state = snapshot();
    for (const listener of listeners) listener(state);
  };

  let workerTimer: Timer | undefined = deps.setInterval(
    () => void pollWorker(),
    WORKER_POLL_MS,
  );
  let serverTimer: Timer | undefined = deps.setInterval(
    () => void pollServer(),
    SERVER_POLL_MS,
  );

  const stopTimers = () => {
    if (workerTimer !== undefined) deps.clearInterval(workerTimer);
    if (serverTimer !== undefined) deps.clearInterval(serverTimer);
    workerTimer = undefined;
    serverTimer = undefined;
  };

  const finish = (reason: EndedReason) => {
    ended = true;
    endedReason = reason;
    stopTimers();
    void ignore(() => deps.markRecentEnded(runId, reason));
    emit();
  };

  const isOurs = (status: RunStatus | undefined) =>
    status?.active === true && status.runId === runId;

  const refreshDue = () =>
    !refreshPromise &&
    !stopping &&
    !ended &&
    expiresAt * 1000 - deps.now() <= REFRESH_BEFORE_MS &&
    (refreshFailedAt === undefined ||
      deps.now() - refreshFailedAt >= REFRESH_RETRY_MS);

  const halted = () => stopping || ended || disposed;

  const refresh = async () => {
    generation += 1;
    try {
      const token = await deps.api.getExtensionToken(runId);
      if (halted()) return;
      await deps.worker.startRun({
        relayUrl: token.relayUrl,
        token: token.extensionToken,
        tabId,
      });
      if (halted()) return;
      // The restarted run has no registered values (plan 4b): screenshots
      // stay refused until this arrives.
      await deps.worker.registrationComplete(runId, 0);
      if (!Number.isFinite(token.expiresAt)) throw new Error('bad expiry');
      expiresAt = token.expiresAt;
      estimated = false;
      refreshFailed = false;
      refreshFailedAt = undefined;
    } catch {
      refreshFailed = true;
      refreshFailedAt = deps.now();
    } finally {
      generation += 1;
      refreshPromise = undefined;
      if (!ended) emit();
    }
  };

  const endByWorker = () => {
    // The server row may still be live: end it, once, best effort.
    void ignore(() => deps.api.stopRun(runId));
    finish('worker-ended');
  };

  const onWorkerStatus = (status: RunStatus) => {
    if (status.active && status.runId !== runId) {
      lastWorker = status;
      endByWorker();
      return;
    }
    if (!status.active) {
      lastWorker = status;
      if (!seenActive && deps.now() - createdAt < GRACE_MS) {
        emit();
        return;
      }
      endByWorker();
      return;
    }
    seenActive = true;
    lastWorker = status;
    if (status.needsRegistration && !registering) {
      registering = true;
      void ignore(() => deps.worker.registrationComplete(runId, 0)).then(() => {
        registering = false;
      });
    }
    if (refreshDue()) refreshPromise = refresh();
    emit();
  };

  const pollWorker = async () => {
    if (workerInFlight || halted()) return;
    workerInFlight = true;
    const issuedUnder = generation;
    let status: RunStatus | undefined;
    try {
      status = await deps.worker.getRunStatus();
    } catch {
      status = undefined;
    } finally {
      workerInFlight = false;
    }
    if (halted()) return;
    if (!status || refreshPromise || generation !== issuedUnder) {
      emit();
      return;
    }
    onWorkerStatus(status);
  };

  const pollServer = async () => {
    if (serverInFlight || halted()) return;
    serverInFlight = true;
    let rowEnded = false;
    try {
      rowEnded = (await deps.api.getRunState(runId)).ended === true;
    } catch {
      rowEnded = false;
    } finally {
      serverInFlight = false;
    }
    if (halted() || !rowEnded) return;
    // The worker may still hold this run: release the tab's debugger. Never
    // when it shows another run (its stopRun takes no run id).
    if (isOurs(lastWorker)) void ignore(() => deps.worker.stopRun());
    finish('stopped');
  };

  const stopNow = async () => {
    let failed = !(await ignore(() => deps.api.stopRun(runId)));
    // A refresh in flight would restart the worker after its stop.
    if (refreshPromise) await refreshPromise;
    // A worker last seen with another run already ended this session (Stop is
    // then a no-op). The worker's stopRun takes no run id, so another window
    // starting a run within the last poll interval would still be stopped.
    if (!(await ignore(() => deps.worker.stopRun()))) failed = true;
    stopFailed = failed;
    finish('user');
  };

  return {
    subscribe(cb) {
      if (disposed) return () => {};
      listeners.add(cb);
      cb(snapshot());
      return () => {
        listeners.delete(cb);
      };
    },
    stop() {
      if (stopPromise) return stopPromise;
      if (ended) return Promise.resolve();
      stopping = true;
      emit();
      stopPromise = stopNow();
      return stopPromise;
    },
    dispose() {
      disposed = true;
      stopTimers();
      listeners.clear();
    },
  };
};

const MAX_SITE_NAME = 100;

/**
 * The panel reopened while the worker holds a run: rebuild the Running screen
 * when the worker's run is live on the server, on the same origin. The expiry
 * is unknown, so a conservative 5 minutes is assumed and marked estimated
 * (the session then refreshes the token at its first poll). Any failure or
 * mismatch rebuilds nothing.
 */
export const rebuildRun = async (deps: {
  api: Pick<AppApi, 'getRunState'>;
  worker: Pick<WorkerApi, 'getRunStatus'>;
  now: () => number;
}): Promise<{ run: SessionRun; target: Target } | null> => {
  try {
    const status = await deps.worker.getRunStatus();
    if (
      !status.active ||
      typeof status.runId !== 'string' ||
      typeof status.origin !== 'string' ||
      !Number.isInteger(status.tabId)
    ) {
      return null;
    }
    const row = await deps.api.getRunState(status.runId);
    if (row.ended || row.origin !== status.origin) return null;
    const tabId = status.tabId as number;
    return {
      run: {
        runId: status.runId,
        tabId,
        expiresAt: Math.floor(deps.now() / 1000) + ESTIMATED_LEFT_S,
        expiresAtEstimated: true,
      },
      target: {
        tabId,
        origin: status.origin,
        siteName: new URL(status.origin).hostname.slice(0, MAX_SITE_NAME),
      },
    };
  } catch {
    return null;
  }
};
