import type { PendingInputRequest } from '../../../lib/ai/eve/input-requests';
import type { RunProgress } from '../../../lib/relay/progress';
import type { AppApi, RunPhase, RunState } from '../../src/app-api';
import type { RunStatus } from '../../src/bridge';
import { createActionLog } from './action-log';
import type { Target } from './state';
import {
  type Confirmation,
  reduceConfirmation,
  type WorkerApi,
} from './worker-api';

// Watches a run the panel started (or rebuilt after reopening) until it ends.
//
// - Polls the worker's getRunStatus every 2 s and the server's run state every
//   3 s; a poll is skipped while the previous one of its kind is in flight,
//   and a failed poll keeps the last state (errors never end a run).
// - The server row says `ended`: the run ended (reason `stopped`).
// - The worker holds no run, or another run (another window replaced it),
//   while the server row is live: the worker ended it by itself (left the
//   origin, token expired, debugger canceled or detached, tab closed). Those
//   reach only the hub, so the server stop route is called once here, or the
//   Eve turn keeps running and agent tokens keep being minted. Another runId
//   ends it at once. No run must be answered continuously for
//   WORKER_GONE_AFTER_MS (see there), and the server is read once more first:
//   a row that already ended wins (reason `stopped`, no stop call).
// - The extension token is refreshed at 5 minutes left (P4c-e): a new token,
//   a same-run-id worker startRun on the run's tab, registrationComplete(runId,
//   0). A refusal shows a notice and is tried again at most once a minute.
// - stop(): the server stop, then the worker stop (P4c-f). Every worker stop
//   names this run, so it never ends another window's run.
// - The submit watch (plan 4c-4): while the server phase is `review` the
//   worker's submit watch is armed (`reset` on entering review), re-armed when
//   a poll shows it lost (`armed: false`), and disarmed on leaving review or
//   when the run ends. Every arm and disarm goes through one chain (one call
//   at a time), and an arm is sent only while review still wants it, so no
//   arm follows a disarm. A submission the worker saw counts only from polls
//   issued after the last reset arm succeeded. With a number after a
//   `confirmation` label the run is stopped as submitted at once (the number
//   copied first); otherwise, once the number read is over (a number after
//   another label, or CONFIRMATION_WAIT_MS with none), the caseworker is
//   asked. A wrong stop cannot be undone, so only that one label stops it.
//
// Nothing here ever sees an applicant value: only ids, origins and counts.
// No I/O runs at creation (only timers), so a StrictMode mount, dispose and
// mount sends nothing for the disposed session.

export const WORKER_POLL_MS = 2000;
export const SERVER_POLL_MS = 3000;
/**
 * A fresh run may read inactive briefly before the worker holds it. Covered by
 * WORKER_GONE_AFTER_MS, which is longer: no separate check is needed.
 */
export const GRACE_MS = 5000;
/**
 * How long the worker must answer "no run" without a break before the run
 * counts as ended by the worker. The bridge answers getRunStatus outside its
 * startRun queue, and a same-run restart (another panel window refreshing the
 * token, or rebuilding the run) clears the run at once, then waits up to its
 * STOP_WAIT_MS (5 s, extension/src/bridge.ts) for an in-flight screenshot
 * before the new run begins. Kept at STOP_WAIT_MS plus one worker poll; a test
 * reads the bridge constant.
 */
export const WORKER_GONE_AFTER_MS = 7_000;
/** Longest continuous `restarting` that still holds off the 7 s rule. */
export const RESTARTING_MAX_MS = 30_000;
export const REFRESH_BEFORE_MS = 5 * 60_000;
export const REFRESH_RETRY_MS = 60_000;
/** A rebuilt run's expiry is unknown: assume this little is left. */
export const ESTIMATED_LEFT_S = 5 * 60;
// esbuild defines this only in a test-hooks build given
// EXTENSION_TEST_ORIGIN_HOLD_MS (scripts/extension-build-config.ts), for the
// panel exactly as for the worker; undeclared otherwise (a unit test too).
declare const __ORIGIN_HOLD_MS__: number | undefined;
/**
 * The bridge's origin hold (extension/src/bridge.ts ORIGIN_HOLD_MS and
 * ORIGIN_HOLD_CAP_MS): 60 s per pending origin in a shipped build, restarted
 * by a move to another origin, twice that in total. The worker status
 * carries no start time, so the panel counts from when it first saw each
 * origin. The countdown is informational only: the worker decides.
 */
export const ORIGIN_HOLD_MS =
  typeof __ORIGIN_HOLD_MS__ === 'number' ? __ORIGIN_HOLD_MS__ : 60_000;
export const ORIGIN_HOLD_CAP_MS = 2 * ORIGIN_HOLD_MS;
/**
 * A hold's countdown is known only when an accepted worker answer within
 * this long before showed this run with no pending origin: the hold began in
 * that gap, so a countdown from the first sight overstates by less than it.
 */
export const HOLD_KNOWN_GAP_MS = 2 * WORKER_POLL_MS + 1000;

/**
 * The worker's number read ends within this of the submission
 * (extension/src/bridge.ts CONFIRMATION_READ_MS; a test compares them). The
 * status cannot tell "not read yet" from "none found", so the panel waits
 * this long from first seeing a submission before it asks.
 */
export const CONFIRMATION_WAIT_MS = 12_000;

export type EndedReason = 'user' | 'submitted' | 'stopped' | 'worker-ended';

/** How a submission was confirmed: from the page, or the caseworker's word. */
export type ConfirmedHow = 'detected' | 'manual';

/**
 * A submission the worker saw while in review: `checking` while the number
 * read may still answer, `ask` once the caseworker must say.
 */
export type SubmissionView =
  | { stage: 'checking' }
  | { stage: 'ask'; confirmation?: Confirmation };

export type SessionRun = {
  runId: string;
  /** Unix time in SECONDS, as the token route returns it. */
  expiresAt: number;
  tabId: number;
  /** True for a run rebuilt after reopening: the expiry is a guess. */
  expiresAtEstimated?: boolean;
};

const SERVER_PHASES = new Set(['working', 'needs_you', 'review']);

export type RunSessionState = {
  status: 'running' | 'ended';
  /** Latest progress the server holds, null until the first server poll. */
  progress: RunProgress | null;
  /** Open questions; always empty once the run ended. */
  pending: PendingInputRequest[];
  /** From the server; `ended` whenever the session itself ended. */
  phase: RunPhase;
  /** Session-local log of reported actions (untrusted text, never stored). */
  actions: string[];
  endedReason?: EndedReason;
  /** The https origin the worker holds the tab on, waiting for a decision. */
  pendingOrigin?: string;
  /**
   * With pendingOrigin: when this session first saw the CURRENT pending
   * origin (ms epoch). Reset when the pending origin changes or clears.
   */
  holdSeenAt?: number;
  /**
   * With pendingOrigin: whether this session saw the hold begin, i.e. an
   * accepted worker answer at most HOLD_KNOWN_GAP_MS before the first
   * pending one showed this run with no pending origin. False for a hold
   * already under way at the session's first answer (a panel opened
   * mid-hold) and for every later origin in such a hold.
   */
  holdCountdownKnown?: boolean;
  /**
   * Only when holdCountdownKnown: when the hold ends at the latest (ms
   * epoch), min(holdSeenAt + 60 s, first sight of the hold + 120 s). Never
   * earlier than the bridge's own deadline, since every sight is at or after
   * the event it sees.
   */
  holdEndsAt?: number;
  jevFailures?: number;
  refreshFailed?: boolean;
  expiresAtEstimated?: boolean;
  /** The run's live expiry, Unix time in SECONDS (a refresh moves it). */
  expiresAt: number;
  secondsLeft: number;
  stopping: boolean;
  stopFailed?: boolean;
  /** Review only: a submission the worker saw, while the panel decides. */
  submission?: SubmissionView;
  /** Ended as submitted: how it was confirmed. */
  confirmedHow?: ConfirmedHow;
  /**
   * Ended as submitted from the page: the number it showed, for the
   * Confirmed screen only (never stored, logged or sent anywhere).
   */
  confirmation?: Confirmation;
};

type Timer = ReturnType<typeof setTimeout>;

export type RunSessionDeps = {
  api: Pick<AppApi, 'getRunState' | 'stopRun' | 'getExtensionToken'>;
  worker: Pick<
    WorkerApi,
    | 'getRunStatus'
    | 'stopRun'
    | 'startRun'
    | 'registrationComplete'
    | 'armSubmitWatch'
    | 'disarmSubmitWatch'
  >;
  /** Milliseconds. */
  now: () => number;
  setInterval: (fn: () => void, ms: number) => Timer;
  clearInterval: (id: Timer) => void;
  setTimeout: (fn: () => void, ms: number) => Timer;
  clearTimeout: (id: Timer) => void;
  markRecentEnded: (runId: string, reason: EndedReason) => Promise<void>;
};

export type RunSession = {
  subscribe(cb: (state: RunSessionState) => void): () => void;
  /** `submitted` only when the caseworker pressed "I submitted it". */
  stop(reason?: 'user' | 'submitted'): Promise<void>;
  /** "Yes" to "Did you submit the form?": stops as a detected submission. */
  confirmSubmission(): Promise<void>;
  /** "No": forgets the submission and watches for the next one. */
  dismissSubmission(): void;
  dispose(): void;
};

const isObject = (v: unknown): v is RunProgress =>
  typeof v === 'object' && v !== null && !Array.isArray(v);

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
  let expiresAt = run.expiresAt;
  let estimated = run.expiresAtEstimated === true;

  const listeners = new Set<(state: RunSessionState) => void>();
  let disposed = false;
  let ended = false;
  let endedReason: EndedReason | undefined;
  let stopping = false;
  let stopFailed = false;
  let stopPromise: Promise<void> | undefined;
  // When the worker first answered "no run" since it last showed this run.
  // Failed and discarded polls neither start nor reset it.
  let inactiveSince: number | undefined;
  let restartingSince: number | undefined;
  // Reading the server once more before ending the run as worker-ended.
  let confirming = false;
  let lastWorker: RunStatus | undefined;
  let serverProgress: RunProgress | null = null;
  let serverPending: PendingInputRequest[] = [];
  let serverPhase: RunPhase = 'working';
  const actionLog = createActionLog();
  // The origin hold as this session saw it (see RunSessionState). A chain is
  // an unbroken run of accepted answers that show a pending origin.
  let lastClearAt: number | undefined;
  let hold:
    | { origin: string; seenAt: number; chainSeenAt: number; known: boolean }
    | undefined;

  let workerInFlight = false;
  let serverInFlight = false;
  let registering = false;

  // The submit watch (see the top of this file).
  let wantArmed = false;
  let watchUsed = false;
  let watchChain: Promise<void> = Promise.resolve();
  let armQueued = false;
  let armInFlight = false;
  // A reset arm is owed: no submission counts until one succeeds.
  let resetPending = false;
  let resetGeneration = 0;
  // Worker polls are numbered when issued. A submission counts only from a
  // poll issued at or after countFrom, a re-arm only from one issued at or
  // after rearmFrom (an older one may predate the arm that just landed).
  let pollSeq = 0;
  let countFrom = Number.POSITIVE_INFINITY;
  let rearmFrom = 0;
  let dismissedAt: number | undefined;
  let seen: { at: number; confirmation?: Confirmation } | undefined;
  let asking = false;
  let askTimer: Timer | undefined;
  let confirmedHow: ConfirmedHow | undefined;
  let confirmation: Confirmation | undefined;

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
      progress: serverProgress,
      pending: ended ? [] : serverPending,
      phase: ended ? 'ended' : serverPhase,
      actions: actionLog.entries(),
      expiresAt,
      secondsLeft: secondsLeft(),
      stopping: stopping && !ended,
    };
    if (endedReason) state.endedReason = endedReason;
    if (!ended && lastWorker?.active && lastWorker.runId === runId) {
      if (lastWorker.pendingOrigin) {
        state.pendingOrigin = lastWorker.pendingOrigin;
      }
      if (hold && hold.origin === lastWorker.pendingOrigin) {
        state.holdSeenAt = hold.seenAt;
        state.holdCountdownKnown = hold.known;
        if (hold.known) {
          state.holdEndsAt = Math.min(
            hold.seenAt + ORIGIN_HOLD_MS,
            hold.chainSeenAt + ORIGIN_HOLD_CAP_MS,
          );
        }
      }
      if (typeof lastWorker.jevFailures === 'number') {
        state.jevFailures = lastWorker.jevFailures;
      }
    }
    if (refreshFailed && !ended) state.refreshFailed = true;
    if (estimated && !ended) state.expiresAtEstimated = true;
    if (stopFailed) state.stopFailed = true;
    if (!ended && !stopping && seen) {
      state.submission = asking
        ? seen.confirmation
          ? { stage: 'ask', confirmation: { ...seen.confirmation } }
          : { stage: 'ask' }
        : { stage: 'checking' };
    }
    if (ended && endedReason === 'submitted' && confirmedHow) {
      state.confirmedHow = confirmedHow;
      if (confirmation) state.confirmation = { ...confirmation };
    }
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

  const clearSubmission = () => {
    seen = undefined;
    asking = false;
    if (askTimer !== undefined) deps.clearTimeout(askTimer);
    askTimer = undefined;
  };

  /** Queues one watch call behind the others; the chain never rejects. */
  const enqueueWatch = (task: () => Promise<void>) => {
    watchChain = watchChain.then(task).catch(() => {});
  };

  const enqueueDisarm = () => {
    if (!watchUsed) return;
    enqueueWatch(async () => {
      // Review came back meanwhile: the reset arm queued after this wins.
      if (wantArmed) return;
      await ignore(() => deps.worker.disarmSubmitWatch(runId));
    });
  };

  const finish = (reason: EndedReason) => {
    ended = true;
    endedReason = reason;
    stopTimers();
    wantArmed = false;
    clearSubmission();
    // Best effort: the worker stop already ended the watch with the run.
    enqueueDisarm();
    void ignore(() => deps.markRecentEnded(runId, reason));
    emit();
  };

  const refreshDue = () =>
    !refreshPromise &&
    !stopping &&
    !ended &&
    expiresAt * 1000 - deps.now() <= REFRESH_BEFORE_MS &&
    (refreshFailedAt === undefined ||
      deps.now() - refreshFailedAt >= REFRESH_RETRY_MS);

  const done = () => stopping || ended || disposed;
  /** No new poll starts, and worker answers are ignored, while this holds. */
  const halted = () => done() || confirming;

  const enqueueArm = () => {
    if (armQueued) return;
    armQueued = true;
    enqueueWatch(async () => {
      armQueued = false;
      // Checked right before the send: a disarm queued meanwhile, a stop or
      // a refresh (which replaces the worker run) wins. The poll arms again.
      if (!wantArmed || done() || refreshPromise) return;
      const reset = resetPending;
      const generation = resetGeneration;
      watchUsed = true;
      armInFlight = true;
      let ok = false;
      try {
        const result = await deps.worker.armSubmitWatch(
          runId,
          reset ? { reset: true } : undefined,
        );
        ok = result.ok === true;
      } catch {
        ok = false;
      } finally {
        armInFlight = false;
        rearmFrom = pollSeq + 1;
      }
      if (ok && reset && generation === resetGeneration && wantArmed) {
        resetPending = false;
        countFrom = pollSeq + 1;
      }
    });
  };

  /** Arms on entering review (with reset) and disarms on leaving it. */
  const syncWatch = () => {
    if (done()) {
      wantArmed = false;
      return;
    }
    const want = serverPhase === 'review';
    if (want === wantArmed) return;
    wantArmed = want;
    clearSubmission();
    if (want) {
      resetPending = true;
      resetGeneration += 1;
      countFrom = Number.POSITIVE_INFINITY;
      enqueueArm();
      return;
    }
    resetPending = false;
    enqueueDisarm();
  };

  /** Stops the run; for a submission, records how and the number first. */
  const beginStop = (
    reason: 'user' | 'submitted',
    how?: ConfirmedHow,
    number?: Confirmation,
  ): Promise<void> => {
    if (stopPromise) return stopPromise;
    if (ended) return Promise.resolve();
    if (reason === 'submitted') {
      confirmedHow = how ?? 'manual';
      confirmation = number ? { ...number } : undefined;
    }
    stopping = true;
    wantArmed = false;
    clearSubmission();
    emit();
    stopPromise = stopNow(reason);
    return stopPromise;
  };

  /** Follows a submission the worker reported (an accepted answer). */
  const takeSubmission = (status: RunStatus, seq: number) => {
    if (!wantArmed || resetPending || seq < countFrom || done()) return;
    // Held on another origin: the Review panel is not shown.
    if (status.pendingOrigin) return;
    const raw = status.submitted;
    if (!raw || typeof raw.at !== 'number' || !Number.isFinite(raw.at)) return;
    if (dismissedAt !== undefined && raw.at <= dismissedAt) return;
    if (!seen || seen.at !== raw.at) {
      clearSubmission();
      const at = raw.at;
      seen = { at };
      askTimer = deps.setTimeout(() => {
        askTimer = undefined;
        if (done() || seen?.at !== at) return;
        asking = true;
        emit();
      }, CONFIRMATION_WAIT_MS);
    }
    // Copied, reduced to {token, kind}: nothing else the status carries.
    const number = reduceConfirmation(raw.confirmation);
    if (!number) return;
    seen.confirmation = number;
    // Under an open prompt the caseworker decides, whatever arrives late:
    // the panel never stops the run while it is asking.
    if (asking) return;
    if (number.kind === 'confirmation') {
      // The number is held before the stop, which ends the worker's read.
      void beginStop('submitted', 'detected', number);
      return;
    }
    // Another label is the read's last word: ask now.
    asking = true;
  };

  const refresh = async () => {
    generation += 1;
    // Nothing settles before the caller has stored refreshPromise, even if a
    // dependency throws synchronously.
    await Promise.resolve();
    try {
      const token = await deps.api.getExtensionToken(runId);
      if (!Number.isFinite(token.expiresAt)) throw new Error('bad expiry');
      if (done()) return;
      await deps.worker.startRun({
        relayUrl: token.relayUrl,
        token: token.extensionToken,
        tabId,
      });
      // The worker now holds the new token: its expiry is the run's.
      expiresAt = token.expiresAt;
      estimated = false;
      refreshFailed = false;
      refreshFailedAt = undefined;
      if (done()) return;
      // A same-run refresh carries the worker's registered values, and the
      // panel holds none itself (the worker registers them), so
      // the count is 0. Screenshots stay refused until this arrives. A
      // failure here is retried by the poll, which sees `needsRegistration`.
      await ignore(() => deps.worker.registrationComplete(runId, 0));
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

  /** The worker answered "no run" long enough: ask the server once more. */
  const workerGone = async () => {
    confirming = true;
    let rowEnded = false;
    try {
      rowEnded = (await deps.api.getRunState(runId)).ended === true;
    } catch {
      rowEnded = false;
    } finally {
      confirming = false;
    }
    if (done()) return;
    if (rowEnded) {
      finish('stopped');
      return;
    }
    endByWorker();
  };

  /** Follows the hold through one accepted answer for this run. */
  const trackHold = (status: RunStatus) => {
    const now = deps.now();
    const pendingOrigin =
      status.active && typeof status.pendingOrigin === 'string'
        ? status.pendingOrigin
        : undefined;
    if (!pendingOrigin) {
      hold = undefined;
      lastClearAt = status.active ? now : undefined;
      return;
    }
    if (hold?.origin === pendingOrigin) return;
    if (hold) {
      // Another origin in the same hold: the bridge restarts its wait.
      hold = { ...hold, origin: pendingOrigin, seenAt: now };
      return;
    }
    const known =
      lastClearAt !== undefined && now - lastClearAt <= HOLD_KNOWN_GAP_MS;
    hold = { origin: pendingOrigin, seenAt: now, chainSeenAt: now, known };
    lastClearAt = undefined;
  };

  const onWorkerStatus = (status: RunStatus, seq: number) => {
    if (status.active && status.runId === runId) trackHold(status);
    else trackHold({ ok: true, active: false });
    if (status.active && status.runId !== runId) {
      lastWorker = status;
      endByWorker();
      return;
    }
    if (!status.active) {
      lastWorker = status;
      const now = deps.now();
      // A restart in progress (same-run startRun or worker resume) is not a
      // gone run: it holds the 7 s clock, up to RESTARTING_MAX_MS in total so
      // a hung restart cannot hide a dead run.
      if (status.restarting) {
        if (restartingSince === undefined) restartingSince = now;
        if (
          now - restartingSince < RESTARTING_MAX_MS &&
          (status.runId === undefined || status.runId === runId)
        ) {
          inactiveSince = undefined;
          emit();
          return;
        }
      } else {
        restartingSince = undefined;
      }
      if (inactiveSince === undefined) inactiveSince = now;
      if (now - inactiveSince >= WORKER_GONE_AFTER_MS) {
        void workerGone();
        return;
      }
      emit();
      return;
    }
    inactiveSince = undefined;
    restartingSince = undefined;
    lastWorker = status;
    if (status.needsRegistration && !registering) {
      // Count 0: the panel holds no values (the worker registers them in
      // the worker). After a worker restart the worker has lost any typed
      // values and this still sends 0, which the worker does not verify.
      registering = true;
      void ignore(() => deps.worker.registrationComplete(runId, 0)).then(() => {
        registering = false;
      });
    }
    if (refreshDue()) refreshPromise = refresh();
    if (
      wantArmed &&
      !armQueued &&
      !armInFlight &&
      seq >= rearmFrom &&
      (status.armed !== true || resetPending)
    ) {
      enqueueArm();
    }
    takeSubmission(status, seq);
    emit();
  };

  const pollWorker = async () => {
    if (workerInFlight || halted()) return;
    workerInFlight = true;
    const issuedUnder = generation;
    pollSeq += 1;
    const seq = pollSeq;
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
    onWorkerStatus(status, seq);
  };

  /** Keeps what a live server row says; a bad shape never throws. */
  const takeRow = (row: RunState) => {
    serverProgress = isObject(row.progress) ? row.progress : null;
    serverPending = Array.isArray(row.pending) ? row.pending : [];
    serverPhase =
      typeof row.phase === 'string' && SERVER_PHASES.has(row.phase)
        ? (row.phase as RunPhase)
        : 'working';
    const actions = serverProgress?.actions;
    if (Array.isArray(actions)) actionLog.add(actions);
  };

  const pollServer = async () => {
    if (serverInFlight || halted()) return;
    serverInFlight = true;
    let rowEnded = false;
    let row: RunState | undefined;
    try {
      row = await deps.api.getRunState(runId);
      rowEnded = row.ended === true;
    } catch {
      row = undefined;
      rowEnded = false;
    } finally {
      serverInFlight = false;
    }
    if (done()) return;
    if (row && !rowEnded) {
      takeRow(row);
      syncWatch();
      emit();
    }
    if (!rowEnded) return;
    // The worker may still hold this run: release the tab's debugger. The
    // stop names this run, so another window's run is left alone.
    void ignore(() => deps.worker.stopRun(runId));
    finish('stopped');
  };

  const stopNow = async (reason: 'user' | 'submitted') => {
    let failed = !(await ignore(() => deps.api.stopRun(runId)));
    // A refresh in flight would restart the worker after its stop.
    if (refreshPromise) await refreshPromise;
    // Scoped to this run: a run another window started is left alone.
    if (!(await ignore(() => deps.worker.stopRun(runId)))) failed = true;
    stopFailed = failed;
    finish(reason);
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
    stop(reason = 'user') {
      if (reason !== 'submitted') return beginStop(reason);
      // "I submitted it" under a prompt that shows a number after another
      // label keeps that number for Confirmed (shown as a number on the
      // page, never as a confirmation number).
      const shown =
        asking && seen?.confirmation?.kind === 'other'
          ? seen.confirmation
          : undefined;
      return beginStop('submitted', 'manual', shown);
    },
    confirmSubmission() {
      if (!seen) return beginStop('submitted', 'manual');
      return beginStop('submitted', 'detected', seen.confirmation);
    },
    dismissSubmission() {
      if (!seen || done()) return;
      dismissedAt = seen.at;
      clearSubmission();
      // The reset makes the worker forget it; until that lands, nothing
      // the worker reports counts.
      resetPending = true;
      resetGeneration += 1;
      countFrom = Number.POSITIVE_INFINITY;
      enqueueArm();
      emit();
    },
    dispose() {
      disposed = true;
      stopTimers();
      clearSubmission();
      // Unmounted in review (the run screen went away): let go of the
      // watch's attach. A page unload does not run React cleanup, so on a
      // real panel close this disarm is effectively never sent and the
      // worker's review attach lasts until the run ends (or a reopened
      // panel arms it again).
      if (wantArmed && !ended) {
        wantArmed = false;
        enqueueDisarm();
      }
      listeners.clear();
    },
  };
};

const MAX_SITE_NAME = 100;

/**
 * The panel reopened while the worker holds a run: rebuild the Running screen
 * when the worker's run is live on the server, on the same origin. The expiry
 * is the worker's reported `exp` (Unix seconds); only a worker that reports none
 * gets a conservative 5 minutes, marked estimated (the session then refreshes
 * the token at its first poll). Any failure or
 * mismatch rebuilds nothing. A worker run whose server row already ended is
 * stopped (by its run id): nothing else would end it before its token expires.
 */
export const rebuildRun = async (deps: {
  api: Pick<AppApi, 'getRunState'>;
  worker: Pick<WorkerApi, 'getRunStatus' | 'stopRun'>;
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
    const runId = status.runId;
    const row = await deps.api.getRunState(runId);
    if (row.ended) {
      await ignore(() => deps.worker.stopRun(runId));
      return null;
    }
    if (row.origin !== status.origin) return null;
    const tabId = status.tabId as number;
    return {
      run: {
        runId: status.runId,
        tabId,
        ...(typeof status.exp === 'number' && Number.isFinite(status.exp)
          ? { expiresAt: status.exp }
          : {
              expiresAt: Math.floor(deps.now() / 1000) + ESTIMATED_LEFT_S,
              expiresAtEstimated: true,
            }),
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
