import { AppApiError } from '../../src/app-api';
import type { RunStatus } from '../../src/bridge';
import { type Recent, SWEPT } from './recents';

// Stops server runs that nothing holds, once per panel open (plan 4d, I1/I2).
//
// A run reaches Recents (no end reason) before its first-turn POST, and gets
// an end reason only from a panel that watched it end. So a recent with no
// end reason whose run the worker does not hold is a run whose panel was
// closed: mid-start (the POST or its stop was lost), or while the worker ended
// it by itself. Its server row stays live (the agent keeps minting tokens and
// counts against the hourly cap) until the idempotent stop route is called.
//
// Never stopped, because it may be a healthy run:
// - any recent while the worker cannot be read, or says `restarting` (a token
//   refresh or a service-worker resume: `run` is briefly empty there). The
//   whole sweep is skipped;
// - the run the worker holds (re-read before every stop, so a run another
//   panel window starts during the sweep is seen);
// - a recent younger than 90 s: another window may be mid-start (the start
//   is bounded well under that: 15 s POST, 10 s token, then the worker);
// - this panel's own start in progress.
//
// A 404 (no row, or not this user's) is final only after 5 minutes: before
// that the row may still be created by a slow first turn. Other failures are
// left for the next open. At most 10 stop calls; nothing logged, nothing
// thrown, ids only.

export const SWEEP_MIN_AGE_MS = 90_000;
export const NOT_FOUND_FINAL_AFTER_MS = 5 * 60_000;
export const MAX_SWEEP_STOPS = 10;

export type SweepDeps = {
  api: { stopRun(runId: string): Promise<unknown> };
  worker: { getRunStatus(): Promise<RunStatus> };
  loadRecents: (now: number) => Promise<Recent[]>;
  markRecentEnded: (runId: string, reason: string) => Promise<void>;
  now: () => number;
  /** This panel's start in progress (see run-starter), never stopped. */
  startingRunId?: () => string | undefined;
};

/**
 * The worker's status, or null when it cannot be trusted to say a run is not
 * held: unreadable, malformed, or `restarting`.
 */
const readStatus = async (
  deps: SweepDeps,
): Promise<{ heldRunId?: string } | null> => {
  let status: RunStatus;
  try {
    status = await deps.worker.getRunStatus();
  } catch {
    return null;
  }
  if (typeof status !== 'object' || status === null) return null;
  if (status.ok !== true || typeof status.active !== 'boolean') return null;
  if (status.restarting === true) return null;
  if (!status.active) return {};
  // An active run with no id cannot be compared, so nothing is stopped.
  if (typeof status.runId !== 'string' || status.runId.length === 0) {
    return null;
  }
  return { heldRunId: status.runId };
};

export const sweepStaleRecents = async (deps: SweepDeps): Promise<void> => {
  try {
    if ((await readStatus(deps)) === null) return;
    const now = deps.now();
    const recents = await deps.loadRecents(now);
    let stops = 0;
    for (const recent of recents) {
      if (stops >= MAX_SWEEP_STOPS) return;
      if (recent.endedReason) continue;
      const age = now - recent.startedAt;
      // A start time in the future gives a negative age: skipped too.
      if (age < SWEEP_MIN_AGE_MS) continue;
      if (deps.startingRunId?.() === recent.runId) continue;
      // Re-read right before the stop: the worker may have taken this run
      // (another window) or begun a refresh since the last read.
      const status = await readStatus(deps);
      if (status === null) return;
      if (status.heldRunId === recent.runId) continue;
      stops += 1;
      try {
        await deps.api.stopRun(recent.runId);
      } catch (error) {
        const final =
          error instanceof AppApiError &&
          error.status === 404 &&
          age > NOT_FOUND_FINAL_AFTER_MS;
        if (!final) continue;
      }
      await deps.markRecentEnded(recent.runId, SWEPT);
    }
  } catch {
    // Best effort: the sweep never changes what the panel shows.
  }
};
