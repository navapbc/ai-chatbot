import { inScope } from '../../../lib/relay/policy';
import { type AppApi, AppApiError } from '../../src/app-api';
import { StartInFlightError, TabGoneError, TabMovedError } from './copy';
import { FIRST_CHUNK_TIMEOUT_MS, postEveChatTurn } from './eve-chat-post';
import { addRecent, markRecentEnded, START_FAILED } from './recents';
import type { StartRequest } from './start-request';
import type { WorkerApi } from './worker-api';

// Starts a run from the panel, in the order the server and the agent expect:
//
//   1. the chosen tab is still open and still on the request's origin;
//   2. a fresh run id (a UUID: the chat id is the run id);
//   3. a Recent entry with no end reason (best effort): written BEFORE the
//      request, so a panel closed mid-start, or a start whose stop is lost,
//      leaves a record the next panel open stops (see orphan-sweep.ts);
//   4. POST /api/eve-chat with `{ id, relay }`. The route creates the run row
//      and the Eve session before it answers, and keeps reading the turn after
//      the client leaves, so the panel reads only until the first chunk;
//   5. an extension token for the run;
//   6. the worker opens the relay socket for the tab;
//   7. registrationComplete(runId, 0): a fresh run has no registered values.
//
// Steps 5 and 6 follow the first chunk at once: the agent may already be
// calling the browser, and its commands fail until the extension attaches.
//
// A failure once step 4 has begun may leave a run on the server, so the server
// stop route is called (and the worker's stopRun once step 6 succeeded) before
// the error is returned. The Recent is marked `start-failed` (hidden on Home)
// only when no run can be left: the server stop answered, or eve-chat refused
// the request with a 4xx (it refuses before it creates the row). Otherwise
// (a timeout, a 5xx, a failed stop) it stays unended for the sweep: the row
// may still be created after the stop's 404. Errors are typed and carry fixed
// messages; nothing here logs, and the request (the applicant's details) is
// only serialized into the request body, never kept or copied into an error.

export type StartDeps = {
  api: AppApi;
  worker: WorkerApi;
  fetch: typeof fetch;
  appOrigin: string;
  newId: () => string;
  now: () => number;
  tabs: {
    get(tabId: number): Promise<{ id?: number; url?: string } | undefined>;
  };
};

export type StartArgs = {
  tabId: number;
  request: StartRequest;
  siteName: string;
};

export type StartResult =
  | { ok: true; runId: string; expiresAt: number; tabId: number }
  | { ok: false; error: unknown; runId?: string };

export { FIRST_CHUNK_TIMEOUT_MS };
const MAX_SITE_NAME = 100;

// One start at a time per panel. Module-level: the shell builds fresh deps for
// every press, so a flag on deps would never see the other start.
let inFlight = false;
// The run id of this panel's start in progress, from step 2 until the start
// returns; the orphan sweep never stops it.
let startingId: string | undefined;

/** The run id this panel is starting right now, if any (an id only). */
export const startingRunId = (): string | undefined => startingId;

const checkTab = async (deps: StartDeps, tabId: number, origin: string) => {
  let tab: { id?: number; url?: string } | undefined;
  try {
    tab = await deps.tabs.get(tabId);
  } catch {
    tab = undefined;
  }
  if (!tab) throw new TabGoneError();
  // The worker's own scope rule: an https URL on exactly this origin.
  if (typeof tab.url !== 'string' || !inScope(tab.url, origin)) {
    throw new TabMovedError();
  }
};

/**
 * Creates the run on the server. Resolves once the first chunk of the turn's
 * stream arrives; the stream is then cancelled, which the route ignores. An
 * error body is never read: its text must not reach the panel.
 */
const createRun = (
  deps: StartDeps,
  runId: string,
  request: StartRequest,
): Promise<void> =>
  postEveChatTurn(deps, JSON.stringify({ id: runId, relay: request }));

/** Runs a cleanup step; true when it succeeded. */
const ignore = async (task: () => Promise<unknown>): Promise<boolean> => {
  try {
    await task();
    return true;
  } catch {
    // Cleanup is best effort; the original error is what the panel shows.
    return false;
  }
};

// eve-chat refuses a first turn with a 4xx before it creates the row (or when
// creating it lost a race to another request). A 5xx may come from a proxy
// while the route is still running, and a network error may be a timeout, so
// neither proves there is no run.
const refusedBeforeCreate = (error: unknown, posted: boolean) =>
  !posted &&
  error instanceof AppApiError &&
  error.status >= 400 &&
  error.status < 500;

export const startRun = async (
  deps: StartDeps,
  args: StartArgs,
): Promise<StartResult> => {
  if (inFlight) return { ok: false, error: new StartInFlightError() };
  inFlight = true;
  try {
    return await startRunNow(deps, args);
  } finally {
    inFlight = false;
    startingId = undefined;
  }
};

const startRunNow = async (
  deps: StartDeps,
  { tabId, request, siteName }: StartArgs,
): Promise<StartResult> => {
  try {
    await checkTab(deps, tabId, request.origin);
  } catch (error) {
    return { ok: false, error };
  }

  const runId = deps.newId();
  startingId = runId;
  try {
    const now = deps.now();
    await addRecent(
      { runId, siteName: siteName.slice(0, MAX_SITE_NAME), startedAt: now },
      now,
    );
  } catch {
    // The Recent list is a convenience; the start goes on without it.
  }

  let posted = false;
  let workerStarted = false;
  let expiresAt: number;
  try {
    await createRun(deps, runId, request);
    posted = true;
    const token = await deps.api.getExtensionToken(runId);
    expiresAt = token.expiresAt;
    await deps.worker.startRun({
      relayUrl: token.relayUrl,
      token: token.extensionToken,
      tabId,
    });
    workerStarted = true;
    await deps.worker.registrationComplete(runId, 0);
  } catch (error) {
    // Server first, then the worker (P4c-f).
    const stopped = await ignore(() => deps.api.stopRun(runId));
    if (workerStarted) await ignore(() => deps.worker.stopRun(runId));
    if (stopped || refusedBeforeCreate(error, posted)) {
      await markRecentEnded(runId, START_FAILED);
    }
    return { ok: false, error, runId };
  }

  return { ok: true, runId, expiresAt, tabId };
};
