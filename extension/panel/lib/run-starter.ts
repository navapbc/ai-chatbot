import { inScope } from '../../../lib/relay/policy';
import type { AppApi } from '../../src/app-api';
import { StartInFlightError, TabGoneError, TabMovedError } from './copy';
import { FIRST_CHUNK_TIMEOUT_MS, postEveChatTurn } from './eve-chat-post';
import { addRecent } from './recents';
import type { StartRequest } from './start-request';
import type { WorkerApi } from './worker-api';

// Starts a run from the panel, in the order the server and the agent expect:
//
//   1. the chosen tab is still open and still on the request's origin;
//   2. a fresh run id (a UUID: the chat id is the run id);
//   3. POST /api/eve-chat with `{ id, relay }`. The route creates the run row
//      and the Eve session before it answers, and keeps reading the turn after
//      the client leaves, so the panel reads only until the first chunk;
//   4. an extension token for the run;
//   5. the worker opens the relay socket for the tab;
//   6. registrationComplete(runId, 0): a fresh run has no registered values;
//   7. a Recent entry (best effort).
//
// Steps 4 and 5 follow the first chunk at once: the agent may already be
// calling the browser, and its commands fail until the extension attaches.
//
// A failure once step 3 has begun may leave a run on the server, so the server
// stop route is called (and the worker's stopRun once step 5 succeeded) before
// the error is returned. Errors are typed and carry fixed messages; nothing
// here logs, and the request (the applicant's details) is only serialized into
// the request body, never kept or copied into an error.

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

const ignore = async (task: () => Promise<unknown>) => {
  try {
    await task();
  } catch {
    // Cleanup is best effort; the original error is what the panel shows.
  }
};

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
  let workerStarted = false;
  let expiresAt: number;
  try {
    await createRun(deps, runId, request);
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
    await ignore(() => deps.api.stopRun(runId));
    if (workerStarted) await ignore(() => deps.worker.stopRun(runId));
    return { ok: false, error, runId };
  }

  try {
    const now = deps.now();
    await addRecent(
      { runId, siteName: siteName.slice(0, MAX_SITE_NAME), startedAt: now },
      now,
    );
  } catch {
    // The Recent list is a convenience; the run has started.
  }
  return { ok: true, runId, expiresAt, tabId };
};
