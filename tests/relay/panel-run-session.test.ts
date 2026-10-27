import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  createRunSession,
  GRACE_MS,
  REFRESH_BEFORE_MS,
  REFRESH_RETRY_MS,
  type RunSessionDeps,
  type RunSessionState,
  rebuildRun,
  SERVER_POLL_MS,
  type SessionRun,
  WORKER_POLL_MS,
} from '../../extension/panel/lib/run-session';
import { createControllerLifecycle } from '../../extension/panel/lib/search-controller';
import {
  AppApiError,
  type ExtensionToken,
  type RunState,
} from '../../extension/src/app-api';
import type { RunStatus } from '../../extension/src/bridge';
import { WorkerError } from '../../extension/panel/lib/worker-api';

// The run session against fakes, on vitest's fake clock. The clock is in
// milliseconds near 1.8e12 and `expiresAt` in SECONDS near 1.8e9, so a
// seconds/milliseconds mix-up is off by a factor of 1000 and fails loudly.

const RUN_ID = '11111111-2222-4333-8444-555555555555';
const OTHER_RUN = '99999999-2222-4333-8444-555555555555';
const ORIGIN = 'https://example.gov';
const NOW = 1_800_000_000_000;
const EXPIRES = NOW / 1000 + 1800;

type Deferred<T> = {
  promise: Promise<T>;
  resolve: (v: T) => void;
  reject: (e: unknown) => void;
};
const deferred = <T>(): Deferred<T> => {
  let resolve!: (v: T) => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
};

const ours = (extra: Partial<RunStatus> = {}): RunStatus => ({
  ok: true,
  active: true,
  runId: RUN_ID,
  origin: ORIGIN,
  attached: true,
  needsRegistration: false,
  tabId: 7,
  ...extra,
});
const inactive: RunStatus = { ok: true, active: false };

const liveRow = (extra: Partial<RunState> = {}): RunState => ({
  runId: RUN_ID,
  origin: ORIGIN,
  ended: false,
  endedAt: null,
  progress: null,
  pending: [],
  ...extra,
});

const setup = (run: Partial<SessionRun> = {}) => {
  const calls: string[] = [];
  const plan = {
    status: async (): Promise<RunStatus> => ours(),
    row: async (): Promise<RunState> => liveRow(),
    token: async (): Promise<ExtensionToken> => ({
      extensionToken: 'tok-new',
      relayUrl: 'wss://relay.example.org',
      expiresAt: Math.floor(Date.now() / 1000) + 1800,
    }),
    workerStart: async (): Promise<void> => {},
    registration: async (): Promise<void> => {},
    serverStop: async () => ({ ended: true, alreadyEnded: false }),
    workerStop: async (): Promise<void> => {},
  };
  const api = {
    getRunState: vi.fn(async (id: string) => {
      calls.push(`row:${id}`);
      return plan.row();
    }),
    stopRun: vi.fn(async (id: string) => {
      calls.push(`serverStop:${id}`);
      return plan.serverStop();
    }),
    getExtensionToken: vi.fn(async (id: string) => {
      calls.push(`token:${id}`);
      return plan.token();
    }),
  };
  const worker = {
    getRunStatus: vi.fn(async () => {
      calls.push('status');
      return plan.status();
    }),
    stopRun: vi.fn(async () => {
      calls.push('workerStop');
      return plan.workerStop();
    }),
    startRun: vi.fn(async (args: { token: string; tabId: number }) => {
      calls.push(`workerStart:${args.tabId}:${args.token}`);
      return plan.workerStart();
    }),
    registrationComplete: vi.fn(async (id: string, count: number) => {
      calls.push(`registration:${id}:${count}`);
      return plan.registration();
    }),
  };
  const markRecentEnded = vi.fn(async (id: string, reason: string) => {
    calls.push(`recent:${id}:${reason}`);
  });
  const deps: RunSessionDeps = {
    api,
    worker,
    now: () => Date.now(),
    setInterval: (fn, ms) => setInterval(fn, ms),
    clearInterval: (id) => clearInterval(id),
    setTimeout: (fn, ms) => setTimeout(fn, ms),
    clearTimeout: (id) => clearTimeout(id),
    markRecentEnded,
  };
  const session = createRunSession(deps, {
    runId: RUN_ID,
    expiresAt: EXPIRES,
    tabId: 7,
    ...run,
  });
  const states: RunSessionState[] = [];
  session.subscribe((s) => states.push(s));
  const last = () => states[states.length - 1];
  return {
    calls,
    plan,
    api,
    worker,
    markRecentEnded,
    session,
    states,
    last,
  };
};

const tick = (ms: number) => vi.advanceTimersByTimeAsync(ms);
const count = (calls: string[], name: string) =>
  calls.filter((c) => c === name || c.startsWith(`${name}:`)).length;

beforeEach(() => {
  vi.useFakeTimers({ now: NOW });
});
afterEach(() => {
  vi.useRealTimers();
});

describe('run session: polling', () => {
  it('polls the worker every 2 s and the server every 5 s', async () => {
    const t = setup();
    expect(WORKER_POLL_MS).toBe(2000);
    expect(SERVER_POLL_MS).toBe(5000);
    // Nothing runs synchronously at creation.
    expect(t.calls).toEqual([]);
    await tick(1999);
    expect(count(t.calls, 'status')).toBe(0);
    await tick(1);
    expect(count(t.calls, 'status')).toBe(1);
    await tick(3000);
    expect(count(t.calls, 'status')).toBe(2);
    expect(count(t.calls, 'row')).toBe(1);
    await tick(5000);
    expect(count(t.calls, 'status')).toBe(5);
    expect(count(t.calls, 'row')).toBe(2);
    expect(t.last()?.status).toBe('running');
    t.session.dispose();
  });

  it('skips a poll while the previous one of the same kind is in flight', async () => {
    const t = setup();
    const status = deferred<RunStatus>();
    const row = deferred<RunState>();
    t.plan.status = () => status.promise;
    t.plan.row = () => row.promise;
    await tick(20_000);
    expect(count(t.calls, 'status')).toBe(1);
    expect(count(t.calls, 'row')).toBe(1);
    status.resolve(ours());
    row.resolve(liveRow());
    t.plan.status = async () => ours();
    t.plan.row = async () => liveRow();
    await tick(2000);
    expect(count(t.calls, 'status')).toBe(2);
    await tick(3000);
    expect(count(t.calls, 'row')).toBe(2);
    t.session.dispose();
  });

  it('shows pendingOrigin and jevFailures from the worker status', async () => {
    const t = setup();
    t.plan.status = async () =>
      ours({ pendingOrigin: 'https://login.example.gov', jevFailures: 3 });
    await tick(2000);
    expect(t.last()).toMatchObject({
      status: 'running',
      pendingOrigin: 'https://login.example.gov',
      jevFailures: 3,
    });
    t.plan.status = async () => ours();
    await tick(2000);
    expect(t.last()?.pendingOrigin).toBeUndefined();
    t.session.dispose();
  });

  it('transient errors from either poll never end the run', async () => {
    const t = setup();
    await tick(2000);
    t.plan.status = async () => {
      throw new WorkerError('worker_unreachable');
    };
    t.plan.row = async () => {
      throw new AppApiError('network', 0);
    };
    await tick(30_000);
    expect(t.last()?.status).toBe('running');
    t.plan.row = async () => {
      throw new AppApiError('not_found', 404);
    };
    await tick(10_000);
    expect(t.last()?.status).toBe('running');
    expect(t.api.stopRun).not.toHaveBeenCalled();
    expect(t.worker.stopRun).not.toHaveBeenCalled();
    expect(t.markRecentEnded).not.toHaveBeenCalled();
    t.session.dispose();
  });

  it('subscribe answers with the current state at once and can unsubscribe', async () => {
    const t = setup();
    const seen: RunSessionState[] = [];
    const off = t.session.subscribe((s) => seen.push(s));
    expect(seen).toEqual([
      {
        status: 'running',
        secondsLeft: 1800,
        stopping: false,
      },
    ]);
    off();
    t.plan.status = async () => ours({ jevFailures: 1 });
    await tick(2000);
    expect(seen).toHaveLength(1);
    t.session.dispose();
  });

  it('calls registrationComplete(runId, 0) when the worker restarted and needs it', async () => {
    const t = setup();
    t.plan.status = async () => ours({ needsRegistration: true });
    const reg = deferred<void>();
    t.plan.registration = () => reg.promise;
    await tick(2000);
    await tick(2000);
    // One at a time.
    expect(t.calls.filter((c) => c.startsWith('registration:'))).toEqual([
      `registration:${RUN_ID}:0`,
    ]);
    reg.resolve();
    t.plan.status = async () => ours();
    await tick(2000);
    expect(count(t.calls, 'registration')).toBe(1);
    t.session.dispose();
  });
});

describe('run session: worker-initiated end', () => {
  it('worker inactive on two polls in a row after the grace: one server stop, worker-ended, recents', async () => {
    const t = setup();
    await tick(2000);
    t.plan.status = async () => inactive;
    await tick(2000);
    // One inactive answer may be a gap (a same-run restart from another
    // panel, a worker cold start): not an end yet.
    expect(t.last()?.status).toBe('running');
    expect(t.api.stopRun).not.toHaveBeenCalled();
    await tick(2000);
    expect(t.last()).toMatchObject({
      status: 'ended',
      endedReason: 'worker-ended',
      stopping: false,
    });
    await tick(20_000);
    expect(t.api.stopRun).toHaveBeenCalledTimes(1);
    expect(t.api.stopRun).toHaveBeenCalledWith(RUN_ID);
    expect(t.worker.stopRun).not.toHaveBeenCalled();
    expect(t.markRecentEnded).toHaveBeenCalledTimes(1);
    expect(t.markRecentEnded).toHaveBeenCalledWith(RUN_ID, 'worker-ended');
    // Polling stopped once ended.
    const polls = count(t.calls, 'status');
    await tick(10_000);
    expect(count(t.calls, 'status')).toBe(polls);
    t.session.dispose();
  });

  it('a server stop that fails is still called only once', async () => {
    const t = setup();
    t.plan.serverStop = async () => {
      throw new AppApiError('network', 0);
    };
    t.plan.status = async () => inactive;
    await tick(GRACE_MS + 10_000);
    expect(t.last()?.status).toBe('ended');
    expect(t.api.stopRun).toHaveBeenCalledTimes(1);
    expect(t.calls.filter((c) => c.startsWith('serverStop'))).toHaveLength(1);
    t.session.dispose();
  });

  it('replaced by another window (different runId): server stop once, never the worker stop', async () => {
    const t = setup();
    await tick(2000);
    t.plan.status = async () => ours({ runId: OTHER_RUN });
    await tick(2000);
    expect(t.last()).toMatchObject({
      status: 'ended',
      endedReason: 'worker-ended',
    });
    await tick(10_000);
    expect(t.api.stopRun).toHaveBeenCalledTimes(1);
    expect(t.api.stopRun).toHaveBeenCalledWith(RUN_ID);
    expect(t.worker.stopRun).not.toHaveBeenCalled();
    expect(t.markRecentEnded).toHaveBeenCalledWith(RUN_ID, 'worker-ended');
    t.session.dispose();
  });

  it('a different runId ends the run even within the grace', async () => {
    const t = setup();
    t.plan.status = async () => ours({ runId: OTHER_RUN });
    await tick(2000);
    expect(t.last()?.status).toBe('ended');
    expect(t.worker.stopRun).not.toHaveBeenCalled();
    t.session.dispose();
  });

  it('grace: active:false within 5 s is not an end; after it, it is', async () => {
    expect(GRACE_MS).toBe(5000);
    const t = setup();
    t.plan.status = async () => inactive;
    await tick(4000);
    expect(count(t.calls, 'status')).toBe(2);
    expect(t.last()?.status).toBe('running');
    expect(t.api.stopRun).not.toHaveBeenCalled();
    await tick(2000);
    expect(t.last()?.status).toBe('ended');
    expect(t.api.stopRun).toHaveBeenCalledTimes(1);
    t.session.dispose();
  });

  it('a single inactive answer between active ones never ends the run', async () => {
    const t = setup();
    const answers = [ours(), inactive, ours(), inactive, ours(), inactive];
    t.plan.status = async () => answers.shift() ?? ours();
    await tick(14_000);
    expect(count(t.calls, 'status')).toBe(7);
    expect(t.last()?.status).toBe('running');
    expect(t.api.stopRun).not.toHaveBeenCalled();
    t.session.dispose();
  });

  it('a failed poll neither counts toward nor resets the inactive streak', async () => {
    const t = setup();
    const answers: (RunStatus | Error)[] = [
      ours(),
      inactive,
      new WorkerError('worker_unreachable'),
      inactive,
    ];
    t.plan.status = async () => {
      const next = answers.shift() ?? ours();
      if (next instanceof Error) throw next;
      return next;
    };
    await tick(6000);
    expect(t.last()?.status).toBe('running');
    await tick(2000);
    expect(t.last()?.status).toBe('ended');
    expect(t.api.stopRun).toHaveBeenCalledTimes(1);
    t.session.dispose();
  });
});

describe('run session: the server ends it', () => {
  it('server ended: true ends the run without calling the server stop', async () => {
    const t = setup();
    await tick(2000);
    t.plan.row = async () =>
      liveRow({ ended: true, endedAt: '2027-01-15T08:00:00.000Z' });
    await tick(3000);
    expect(t.last()).toMatchObject({
      status: 'ended',
      endedReason: 'stopped',
    });
    await tick(20_000);
    expect(t.api.stopRun).not.toHaveBeenCalled();
    expect(t.markRecentEnded).toHaveBeenCalledWith(RUN_ID, 'stopped');
    // The worker still held this run, so its debugger is released.
    expect(t.worker.stopRun).toHaveBeenCalledTimes(1);
    t.session.dispose();
  });

  it('server ended while the worker shows another run: the worker is left alone', async () => {
    const t = setup();
    t.plan.status = async () => {
      throw new WorkerError('worker_unreachable');
    };
    t.plan.row = async () => liveRow({ ended: true, endedAt: 'x' });
    await tick(5000);
    expect(t.last()?.status).toBe('ended');
    expect(t.worker.stopRun).not.toHaveBeenCalled();
    t.session.dispose();
  });
});

describe('run session: stop', () => {
  it('calls the server stop, then the worker stop, then ends with reason user', async () => {
    const t = setup();
    await tick(2000);
    const done = t.session.stop();
    expect(t.last()).toMatchObject({ status: 'running', stopping: true });
    await done;
    expect(t.calls.filter((c) => /Stop|recent/.test(c))).toEqual([
      `serverStop:${RUN_ID}`,
      'workerStop',
      `recent:${RUN_ID}:user`,
    ]);
    expect(t.last()).toEqual({
      status: 'ended',
      endedReason: 'user',
      secondsLeft: expect.any(Number),
      stopping: false,
    });
    t.session.dispose();
  });

  it('a second Stop while stopping returns the same promise; Stop after ended is a no-op', async () => {
    const t = setup();
    const server = deferred<{ ended: boolean; alreadyEnded: boolean }>();
    t.plan.serverStop = () => server.promise;
    const a = t.session.stop();
    const b = t.session.stop();
    expect(b).toBe(a);
    server.resolve({ ended: true, alreadyEnded: false });
    await a;
    await t.session.stop();
    expect(t.api.stopRun).toHaveBeenCalledTimes(1);
    expect(t.worker.stopRun).toHaveBeenCalledTimes(1);
    expect(t.markRecentEnded).toHaveBeenCalledTimes(1);
    t.session.dispose();
  });

  it('records stopFailed when either call fails, and still ends', async () => {
    const t = setup();
    t.plan.serverStop = async () => {
      throw new AppApiError('network', 0);
    };
    await t.session.stop();
    expect(t.worker.stopRun).toHaveBeenCalledTimes(1);
    expect(t.last()).toMatchObject({
      status: 'ended',
      endedReason: 'user',
      stopFailed: true,
    });

    const u = setup();
    u.plan.workerStop = async () => {
      throw new WorkerError('worker_unreachable');
    };
    await u.session.stop();
    expect(u.last()).toMatchObject({ status: 'ended', stopFailed: true });
    t.session.dispose();
    u.session.dispose();
  });

  it('Stop after a worker-initiated end calls nothing more', async () => {
    const t = setup();
    t.plan.status = async () => ours({ runId: OTHER_RUN });
    await tick(2000);
    await t.session.stop();
    expect(t.api.stopRun).toHaveBeenCalledTimes(1);
    expect(t.worker.stopRun).not.toHaveBeenCalled();
    t.session.dispose();
  });

  it('polls landing while stopping do not end the run as worker-ended', async () => {
    const t = setup();
    await tick(2000);
    const server = deferred<{ ended: boolean; alreadyEnded: boolean }>();
    t.plan.serverStop = () => server.promise;
    const done = t.session.stop();
    t.plan.status = async () => inactive;
    t.plan.row = async () => liveRow({ ended: true, endedAt: 'x' });
    await tick(6000);
    server.resolve({ ended: true, alreadyEnded: false });
    await done;
    expect(t.last()?.endedReason).toBe('user');
    expect(t.markRecentEnded).toHaveBeenCalledTimes(1);
    expect(t.api.stopRun).toHaveBeenCalledTimes(1);
    t.session.dispose();
  });
});

describe('run session: token refresh', () => {
  const SOON = NOW / 1000 + 6 * 60; // 6 minutes left at creation

  it('refreshes at 5 minutes left: token, same-run startRun on the tab, registrationComplete(runId, 0)', async () => {
    expect(REFRESH_BEFORE_MS).toBe(5 * 60_000);
    const t = setup({ expiresAt: SOON });
    await tick(58_000);
    expect(t.api.getExtensionToken).not.toHaveBeenCalled();
    expect(t.last()?.secondsLeft).toBe(6 * 60 - 58);
    await tick(2000);
    // Exactly 5 minutes left at this poll.
    await vi.waitFor(() =>
      expect(
        t.calls.filter((c) => /token|workerStart|registration/.test(c)),
      ).toEqual([
        `token:${RUN_ID}`,
        'workerStart:7:tok-new',
        `registration:${RUN_ID}:0`,
      ]),
    );
    await tick(0);
    // The new expiry (30 minutes from the refresh) replaces the old one.
    expect(t.last()?.secondsLeft).toBe(1800);
    expect(t.last()?.refreshFailed).toBeUndefined();
    await tick(60_000);
    expect(t.api.getExtensionToken).toHaveBeenCalledTimes(1);
    t.session.dispose();
  });

  it('a refusal sets refreshFailed and retries at most once per minute', async () => {
    expect(REFRESH_RETRY_MS).toBe(60_000);
    const t = setup({ expiresAt: NOW / 1000 + 4 * 60 });
    t.plan.workerStart = async () => {
      throw new WorkerError('tab_out_of_scope');
    };
    await tick(2000);
    expect(t.worker.startRun).toHaveBeenCalledTimes(1);
    expect(t.last()).toMatchObject({ status: 'running', refreshFailed: true });
    expect(t.worker.registrationComplete).not.toHaveBeenCalled();
    await tick(58_000);
    expect(t.worker.startRun).toHaveBeenCalledTimes(1);
    await tick(2000);
    expect(t.worker.startRun).toHaveBeenCalledTimes(2);
    // A later success clears the notice.
    t.plan.workerStart = async () => {};
    await tick(60_000);
    expect(t.worker.startRun).toHaveBeenCalledTimes(3);
    expect(t.last()?.refreshFailed).toBeUndefined();
    expect(t.last()?.secondsLeft).toBe(1800);
    t.session.dispose();
  });

  it('a token failure is a refusal like any other', async () => {
    const t = setup({ expiresAt: NOW / 1000 + 60 });
    t.plan.token = async () => {
      throw new AppApiError('forbidden', 403);
    };
    await tick(2000);
    expect(t.last()?.refreshFailed).toBe(true);
    expect(t.worker.startRun).not.toHaveBeenCalled();
    t.session.dispose();
  });

  it('only refreshes while the worker shows this run', async () => {
    const t = setup({ expiresAt: NOW / 1000 + 60 });
    t.plan.status = async () => inactive;
    await tick(4000);
    expect(t.api.getExtensionToken).not.toHaveBeenCalled();
    t.session.dispose();
  });

  it('a poll landing mid-refresh (worker briefly inactive) does not end the run', async () => {
    const t = setup({ expiresAt: NOW / 1000 + 60 });
    const start = deferred<void>();
    t.plan.workerStart = () => start.promise;
    await tick(2000);
    expect(t.worker.startRun).toHaveBeenCalledTimes(1);
    // The bridge clears its run before beginning the new one.
    t.plan.status = async () => inactive;
    await tick(6000);
    expect(t.last()?.status).toBe('running');
    expect(t.api.stopRun).not.toHaveBeenCalled();
    t.plan.status = async () => ours();
    start.resolve();
    await tick(2000);
    expect(t.last()?.status).toBe('running');
    expect(t.api.stopRun).not.toHaveBeenCalled();
    t.session.dispose();
  });

  it('a poll issued during the refresh and answered after it is ignored', async () => {
    const t = setup({ expiresAt: NOW / 1000 + 60 });
    const token = deferred<ExtensionToken>();
    t.plan.token = () => token.promise;
    await tick(2000);
    expect(t.api.getExtensionToken).toHaveBeenCalledTimes(1);
    // Issued while the refresh runs, answered (inactive) after it finished.
    const late = deferred<RunStatus>();
    t.plan.status = () => late.promise;
    await tick(2000);
    expect(count(t.calls, 'status')).toBe(2);
    token.resolve({
      extensionToken: 'tok-new',
      relayUrl: 'wss://relay.example.org',
      expiresAt: Math.floor(Date.now() / 1000) + 1800,
    });
    await tick(0);
    expect(t.worker.registrationComplete).toHaveBeenCalledTimes(1);
    late.resolve(inactive);
    await tick(0);
    expect(t.last()?.status).toBe('running');
    expect(t.api.stopRun).not.toHaveBeenCalled();
    t.session.dispose();
  });

  it('Stop during a refresh waits for it and the refresh does not restart the worker', async () => {
    const t = setup({ expiresAt: NOW / 1000 + 60 });
    const token = deferred<ExtensionToken>();
    t.plan.token = () => token.promise;
    await tick(2000);
    const done = t.session.stop();
    await tick(0);
    token.resolve({
      extensionToken: 'tok-new',
      relayUrl: 'wss://relay.example.org',
      expiresAt: NOW / 1000 + 1800,
    });
    await done;
    expect(t.worker.startRun).not.toHaveBeenCalled();
    expect(t.calls.filter((c) => /Stop|workerStart/.test(c))).toEqual([
      `serverStop:${RUN_ID}`,
      'workerStop',
    ]);
    t.session.dispose();
  });

  it('Stop while the worker restart is in flight stops the worker after it', async () => {
    const t = setup({ expiresAt: NOW / 1000 + 60 });
    const start = deferred<void>();
    t.plan.workerStart = () => start.promise;
    await tick(2000);
    const done = t.session.stop();
    await tick(0);
    expect(t.calls).toContain(`serverStop:${RUN_ID}`);
    expect(t.calls).not.toContain('workerStop');
    start.resolve();
    await done;
    const order = t.calls.filter((c) => /Stop|workerStart/.test(c));
    expect(order).toEqual([
      'workerStart:7:tok-new',
      `serverStop:${RUN_ID}`,
      'workerStop',
    ]);
    t.session.dispose();
  });

  it('an estimated expiry refreshes at the first poll and says so until then', async () => {
    const t = setup({
      expiresAt: NOW / 1000 + 300,
      expiresAtEstimated: true,
    });
    expect(t.last()).toMatchObject({
      expiresAtEstimated: true,
      secondsLeft: 300,
    });
    await tick(2000);
    expect(t.api.getExtensionToken).toHaveBeenCalledTimes(1);
    expect(t.last()?.expiresAtEstimated).toBeUndefined();
    expect(t.last()?.secondsLeft).toBe(1800);
    t.session.dispose();
  });
});

describe('run session: secondsLeft', () => {
  it('converts the SECONDS expiry against the millisecond clock', () => {
    const t = setup();
    expect(t.last()?.secondsLeft).toBe(1800);
    t.session.dispose();
    const u = setup({ expiresAt: NOW / 1000 + 90.5 });
    expect(u.last()?.secondsLeft).toBe(90);
    u.session.dispose();
  });

  it('is 0 past expiry, never negative', async () => {
    const t = setup({ expiresAt: NOW / 1000 - 30 });
    expect(t.last()?.secondsLeft).toBe(0);
    t.session.dispose();
  });

  it('counts down as polls land', async () => {
    const t = setup();
    await tick(10_000);
    expect(t.last()?.secondsLeft).toBe(1790);
    t.session.dispose();
  });
});

describe('run session: no applicant data', () => {
  const STATE_KEYS = new Set([
    'status',
    'endedReason',
    'pendingOrigin',
    'jevFailures',
    'refreshFailed',
    'expiresAtEstimated',
    'secondsLeft',
    'stopping',
    'stopFailed',
  ]);

  it('every emitted state holds only the fixed keys', async () => {
    const t = setup({ expiresAt: NOW / 1000 + 60 });
    t.plan.status = async () =>
      ({
        ...ours({ pendingOrigin: 'https://login.example.gov', jevFailures: 2 }),
        // A worker answering extra fields: none of them may pass through.
        participant: 'Zelda Quill',
        token: 'tok-secret',
      }) as RunStatus;
    t.plan.row = async () =>
      ({ ...liveRow(), pending: [{ prompt: 'Zelda Quill' }] }) as RunState;
    await tick(10_000);
    await t.session.stop();
    expect(t.states.length).toBeGreaterThan(3);
    for (const state of t.states) {
      for (const key of Object.keys(state)) expect(STATE_KEYS).toContain(key);
    }
    expect(JSON.stringify(t.states)).not.toMatch(/Zelda|tok-/);
    t.session.dispose();
  });

  it('rebuildRun returns only ids, the origin and the hostname', async () => {
    const rebuilt = await rebuildRun({
      api: {
        getRunState: async () =>
          ({ ...liveRow(), pending: [{ prompt: 'Zelda Quill' }] }) as RunState,
      },
      worker: {
        getRunStatus: async () =>
          ({ ...ours(), participant: 'Zelda Quill' }) as RunStatus,
      },
      now: () => NOW,
    });
    expect(Object.keys(rebuilt?.run ?? {}).sort()).toEqual([
      'expiresAt',
      'expiresAtEstimated',
      'runId',
      'tabId',
    ]);
    expect(Object.keys(rebuilt?.target ?? {}).sort()).toEqual([
      'origin',
      'siteName',
      'tabId',
    ]);
    expect(JSON.stringify(rebuilt)).not.toContain('Zelda');
  });
});

describe('run session: dispose', () => {
  it('leaves no timers and sends no callback afterwards', async () => {
    const t = setup();
    const status = deferred<RunStatus>();
    t.plan.status = () => status.promise;
    await tick(2000);
    const before = t.states.length;
    t.session.dispose();
    expect(vi.getTimerCount()).toBe(0);
    status.resolve(inactive);
    await tick(30_000);
    expect(t.states.length).toBe(before);
    expect(t.api.stopRun).not.toHaveBeenCalled();
    expect(count(t.calls, 'status')).toBe(1);
  });

  it('a mount/unmount/mount (StrictMode) leaves one live session', async () => {
    const t = setup();
    t.session.dispose();
    const made: { dispose(): void }[] = [];
    const deps = {
      api: t.api,
      worker: t.worker,
      now: () => Date.now(),
      setInterval: (fn: () => void, ms: number) => setInterval(fn, ms),
      clearInterval: (id: ReturnType<typeof setInterval>) => clearInterval(id),
      setTimeout: (fn: () => void, ms: number) => setTimeout(fn, ms),
      clearTimeout: (id: ReturnType<typeof setTimeout>) => clearTimeout(id),
      markRecentEnded: t.markRecentEnded,
    } satisfies RunSessionDeps;
    const lifecycle = createControllerLifecycle(() => {
      const s = createRunSession(deps, {
        runId: RUN_ID,
        expiresAt: EXPIRES,
        tabId: 7,
      });
      made.push(s);
      return s;
    });
    const first = lifecycle.mount();
    const firstStates: RunSessionState[] = [];
    first.subscribe((s) => firstStates.push(s));
    lifecycle.unmount();
    const second = lifecycle.mount();
    const secondStates: RunSessionState[] = [];
    second.subscribe((s) => secondStates.push(s));
    expect(made).toHaveLength(2);
    // Nothing ran for the disposed one: no I/O at creation.
    expect(t.calls).toEqual([]);
    await tick(2000);
    expect(count(t.calls, 'status')).toBe(1);
    expect(firstStates).toHaveLength(1);
    expect(secondStates.length).toBeGreaterThan(1);
    lifecycle.unmount();
    expect(vi.getTimerCount()).toBe(0);
  });
});

describe('rebuildRun', () => {
  const deps = (
    status: () => Promise<RunStatus>,
    row: () => Promise<RunState>,
  ) => ({
    api: { getRunState: vi.fn(async (_id: string) => row()) },
    worker: { getRunStatus: vi.fn(status) },
    now: () => NOW,
  });

  it('rebuilds a live run with an estimated expiry five minutes out', async () => {
    const d = deps(
      async () => ours(),
      async () => liveRow(),
    );
    expect(await rebuildRun(d)).toEqual({
      run: {
        runId: RUN_ID,
        tabId: 7,
        expiresAt: NOW / 1000 + 300,
        expiresAtEstimated: true,
      },
      target: { tabId: 7, origin: ORIGIN, siteName: 'example.gov' },
    });
    expect(d.api.getRunState).toHaveBeenCalledWith(RUN_ID);
  });

  it('an inactive worker rebuilds nothing and asks the server nothing', async () => {
    const d = deps(
      async () => inactive,
      async () => liveRow(),
    );
    expect(await rebuildRun(d)).toBeNull();
    expect(d.api.getRunState).not.toHaveBeenCalled();
  });

  it('an ended server row rebuilds nothing', async () => {
    const d = deps(
      async () => ours(),
      async () => liveRow({ ended: true, endedAt: 'x' }),
    );
    expect(await rebuildRun(d)).toBeNull();
  });

  it('an origin mismatch rebuilds nothing', async () => {
    const d = deps(
      async () => ours(),
      async () => liveRow({ origin: 'https://other.example.gov' }),
    );
    expect(await rebuildRun(d)).toBeNull();
  });

  it('errors and a status without tabId rebuild nothing', async () => {
    expect(
      await rebuildRun(
        deps(
          async () => {
            throw new WorkerError('worker_unreachable');
          },
          async () => liveRow(),
        ),
      ),
    ).toBeNull();
    expect(
      await rebuildRun(
        deps(
          async () => ours(),
          async () => {
            throw new AppApiError('not_found', 404);
          },
        ),
      ),
    ).toBeNull();
    expect(
      await rebuildRun(
        deps(
          async () => ours({ tabId: undefined }),
          async () => liveRow(),
        ),
      ),
    ).toBeNull();
  });
});
