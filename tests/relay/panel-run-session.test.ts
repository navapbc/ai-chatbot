import { readFileSync } from 'node:fs';
import path from 'node:path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  CONFIRMATION_LOOK_MS,
  createRunSession,
  GRACE_MS,
  ORIGIN_HOLD_CAP_MS,
  ORIGIN_HOLD_MS,
  REFRESH_BEFORE_MS,
  REFRESH_RETRY_MS,
  type RunSessionDeps,
  type RunSessionState,
  rebuildRun,
  SERVER_POLL_MS,
  type SessionRun,
  WORKER_GONE_AFTER_MS,
  WORKER_POLL_MS,
} from '../../extension/panel/lib/run-session';
import type { PendingInputRequest } from '../../lib/ai/eve/input-requests';
import { createControllerLifecycle } from '../../extension/panel/lib/search-controller';
import {
  AppApiError,
  type ExtensionToken,
  type RunState,
} from '../../extension/src/app-api';
import type { RunStatus } from '../../extension/src/bridge';
import {
  type SubmitWatchResult,
  WorkerError,
} from '../../extension/panel/lib/worker-api';

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
    arm: async (_options?: {
      reset?: boolean;
    }): Promise<SubmitWatchResult> => ({ ok: true }),
    disarm: async (): Promise<SubmitWatchResult> => ({ ok: true }),
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
    stopRun: vi.fn(async (id?: string) => {
      calls.push(id === undefined ? 'workerStop' : `workerStop:${id}`);
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
    armSubmitWatch: vi.fn(async (id: string, options?: { reset?: boolean }) => {
      calls.push(`arm:${id}${options?.reset ? ':reset' : ''}`);
      return plan.arm(options);
    }),
    disarmSubmitWatch: vi.fn(async (id: string) => {
      calls.push(`disarm:${id}`);
      return plan.disarm();
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
  it('polls the worker every 2 s and the server every 3 s', async () => {
    const t = setup();
    expect(WORKER_POLL_MS).toBe(2000);
    expect(SERVER_POLL_MS).toBe(3000);
    // Nothing runs synchronously at creation.
    expect(t.calls).toEqual([]);
    await tick(1999);
    expect(count(t.calls, 'status')).toBe(0);
    await tick(1);
    expect(count(t.calls, 'status')).toBe(1);
    await tick(1000);
    expect(count(t.calls, 'status')).toBe(1);
    expect(count(t.calls, 'row')).toBe(1);
    await tick(1000);
    expect(count(t.calls, 'status')).toBe(2);
    await tick(5000);
    expect(count(t.calls, 'status')).toBe(4);
    expect(count(t.calls, 'row')).toBe(3);
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
    expect(count(t.calls, 'row')).toBe(2);
    await tick(3000);
    expect(count(t.calls, 'row')).toBe(3);
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
        progress: null,
        pending: [],
        phase: 'working',
        actions: [],
        expiresAt: EXPIRES,
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

describe('run session: progress, questions and phase', () => {
  const progress = (actions?: string[]) => ({
    sections: [{ name: 'Contact', status: 'filling' as const, filled: 1 }],
    current: 'Contact',
    ...(actions ? { actions } : {}),
  });
  const question = (id: string) =>
    ({ id, prompt: 'Which county?' }) as unknown as RunState['pending'][number];

  it('starts with no progress, no questions, working and an empty log', () => {
    const t = setup();
    expect(t.last()).toMatchObject({
      progress: null,
      pending: [],
      phase: 'working',
      actions: [],
    });
    t.session.dispose();
  });

  it('carries the latest progress and pending from the server poll', async () => {
    const t = setup();
    const p1 = progress();
    t.plan.row = async () =>
      liveRow({ progress: p1, pending: [question('q1')], phase: 'needs_you' });
    await tick(3000);
    expect(t.last()).toMatchObject({
      progress: p1,
      pending: [question('q1')],
      phase: 'needs_you',
    });
    const p2 = { ...progress(), current: 'Income' };
    t.plan.row = async () => liveRow({ progress: p2, phase: 'working' });
    await tick(3000);
    expect(t.last()).toMatchObject({
      progress: p2,
      pending: [],
      phase: 'working',
    });
    t.session.dispose();
  });

  it.each(['working', 'needs_you', 'review'] as const)(
    'maps the server phase %s',
    async (phase) => {
      const t = setup();
      t.plan.row = async () => liveRow({ phase });
      await tick(3000);
      expect(t.last()?.phase).toBe(phase);
      t.session.dispose();
    },
  );

  it('maps an unknown, non-string or missing phase to working', async () => {
    const t = setup();
    t.plan.row = async () => liveRow({ phase: 'review' });
    await tick(3000);
    expect(t.last()?.phase).toBe('review');
    for (const phase of ['bogus', 7, null, 'ended', undefined]) {
      t.plan.row = async () => liveRow({ phase } as Partial<RunState>);
      await tick(3000);
      expect(t.last()?.phase).toBe('working');
      t.plan.row = async () => liveRow({ phase: 'review' });
      await tick(3000);
      expect(t.last()?.phase).toBe('review');
    }
    t.session.dispose();
  });

  it('a failed server poll keeps the last progress and phase', async () => {
    const t = setup();
    t.plan.row = async () => liveRow({ progress: progress(), phase: 'review' });
    await tick(3000);
    t.plan.row = async () => {
      throw new AppApiError('network', 0);
    };
    await tick(9000);
    expect(t.last()).toMatchObject({ phase: 'review', status: 'running' });
    expect(t.last()?.progress).not.toBeNull();
    t.session.dispose();
  });

  it('server ended: phase ended and pending empty', async () => {
    const t = setup();
    t.plan.row = async () =>
      liveRow({ pending: [question('q1')], phase: 'needs_you' });
    await tick(3000);
    expect(t.last()?.pending).toHaveLength(1);
    t.plan.row = async () =>
      liveRow({
        ended: true,
        endedAt: 'x',
        pending: [question('q1')],
        phase: 'ended',
      });
    await tick(3000);
    expect(t.last()).toMatchObject({
      status: 'ended',
      phase: 'ended',
      pending: [],
    });
    t.session.dispose();
  });

  it('worker-ended: phase ended and pending empty', async () => {
    const t = setup();
    t.plan.row = async () =>
      liveRow({ pending: [question('q1')], phase: 'needs_you' });
    await tick(3000);
    t.plan.status = async () => inactive;
    await tick(30_000);
    expect(t.last()).toMatchObject({
      status: 'ended',
      endedReason: 'worker-ended',
      phase: 'ended',
      pending: [],
    });
    t.session.dispose();
  });

  it('user Stop: phase ended and pending empty', async () => {
    const t = setup();
    t.plan.row = async () =>
      liveRow({ pending: [question('q1')], phase: 'needs_you' });
    await tick(3000);
    await t.session.stop();
    expect(t.last()).toMatchObject({
      status: 'ended',
      endedReason: 'user',
      phase: 'ended',
      pending: [],
    });
    t.session.dispose();
  });

  it("stop('submitted'): server stop then worker stop once each, ends submitted", async () => {
    const t = setup();
    await tick(3000);
    const order: string[] = [];
    t.api.stopRun.mockImplementation(async () => {
      order.push('server');
      return { ended: true, alreadyEnded: false };
    });
    t.worker.stopRun.mockImplementation(async () => {
      order.push('worker');
    });
    await t.session.stop('submitted');
    expect(order).toEqual(['server', 'worker']);
    expect(t.last()).toMatchObject({
      status: 'ended',
      endedReason: 'submitted',
    });
    expect(t.markRecentEnded).toHaveBeenCalledWith(RUN_ID, 'submitted');
    t.session.dispose();
  });

  it('stop() defaults to the user reason', async () => {
    const t = setup();
    await t.session.stop();
    expect(t.markRecentEnded).toHaveBeenCalledWith(RUN_ID, 'user');
    t.session.dispose();
  });

  it("stop('submitted') with a failing server stop still ends and sets stopFailed", async () => {
    const t = setup();
    t.api.stopRun.mockRejectedValue(new AppApiError('network', 0));
    await t.session.stop('submitted');
    expect(t.last()).toMatchObject({
      status: 'ended',
      endedReason: 'submitted',
      stopFailed: true,
    });
    expect(t.worker.stopRun).toHaveBeenCalledTimes(1);
    t.session.dispose();
  });

  it('a double press calls each stop once', async () => {
    const t = setup();
    await Promise.all([t.session.stop('submitted'), t.session.stop('user')]);
    expect(t.api.stopRun).toHaveBeenCalledTimes(1);
    expect(t.worker.stopRun).toHaveBeenCalledTimes(1);
    expect(t.last()?.endedReason).toBe('submitted');
    t.session.dispose();
  });

  it('feeds the action log from every poll without repeats', async () => {
    const t = setup();
    t.plan.row = async () =>
      liveRow({ progress: progress(['a', 'b', 'c', 'd', 'e']) });
    await tick(3000);
    expect(t.last()?.actions).toEqual(['a', 'b', 'c', 'd', 'e']);
    await tick(3000);
    expect(t.last()?.actions).toEqual(['a', 'b', 'c', 'd', 'e']);
    t.plan.row = async () =>
      liveRow({ progress: progress(['b', 'c', 'd', 'e', 'f']) });
    await tick(3000);
    expect(t.last()?.actions).toEqual(['a', 'b', 'c', 'd', 'e', 'f']);
    // Progress without actions, or no progress, keeps the log.
    t.plan.row = async () => liveRow({ progress: progress() });
    await tick(3000);
    t.plan.row = async () => liveRow({ progress: null });
    await tick(3000);
    expect(t.last()?.actions).toEqual(['a', 'b', 'c', 'd', 'e', 'f']);
    t.session.dispose();
  });

  it('progress that goes backwards or repeats does not throw', async () => {
    const t = setup();
    for (const actions of [['a', 'b'], ['a'], [], ['a', 'b'], ['b']]) {
      t.plan.row = async () => liveRow({ progress: progress(actions) });
      await tick(3000);
      expect(t.last()?.status).toBe('running');
    }
    t.session.dispose();
  });

  it('keeps the log after the run ended', async () => {
    const t = setup();
    t.plan.row = async () => liveRow({ progress: progress(['a']) });
    await tick(3000);
    await t.session.stop();
    expect(t.last()?.actions).toEqual(['a']);
    t.session.dispose();
  });
});

describe('run session: worker-initiated end', () => {
  it('WORKER_GONE_AFTER_MS is the bridge STOP_WAIT_MS plus one poll', () => {
    const source = readFileSync(
      path.resolve(__dirname, '../../extension/src/bridge.ts'),
      'utf8',
    );
    const wait = Number(
      source.match(/const STOP_WAIT_MS = ([\d_]+);/)?.[1]?.replace(/_/g, ''),
    );
    expect(wait).toBe(5000);
    expect(WORKER_GONE_AFTER_MS).toBe(wait + WORKER_POLL_MS);
  });

  it('inactive continuously for 7 s: one server stop, worker-ended, recents', async () => {
    const t = setup();
    await tick(2000);
    t.plan.status = async () => inactive;
    // First inactive answer at 4 s; still running 6 s later.
    await tick(8000);
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

  it('inactive for 5 s (a bridge same-run restart waiting on a screenshot), then active: not ended', async () => {
    const t = setup();
    await tick(2000);
    t.plan.status = async () => inactive;
    await tick(6000); // inactive answers at 4, 6 and 8 s
    t.plan.status = async () => ours();
    await tick(20_000);
    expect(t.last()?.status).toBe('running');
    expect(t.api.stopRun).not.toHaveBeenCalled();
    t.session.dispose();
  });

  it('inactive 3 s, active, inactive 5 s: the active answer resets the clock', async () => {
    const t = setup();
    const answers = [
      ours(),
      inactive,
      inactive,
      ours(),
      inactive,
      inactive,
      inactive,
    ];
    t.plan.status = async () => answers.shift() ?? ours();
    await tick(14_000);
    expect(count(t.calls, 'status')).toBe(7);
    expect(t.last()?.status).toBe('running');
    expect(t.api.stopRun).not.toHaveBeenCalled();
    t.session.dispose();
  });

  it('a failed poll neither counts nor resets the inactive clock', async () => {
    const t = setup();
    const answers: (RunStatus | Error)[] = [
      ours(),
      inactive, // 4 s: the clock starts
      new WorkerError('worker_unreachable'),
      inactive,
      new WorkerError('worker_unreachable'),
      inactive, // 12 s: 8 s inactive
    ];
    t.plan.status = async () => {
      const next = answers.shift() ?? ours();
      if (next instanceof Error) throw next;
      return next;
    };
    await tick(10_000);
    expect(t.last()?.status).toBe('running');
    await tick(2000);
    expect(t.last()?.status).toBe('ended');
    expect(t.api.stopRun).toHaveBeenCalledTimes(1);
    t.session.dispose();
  });

  it('inactive from the start is not an end within the first 5 s', async () => {
    expect(GRACE_MS).toBe(5000);
    const t = setup();
    t.plan.status = async () => inactive;
    await tick(4000);
    expect(t.last()?.status).toBe('running');
    // First inactive answer at 2 s: gone at the 10 s poll.
    await tick(4000);
    expect(t.last()?.status).toBe('running');
    await tick(2000);
    expect(t.last()?.status).toBe('ended');
    t.session.dispose();
  });

  it('a server stop that fails is still called only once', async () => {
    const t = setup();
    t.plan.serverStop = async () => {
      throw new AppApiError('network', 0);
    };
    t.plan.status = async () => inactive;
    await tick(30_000);
    expect(t.last()?.status).toBe('ended');
    expect(t.api.stopRun).toHaveBeenCalledTimes(1);
    t.session.dispose();
  });

  it('replaced by another window (different runId): ends at once, server stop once, never the worker stop', async () => {
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

  it('a different runId ends the run even within the first 5 s', async () => {
    const t = setup();
    t.plan.status = async () => ours({ runId: OTHER_RUN });
    await tick(2000);
    expect(t.last()?.status).toBe('ended');
    expect(t.worker.stopRun).not.toHaveBeenCalled();
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
    // The worker's run (this run id only) is released.
    expect(t.worker.stopRun).toHaveBeenCalledTimes(1);
    expect(t.worker.stopRun).toHaveBeenCalledWith(RUN_ID);
    t.session.dispose();
  });

  it('server ended first, worker inactive after: stopped, no server stop', async () => {
    const t = setup();
    await tick(2000);
    t.plan.status = async () => inactive;
    t.plan.row = async () => liveRow({ ended: true, endedAt: 'x' });
    await tick(30_000);
    expect(t.last()?.endedReason).toBe('stopped');
    expect(t.api.stopRun).not.toHaveBeenCalled();
    expect(t.markRecentEnded).toHaveBeenCalledTimes(1);
    t.session.dispose();
  });

  it('worker gone first, server ended meanwhile: the server is asked before stopping, and wins', async () => {
    const t = setup();
    await tick(2000);
    t.plan.status = async () => inactive;
    // Inactive from 4 s, gone at 12 s; the row ends at 11 s, after the 10 s
    // server poll and before the next one at 15 s.
    await tick(9000);
    expect(t.last()?.status).toBe('running');
    t.plan.row = async () => liveRow({ ended: true, endedAt: 'x' });
    await tick(1000);
    expect(t.last()).toMatchObject({ status: 'ended', endedReason: 'stopped' });
    await tick(20_000);
    expect(t.api.stopRun).not.toHaveBeenCalled();
    expect(t.markRecentEnded).toHaveBeenCalledTimes(1);
    expect(t.markRecentEnded).toHaveBeenCalledWith(RUN_ID, 'stopped');
    t.session.dispose();
  });

  it('worker gone while the server cannot be read: worker-ended, one server stop', async () => {
    const t = setup();
    t.plan.row = async () => {
      throw new AppApiError('network', 0);
    };
    await tick(2000);
    t.plan.status = async () => inactive;
    await tick(30_000);
    expect(t.last()?.endedReason).toBe('worker-ended');
    expect(t.api.stopRun).toHaveBeenCalledTimes(1);
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
      `workerStop:${RUN_ID}`,
      `recent:${RUN_ID}:user`,
    ]);
    expect(t.last()).toEqual({
      status: 'ended',
      endedReason: 'user',
      progress: null,
      pending: [],
      phase: 'ended',
      actions: [],
      expiresAt: EXPIRES,
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
      `workerStop:${RUN_ID}`,
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
    expect(t.calls).not.toContain(`workerStop:${RUN_ID}`);
    start.resolve();
    await done;
    const order = t.calls.filter((c) => /Stop|workerStart/.test(c));
    expect(order).toEqual([
      'workerStart:7:tok-new',
      `serverStop:${RUN_ID}`,
      `workerStop:${RUN_ID}`,
    ]);
    t.session.dispose();
  });

  it('commits the new expiry once startRun succeeds; a registrationComplete failure is retried by the poll', async () => {
    const t = setup({ expiresAt: NOW / 1000 + 60 });
    t.plan.registration = async () => {
      throw new WorkerError('worker_unreachable');
    };
    await tick(2000);
    expect(t.worker.startRun).toHaveBeenCalledTimes(1);
    expect(t.last()?.refreshFailed).toBeUndefined();
    expect(t.last()?.secondsLeft).toBe(1800);
    // The restarted run says it needs registration: the poll sends it again.
    t.plan.registration = async () => {};
    t.plan.status = async () => ours({ needsRegistration: true });
    await tick(2000);
    expect(count(t.calls, 'registration')).toBe(2);
    expect(t.api.getExtensionToken).toHaveBeenCalledTimes(1);
    t.session.dispose();
  });

  it('a synchronous throw in the token fetch is a refusal, and is retried a minute later', async () => {
    const t = setup({ expiresAt: NOW / 1000 + 4 * 60 });
    t.api.getExtensionToken.mockImplementation(() => {
      throw new Error('sync');
    });
    await tick(2000);
    expect(t.last()?.refreshFailed).toBe(true);
    await tick(60_000);
    expect(t.api.getExtensionToken).toHaveBeenCalledTimes(2);
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

describe('run session: restarting worker', () => {
  const restarting = (runId?: string): RunStatus => ({
    ok: true,
    active: false,
    restarting: true,
    ...(runId ? { runId } : {}),
  });

  it('an inactive restarting status for 10 s does not end the session', async () => {
    const t = setup();
    t.plan.status = async () => restarting(RUN_ID);
    await tick(10_000);
    expect(t.last()?.status).toBe('running');
    expect(t.worker.stopRun).not.toHaveBeenCalled();
    t.session.dispose();
  });

  it('a restarting status without a run id (a worker resume) also holds', async () => {
    const t = setup();
    t.plan.status = async () => restarting();
    await tick(20_000);
    expect(t.last()?.status).toBe('running');
    t.session.dispose();
  });

  it('after 30 s of continuous restarting the 7 s rule applies and the run ends', async () => {
    const t = setup();
    t.plan.status = async () => restarting(RUN_ID);
    await tick(30_000);
    expect(t.last()?.status).toBe('running');
    await tick(WORKER_GONE_AFTER_MS + 4000);
    expect(t.last()).toMatchObject({
      status: 'ended',
      endedReason: 'worker-ended',
    });
    t.session.dispose();
  });

  it('restarting for another run id does not hold this run', async () => {
    const t = setup();
    t.plan.status = async () => restarting('other-run');
    await tick(WORKER_GONE_AFTER_MS + 4000);
    expect(t.last()?.status).toBe('ended');
    t.session.dispose();
  });

  it('a plain inactive status after a restart still ends at 7 s', async () => {
    const t = setup();
    t.plan.status = async () => restarting(RUN_ID);
    await tick(10_000);
    t.plan.status = async () => inactive;
    await tick(WORKER_GONE_AFTER_MS + 4000);
    expect(t.last()?.status).toBe('ended');
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

describe('run session: origin hold', () => {
  const A = 'https://login.example.gov';
  const B = 'https://idp.example.gov';
  const C = 'https://mfa.example.gov';

  it('mirrors the bridge hold: 60 s per origin, 120 s in total', () => {
    expect(ORIGIN_HOLD_MS).toBe(60_000);
    expect(ORIGIN_HOLD_CAP_MS).toBe(120_000);
  });

  it('no hold: no hold fields', async () => {
    const t = setup();
    await tick(4000);
    const state = t.last();
    expect(state?.holdSeenAt).toBeUndefined();
    expect(state?.holdEndsAt).toBeUndefined();
    expect(state?.holdCountdownKnown).toBeUndefined();
    t.session.dispose();
  });

  it('a hold seen start at runtime: holdSeenAt is the first sight, the countdown is known', async () => {
    const t = setup();
    await tick(2000); // clear
    t.plan.status = async () => ours({ pendingOrigin: A });
    await tick(2000); // first sight at NOW + 4000
    expect(t.last()).toMatchObject({
      pendingOrigin: A,
      holdSeenAt: NOW + 4000,
      holdCountdownKnown: true,
      holdEndsAt: NOW + 4000 + ORIGIN_HOLD_MS,
    });
    await tick(4000); // the same origin again: unchanged
    expect(t.last()).toMatchObject({
      holdSeenAt: NOW + 4000,
      holdEndsAt: NOW + 4000 + ORIGIN_HOLD_MS,
    });
    t.session.dispose();
  });

  it('the first worker answer already shows a hold: no countdown', async () => {
    const t = setup();
    t.plan.status = async () => ours({ pendingOrigin: A });
    await tick(2000);
    expect(t.last()).toMatchObject({
      pendingOrigin: A,
      holdSeenAt: NOW + 2000,
      holdCountdownKnown: false,
    });
    expect(t.last()?.holdEndsAt).toBeUndefined();
    t.session.dispose();
  });

  it('a failed first poll does not count as seeing the run clear', async () => {
    const t = setup();
    t.plan.status = async () => {
      throw new WorkerError('worker_unreachable');
    };
    await tick(2000);
    t.plan.status = async () => ours({ pendingOrigin: A });
    await tick(2000);
    expect(t.last()?.holdCountdownKnown).toBe(false);
    t.session.dispose();
  });

  it('a clear answer too long before the hold leaves the countdown unknown', async () => {
    const t = setup();
    await tick(2000); // clear at NOW + 2000
    t.plan.status = async () => {
      throw new WorkerError('worker_unreachable');
    };
    await tick(4000); // two failed polls
    t.plan.status = async () => ours({ pendingOrigin: A });
    await tick(2000); // 6 s after the last clear answer
    expect(t.last()?.holdCountdownKnown).toBe(false);
    t.session.dispose();
  });

  it('one failed poll between a clear answer and the hold keeps it known', async () => {
    const t = setup();
    await tick(2000); // clear at NOW + 2000
    t.plan.status = async () => {
      throw new WorkerError('worker_unreachable');
    };
    await tick(2000);
    t.plan.status = async () => ours({ pendingOrigin: A });
    await tick(2000); // seen at NOW + 6000, 4 s after the clear answer
    expect(t.last()).toMatchObject({
      holdCountdownKnown: true,
      holdSeenAt: NOW + 6000,
    });
    t.session.dispose();
  });

  it('a failed poll mid-hold changes nothing', async () => {
    const t = setup();
    await tick(2000);
    t.plan.status = async () => ours({ pendingOrigin: A });
    await tick(2000);
    t.plan.status = async () => {
      throw new WorkerError('worker_unreachable');
    };
    await tick(2000);
    t.plan.status = async () => ours({ pendingOrigin: A });
    await tick(2000);
    expect(t.last()).toMatchObject({
      pendingOrigin: A,
      holdSeenAt: NOW + 4000,
      holdCountdownKnown: true,
    });
    t.session.dispose();
  });

  it('a changed pending origin resets holdSeenAt; the 120 s cap counts from the first', async () => {
    const t = setup();
    await tick(2000);
    t.plan.status = async () => ours({ pendingOrigin: A });
    await tick(2000); // A at NOW + 4000
    await tick(44_000);
    t.plan.status = async () => ours({ pendingOrigin: B });
    await tick(2000); // B at NOW + 50_000
    expect(t.last()).toMatchObject({
      pendingOrigin: B,
      holdSeenAt: NOW + 50_000,
      holdCountdownKnown: true,
      holdEndsAt: NOW + 50_000 + ORIGIN_HOLD_MS,
    });
    await tick(48_000);
    t.plan.status = async () => ours({ pendingOrigin: C });
    await tick(2000); // C at NOW + 100_000: the cap binds
    expect(t.last()).toMatchObject({
      pendingOrigin: C,
      holdSeenAt: NOW + 100_000,
      holdCountdownKnown: true,
      holdEndsAt: NOW + 4000 + ORIGIN_HOLD_CAP_MS,
    });
    t.session.dispose();
  });

  it('a change in a hold whose start was not seen stays unknown', async () => {
    const t = setup();
    t.plan.status = async () => ours({ pendingOrigin: A });
    await tick(2000);
    t.plan.status = async () => ours({ pendingOrigin: B });
    await tick(2000);
    expect(t.last()).toMatchObject({
      pendingOrigin: B,
      holdSeenAt: NOW + 4000,
      holdCountdownKnown: false,
    });
    expect(t.last()?.holdEndsAt).toBeUndefined();
    t.session.dispose();
  });

  it('a cleared hold clears the fields; a later hold starts fresh', async () => {
    const t = setup();
    t.plan.status = async () => ours({ pendingOrigin: A });
    await tick(2000); // unknown
    t.plan.status = async () => ours();
    await tick(2000);
    expect(t.last()?.pendingOrigin).toBeUndefined();
    expect(t.last()?.holdSeenAt).toBeUndefined();
    expect(t.last()?.holdCountdownKnown).toBeUndefined();
    t.plan.status = async () => ours({ pendingOrigin: A });
    await tick(2000);
    expect(t.last()).toMatchObject({
      pendingOrigin: A,
      holdSeenAt: NOW + 6000,
      holdCountdownKnown: true,
      holdEndsAt: NOW + 6000 + ORIGIN_HOLD_MS,
    });
    t.session.dispose();
  });

  it('an inactive answer clears the hold and does not count as clear', async () => {
    const t = setup();
    await tick(2000);
    t.plan.status = async () => ours({ pendingOrigin: A });
    await tick(2000);
    t.plan.status = async () => inactive;
    await tick(2000);
    expect(t.last()?.holdSeenAt).toBeUndefined();
    t.plan.status = async () => ours({ pendingOrigin: A });
    await tick(2000);
    expect(t.last()?.holdCountdownKnown).toBe(false);
    t.session.dispose();
  });

  it('an ended run shows no hold', async () => {
    const t = setup();
    await tick(2000);
    t.plan.status = async () => ours({ pendingOrigin: A });
    await tick(2000);
    await t.session.stop();
    expect(t.last()?.status).toBe('ended');
    expect(t.last()?.pendingOrigin).toBeUndefined();
    expect(t.last()?.holdSeenAt).toBeUndefined();
    expect(t.last()?.holdEndsAt).toBeUndefined();
    t.session.dispose();
  });

  it('exposes the live expiry in seconds, updated by a refresh', async () => {
    const t = setup({ expiresAt: NOW / 1000 + 60 });
    expect(t.last()?.expiresAt).toBe(NOW / 1000 + 60);
    await tick(2000); // refresh: the token route answers now + 1800
    await tick(2000);
    expect(t.last()?.expiresAt).toBe(Math.floor((NOW + 2000) / 1000) + 1800);
    t.session.dispose();
  });
});

describe('run session: no applicant data', () => {
  const STATE_KEYS = new Set([
    'status',
    'progress',
    'pending',
    'phase',
    'actions',
    'endedReason',
    'pendingOrigin',
    'holdSeenAt',
    'holdEndsAt',
    'holdCountdownKnown',
    'jevFailures',
    'refreshFailed',
    'expiresAtEstimated',
    'expiresAt',
    'secondsLeft',
    'stopping',
    'stopFailed',
    'submission',
    'confirmedHow',
    'confirmation',
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
      ({ ...liveRow(), participant: 'Zelda Quill' }) as RunState;
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
        stopRun: async () => {},
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
  const rebuiltSetupFor = async (exp: number | undefined) => {
    const rebuilt = await rebuildRun({
      api: { getRunState: async () => liveRow() },
      worker: {
        getRunStatus: async () => ours(exp === undefined ? {} : { exp }),
        stopRun: async () => {},
      },
      now: () => NOW,
    });
    if (!rebuilt) throw new Error('not rebuilt');
    const t = setup(rebuilt.run);
    return { t, rebuilt };
  };
  const deps = (
    status: () => Promise<RunStatus>,
    row: () => Promise<RunState>,
  ) => ({
    api: { getRunState: vi.fn(async (_id: string) => row()) },
    worker: {
      getRunStatus: vi.fn(status),
      stopRun: vi.fn(async (_id?: string) => {}),
    },
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

  it('uses the worker-reported exp, not an estimate', async () => {
    const exp = NOW / 1000 + 25 * 60;
    const d = deps(
      async () => ours({ exp }),
      async () => liveRow(),
    );
    const rebuilt = await rebuildRun(d);
    expect(rebuilt?.run).toEqual({ runId: RUN_ID, tabId: 7, expiresAt: exp });
    expect(rebuilt?.run.expiresAtEstimated).toBeUndefined();
  });

  it('a panel opened with exp 25 minutes away does not refresh', async () => {
    const exp = NOW / 1000 + 25 * 60;
    const { t, rebuilt } = await rebuiltSetupFor(exp);
    expect(rebuilt.run.expiresAtEstimated).toBeUndefined();
    expect(t.last()?.expiresAtEstimated).toBeUndefined();
    await tick(4000);
    expect(t.worker.startRun).not.toHaveBeenCalled();
    expect(t.api.getExtensionToken).not.toHaveBeenCalled();
    t.session.dispose();
  });

  it('without exp the estimate and the first-poll refresh remain', async () => {
    const { t, rebuilt } = await rebuiltSetupFor(undefined);
    expect(rebuilt.run.expiresAtEstimated).toBe(true);
    await tick(2000);
    expect(t.api.getExtensionToken).toHaveBeenCalledTimes(1);
    t.session.dispose();
  });

  it('an inactive worker rebuilds nothing and asks the server nothing', async () => {
    const d = deps(
      async () => inactive,
      async () => liveRow(),
    );
    expect(await rebuildRun(d)).toBeNull();
    expect(d.api.getRunState).not.toHaveBeenCalled();
  });

  it('an ended server row rebuilds nothing and stops that worker run by id', async () => {
    const d = deps(
      async () => ours(),
      async () => liveRow({ ended: true, endedAt: 'x' }),
    );
    expect(await rebuildRun(d)).toBeNull();
    expect(d.worker.stopRun).toHaveBeenCalledTimes(1);
    expect(d.worker.stopRun).toHaveBeenCalledWith(RUN_ID);
  });

  it('a live row on another origin, or an inactive worker, stops nothing', async () => {
    const d = deps(
      async () => ours(),
      async () => liveRow({ origin: 'https://other.example.gov' }),
    );
    expect(await rebuildRun(d)).toBeNull();
    expect(d.worker.stopRun).not.toHaveBeenCalled();
    const e = deps(
      async () => inactive,
      async () => liveRow(),
    );
    expect(await rebuildRun(e)).toBeNull();
    expect(e.worker.stopRun).not.toHaveBeenCalled();
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

describe('a reopened panel lands on the right card', () => {
  const rebuiltSetup = async (
    status: () => Promise<RunStatus>,
    row: () => Promise<RunState>,
  ) => {
    const calls: string[] = [];
    const rebuilt = await rebuildRun({
      api: {
        getRunState: vi.fn(async (id: string) => {
          calls.push(`row:${id}`);
          return row();
        }),
      },
      worker: {
        getRunStatus: vi.fn(async () => {
          calls.push('status');
          return status();
        }),
        stopRun: vi.fn(async () => {}),
      },
      now: () => NOW,
    });
    if (!rebuilt) throw new Error('not rebuilt');
    const t = setup(rebuilt.run);
    t.plan.status = status;
    t.plan.row = row;
    return { t, rebuilt, calls };
  };
  const question = {
    requestId: 'r1',
    prompt: 'Which income?',
    options: ['a', 'b'],
  } as unknown as PendingInputRequest;

  it('pending questions land on needs_you with an empty log', async () => {
    const { t, rebuilt } = await rebuiltSetup(
      async () => ours(),
      async () =>
        liveRow({
          phase: 'needs_you',
          pending: [question],
          progress: { actions: ['clicked A'], sections: [] },
        } as Partial<RunState>),
    );
    expect(rebuilt.run.expiresAtEstimated).toBe(true);
    expect(t.last().actions).toEqual([]);
    expect(t.last().progress).toBeNull();
    await tick(SERVER_POLL_MS);
    expect(t.last().phase).toBe('needs_you');
    expect(t.last().pending).toHaveLength(1);
    expect(t.last().progress).not.toBeNull();
    t.session.dispose();
  });

  it('phase review lands on review', async () => {
    const { t } = await rebuiltSetup(
      async () => ours(),
      async () => liveRow({ phase: 'review' } as Partial<RunState>),
    );
    await tick(SERVER_POLL_MS);
    expect(t.last().phase).toBe('review');
    t.session.dispose();
  });

  it('a worker origin hold shows the pending origin on the first poll', async () => {
    const { t } = await rebuiltSetup(
      async () => ours({ pendingOrigin: 'https://other.example.gov' }),
      async () => liveRow(),
    );
    await tick(WORKER_POLL_MS);
    expect(t.last().pendingOrigin).toBe('https://other.example.gov');
    expect(t.last().holdCountdownKnown).toBe(false);
    t.session.dispose();
  });

  it('an ended row is not rebuilt, and a rebuild calls only the status and the row', async () => {
    const ended = await rebuildRun({
      api: { getRunState: async () => liveRow({ ended: true }) },
      worker: { getRunStatus: async () => ours(), stopRun: async () => {} },
      now: () => NOW,
    });
    expect(ended).toBeNull();
    const { calls } = await rebuiltSetup(
      async () => ours(),
      async () => liveRow(),
    );
    expect(calls).toEqual(['status', `row:${RUN_ID}`]);
  });
});

describe('run session: the submit watch', () => {
  const NUMBER = { token: 'AB-1234', kind: 'confirmation' as const };
  const OTHER = { token: '20240017', kind: 'other' as const };
  type Watch = { armed: boolean; submitted?: RunStatus['submitted'] };

  /**
   * Review from the first server poll (3 s). `w` is the worker's watch: an
   * arm sets it, a reset arm or a disarm forgets the submission.
   */
  const review = (run: Partial<SessionRun> = {}) => {
    const t = setup(run);
    const w: Watch = { armed: false };
    t.plan.row = async () => liveRow({ phase: 'review' });
    t.plan.status = async () =>
      ours({
        armed: w.armed,
        ...(w.submitted ? { submitted: w.submitted } : {}),
      });
    t.plan.arm = async (options) => {
      if (options?.reset) w.submitted = undefined;
      w.armed = true;
      return { ok: true };
    };
    t.plan.disarm = async () => {
      w.armed = false;
      w.submitted = undefined;
      return { ok: true };
    };
    return { t, w };
  };
  const watchCalls = (calls: string[]) =>
    calls.filter((c) => c.startsWith('arm:') || c.startsWith('disarm:'));
  const ARM_RESET = `arm:${RUN_ID}:reset`;
  const ARM = `arm:${RUN_ID}`;
  const DISARM = `disarm:${RUN_ID}`;

  it('CONFIRMATION_LOOK_MS mirrors the bridge CONFIRMATION_READ_MS', () => {
    const source = readFileSync(
      path.resolve(__dirname, '../../extension/src/bridge.ts'),
      'utf8',
    );
    const read = Number(
      source
        .match(/export const CONFIRMATION_READ_MS = ([\d_]+);/)?.[1]
        ?.replace(/_/g, ''),
    );
    expect(read).toBe(12_000);
    expect(CONFIRMATION_LOOK_MS).toBe(read);
  });

  it('arms with reset on entering review, holds while armed, disarms on leaving, arms afresh on return', async () => {
    const { t } = review();
    await tick(2000);
    // Working at the first worker poll: nothing armed.
    expect(watchCalls(t.calls)).toEqual([]);
    await tick(1000);
    expect(watchCalls(t.calls)).toEqual([ARM_RESET]);
    await tick(10_000);
    expect(watchCalls(t.calls)).toEqual([ARM_RESET]);
    t.plan.row = async () => liveRow({ phase: 'working' });
    await tick(3000);
    expect(watchCalls(t.calls)).toEqual([ARM_RESET, DISARM]);
    // Not in review: a lost watch is not armed again.
    await tick(10_000);
    expect(watchCalls(t.calls)).toEqual([ARM_RESET, DISARM]);
    t.plan.row = async () => liveRow({ phase: 'review' });
    await tick(3000);
    expect(watchCalls(t.calls)).toEqual([ARM_RESET, DISARM, ARM_RESET]);
    t.session.dispose();
  });

  it('a run that ends disarms after the worker stop; a panel closed in review disarms', async () => {
    const { t } = review();
    await tick(3000);
    await t.session.stop();
    await tick(0);
    expect(watchCalls(t.calls)).toEqual([ARM_RESET, DISARM]);
    expect(t.calls.indexOf(`workerStop:${RUN_ID}`)).toBeLessThan(
      t.calls.indexOf(DISARM),
    );
    t.session.dispose();

    const closed = review();
    await tick(3000);
    closed.t.session.dispose();
    await tick(0);
    expect(watchCalls(closed.t.calls)).toEqual([ARM_RESET, DISARM]);
    expect(vi.getTimerCount()).toBe(0);
  });

  it('re-arms only while armed reads false, never two at a time', async () => {
    const { t, w } = review();
    await tick(3000);
    // A detach lost the watch; the re-arm hangs.
    w.armed = false;
    const hang = deferred<SubmitWatchResult>();
    t.plan.arm = () => hang.promise;
    await tick(10_000);
    expect(watchCalls(t.calls)).toEqual([ARM_RESET, ARM]);
    hang.resolve({ ok: true });
    w.armed = true;
    await tick(6000);
    expect(watchCalls(t.calls)).toEqual([ARM_RESET, ARM]);
    // Refused arms are tried again on the next poll that reads false.
    w.armed = false;
    t.plan.arm = async () => ({ ok: false, code: 'not_attached' });
    await tick(4000);
    expect(count(t.calls, 'arm')).toBe(4);
    t.session.dispose();
  });

  it('leaving review waits out an arm in flight, then disarms, and no arm follows the disarm', async () => {
    const { t, w } = review();
    await tick(3000);
    w.armed = false;
    const hang = deferred<SubmitWatchResult>();
    t.plan.arm = () => hang.promise;
    await tick(2000);
    expect(watchCalls(t.calls)).toEqual([ARM_RESET, ARM]);
    const disarm = deferred<SubmitWatchResult>();
    t.plan.disarm = () => disarm.promise;
    t.plan.row = async () => liveRow({ phase: 'working' });
    await tick(1000);
    // Out of review, but the arm is still in flight: no disarm yet.
    expect(t.calls).not.toContain(DISARM);
    hang.resolve({ ok: true });
    await tick(0);
    expect(watchCalls(t.calls)).toEqual([ARM_RESET, ARM, DISARM]);
    // The worker keeps answering armed: false while the disarm hangs.
    await tick(10_000);
    disarm.resolve({ ok: true });
    await tick(6000);
    expect(watchCalls(t.calls)).toEqual([ARM_RESET, ARM, DISARM]);
    t.session.dispose();
  });

  it('an arm still queued when review ends is dropped, not sent before the disarm', async () => {
    const { t, w } = review();
    await tick(3000);
    w.submitted = { at: Date.now(), confirmation: OTHER };
    await tick(1000);
    expect(t.last()?.submission).toBeDefined();
    // The watch is lost and the re-arm hangs; No queues a reset arm behind.
    w.armed = false;
    const hang = deferred<SubmitWatchResult>();
    t.plan.arm = () => hang.promise;
    await tick(2000);
    expect(watchCalls(t.calls)).toEqual([ARM_RESET, ARM]);
    t.session.dismissSubmission();
    t.plan.row = async () => liveRow({ phase: 'working' });
    await tick(3000);
    hang.resolve({ ok: true });
    await tick(0);
    expect(watchCalls(t.calls)).toEqual([ARM_RESET, ARM, DISARM]);
    await tick(6000);
    expect(watchCalls(t.calls)).toEqual([ARM_RESET, ARM, DISARM]);
    t.session.dispose();
  });

  it('a confirmation number never stops the run without a press; the prompt shows it at once', async () => {
    const { t, w } = review();
    await tick(3000);
    w.submitted = { at: Date.now(), confirmation: NUMBER };
    await tick(1000);
    expect(t.last()?.submission).toEqual({
      looking: false,
      confirmation: NUMBER,
    });
    await tick(30_000);
    expect(count(t.calls, 'serverStop')).toBe(0);
    expect(count(t.calls, 'workerStop')).toBe(0);
    expect(t.last()?.status).toBe('running');
    t.session.dispose();
  });

  it('Yes stops once (server then worker) and carries the number known at that moment', async () => {
    const { t, w } = review();
    await tick(3000);
    const order: string[] = [];
    t.plan.serverStop = async () => {
      order.push('server');
      return { ended: true, alreadyEnded: false };
    };
    t.plan.workerStop = async () => {
      // The stop ends the worker's run, and the number with it.
      order.push('worker');
      w.submitted = undefined;
      w.armed = false;
    };
    w.submitted = { at: Date.now(), confirmation: NUMBER };
    await tick(1000);
    await t.session.confirmSubmission();
    expect(order).toEqual(['server', 'worker']);
    expect(t.last()).toMatchObject({
      status: 'ended',
      endedReason: 'submitted',
      confirmedHow: 'detected',
      confirmation: NUMBER,
    });
    expect(t.markRecentEnded).toHaveBeenCalledWith(RUN_ID, 'submitted');
    await tick(10_000);
    expect(order).toEqual(['server', 'worker']);
    t.session.dispose();
  });

  it('Yes pressed twice, and I submitted it during the stop, stop once', async () => {
    const { t, w } = review();
    await tick(3000);
    const server = deferred<{ ended: boolean; alreadyEnded: boolean }>();
    t.plan.serverStop = () => server.promise;
    w.submitted = { at: Date.now(), confirmation: NUMBER };
    await tick(1000);
    void t.session.confirmSubmission();
    expect(t.last()?.stopping).toBe(true);
    await tick(6000);
    void t.session.confirmSubmission();
    void t.session.stop('submitted');
    server.resolve({ ended: true, alreadyEnded: false });
    await tick(0);
    expect(count(t.calls, 'serverStop')).toBe(1);
    expect(count(t.calls, 'workerStop')).toBe(1);
    expect(t.last()).toMatchObject({
      confirmedHow: 'detected',
      confirmation: NUMBER,
    });
    t.session.dispose();
  });

  it('a submission seen before the reset arm lands never counts (no prompt, no stop)', async () => {
    const { t, w } = review();
    // The worker still holds a confirmed submission from an earlier review.
    w.armed = true;
    w.submitted = { at: Date.now() - 60_000, confirmation: NUMBER };
    const hang = deferred<SubmitWatchResult>();
    t.plan.arm = () => hang.promise;
    await tick(10_000);
    expect(watchCalls(t.calls)).toEqual([ARM_RESET]);
    expect(count(t.calls, 'serverStop')).toBe(0);
    expect(t.last()?.submission).toBeUndefined();
    w.submitted = undefined;
    hang.resolve({ ok: true });
    await tick(4000);
    expect(count(t.calls, 'serverStop')).toBe(0);
    expect(t.last()?.status).toBe('running');
    t.session.dispose();
  });

  it('a refused reset arm is tried again, still with reset, while the old submission is ignored', async () => {
    const { t, w } = review();
    w.armed = true;
    w.submitted = { at: Date.now() - 60_000, confirmation: NUMBER };
    t.plan.arm = async () => ({ ok: false, code: 'run_held' });
    // Entering review at 3 s, then the 4 s and 6 s polls.
    await tick(7000);
    expect(watchCalls(t.calls)).toEqual([ARM_RESET, ARM_RESET, ARM_RESET]);
    expect(t.last()?.submission).toBeUndefined();
    t.session.dispose();
  });

  it('a number after another label shows as such; Yes confirms as detected with it', async () => {
    const { t, w } = review();
    await tick(3000);
    w.submitted = { at: Date.now(), confirmation: OTHER };
    await tick(1000);
    expect(t.last()?.submission).toEqual({
      looking: false,
      confirmation: OTHER,
    });
    expect(count(t.calls, 'serverStop')).toBe(0);
    await t.session.confirmSubmission();
    expect(t.calls.filter((c) => /Stop/.test(c))).toEqual([
      `serverStop:${RUN_ID}`,
      `workerStop:${RUN_ID}`,
    ]);
    expect(t.last()).toMatchObject({
      status: 'ended',
      endedReason: 'submitted',
      confirmedHow: 'detected',
      confirmation: OTHER,
    });
    t.session.dispose();
  });

  it('no number: the prompt shows at once, looking for 12 s from first sight, then not; Yes confirms with no number', async () => {
    const { t, w } = review();
    await tick(3000);
    w.submitted = { at: Date.now() };
    await tick(1000);
    // First seen at the 4 s poll.
    expect(t.last()?.submission).toEqual({ looking: true });
    await tick(CONFIRMATION_LOOK_MS - 1);
    expect(t.last()?.submission).toEqual({ looking: true });
    await tick(1);
    expect(t.last()?.submission).toEqual({ looking: false });
    expect(count(t.calls, 'serverStop')).toBe(0);
    await t.session.confirmSubmission();
    expect(t.last()).toMatchObject({
      endedReason: 'submitted',
      confirmedHow: 'detected',
    });
    expect(t.last()?.confirmation).toBeUndefined();
    t.session.dispose();
  });

  it('a number arriving later updates the open prompt and never stops the run', async () => {
    for (const number of [NUMBER, OTHER]) {
      const { t, w } = review();
      await tick(3000);
      const at = Date.now();
      w.submitted = { at };
      await tick(5000);
      expect(t.last()?.submission).toEqual({ looking: true });
      w.submitted = { at, confirmation: number };
      await tick(2000);
      expect(t.last()?.submission).toEqual({
        looking: false,
        confirmation: number,
      });
      // Even after the looking window.
      await tick(CONFIRMATION_LOOK_MS);
      expect(t.last()?.submission).toEqual({
        looking: false,
        confirmation: number,
      });
      expect(count(t.calls, 'serverStop')).toBe(0);
      expect(t.last()?.status).toBe('running');
      t.session.dispose();
    }
  });

  it('No clears the prompt and re-arms with reset; the old submission never comes back; a later one is seen', async () => {
    const { t, w } = review();
    await tick(3000);
    const first = Date.now();
    w.submitted = { at: first, confirmation: OTHER };
    await tick(1000);
    expect(t.last()?.submission).toBeDefined();
    // The reset arm hangs, and meanwhile the worker still reports it.
    const hang = deferred<SubmitWatchResult>();
    t.plan.arm = () => hang.promise;
    t.session.dismissSubmission();
    expect(t.last()?.submission).toBeUndefined();
    await tick(6000);
    expect(watchCalls(t.calls)).toEqual([ARM_RESET, ARM_RESET]);
    expect(t.last()?.submission).toBeUndefined();
    w.submitted = undefined;
    hang.resolve({ ok: true });
    await tick(4000);
    expect(t.last()?.submission).toBeUndefined();
    // A worker that reports the dismissed one again is still ignored.
    w.submitted = { at: first, confirmation: NUMBER };
    await tick(4000);
    expect(count(t.calls, 'serverStop')).toBe(0);
    // The next submission is seen.
    w.submitted = { at: first + 30_000 };
    await tick(2000);
    expect(t.last()?.submission).toEqual({ looking: true });
    expect(t.last()?.status).toBe('running');
    t.session.dispose();
  });

  it('leaving review drops the prompt', async () => {
    const { t, w } = review();
    await tick(3000);
    w.submitted = { at: Date.now(), confirmation: OTHER };
    await tick(1000);
    expect(t.last()?.submission).toBeDefined();
    t.plan.row = async () => liveRow({ phase: 'working' });
    await tick(2000);
    expect(t.last()?.submission).toBeUndefined();
    t.session.dispose();
  });

  it('a held tab ignores a reported submission', async () => {
    const { t, w } = review();
    await tick(3000);
    t.plan.status = async () =>
      ours({
        armed: true,
        pendingOrigin: 'https://login.example.gov',
        submitted: { at: Date.now(), confirmation: NUMBER },
      });
    await tick(6000);
    expect(count(t.calls, 'serverStop')).toBe(0);
    expect(t.last()?.submission).toBeUndefined();
    void w;
    t.session.dispose();
  });

  it('a failed stop after Yes still ends as a detected submission with the number and stopFailed', async () => {
    const { t, w } = review();
    await tick(3000);
    t.plan.serverStop = async () => {
      throw new AppApiError('network', 0);
    };
    w.submitted = { at: Date.now(), confirmation: NUMBER };
    await tick(1000);
    await t.session.confirmSubmission();
    expect(t.last()).toMatchObject({
      status: 'ended',
      endedReason: 'submitted',
      stopFailed: true,
      confirmedHow: 'detected',
      confirmation: NUMBER,
    });
    expect(count(t.calls, 'workerStop')).toBe(1);
    t.session.dispose();
  });

  it('a confirmation number while the last server phase is working never stops the run', async () => {
    const t = setup();
    t.plan.row = async () => liveRow({ phase: 'working' });
    t.plan.status = async () =>
      ours({
        armed: true,
        submitted: { at: Date.now(), confirmation: NUMBER },
      });
    await tick(20_000);
    expect(count(t.calls, 'serverStop')).toBe(0);
    expect(count(t.calls, 'arm')).toBe(0);
    expect(t.last()?.status).toBe('running');
    expect(t.last()?.submission).toBeUndefined();
    t.session.dispose();
  });

  it('"I submitted it" under a prompt showing a number keeps it, as manual', async () => {
    const { t, w } = review();
    await tick(3000);
    w.submitted = { at: Date.now(), confirmation: OTHER };
    await tick(1000);
    expect(t.last()?.submission).toBeDefined();
    await t.session.stop('submitted');
    expect(t.last()).toMatchObject({
      endedReason: 'submitted',
      confirmedHow: 'manual',
      confirmation: OTHER,
    });
    expect(count(t.calls, 'serverStop')).toBe(1);
    expect(count(t.calls, 'workerStop')).toBe(1);
    t.session.dispose();
  });

  it('"I submitted it" is a manual confirmation with no number, even while checking', async () => {
    const { t, w } = review();
    await tick(3000);
    w.submitted = { at: Date.now() };
    await tick(1000);
    await t.session.stop('submitted');
    expect(t.last()).toMatchObject({ confirmedHow: 'manual' });
    expect(t.last()?.confirmation).toBeUndefined();
    t.session.dispose();
  });

  it('the number never reaches the recents, the server stop or any worker call', async () => {
    const { t, w } = review();
    await tick(3000);
    w.submitted = { at: Date.now(), confirmation: NUMBER };
    await tick(1000);
    await t.session.confirmSubmission();
    expect(t.last()?.confirmation).toEqual(NUMBER);
    const sent = JSON.stringify([
      t.calls,
      t.api.stopRun.mock.calls,
      t.api.getRunState.mock.calls,
      t.markRecentEnded.mock.calls,
      t.worker.stopRun.mock.calls,
      t.worker.armSubmitWatch.mock.calls,
      t.worker.disarmSubmitWatch.mock.calls,
      t.worker.registrationComplete.mock.calls,
    ]);
    expect(sent).not.toContain('AB-1234');
    t.session.dispose();
  });

  it('a token refresh in review loses the watch and it is armed again', async () => {
    // Refresh due at the 6 s poll (5 minutes and 5 seconds left at start).
    const { t, w } = review({ expiresAt: NOW / 1000 + 305 });
    t.plan.workerStart = async () => {
      // A same-run startRun is an endRun plus begin: the watch is gone.
      w.armed = false;
      w.submitted = undefined;
    };
    await tick(12_000);
    const start = t.calls.findIndex((c) => c.startsWith('workerStart:'));
    expect(start).toBeGreaterThan(-1);
    const after = t.calls.slice(start).filter((c) => c.startsWith('arm:'));
    expect(after).toEqual([ARM]);
    expect(w.armed).toBe(true);
    t.session.dispose();
  });
});
