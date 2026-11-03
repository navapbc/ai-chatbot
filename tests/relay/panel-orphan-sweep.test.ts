import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  MAX_SWEEP_STOPS,
  type SweepDeps,
  sweepStaleRecents,
} from '../../extension/panel/lib/orphan-sweep';
import {
  addRecent,
  loadRecents,
  markRecentEnded,
  type Recent,
} from '../../extension/panel/lib/recents';
import { AppApiError } from '../../extension/src/app-api';
import type { RunStatus } from '../../extension/src/bridge';

const NOW = 1_800_000_000_000;
const SEC = 1000;
const MIN = 60 * SEC;

const id = (n: number) =>
  `00000000-0000-4000-8000-${String(n).padStart(12, '0')}`;
const rec = (n: number, ageMs: number, over: Partial<Recent> = {}): Recent => ({
  runId: id(n),
  siteName: `Site ${n}`,
  startedAt: NOW - ageMs,
  ...over,
});

const OLD = 2 * MIN;
const IDLE: RunStatus = { ok: true, active: false };
const holding = (n: number): RunStatus => ({
  ok: true,
  active: true,
  runId: id(n),
  origin: 'https://example.gov',
  tabId: 7,
});

let store: Record<string, unknown>;
let stops: string[];
const consoleSpies: ReturnType<typeof vi.spyOn>[] = [];

beforeEach(() => {
  store = {};
  stops = [];
  (globalThis as unknown as { chrome: unknown }).chrome = {
    storage: {
      local: {
        get: async (key: string) => (key in store ? { [key]: store[key] } : {}),
        set: async (items: Record<string, unknown>) => {
          Object.assign(store, JSON.parse(JSON.stringify(items)));
        },
      },
    },
  };
  for (const method of ['log', 'info', 'warn', 'error', 'debug'] as const) {
    consoleSpies.push(vi.spyOn(console, method));
  }
});

afterEach(() => {
  for (const spy of consoleSpies) expect(spy).not.toHaveBeenCalled();
  for (const spy of consoleSpies.splice(0)) spy.mockRestore();
});

const seed = async (...recents: Recent[]) => {
  for (const r of recents) await addRecent(r, NOW);
};

const reasonOf = async (n: number) =>
  (await loadRecents(NOW)).find((r) => r.runId === id(n))?.endedReason;

type Over = {
  status?: RunStatus | (() => Promise<RunStatus>);
  stop?: (runId: string) => Promise<unknown>;
  loadRecents?: SweepDeps['loadRecents'];
  startingRunId?: () => string | undefined;
};

const sweepDeps = (over: Over = {}): SweepDeps => {
  const status = ('status' in over ? over.status : IDLE) as
    | RunStatus
    | (() => Promise<RunStatus>);
  return {
    api: {
      stopRun: vi.fn(async (runId: string) => {
        stops.push(runId);
        if (over.stop) return over.stop(runId);
        return { ended: true, alreadyEnded: false };
      }),
    },
    worker: {
      getRunStatus: vi.fn(
        typeof status === 'function' ? status : async () => status,
      ),
    },
    loadRecents: over.loadRecents ?? loadRecents,
    markRecentEnded,
    now: () => NOW,
    startingRunId: over.startingRunId,
  };
};

describe('sweepStaleRecents: what is stopped', () => {
  it('a recent left by a panel closed during Starting is stopped on the next open', async () => {
    // The starter wrote the recent before the POST; the panel closed, so no
    // end reason was ever written and the worker holds nothing.
    await seed(rec(1, OLD));
    await sweepStaleRecents(sweepDeps());
    expect(stops).toEqual([id(1)]);
    expect(await reasonOf(1)).toBe('swept');
  });

  it('a run the worker ended while the panel was closed (I1) is stopped', async () => {
    await seed(rec(1, 40 * MIN));
    await sweepStaleRecents(sweepDeps());
    expect(stops).toEqual([id(1)]);
    expect(await reasonOf(1)).toBe('swept');
  });

  it('a recent that already ended is never stopped', async () => {
    await seed(
      rec(1, OLD, { endedReason: 'user' }),
      rec(2, OLD, { endedReason: 'start-failed' }),
      rec(3, OLD, { endedReason: 'swept' }),
    );
    await sweepStaleRecents(sweepDeps());
    expect(stops).toEqual([]);
  });

  it('marks nothing and stops nothing when every recent ended (no requests)', async () => {
    const d = sweepDeps();
    await sweepStaleRecents(d);
    expect(d.api.stopRun).not.toHaveBeenCalled();
  });
});

describe('sweepStaleRecents: never a healthy run', () => {
  it('never stops the run the worker holds (and still stops one it does not)', async () => {
    await seed(rec(1, 30 * MIN), rec(2, 20 * MIN));
    await sweepStaleRecents(sweepDeps({ status: holding(1) }));
    expect(stops).toEqual([id(2)]);
    expect(await reasonOf(1)).toBeUndefined();
    expect(await reasonOf(2)).toBe('swept');
  });

  it('skips a recent younger than 90 seconds (and still stops an old one)', async () => {
    await seed(rec(1, 89 * SEC), rec(2, OLD));
    await sweepStaleRecents(sweepDeps());
    expect(stops).toEqual([id(2)]);
    expect(await reasonOf(1)).toBeUndefined();
  });

  it('skips a recent whose start time is in the future', async () => {
    await seed(rec(1, -10 * MIN), rec(2, OLD));
    await sweepStaleRecents(sweepDeps());
    expect(stops).toEqual([id(2)]);
  });

  it('skips the whole sweep while the worker is restarting (a resume, no run id)', async () => {
    await seed(rec(1, OLD), rec(2, 30 * MIN));
    const d = sweepDeps({
      status: { ok: true, active: false, restarting: true },
    });
    await sweepStaleRecents(d);
    expect(d.api.stopRun).not.toHaveBeenCalled();
    expect(await reasonOf(1)).toBeUndefined();
  });

  it('skips the whole sweep during a same-run refresh of another recent', async () => {
    await seed(rec(1, OLD), rec(2, 30 * MIN));
    const d = sweepDeps({
      status: { ok: true, active: false, restarting: true, runId: id(2) },
    });
    await sweepStaleRecents(d);
    expect(d.api.stopRun).not.toHaveBeenCalled();
  });

  it('skips the whole sweep when the worker status cannot be read', async () => {
    await seed(rec(1, OLD));
    const d = sweepDeps({
      status: async () => {
        throw new Error('unreachable');
      },
    });
    await sweepStaleRecents(d);
    expect(d.api.stopRun).not.toHaveBeenCalled();
  });

  it('skips the whole sweep on a malformed status', async () => {
    await seed(rec(1, OLD));
    for (const bad of [
      { ok: true } as unknown as RunStatus,
      { ok: false, active: false } as unknown as RunStatus,
      { ok: true, active: true } as RunStatus,
      null as unknown as RunStatus,
    ]) {
      const d = sweepDeps({ status: bad });
      await sweepStaleRecents(d);
      expect(d.api.stopRun).not.toHaveBeenCalled();
    }
  });

  it("never stops this panel's own start in progress", async () => {
    await seed(rec(1, OLD), rec(2, OLD + SEC));
    await sweepStaleRecents(sweepDeps({ startingRunId: () => id(1) }));
    expect(stops).toEqual([id(2)]);
  });

  it('re-reads the worker before each stop: a run taken mid-sweep is left alone', async () => {
    await seed(rec(1, OLD), rec(2, 30 * MIN));
    let reads = 0;
    const d = sweepDeps({
      // Idle at first; by the second candidate the worker holds run 2.
      status: async () => (++reads <= 2 ? IDLE : holding(2)),
    });
    await sweepStaleRecents(d);
    expect(stops).toEqual([id(1)]);
  });

  it('stops the sweep when the worker starts restarting mid-sweep', async () => {
    await seed(rec(1, OLD), rec(2, 30 * MIN));
    let reads = 0;
    const d = sweepDeps({
      status: async () =>
        ++reads <= 2 ? IDLE : { ok: true, active: false, restarting: true },
    });
    await sweepStaleRecents(d);
    expect(stops).toEqual([id(1)]);
  });

  it("two windows: the second open does not stop the first window's live run", async () => {
    // Window A started run 1 a while ago and is Running: the worker holds it.
    // Window A's newer start (run 2) is still Starting: recorded, not yet
    // held, and young.
    await seed(rec(1, 10 * MIN), rec(2, 20 * SEC));
    // Window B opens.
    await sweepStaleRecents(sweepDeps({ status: holding(1) }));
    expect(stops).toEqual([]);
    expect(await reasonOf(1)).toBeUndefined();
    expect(await reasonOf(2)).toBeUndefined();
  });

  it('two windows: a run Starting in the other window (worker idle) is young, so skipped', async () => {
    await seed(rec(1, 30 * SEC));
    await sweepStaleRecents(sweepDeps());
    expect(stops).toEqual([]);
  });
});

describe('sweepStaleRecents: answers from the stop route', () => {
  it('an already-ended answer marks it swept', async () => {
    await seed(rec(1, OLD));
    await sweepStaleRecents(
      sweepDeps({ stop: async () => ({ ended: true, alreadyEnded: true }) }),
    );
    expect(await reasonOf(1)).toBe('swept');
  });

  it('a 404 is final only for a recent older than 5 minutes', async () => {
    await seed(rec(1, 5 * MIN + SEC), rec(2, 4 * MIN));
    await sweepStaleRecents(
      sweepDeps({
        stop: async () => {
          throw new AppApiError('not_found', 404);
        },
      }),
    );
    expect(stops).toEqual([id(2), id(1)]);
    expect(await reasonOf(1)).toBe('swept');
    // Left for the next open: the row may still be created.
    expect(await reasonOf(2)).toBeUndefined();
  });

  it('other failures leave the recent for the next open', async () => {
    const errors = [
      new AppApiError('signed_out', 401),
      new AppApiError('forbidden', 403),
      new AppApiError('server_error', 500),
      new AppApiError('network', 0),
      new Error('boom'),
    ];
    for (const error of errors) {
      store = {};
      await seed(rec(1, 30 * MIN));
      await sweepStaleRecents(
        sweepDeps({
          stop: async () => {
            throw error;
          },
        }),
      );
      expect(await reasonOf(1)).toBeUndefined();
    }
  });

  it('a throwing stop does not break the sweep or the open', async () => {
    await seed(rec(1, 30 * MIN), rec(2, 20 * MIN));
    const result = sweepStaleRecents(
      sweepDeps({
        stop: async (runId) => {
          if (runId === id(2)) throw new TypeError('boom');
          return { ended: true, alreadyEnded: false };
        },
      }),
    );
    await expect(result).resolves.toBeUndefined();
    expect(stops).toEqual([id(2), id(1)]);
    expect(await reasonOf(1)).toBe('swept');
  });

  it('a throwing recents load resolves quietly', async () => {
    const d = sweepDeps({
      loadRecents: async () => {
        throw new Error('boom');
      },
    });
    await expect(sweepStaleRecents(d)).resolves.toBeUndefined();
    expect(d.api.stopRun).not.toHaveBeenCalled();
  });

  it(`makes at most ${MAX_SWEEP_STOPS} stop calls`, async () => {
    const many = Array.from({ length: 15 }, (_, i) => rec(i + 1, OLD + i));
    const d = sweepDeps({ loadRecents: async () => many });
    await sweepStaleRecents(d);
    expect(d.api.stopRun).toHaveBeenCalledTimes(MAX_SWEEP_STOPS);
  });

  it('sends only run ids to the stop route', async () => {
    await seed(rec(1, OLD));
    const d = sweepDeps();
    await sweepStaleRecents(d);
    expect(d.api.stopRun).toHaveBeenCalledWith(id(1));
  });
});
