import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { AppApiError, type AppApi } from '../../extension/src/app-api';
import {
  messageFor,
  StartInFlightError,
  TabGoneError,
  TabMovedError,
} from '../../extension/panel/lib/copy';
import { startRun } from '../../extension/panel/lib/run-starter';
import type { StartRequest } from '../../extension/panel/lib/start-request';
import {
  type WorkerApi,
  WorkerError,
} from '../../extension/panel/lib/worker-api';

const APP = 'https://app.example.org';
const RUN_ID = '11111111-2222-4333-8444-555555555555';
const NOW = 1_800_000_000_000;
const SECRET_NAME = 'Zelda Quill-Secret';
const SECRET_CITY = 'Hyrule-Town';

const request: StartRequest = {
  origin: 'https://example.gov',
  participant: { details: { first_name: SECRET_NAME, city: SECRET_CITY } },
  model: 'claude-haiku-4-5',
};

let calls: string[];
let store: Record<string, unknown>;
let storageFails: boolean;

type FetchPlan =
  | { kind: 'status'; status: number }
  | { kind: 'reject' }
  | { kind: 'chunk' }
  | { kind: 'empty' }
  | { kind: 'readRejects' }
  | { kind: 'hang' }
  | { kind: 'nullBody' };

const makeFetch = (plan: FetchPlan) => {
  const state = {
    cancelled: false,
    reads: 0,
    init: undefined as RequestInit | undefined,
    url: '',
    bodyCancelled: false,
  };
  const fetchFn = vi.fn(async (url: string, init?: RequestInit) => {
    calls.push('fetch');
    state.url = url;
    state.init = init;
    const signal = init?.signal;
    if (plan.kind === 'reject') throw new TypeError('Failed to fetch');
    if (plan.kind === 'status') {
      return {
        ok: plan.status >= 200 && plan.status < 300,
        status: plan.status,
        text: async () => `server says ${SECRET_NAME}`,
        body: {
          cancel: async () => {
            state.bodyCancelled = true;
          },
          getReader: () => {
            throw new Error('must not read an error body');
          },
        },
      } as unknown as Response;
    }
    if (plan.kind === 'hang') {
      return new Promise<Response>((_resolve, reject) => {
        signal?.addEventListener('abort', () =>
          reject(new DOMException('aborted', 'AbortError')),
        );
      });
    }
    if (plan.kind === 'nullBody') {
      return { ok: true, status: 200, body: null } as unknown as Response;
    }
    const reader = {
      read: vi.fn(() => {
        state.reads += 1;
        if (plan.kind === 'chunk') {
          if (state.reads === 1) {
            return Promise.resolve({
              done: false,
              value: new TextEncoder().encode('data: {"type":"start"}\n\n'),
            });
          }
          // Never resolves: the turn keeps going after the first chunk.
          return new Promise(() => {});
        }
        if (plan.kind === 'empty') {
          return Promise.resolve({ done: true, value: undefined });
        }
        return Promise.reject(new TypeError('network error'));
      }),
      cancel: vi.fn(async () => {
        state.cancelled = true;
      }),
    };
    return {
      ok: true,
      status: 200,
      body: { getReader: () => reader },
    } as unknown as Response;
  });
  return { fetchFn, state };
};

const makeApi = (over: Partial<AppApi> = {}): AppApi => {
  const api = {
    getExtensionToken: vi.fn(async (runId: string) => {
      calls.push(`token:${runId}`);
      return {
        extensionToken: 'ext-token',
        relayUrl: 'wss://relay.example.org',
        expiresAt: 1_800_001_800,
      };
    }),
    stopRun: vi.fn(async (runId: string) => {
      calls.push(`apiStop:${runId}`);
      return { ended: true, alreadyEnded: false };
    }),
    ...over,
  };
  return api as unknown as AppApi;
};

const makeWorker = (over: Partial<WorkerApi> = {}): WorkerApi => {
  const worker = {
    startRun: vi.fn(async (args: { tabId: number }) => {
      calls.push(`workerStart:${args.tabId}`);
    }),
    registrationComplete: vi.fn(async (runId: string, count: number) => {
      calls.push(`registered:${runId}:${count}`);
    }),
    stopRun: vi.fn(async () => {
      calls.push('workerStop');
    }),
    ...over,
  };
  return worker as unknown as WorkerApi;
};

const tabs = (tab: { id?: number; url?: string } | undefined) => ({
  get: vi.fn(async (tabId: number) => {
    calls.push(`tab:${tabId}`);
    return tab;
  }),
});

const deps = (
  over: {
    plan?: FetchPlan;
    api?: AppApi;
    worker?: WorkerApi;
    tab?: { id?: number; url?: string } | undefined;
  } = {},
) => {
  const { fetchFn, state } = makeFetch(over.plan ?? { kind: 'chunk' });
  return {
    state,
    deps: {
      api: over.api ?? makeApi(),
      worker: over.worker ?? makeWorker(),
      fetch: fetchFn as unknown as typeof fetch,
      appOrigin: APP,
      newId: () => RUN_ID,
      now: () => NOW,
      tabs: tabs(
        'tab' in over ? over.tab : { id: 7, url: 'https://example.gov/apply' },
      ),
    },
  };
};

const args = (siteName = 'Example Benefits') => ({
  tabId: 7,
  request,
  siteName,
});

const consoleSpies: ReturnType<typeof vi.spyOn>[] = [];

beforeEach(() => {
  calls = [];
  store = {};
  storageFails = false;
  (globalThis as unknown as { chrome: unknown }).chrome = {
    storage: {
      local: {
        get: async (key: string) => {
          if (storageFails) throw new Error('boom');
          return key in store ? { [key]: store[key] } : {};
        },
        set: async (items: Record<string, unknown>) => {
          if (storageFails) throw new Error('boom');
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
  // No console call at all from the starter, so no applicant detail either.
  for (const spy of consoleSpies) expect(spy).not.toHaveBeenCalled();
  for (const spy of consoleSpies.splice(0)) spy.mockRestore();
  vi.useRealTimers();
});

const expectNoDetail = (value: unknown) => {
  const text = JSON.stringify(value, Object.getOwnPropertyNames(value ?? {}));
  expect(text).not.toContain(SECRET_NAME);
  expect(text).not.toContain(SECRET_CITY);
  if (value instanceof Error) {
    expect(value.message).not.toContain(SECRET_NAME);
    expect(value.message).not.toContain(SECRET_CITY);
    expect(String(value.stack)).not.toContain(SECRET_NAME);
  }
};

describe('startRun: success', () => {
  it('runs the steps in order with the exact request', async () => {
    const { deps: d, state } = deps();
    const result = await startRun(d, args());
    expect(result).toEqual({
      ok: true,
      runId: RUN_ID,
      expiresAt: 1_800_001_800,
      tabId: 7,
    });
    expect(calls).toEqual([
      'tab:7',
      'fetch',
      `token:${RUN_ID}`,
      'workerStart:7',
      `registered:${RUN_ID}:0`,
    ]);
    expect(state.url).toBe(`${APP}/api/eve-chat`);
    const init = state.init as RequestInit;
    expect(init.method).toBe('POST');
    expect(init.headers).toEqual({ 'Content-Type': 'application/json' });
    expect(init.credentials).toBe('include');
    expect(init.cache).toBe('no-store');
    expect(init.redirect).toBe('error');
    expect(init.signal).toBeInstanceOf(AbortSignal);
    // Sent verbatim, details included, and nothing else.
    expect(JSON.parse(init.body as string)).toEqual({
      id: RUN_ID,
      relay: request,
    });
    expect(d.worker.startRun).toHaveBeenCalledWith({
      relayUrl: 'wss://relay.example.org',
      token: 'ext-token',
      tabId: 7,
    });
    expect(d.api.stopRun).not.toHaveBeenCalled();
    expect(d.worker.stopRun).not.toHaveBeenCalled();
    expectNoDetail(result);
  });

  it('cancels the reader after the first chunk and does not wait for the turn', async () => {
    const { deps: d, state } = deps();
    const result = await startRun(d, args());
    expect(result.ok).toBe(true);
    expect(state.reads).toBe(1);
    expect(state.cancelled).toBe(true);
  });

  it('records the run in recents, without the participant', async () => {
    const { deps: d } = deps();
    await startRun(d, args());
    expect(store.recentRuns).toEqual([
      { runId: RUN_ID, siteName: 'Example Benefits', startedAt: NOW },
    ]);
    expectNoDetail(store);
  });

  it('truncates the site name to 100 characters', async () => {
    const { deps: d } = deps();
    const long = `${'a'.repeat(240)}.example.gov`;
    const result = await startRun(d, args(long));
    expect(result.ok).toBe(true);
    const stored = store.recentRuns as { siteName: string }[];
    expect(stored[0].siteName).toBe('a'.repeat(100));
  });

  it('a storage failure in addRecent does not fail the start', async () => {
    storageFails = true;
    const { deps: d } = deps();
    const result = await startRun(d, args());
    expect(result).toEqual({
      ok: true,
      runId: RUN_ID,
      expiresAt: 1_800_001_800,
      tabId: 7,
    });
    expect(d.api.stopRun).not.toHaveBeenCalled();
  });

  it('an invalid recent (addRecent throws) does not fail the start', async () => {
    const { deps: d } = deps();
    // An id that is not a UUID makes addRecent throw 'invalid recent'.
    const result = await startRun({ ...d, newId: () => 'not-a-uuid' }, args());
    expect(result.ok).toBe(true);
    expect(store.recentRuns).toBeUndefined();
  });
});

describe('startRun: before any request', () => {
  it('refuses a closed tab with no request', async () => {
    const { deps: d } = deps({ tab: undefined });
    const result = await startRun(d, args());
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.error).toBeInstanceOf(TabGoneError);
    expect(result.runId).toBeUndefined();
    expect(messageFor(result.error).title).toBe('The tab was closed');
    expect(calls).toEqual(['tab:7']);
  });

  it('treats a tabs.get rejection as a closed tab', async () => {
    const { deps: d } = deps();
    const result = await startRun(
      {
        ...d,
        tabs: {
          get: async () => {
            throw new Error('No tab with id: 7');
          },
        },
      },
      args(),
    );
    expect(!result.ok && result.error).toBeInstanceOf(TabGoneError);
    expect(d.fetch).not.toHaveBeenCalled();
  });

  it('refuses a tab on another origin with no request', async () => {
    const { deps: d } = deps({
      tab: { id: 7, url: 'https://elsewhere.example.com/x' },
    });
    const result = await startRun(d, args());
    expect(!result.ok && result.error).toBeInstanceOf(TabMovedError);
    if (result.ok) return;
    expect(messageFor(result.error).title).toBe('The tab left the site');
    expect(calls).toEqual(['tab:7']);
    expect(d.api.stopRun).not.toHaveBeenCalled();
  });

  it('refuses a tab with no readable URL', async () => {
    const { deps: d } = deps({ tab: { id: 7 } });
    const result = await startRun(d, args());
    expect(!result.ok && result.error).toBeInstanceOf(TabMovedError);
  });
});

describe('startRun: the eve-chat request fails', () => {
  const statuses: [number, AppApiError['code']][] = [
    [401, 'signed_out'],
    [403, 'forbidden'],
    [429, 'rate_limited'],
    [400, 'bad_request'],
    [500, 'server_error'],
    [503, 'server_error'],
    [404, 'server_error'],
  ];
  for (const [status, code] of statuses) {
    it(`maps ${status} to ${code} and stops the server run`, async () => {
      const { deps: d, state } = deps({ plan: { kind: 'status', status } });
      const result = await startRun(d, args());
      expect(result.ok).toBe(false);
      if (result.ok) return;
      expect(result.error).toBeInstanceOf(AppApiError);
      expect((result.error as AppApiError).code).toBe(code);
      expect((result.error as AppApiError).status).toBe(status);
      expect(result.runId).toBe(RUN_ID);
      expect(calls).toEqual(['tab:7', 'fetch', `apiStop:${RUN_ID}`]);
      expect(d.worker.stopRun).not.toHaveBeenCalled();
      // The body is never read; it is released.
      expect(state.bodyCancelled).toBe(true);
      expectNoDetail(result.error);
    });
  }

  it('the run limit shows the run-limit copy', async () => {
    const { deps: d } = deps({ plan: { kind: 'status', status: 429 } });
    const result = await startRun(d, args());
    expect(!result.ok && messageFor(result.error).title).toBe(
      'Too many runs right now',
    );
  });

  const network: [string, FetchPlan][] = [
    ['fetch rejects', { kind: 'reject' }],
    ['the body ends without a chunk', { kind: 'empty' }],
    ['the read rejects before the first chunk', { kind: 'readRejects' }],
    ['there is no body', { kind: 'nullBody' }],
  ];
  for (const [name, plan] of network) {
    it(`is a network error when ${name}, and stops the server run`, async () => {
      const { deps: d } = deps({ plan });
      const result = await startRun(d, args());
      expect(result.ok).toBe(false);
      if (result.ok) return;
      expect(result.error).toBeInstanceOf(AppApiError);
      expect((result.error as AppApiError).code).toBe('network');
      expect(result.runId).toBe(RUN_ID);
      expect(d.api.stopRun).toHaveBeenCalledTimes(1);
      expect(d.api.stopRun).toHaveBeenCalledWith(RUN_ID);
      expect(d.api.getExtensionToken).not.toHaveBeenCalled();
      expect(d.worker.stopRun).not.toHaveBeenCalled();
      expectNoDetail(result.error);
    });
  }

  it('times out after 15 seconds and stops the server run', async () => {
    vi.useFakeTimers();
    const { deps: d } = deps({ plan: { kind: 'hang' } });
    const pending = startRun(d, args());
    await vi.advanceTimersByTimeAsync(14_999);
    expect(d.api.stopRun).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(1);
    const result = await pending;
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect((result.error as AppApiError).code).toBe('network');
    expect(d.api.stopRun).toHaveBeenCalledWith(RUN_ID);
  });

  it('times out when the first chunk never comes', async () => {
    vi.useFakeTimers();
    const { deps: d } = deps();
    // A reader whose first read never resolves until aborted.
    const hangingRead = vi.fn(async (_url: string, init?: RequestInit) => {
      calls.push('fetch');
      return {
        ok: true,
        status: 200,
        body: {
          getReader: () => ({
            read: () =>
              new Promise((_resolve, reject) => {
                init?.signal?.addEventListener('abort', () =>
                  reject(new DOMException('aborted', 'AbortError')),
                );
              }),
            cancel: async () => {},
          }),
        },
      } as unknown as Response;
    });
    const pending = startRun(
      { ...d, fetch: hangingRead as unknown as typeof fetch },
      args(),
    );
    await vi.advanceTimersByTimeAsync(15_000);
    const result = await pending;
    expect(!result.ok && (result.error as AppApiError).code).toBe('network');
    expect(d.api.stopRun).toHaveBeenCalledWith(RUN_ID);
  });

  it('ignores a failing server stop', async () => {
    const api = makeApi({
      stopRun: vi.fn(async () => {
        throw new AppApiError('network', 0);
      }),
    });
    const { deps: d } = deps({ plan: { kind: 'status', status: 500 } });
    const result = await startRun({ ...d, api }, args());
    expect(!result.ok && (result.error as AppApiError).code).toBe(
      'server_error',
    );
  });
});

describe('startRun: after the run exists', () => {
  it('a token route 404 stops the server run, not the worker', async () => {
    const api = makeApi({
      getExtensionToken: vi.fn(async () => {
        calls.push('token');
        throw new AppApiError('not_found', 404);
      }),
    });
    const { deps: d } = deps({ api });
    const result = await startRun(d, args());
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect((result.error as AppApiError).code).toBe('not_found');
    expect(result.runId).toBe(RUN_ID);
    expect(calls).toEqual(['tab:7', 'fetch', 'token', `apiStop:${RUN_ID}`]);
    expect(d.worker.stopRun).not.toHaveBeenCalled();
  });

  it('a worker refusal stops the server run, not the worker', async () => {
    const worker = makeWorker({
      startRun: vi.fn(async () => {
        calls.push('workerStart');
        throw new WorkerError('tab_out_of_scope');
      }),
    });
    const { deps: d } = deps({ worker });
    const result = await startRun(d, args());
    expect(!result.ok && result.error).toBeInstanceOf(WorkerError);
    expect(calls).toEqual([
      'tab:7',
      'fetch',
      `token:${RUN_ID}`,
      'workerStart',
      `apiStop:${RUN_ID}`,
    ]);
    expect(worker.stopRun).not.toHaveBeenCalled();
  });

  it('a registrationComplete refusal stops the server run, then the worker', async () => {
    const worker = makeWorker({
      registrationComplete: vi.fn(async () => {
        calls.push('registered');
        throw new WorkerError('wrong_run');
      }),
    });
    const { deps: d } = deps({ worker });
    const result = await startRun(d, args());
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect((result.error as WorkerError).code).toBe('wrong_run');
    expect(result.runId).toBe(RUN_ID);
    expect(calls).toEqual([
      'tab:7',
      'fetch',
      `token:${RUN_ID}`,
      'workerStart:7',
      'registered',
      `apiStop:${RUN_ID}`,
      'workerStop',
    ]);
  });

  it('ignores a failing worker stop', async () => {
    const worker = makeWorker({
      registrationComplete: vi.fn(async () => {
        throw new WorkerError('worker_error');
      }),
      stopRun: vi.fn(async () => {
        throw new WorkerError('worker_unreachable');
      }),
    });
    const { deps: d } = deps({ worker });
    const result = await startRun(d, args());
    expect(!result.ok && (result.error as WorkerError).code).toBe(
      'worker_error',
    );
  });
});

describe('startRun: one start at a time', () => {
  it('refuses a second start with zero requests, and the first completes', async () => {
    let release: () => void = () => {};
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    const first = deps();
    const slowTabs = {
      get: vi.fn(async (tabId: number) => {
        calls.push(`tab:${tabId}`);
        await gate;
        return { id: 7, url: 'https://example.gov/apply' };
      }),
    };
    const pending = startRun({ ...first.deps, tabs: slowTabs }, args());

    const second = deps();
    const refused = await startRun(second.deps, args());
    expect(refused.ok).toBe(false);
    if (refused.ok) return;
    expect(refused.error).toBeInstanceOf(StartInFlightError);
    expect(refused.runId).toBeUndefined();
    expect(second.deps.tabs.get).not.toHaveBeenCalled();
    expect(second.deps.fetch).not.toHaveBeenCalled();
    expect(second.deps.api.stopRun).not.toHaveBeenCalled();
    expect(second.deps.worker.startRun).not.toHaveBeenCalled();
    expect(messageFor(refused.error).title).toBe('A run is already starting');

    release();
    expect((await pending).ok).toBe(true);
    expect(first.deps.api.stopRun).not.toHaveBeenCalled();
  });

  it('clears the in-flight flag after a success', async () => {
    expect((await startRun(deps().deps, args())).ok).toBe(true);
    expect((await startRun(deps().deps, args())).ok).toBe(true);
  });

  it('clears the in-flight flag after a failure', async () => {
    const failed = await startRun(
      deps({ plan: { kind: 'status', status: 500 } }).deps,
      args(),
    );
    expect(failed.ok).toBe(false);
    expect((await startRun(deps().deps, args())).ok).toBe(true);
  });

  it('clears the in-flight flag after a tab check failure', async () => {
    await startRun(deps({ tab: undefined }).deps, args());
    expect((await startRun(deps().deps, args())).ok).toBe(true);
  });
});

describe('startRun: no applicant detail leaks', () => {
  it('in any error, for every failure path', async () => {
    const plans: FetchPlan[] = [
      { kind: 'status', status: 400 },
      { kind: 'reject' },
      { kind: 'readRejects' },
      { kind: 'empty' },
    ];
    for (const plan of plans) {
      const result = await startRun(deps({ plan }).deps, args());
      expect(result.ok).toBe(false);
      expectNoDetail(result);
      if (!result.ok) expectNoDetail(result.error);
    }
    for (const tab of [undefined, { id: 7, url: 'https://other.example' }]) {
      const result = await startRun(deps({ tab }).deps, args());
      expect(result.ok).toBe(false);
      expectNoDetail(result);
    }
    expectNoDetail(store);
  });
});
