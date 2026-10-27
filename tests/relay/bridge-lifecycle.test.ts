import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

// The bridge's run lifecycle against a fake chrome and relay socket:
// serialized startRun, a stop issued while the socket is down, the detach
// timer (P4b-d), the run status after a worker restart, and the test hooks.
// Each test loads a fresh bridge module so no run, queue or pending stop
// leaks between tests.

const ORIGIN = 'https://a.gov';
const RELAY = 'ws://127.0.0.1:1';
const REGISTRATION_REFUSED =
  'screenshots are refused until known values are registered again';

interface Sent {
  type: string;
  id?: string;
  reason?: string;
  error?: string;
  result?: unknown;
}

class FakeSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;
  static all: FakeSocket[] = [];
  readyState = 0;
  sent: Sent[] = [];
  closedByClient = false;
  onopen: null | (() => unknown) = null;
  onmessage: null | ((event: { data: string }) => unknown) = null;
  onclose: null | ((event: { code: number }) => unknown) = null;
  constructor(readonly url: string) {
    FakeSocket.all.push(this);
  }
  send(data: string) {
    if (this.readyState !== 1) throw new Error('not open');
    this.sent.push(JSON.parse(data));
  }
  close() {
    this.closedByClient = true;
    this.readyState = 3;
  }
  /** The relay accepts the connection. */
  async open() {
    this.readyState = 1;
    await this.onopen?.();
  }
  /** The connection drops (or the relay closes it) with `code`. */
  drop(code = 1006) {
    this.readyState = 3;
    this.onclose?.({ code });
  }
}

const live = () => FakeSocket.all.filter((s) => s.readyState !== 3);

const b64url = (value: unknown) =>
  Buffer.from(JSON.stringify(value)).toString('base64url');

const tokenFor = (runId: string, expInSeconds = 3600) =>
  `${b64url({
    runId,
    role: 'extension',
    origin: ORIGIN,
    exp: Math.floor(Date.now() / 1000) + expInSeconds,
  })}.sig`;

let storage: Map<string, unknown>;
const attach = vi.fn(async () => {});
const detach = vi.fn(async () => {});

function stubChrome() {
  storage = new Map();
  vi.stubGlobal('WebSocket', FakeSocket);
  vi.stubGlobal('chrome', {
    runtime: {
      id: 'ext',
      onMessage: { addListener: () => {} },
    },
    tabs: {
      get: async () => ({ url: `${ORIGIN}/form` }),
      onUpdated: { addListener: () => {} },
      onRemoved: { addListener: () => {} },
    },
    storage: {
      session: {
        set: async (items: Record<string, unknown>) => {
          for (const [k, v] of Object.entries(items)) storage.set(k, v);
        },
        get: async (key: string) =>
          storage.has(key) ? { [key]: storage.get(key) } : {},
        remove: async (key: string) => {
          storage.delete(key);
        },
      },
    },
    debugger: {
      getTargets: async () => [
        {
          tabId: 1,
          type: 'page',
          id: 'T',
          title: '',
          url: '',
          attached: false,
        },
      ],
      attach,
      detach,
      sendCommand: async (_d: unknown, method: string) => {
        if (method === 'Page.getFrameTree') {
          return { frameTree: { frame: { id: 'T', url: `${ORIGIN}/form` } } };
        }
        return {};
      },
      onEvent: { addListener: () => {} },
      onDetach: { addListener: () => {} },
    },
  });
}

let bridge: typeof import('@/extension/src/bridge');

beforeEach(async () => {
  vi.resetModules();
  FakeSocket.all = [];
  attach.mockClear();
  detach.mockClear();
  stubChrome();
  bridge = await import('@/extension/src/bridge');
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.doUnmock('@/extension/src/config');
  vi.doUnmock('@/extension/src/jev-classifier');
});

const start = (token: string) =>
  bridge.handleMessage({ type: 'startRun', relayUrl: RELAY, token, tabId: 1 });

const status = () =>
  bridge.handleMessage({ type: 'getRunStatus' }) as Promise<
    Record<string, unknown>
  >;

/** Sends a relay command and returns the bridge's reply to it. */
const ask = async (socket: FakeSocket, message: Record<string, unknown>) => {
  await socket.onmessage?.({ data: JSON.stringify(message) });
  const reply = socket.sent.find(
    (m) => m.type === 'reply' && m.id === message.id,
  );
  if (!reply) throw new Error(`no reply to ${String(message.id)}`);
  return reply;
};

/** Starts a run, opens its socket and attaches the tab. */
async function attachedRun(runId = 'run-1') {
  expect(await start(tokenFor(runId))).toEqual({ ok: true });
  const socket = FakeSocket.all.at(-1) as FakeSocket;
  await socket.open();
  expect(
    (await ask(socket, { id: 'a', type: 'attach', targetId: 'T' })).error,
  ).toBeUndefined();
  return socket;
}

describe('startRun serialization', () => {
  it('two concurrent startRun calls leave exactly one run and one socket', async () => {
    const token = tokenFor('run-1');
    const [a, b] = await Promise.all([start(token), start(token)]);
    expect(a).toEqual({ ok: true });
    expect(b).toEqual({ ok: true });
    expect(FakeSocket.all).toHaveLength(2);
    expect(live()).toHaveLength(1);
    // The first run's socket was closed before the second one was made.
    expect(FakeSocket.all[0].closedByClient).toBe(true);
    expect((await status()).active).toBe(true);
  });

  it('a later call replaces the earlier run with reason replaced', async () => {
    const first = await attachedRun('run-1');
    expect(await start(tokenFor('run-2'))).toEqual({ ok: true });
    expect(first.sent.filter((m) => m.type === 'stop')).toEqual([
      { type: 'stop', reason: 'replaced' },
    ]);
    expect((await status()).runId).toBe('run-2');
  });

  it('startRun racing stopRun leaves no run and no live socket', async () => {
    const [started, stopped] = await Promise.all([
      start(tokenFor('run-1')),
      bridge.handleMessage({ type: 'stopRun' }),
    ]);
    expect(started).toEqual({ ok: true });
    expect(stopped).toEqual({ ok: true });
    expect((await status()).active).toBe(false);
    // The run's socket is closed; the one left only delivers the stop (the
    // run's socket had not opened yet).
    expect(FakeSocket.all[0].closedByClient).toBe(true);
    expect(live()).toHaveLength(1);
    const delivery = live()[0];
    await delivery.open();
    expect(delivery.sent.map((m) => m.type)).toEqual(['register', 'stop']);
    expect(live()).toHaveLength(0);
  });

  it('a failing call does not poison the queue', async () => {
    const results = await Promise.all([
      start('not-a-token'),
      start(tokenFor('run-1')),
    ]);
    expect(results[0].ok).toBe(false);
    expect(results[1]).toEqual({ ok: true });
    expect(live()).toHaveLength(1);
  });

  it('startRun racing resumeRun leaves one run and one socket', async () => {
    storage.set('run', {
      relayUrl: RELAY,
      token: tokenFor('run-old'),
      tabId: 1,
      allowedFrameOrigins: [],
    });
    const token = tokenFor('run-new');
    await Promise.all([bridge.resumeRun(), start(token)]);
    // One run socket (the resumed run's was closed); any other live socket
    // only delivers the resumed run's stop.
    const runSockets = live().filter((s) => s.url.includes(token));
    expect(runSockets).toHaveLength(1);
    expect(FakeSocket.all[0].closedByClient).toBe(true);
    expect((await status()).runId).toBe('run-new');
  });
});

describe('stop while the socket is down', () => {
  it('delivers one stop, right after register, once a socket opens', async () => {
    vi.useFakeTimers();
    const socket = await attachedRun('run-1');
    socket.drop(1006);
    expect(await bridge.handleMessage({ type: 'stopRun' })).toEqual({
      ok: true,
    });
    // Local teardown happened at once.
    expect(detach).toHaveBeenCalled();
    expect(storage.has('run')).toBe(false);
    expect((await status()).active).toBe(false);

    const delivery = FakeSocket.all.at(-1) as FakeSocket;
    expect(delivery).not.toBe(socket);
    await delivery.open();
    expect(delivery.sent).toEqual([
      { type: 'register', runId: 'run-1', tabId: 1, origin: ORIGIN },
      { type: 'stop', reason: 'user' },
    ]);
    expect(delivery.closedByClient).toBe(true);
    const count = FakeSocket.all.length;
    await vi.advanceTimersByTimeAsync(60_000);
    expect(FakeSocket.all).toHaveLength(count);
    expect(vi.getTimerCount()).toBe(0);
  });

  it('retries a failed delivery and still sends stop only once', async () => {
    vi.useFakeTimers();
    const socket = await attachedRun('run-1');
    socket.drop(1006);
    await bridge.handleMessage({ type: 'stopRun' });
    (FakeSocket.all.at(-1) as FakeSocket).drop(1006);
    await vi.advanceTimersByTimeAsync(1_000);
    const retry = FakeSocket.all.at(-1) as FakeSocket;
    await retry.open();
    const stops = FakeSocket.all.flatMap((s) =>
      s.sent.filter((m) => m.type === 'stop'),
    );
    expect(stops).toEqual([{ type: 'stop', reason: 'user' }]);
  });

  it('drops the stop once the token has expired', async () => {
    vi.useFakeTimers();
    expect(await start(tokenFor('run-1', 3))).toEqual({ ok: true });
    const socket = FakeSocket.all.at(-1) as FakeSocket;
    await socket.open();
    socket.drop(1006);
    await bridge.handleMessage({ type: 'stopRun' });
    // Every delivery attempt fails until the token expires.
    for (let i = 0; i < 10; i++) {
      const last = FakeSocket.all.at(-1) as FakeSocket;
      if (last.readyState === 0) last.drop(1006);
      await vi.advanceTimersByTimeAsync(1_000);
    }
    const count = FakeSocket.all.length;
    await vi.advanceTimersByTimeAsync(60_000);
    expect(FakeSocket.all).toHaveLength(count);
    for (const s of FakeSocket.all) {
      if (s.readyState === 0) await s.open();
    }
    expect(
      FakeSocket.all.flatMap((s) => s.sent).filter((m) => m.type === 'stop'),
    ).toEqual([]);
    expect(vi.getTimerCount()).toBe(0);
  });

  it('gives up after 30 seconds of trying', async () => {
    vi.useFakeTimers();
    const socket = await attachedRun('run-1');
    socket.drop(1006);
    await bridge.handleMessage({ type: 'stopRun' });
    for (let i = 0; i < 40; i++) {
      const last = FakeSocket.all.at(-1) as FakeSocket;
      if (last.readyState === 0) last.drop(1006);
      await vi.advanceTimersByTimeAsync(1_000);
    }
    const count = FakeSocket.all.length;
    await vi.advanceTimersByTimeAsync(120_000);
    expect(FakeSocket.all).toHaveLength(count);
    expect(
      FakeSocket.all.flatMap((s) => s.sent).filter((m) => m.type === 'stop'),
    ).toEqual([]);
    expect(vi.getTimerCount()).toBe(0);
  });

  it('does not retry when the relay refuses the delivery socket', async () => {
    vi.useFakeTimers();
    const socket = await attachedRun('run-1');
    socket.drop(1006);
    await bridge.handleMessage({ type: 'stopRun' });
    (FakeSocket.all.at(-1) as FakeSocket).drop(1008);
    const count = FakeSocket.all.length;
    await vi.advanceTimersByTimeAsync(60_000);
    expect(FakeSocket.all).toHaveLength(count);
  });

  it('sends no stop when the relay itself closed the run', async () => {
    vi.useFakeTimers();
    const socket = await attachedRun('run-1');
    socket.drop(1000);
    await vi.advanceTimersByTimeAsync(10);
    expect((await status()).active).toBe(false);
    const count = FakeSocket.all.length;
    await vi.advanceTimersByTimeAsync(60_000);
    expect(FakeSocket.all).toHaveLength(count);
  });

  it('never puts the token in a URL it logs or a status', async () => {
    const socket = await attachedRun('run-1');
    socket.drop(1006);
    const warn = vi.spyOn(console, 'warn');
    const log = vi.spyOn(console, 'log');
    await bridge.handleMessage({ type: 'stopRun' });
    const text = JSON.stringify([warn.mock.calls, log.mock.calls]);
    expect(text).not.toContain('sig');
    expect(text).not.toContain('127.0.0.1');
  });
});

describe('debugger detach (P4b-d)', () => {
  it('ends the run 5 s after a target_closed detach with no re-attach', async () => {
    vi.useFakeTimers();
    const socket = await attachedRun('run-1');
    bridge.onDebuggerDetach({ tabId: 1 }, 'target_closed');
    await vi.advanceTimersByTimeAsync(4_999);
    expect((await status()).active).toBe(true);
    await vi.advanceTimersByTimeAsync(1);
    expect((await status()).active).toBe(false);
    expect(socket.sent.filter((m) => m.type === 'stop')).toEqual([
      { type: 'stop', reason: 'debugger-detached' },
    ]);
  });

  it('a re-attach inside 5 s keeps the run', async () => {
    vi.useFakeTimers();
    const socket = await attachedRun('run-1');
    bridge.onDebuggerDetach({ tabId: 1 }, 'target_closed');
    await vi.advanceTimersByTimeAsync(2_000);
    expect(
      (await ask(socket, { id: 'b', type: 'attach', targetId: 'T' })).error,
    ).toBeUndefined();
    await vi.advanceTimersByTimeAsync(10_000);
    expect((await status()).active).toBe(true);
    expect(socket.sent.filter((m) => m.type === 'stop')).toEqual([]);
  });

  it('canceled_by_user still ends the run at once', async () => {
    const socket = await attachedRun('run-1');
    bridge.onDebuggerDetach({ tabId: 1 }, 'canceled_by_user');
    await vi.waitFor(async () => expect((await status()).active).toBe(false));
    expect(socket.sent.filter((m) => m.type === 'stop')).toEqual([
      { type: 'stop', reason: 'debugger-canceled' },
    ]);
  });

  it('still announces the detach to the agent', async () => {
    const socket = await attachedRun('run-1');
    bridge.onDebuggerDetach({ tabId: 1 }, 'target_closed');
    expect(
      socket.sent.some(
        (m) =>
          m.type === 'event' &&
          (m as { method?: string }).method === 'Target.detachedFromTarget',
      ),
    ).toBe(true);
    await bridge.handleMessage({ type: 'stopRun' });
  });
});

describe('run status after a worker restart', () => {
  async function resumed() {
    storage.set('run', {
      relayUrl: RELAY,
      token: tokenFor('run-1'),
      tabId: 1,
      allowedFrameOrigins: [],
    });
    await bridge.resumeRun();
    const socket = FakeSocket.all.at(-1) as FakeSocket;
    await socket.open();
    expect(
      (await ask(socket, { id: 'a', type: 'attach', targetId: 'T' })).error,
    ).toBeUndefined();
    return socket;
  }

  const screenshot = (socket: FakeSocket, id: string) =>
    ask(socket, {
      id,
      type: 'cdp',
      sessionId: 'tab-1',
      method: 'Page.captureScreenshot',
      params: {},
    });

  it('reports needsRegistration and refuses screenshots until registrationComplete', async () => {
    const socket = await resumed();
    const before = await status();
    expect(before).toEqual({
      ok: true,
      active: true,
      runId: 'run-1',
      origin: ORIGIN,
      attached: true,
      needsRegistration: true,
    });
    expect((await screenshot(socket, 's1')).error).toBe(REGISTRATION_REFUSED);
    // Values can be registered again while it is set.
    expect(
      await bridge.handleMessage({ type: 'addKnownValue', value: 'Rosalind' }),
    ).toEqual({ ok: true });
    expect((await screenshot(socket, 's2')).error).toBe(REGISTRATION_REFUSED);

    expect(
      await bridge.handleMessage({ type: 'registrationComplete', count: 1 }),
    ).toEqual({ ok: true });
    expect((await status()).needsRegistration).toBe(false);
    expect((await screenshot(socket, 's3')).error).not.toBe(
      REGISTRATION_REFUSED,
    );
  });

  it('a same-run-id restart (token refresh) after values were registered needs registration', async () => {
    const socket = await attachedRun('run-1');
    expect(
      await bridge.handleMessage({ type: 'addKnownValue', value: 'Rosalind' }),
    ).toEqual({ ok: true });
    expect(await start(tokenFor('run-1', 7200))).toEqual({ ok: true });
    // Not ended on the relay: the new socket's register replaces the old one.
    expect(socket.sent.filter((m) => m.type === 'stop')).toEqual([]);
    expect((await status()).needsRegistration).toBe(true);
    const next = FakeSocket.all.at(-1) as FakeSocket;
    await next.open();
    expect(
      (await ask(next, { id: 'a', type: 'attach', targetId: 'T' })).error,
    ).toBeUndefined();
    expect((await screenshot(next, 's1')).error).toBe(REGISTRATION_REFUSED);
  });

  it('a same-run-id restart with no registered values does not', async () => {
    await attachedRun('run-1');
    expect(await start(tokenFor('run-1', 7200))).toEqual({ ok: true });
    expect((await status()).needsRegistration).toBe(false);
  });

  it('a fresh run does not need registration', async () => {
    await attachedRun('run-1');
    expect((await status()).needsRegistration).toBe(false);
  });

  it('never puts the token or a URL in the status', async () => {
    await resumed();
    const text = JSON.stringify(await status());
    expect(text).not.toContain('sig');
    expect(text).not.toContain('ws:');
    expect(text).not.toContain('127.0.0.1');
    expect(text).not.toContain('token');
  });

  it('reports an inactive bridge with no run', async () => {
    expect(await status()).toEqual({ ok: true, active: false });
  });

  it('refuses registrationComplete without a run, or with a bad count', async () => {
    expect(
      (await bridge.handleMessage({ type: 'registrationComplete' })).ok,
    ).toBe(false);
    await resumed();
    expect(
      (await bridge.handleMessage({ type: 'registrationComplete', count: -1 }))
        .ok,
    ).toBe(false);
    expect(
      (await bridge.handleMessage({ type: 'registrationComplete', count: '1' }))
        .ok,
    ).toBe(false);
    expect((await status()).needsRegistration).toBe(true);
    expect(
      await bridge.handleMessage({ type: 'registrationComplete' }),
    ).toEqual({ ok: true });
  });
});

describe('Jev classifier lifecycle', () => {
  async function withJev() {
    const stops: Array<ReturnType<typeof vi.fn>> = [];
    vi.resetModules();
    vi.doMock('@/extension/src/config', () => ({
      APP_ORIGIN: 'https://app.example',
    }));
    vi.doMock('@/extension/src/jev-classifier', () => ({
      createJevClassifier: () => {
        const stop = vi.fn();
        stops.push(stop);
        return {
          classify: async () => false,
          failures: () => 3,
          requests: () => 7,
          stop,
        };
      },
    }));
    bridge = await import('@/extension/src/bridge');
    return stops;
  }

  it('reports Jev counters and stops the classifier with the run', async () => {
    const stops = await withJev();
    await attachedRun('run-1');
    const s = await status();
    expect(s.jevFailures).toBe(3);
    expect(s.jevRequests).toBe(7);
    await bridge.handleMessage({ type: 'stopRun' });
    expect(stops[0]).toHaveBeenCalledTimes(1);
  });

  it('stops the replaced run classifier', async () => {
    const stops = await withJev();
    await attachedRun('run-1');
    await start(tokenFor('run-2'));
    expect(stops[0]).toHaveBeenCalledTimes(1);
    expect(stops[1]).not.toHaveBeenCalled();
  });

  it('leaves no timer behind after stopRun', async () => {
    vi.useFakeTimers();
    await attachedRun('run-1');
    await bridge.handleMessage({ type: 'stopRun' });
    expect(vi.getTimerCount()).toBe(0);
  });
});

describe('test hooks', () => {
  type Hooked = {
    bridgeHandleMessage?: unknown;
    bridgeOnDebuggerDetach?: unknown;
  };

  afterEach(() => {
    Reflect.deleteProperty(globalThis, 'bridgeHandleMessage');
    Reflect.deleteProperty(globalThis, 'bridgeOnDebuggerDetach');
  });

  it('are absent when the build does not define EXTENSION_TEST_HOOKS', async () => {
    vi.resetModules();
    await import('@/extension/src/service-worker');
    const g = globalThis as Hooked;
    expect(g.bridgeHandleMessage).toBeUndefined();
    expect(g.bridgeOnDebuggerDetach).toBeUndefined();
  });

  it('are absent when it is defined false', async () => {
    vi.resetModules();
    vi.stubGlobal('EXTENSION_TEST_HOOKS', false);
    await import('@/extension/src/service-worker');
    expect((globalThis as Hooked).bridgeHandleMessage).toBeUndefined();
  });

  it('are installed when it is defined true', async () => {
    vi.resetModules();
    vi.stubGlobal('EXTENSION_TEST_HOOKS', true);
    await import('@/extension/src/service-worker');
    const g = globalThis as Hooked;
    expect(typeof g.bridgeHandleMessage).toBe('function');
    expect(typeof g.bridgeOnDebuggerDetach).toBe('function');
  });
});
