import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ALLOWED_CDP_METHODS } from '@/lib/relay/allowed-cdp-methods';

// The bridge's run lifecycle against a fake chrome and relay socket:
// serialized startRun, a stop issued while the socket is down, the detach
// timer (P4b-d), the run status after a worker restart, and the test hooks.
// Each test loads a fresh bridge module so no run, queue or pending stop
// leaks between tests.

const ORIGIN = 'https://a.gov';
const RELAY = 'ws://127.0.0.1:1';
const REGISTRATION_REFUSED = 'registration required';

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
    alarms: { onAlarm: { addListener: () => {} } },
    sidePanel: { setPanelBehavior: async () => {} },
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
          title: 'Form for Rosalind',
          url: `${ORIGIN}/form?name=Rosalind`,
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

describe('run status: exp and restarting', () => {
  it('reports the token exp in Unix seconds and nothing secret', async () => {
    const token = tokenFor('run-1', 1500);
    const exp = JSON.parse(
      Buffer.from(token.split('.')[0], 'base64url').toString(),
    ).exp;
    expect(await start(token)).toEqual({ ok: true });
    const got = await status();
    expect(got.exp).toBe(exp);
    expect(JSON.stringify(got)).not.toContain('sig');
  });

  /** Holds the first storage.session.set after `arm()` until released. */
  const gateSet = () => {
    let release = () => {};
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    const session = (
      (globalThis as unknown as { chrome: unknown }).chrome as {
        storage: {
          session: { set: (i: Record<string, unknown>) => Promise<void> };
        };
      }
    ).storage.session;
    const original = session.set;
    let armed = false;
    session.set = async (items) => {
      if (armed && 'run' in items) {
        armed = false;
        await gate;
      }
      return original(items);
    };
    return {
      arm: () => {
        armed = true;
      },
      release,
    };
  };

  it('is restarting from a same-run startRun until begin finishes', async () => {
    await attachedRun('run-1');
    const g = gateSet();
    g.arm();
    const second = start(tokenFor('run-1'));
    await vi.waitFor(async () =>
      expect(await status()).toEqual({
        ok: true,
        active: false,
        restarting: true,
        runId: 'run-1',
      }),
    );
    g.release();
    expect(await second).toEqual({ ok: true });
    expect(await status()).toMatchObject({ active: true, runId: 'run-1' });
    expect((await status()).restarting).toBeUndefined();
  });

  it('clears restarting when the same-run startRun throws', async () => {
    await attachedRun('run-1');
    const session = (
      (globalThis as unknown as { chrome: unknown }).chrome as {
        storage: {
          session: { set: (i: Record<string, unknown>) => Promise<void> };
        };
      }
    ).storage.session;
    const original = session.set;
    let calls = 0;
    session.set = async (items) => {
      calls += 1;
      // The begun-run record (call 1) passes; the config write (call 2) fails.
      if (calls === 2) throw new Error('storage down');
      return original(items);
    };
    await expect(start(tokenFor('run-1'))).rejects.toThrow('storage down');
    expect(await status()).toEqual({ ok: true, active: false });
  });

  it('a different run id is not restarting, so the old run reads as gone', async () => {
    await attachedRun('run-1');
    const g = gateSet();
    g.arm();
    const second = start(tokenFor('run-2'));
    await vi.waitFor(async () =>
      expect(await status()).toEqual({ ok: true, active: false }),
    );
    g.release();
    await second;
  });
});

describe('stopRun scoped to a run id', () => {
  const stop = (runId?: string) =>
    bridge.handleMessage(
      runId === undefined ? { type: 'stopRun' } : { type: 'stopRun', runId },
    );

  it('a matching run id stops the run', async () => {
    expect(await start(tokenFor('run-1'))).toEqual({ ok: true });
    expect(await stop('run-1')).toEqual({ ok: true });
    expect(await status()).toEqual({ ok: true, active: false });
  });

  it('a different run id succeeds without stopping anything', async () => {
    expect(await start(tokenFor('run-1'))).toEqual({ ok: true });
    const socket = FakeSocket.all.at(-1) as FakeSocket;
    expect(await stop('run-2')).toEqual({ ok: true });
    expect(await status()).toMatchObject({ active: true, runId: 'run-1' });
    expect(socket.closedByClient).toBe(false);
  });

  it('a run id with no run succeeds and does nothing', async () => {
    expect(await stop('run-1')).toEqual({ ok: true });
    expect(await status()).toEqual({ ok: true, active: false });
  });

  it('no run id stops whatever run is current (as before)', async () => {
    expect(await start(tokenFor('run-1'))).toEqual({ ok: true });
    expect(await stop()).toEqual({ ok: true });
    expect(await status()).toEqual({ ok: true, active: false });
  });

  it('a run id stop queued behind a startRun sees the run that start made', async () => {
    const started = start(tokenFor('run-2'));
    const stopped = stop('run-2');
    expect(await started).toEqual({ ok: true });
    expect(await stopped).toEqual({ ok: true });
    expect(await status()).toEqual({ ok: true, active: false });
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
      exp: expect.any(Number),
      ok: true,
      active: true,
      runId: 'run-1',
      origin: ORIGIN,
      tabId: 1,
      attached: true,
      needsRegistration: true,
      armed: false,
    });
    expect((await screenshot(socket, 's1')).error).toBe(REGISTRATION_REFUSED);
    // Values can be registered again while it is set.
    expect(
      await bridge.handleMessage({
        type: 'addKnownValue',
        runId: 'run-1',
        value: 'Rosalind',
      }),
    ).toEqual({ ok: true });
    expect((await screenshot(socket, 's2')).error).toBe(REGISTRATION_REFUSED);

    expect(
      await bridge.handleMessage({
        type: 'registrationComplete',
        runId: 'run-1',
        count: 1,
      }),
    ).toEqual({ ok: true });
    expect((await status()).needsRegistration).toBe(false);
    expect((await screenshot(socket, 's3')).error).not.toBe(
      REGISTRATION_REFUSED,
    );
  });

  const addValue = (value: string, runId = 'run-1') =>
    bridge.handleMessage({ type: 'addKnownValue', runId, value });
  const titleOf = async (socket: FakeSocket, id: string) => {
    const reply = await ask(socket, { id, type: 'listTargets' });
    return (reply.result as { targetInfos: { title: string }[] }).targetInfos[0]
      .title;
  };
  const refreshed = async (runId = 'run-1') => {
    expect(await start(tokenFor(runId, 7200))).toEqual({ ok: true });
    const next = FakeSocket.all.at(-1) as FakeSocket;
    await next.open();
    return next;
  };

  it('a same-run-id restart (token refresh) keeps the registered values', async () => {
    const socket = await attachedRun('run-1');
    expect(await addValue('Rosalind')).toEqual({ ok: true });
    expect(await titleOf(socket, 't0')).not.toContain('Rosalind');
    const next = await refreshed();
    // Not ended on the relay: the new socket's register replaces the old one.
    expect(socket.sent.filter((m) => m.type === 'stop')).toEqual([]);
    expect((await status()).needsRegistration).toBe(false);
    expect(await titleOf(next, 't1')).not.toContain('Rosalind');
    // A run that never registered it would show the title as is.
    await bridge.handleMessage({ type: 'stopRun' });
    const fresh = await attachedRun('run-9');
    expect(await titleOf(fresh, 't2')).toContain('Rosalind');
  });

  it('values are cleared by stopRun and not inherited by a later run', async () => {
    await attachedRun('run-1');
    expect(await addValue('Rosalind')).toEqual({ ok: true });
    await bridge.handleMessage({ type: 'stopRun' });
    const same = await attachedRun('run-1');
    // Same id after stop: values were lost, and none are carried.
    expect((await status()).needsRegistration).toBe(true);
    await bridge.handleMessage({
      type: 'registrationComplete',
      runId: 'run-1',
    });
    expect(await titleOf(same, 't1')).toContain('Rosalind');
    await bridge.handleMessage({ type: 'stopRun' });
    const other = await attachedRun('run-2');
    expect(await titleOf(other, 't2')).toContain('Rosalind');
  });

  it('a different run id does not inherit the values', async () => {
    await attachedRun('run-1');
    expect(await addValue('Rosalind')).toEqual({ ok: true });
    const next = await refreshed('run-2');
    expect((await status()).needsRegistration).toBe(false);
    expect(await titleOf(next, 't1')).toContain('Rosalind');
  });

  it('the redactor cap holds across a refresh', async () => {
    await attachedRun('run-1');
    for (let i = 0; i < 32; i++) {
      expect(await addValue(`value-${i}`)).toEqual({ ok: true });
    }
    expect((await addValue('value-extra')).ok).toBe(false);
    await refreshed();
    expect((await status()).needsRegistration).toBe(false);
    // Still full after the carry: the 33rd is refused, a carried one is not.
    expect((await addValue('value-extra')).ok).toBe(false);
    expect(await addValue('value-0')).toEqual({ ok: true });
  });

  it('a value under 3 characters is carried the way the redactor treats it', async () => {
    await attachedRun('run-1');
    expect(await addValue('ab')).toEqual({ ok: true });
    const next = await refreshed();
    expect((await status()).needsRegistration).toBe(false);
    // Nothing is scrubbed for it, before or after the refresh.
    expect(await titleOf(next, 't1')).toBe('Form for Rosalind');
    expect(await addValue('ab')).toEqual({ ok: true });
  });

  it('a value registered while startRun awaits is still scrubbed on the new run', async () => {
    await attachedRun('run-1');
    expect(await addValue('Rosalind')).toEqual({ ok: true });
    const store = (
      globalThis as unknown as {
        chrome: { storage: { session: { set: (i: object) => Promise<void> } } };
      }
    ).chrome.storage.session;
    const realSet = store.set;
    let release: () => void = () => {};
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    let entered = false;
    store.set = async (items) => {
      entered = true;
      await gate;
      return realSet(items);
    };
    const started = start(tokenFor('run-1', 7200));
    // startRun is now blocked inside the awaits before its snapshot.
    await vi.waitFor(() => expect(entered).toBe(true));
    expect(await addValue('Hypatia')).toEqual({ ok: true });
    release();
    expect(await started).toEqual({ ok: true });
    store.set = realSet;
    const dbg = (
      globalThis as unknown as {
        chrome: { debugger: { getTargets: () => Promise<unknown[]> } };
      }
    ).chrome.debugger;
    dbg.getTargets = async () => [
      {
        tabId: 1,
        type: 'page',
        id: 'T',
        title: 'Rosalind and Hypatia',
        url: `${ORIGIN}/form`,
        attached: false,
      },
    ];
    const next = FakeSocket.all.at(-1) as FakeSocket;
    await next.open();
    await ask(next, { id: 'a', type: 'attach', targetId: 'T' });
    expect((await status()).needsRegistration).toBe(false);
    // Both values, including the one added mid-refresh, are masked again.
    expect(await titleOf(next, 't1')).toBe('[masked] and [masked]');
  });

  it('needsRegistration stays set across a refresh that carries values', async () => {
    await resumed();
    expect(await addValue('Rosalind')).toEqual({ ok: true });
    await refreshed();
    expect((await status()).needsRegistration).toBe(true);
  });

  it('refuses a value once the carried list is full', async () => {
    await attachedRun('run-1');
    for (let i = 0; i < 256; i++) {
      expect((await addValue(' '.repeat(i + 1))).ok).toBe(true);
    }
    expect((await addValue('Rosalind')).ok).toBe(false);
    expect(await addValue(' ')).toEqual({ ok: true });
  });

  it('a same-run-id restart with no registered values does not', async () => {
    await attachedRun('run-1');
    expect(await start(tokenFor('run-1', 7200))).toEqual({ ok: true });
    expect((await status()).needsRegistration).toBe(false);
  });

  it('a same-run-id startRun after the run ended on token expiry needs registration', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'Date'] });
    expect(await start(tokenFor('run-1', 3))).toEqual({ ok: true });
    await (FakeSocket.all.at(-1) as FakeSocket).open();
    expect(
      await bridge.handleMessage({
        type: 'addKnownValue',
        runId: 'run-1',
        value: 'Rosalind',
      }),
    ).toEqual({ ok: true });
    await vi.advanceTimersByTimeAsync(4_000);
    // Ended by the worker itself: no run is left to say values were lost.
    expect((await status()).active).toBe(false);
    vi.useRealTimers();

    expect(await start(tokenFor('run-1'))).toEqual({ ok: true });
    expect((await status()).needsRegistration).toBe(true);
    const next = FakeSocket.all.at(-1) as FakeSocket;
    await next.open();
    expect(
      (await ask(next, { id: 'a', type: 'attach', targetId: 'T' })).error,
    ).toBeUndefined();
    expect((await screenshot(next, 's1')).error).toBe(REGISTRATION_REFUSED);
    expect(
      await bridge.handleMessage({
        type: 'registrationComplete',
        runId: 'run-1',
        count: 1,
      }),
    ).toEqual({ ok: true });
    expect((await screenshot(next, 's2')).error).not.toBe(REGISTRATION_REFUSED);
  });

  it('a same-run-id startRun after stopRun needs registration; a fresh run id does not', async () => {
    await attachedRun('run-1');
    await bridge.handleMessage({ type: 'stopRun' });
    expect(await start(tokenFor('run-2'))).toEqual({ ok: true });
    expect((await status()).needsRegistration).toBe(false);
    await bridge.handleMessage({ type: 'stopRun' });
    expect(await start(tokenFor('run-1'))).toEqual({ ok: true });
    expect((await status()).needsRegistration).toBe(true);
  });

  it('the record of begun run ids survives a worker restart and holds ids only', async () => {
    await attachedRun('run-1');
    await bridge.handleMessage({ type: 'stopRun' });
    const record = JSON.stringify([...storage.entries()]);
    expect(record).toContain('run-1');
    expect(record).not.toContain('sig');
    expect(record).not.toContain('Rosalind');
    // A new worker: fresh module state, the same chrome.storage.session.
    vi.resetModules();
    bridge = await import('@/extension/src/bridge');
    expect(await start(tokenFor('run-1'))).toEqual({ ok: true });
    expect((await status()).needsRegistration).toBe(true);
    expect(await start(tokenFor('run-3'))).toEqual({ ok: true });
    expect((await status()).needsRegistration).toBe(false);
  });

  it('keeps only the 20 most recent run ids', async () => {
    for (let i = 0; i <= 20; i++) {
      expect(await start(tokenFor(`r${i}`))).toEqual({ ok: true });
    }
    await bridge.handleMessage({ type: 'stopRun' });
    // r0 fell out of the record; r20 is in it.
    expect(await start(tokenFor('r0'))).toEqual({ ok: true });
    expect((await status()).needsRegistration).toBe(false);
    await bridge.handleMessage({ type: 'stopRun' });
    expect(await start(tokenFor('r20'))).toEqual({ ok: true });
    expect((await status()).needsRegistration).toBe(true);
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

  it('binds addKnownValue and registrationComplete to the active run id', async () => {
    const socket = await resumed();
    const wrong = { ok: false, error: 'wrong run' };
    for (const runId of [undefined, 42, null, 'run-2', '']) {
      expect(
        await bridge.handleMessage({
          type: 'addKnownValue',
          runId,
          value: 'Rosalind',
        }),
      ).toEqual(wrong);
      expect(
        await bridge.handleMessage({ type: 'registrationComplete', runId }),
      ).toEqual(wrong);
    }
    // Refused messages change nothing.
    expect((await status()).needsRegistration).toBe(true);
    expect((await screenshot(socket, 's1')).error).toBe(REGISTRATION_REFUSED);
    expect(
      await bridge.handleMessage({
        type: 'addKnownValue',
        runId: 'run-1',
        value: 'Rosalind',
      }),
    ).toEqual({ ok: true });
    // The count is still validated once the run id matches.
    expect(
      (
        await bridge.handleMessage({
          type: 'registrationComplete',
          runId: 'run-1',
          count: -1,
        })
      ).ok,
    ).toBe(false);
    expect(
      await bridge.handleMessage({
        type: 'registrationComplete',
        runId: 'run-1',
        count: 1,
      }),
    ).toEqual({ ok: true });
    expect((await status()).needsRegistration).toBe(false);
  });

  it('refuses registrationComplete without a run, or with a bad count', async () => {
    expect(
      (
        await bridge.handleMessage({
          type: 'registrationComplete',
          runId: 'run-1',
        })
      ).ok,
    ).toBe(false);
    await resumed();
    expect(
      (
        await bridge.handleMessage({
          type: 'registrationComplete',
          runId: 'run-1',
          count: -1,
        })
      ).ok,
    ).toBe(false);
    expect(
      (
        await bridge.handleMessage({
          type: 'registrationComplete',
          runId: 'run-1',
          count: '1',
        })
      ).ok,
    ).toBe(false);
    expect((await status()).needsRegistration).toBe(true);
    expect(
      await bridge.handleMessage({
        type: 'registrationComplete',
        runId: 'run-1',
      }),
    ).toEqual({ ok: true });
  });
});

describe('while a resumed run needs registration (fail closed)', () => {
  // Methods whose result is empty and which keep the session alive.
  const STILL_ALLOWED = new Set([
    'Runtime.enable',
    'Runtime.runIfWaitingForDebugger',
    'Page.enable',
    'Page.bringToFront',
    'Page.handleJavaScriptDialog',
    'Network.enable',
    'DOM.enable',
    'Accessibility.enable',
    'Target.setAutoAttach',
    'Input.insertText',
    'Input.dispatchKeyEvent',
    'Input.dispatchMouseEvent',
  ]);
  const REFUSED = [...ALLOWED_CDP_METHODS].filter((m) => !STILL_ALLOWED.has(m));

  const PARAMS: Record<string, Record<string, unknown>> = {
    'Page.navigate': { url: `${ORIGIN}/next` },
    'Runtime.evaluate': { expression: '1' },
    'Runtime.callFunctionOn': {
      functionDeclaration: 'function () { return 1; }',
      objectId: '1.1.1',
    },
    'DOM.describeNode': { backendNodeId: 1 },
    'DOM.resolveNode': { backendNodeId: 1 },
    'DOM.getBoxModel': { backendNodeId: 1 },
    'DOM.scrollIntoViewIfNeeded': { backendNodeId: 1 },
    'Input.insertText': { text: 'x' },
    'Input.dispatchKeyEvent': { type: 'keyDown', key: 'a' },
    'Input.dispatchMouseEvent': { type: 'mouseMoved', x: 1, y: 1 },
    'Page.handleJavaScriptDialog': { accept: true },
    'Target.setAutoAttach': {
      autoAttach: true,
      waitForDebuggerOnStart: false,
      flatten: true,
    },
  };

  let n = 0;
  const cdp = (socket: FakeSocket, method: string) => {
    n += 1;
    return ask(socket, {
      id: `c${n}`,
      type: 'cdp',
      sessionId: 'tab-1',
      method,
      params: PARAMS[method] ?? {},
    });
  };

  async function resumedRun() {
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

  it('refuses every method that can return page text, with no result', async () => {
    expect(REFUSED).toEqual(
      expect.arrayContaining([
        'DOM.describeNode',
        'Accessibility.getFullAXTree',
        'Runtime.evaluate',
        'Runtime.callFunctionOn',
        'Page.getFrameTree',
        'Page.captureScreenshot',
        'Page.navigate',
      ]),
    );
    const socket = await resumedRun();
    for (const method of REFUSED) {
      const reply = await cdp(socket, method);
      expect([method, reply.error]).toEqual([method, REGISTRATION_REFUSED]);
      expect(reply.result).toBeUndefined();
    }
  });

  it('still runs the methods that return no page text', async () => {
    const socket = await resumedRun();
    for (const method of STILL_ALLOWED) {
      const reply = await cdp(socket, method);
      // Page.handleJavaScriptDialog fails only because no dialog is open.
      const expected =
        method === 'Page.handleJavaScriptDialog'
          ? 'no dialog is open'
          : undefined;
      expect([method, reply.error]).toEqual([method, expected]);
    }
  });

  it('refuses an unknown method', async () => {
    const socket = await resumedRun();
    const reply = await cdp(socket, 'Storage.getCookies');
    expect(reply.error).toBeDefined();
    expect(reply.result).toBeUndefined();
  });

  it('allows every method again after registrationComplete (count 0)', async () => {
    const socket = await resumedRun();
    expect(
      await bridge.handleMessage({
        type: 'registrationComplete',
        runId: 'run-1',
        count: 0,
      }),
    ).toEqual({ ok: true });
    for (const method of REFUSED) {
      const reply = await cdp(socket, method);
      expect([method, reply.error]).not.toEqual([method, REGISTRATION_REFUSED]);
    }
  });

  it('a fresh run is unaffected', async () => {
    const socket = await attachedRun('run-1');
    for (const method of REFUSED) {
      const reply = await cdp(socket, method);
      expect([method, reply.error]).not.toEqual([method, REGISTRATION_REFUSED]);
    }
  });

  it('blanks the target URL and title the bridge reports', async () => {
    const socket = await resumedRun();
    const reply = await ask(socket, { id: 'l', type: 'listTargets' });
    const infos = (reply.result as { targetInfos: Record<string, unknown>[] })
      .targetInfos;
    expect(infos[0]).toMatchObject({ url: '', title: '' });
    const attached = await ask(socket, {
      id: 'a2',
      type: 'attach',
      targetId: 'T',
    });
    expect(JSON.stringify(attached)).not.toContain('Rosalind');
  });

  it('forwards only events that carry no page text, with target URLs blanked', async () => {
    const socket = await resumedRun();
    const sentEvents = () =>
      socket.sent
        .filter((m) => m.type === 'event')
        .map((m) => m as unknown as { method: string; params: unknown });
    const fire = (method: string, params: unknown) =>
      bridge.onDebuggerEvent({ tabId: 1 }, method, params);
    fire('Page.frameNavigated', {
      frame: { id: 'T', loaderId: 'L', url: `${ORIGIN}/form?name=Rosalind` },
    });
    fire('Network.requestWillBeSent', {
      requestId: 'r',
      request: { url: `${ORIGIN}/q?name=Rosalind` },
    });
    fire('Page.loadEventFired', { timestamp: 1 });
    fire('Target.targetInfoChanged', {
      targetInfo: {
        targetId: 'T',
        type: 'page',
        url: `${ORIGIN}/x`,
        title: 'Rosalind',
      },
    });
    const events = sentEvents();
    expect(events.map((e) => e.method)).toEqual([
      'Page.loadEventFired',
      'Target.targetInfoChanged',
    ]);
    expect(JSON.stringify(events)).not.toContain('Rosalind');

    await bridge.handleMessage({
      type: 'registrationComplete',
      runId: 'run-1',
    });
    fire('Page.frameNavigated', {
      frame: { id: 'T', loaderId: 'L', url: `${ORIGIN}/form` },
    });
    expect(sentEvents().map((e) => e.method)).toContain('Page.frameNavigated');
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
