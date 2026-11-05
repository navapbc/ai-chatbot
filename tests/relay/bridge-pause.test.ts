import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

// Pause and resume (the caseworker's buttons) against a fake chrome and relay
// socket: while paused every agent CDP command is refused with a fixed string
// and no page event is forwarded; resume lets both through; the flag shows in
// getRunStatus, is refused for another run id, and survives a same-run token
// refresh.

const ORIGIN = 'https://a.gov';
const B = 'https://b.gov';
const C = 'https://c.gov';
const FRAME = 'https://frames.gov';
const RELAY = 'ws://127.0.0.1:1';
const OUT_OF_SCOPE = 'tab is outside the allowed origin';
const FRAME_OUT_OF_SCOPE = 'frame is outside the allowed origins';
const HOLD_MS = 60_000;
const CAP_MS = 120_000;

interface Sent {
  type: string;
  id?: string;
  reason?: string;
  error?: string;
  result?: unknown;
  method?: string;
  params?: unknown;
}

class FakeSocket {
  static OPEN = 1;
  static all: FakeSocket[] = [];
  readyState = 0;
  sent: Sent[] = [];
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
    this.readyState = 3;
  }
  async open() {
    this.readyState = 1;
    await this.onopen?.();
  }
  drop(code = 1006) {
    this.readyState = 3;
    this.onclose?.({ code });
  }
}

const b64url = (value: unknown) =>
  Buffer.from(JSON.stringify(value)).toString('base64url');

const tokenFor = (runId: string) =>
  `${b64url({
    runId,
    role: 'extension',
    origin: ORIGIN,
    exp: Math.floor(Date.now() / 1000) + 3 * 3600,
  })}.sig`;

/** Per-test knobs for the fake chrome. */
const fake: {
  /** The tab's committed URL (chrome.tabs.get). */
  tabUrl: string;
  /** Holds Accessibility.getFullAXTree until released. */
  axGate?: Promise<void>;
  /** Holds the next chrome.tabs.get until released (one-shot). */
  tabsGate?: Promise<void>;
  /** Set when a gated chrome.tabs.get is waiting. */
  tabsWaiting?: boolean;
} = { tabUrl: `${ORIGIN}/form` };

let storage: Map<string, unknown>;

const sendCommand = vi.fn(
  async (
    debuggee: { tabId: number; sessionId?: string },
    method: string,
  ): Promise<unknown> => {
    if (method === 'Page.getFrameTree') {
      return debuggee.sessionId
        ? { frameTree: { frame: { id: 'F', parentId: 'T', url: `${B}/x` } } }
        : { frameTree: { frame: { id: 'T', url: fake.tabUrl } } };
    }
    if (method === 'Accessibility.getFullAXTree') {
      await fake.axGate;
      return { nodes: [] };
    }
    if (method === 'Runtime.evaluate') {
      return { result: { type: 'number', value: 1, description: '1' } };
    }
    return {};
  },
);

function stubChrome() {
  storage = new Map();
  vi.stubGlobal('WebSocket', FakeSocket);
  vi.stubGlobal('chrome', {
    runtime: { id: 'ext' },
    tabs: {
      get: async () => {
        const gate = fake.tabsGate;
        if (gate) {
          fake.tabsGate = undefined;
          fake.tabsWaiting = true;
          await gate;
          fake.tabsWaiting = false;
        }
        return { url: fake.tabUrl };
      },
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
          url: fake.tabUrl,
          attached: false,
        },
      ],
      attach: async () => {},
      detach: async () => {},
      sendCommand,
    },
  });
}

let bridge: typeof import('@/extension/src/bridge');

beforeEach(async () => {
  vi.resetModules();
  FakeSocket.all = [];
  fake.tabUrl = `${ORIGIN}/form`;
  fake.axGate = undefined;
  fake.tabsGate = undefined;
  fake.tabsWaiting = false;
  sendCommand.mockClear();
  stubChrome();
  bridge = await import('@/extension/src/bridge');
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

const status = () =>
  bridge.handleMessage({ type: 'getRunStatus' }) as Promise<
    Record<string, unknown>
  >;

const allow = (origin: unknown) =>
  bridge.handleMessage({ type: 'allowOrigin', origin });

/** Sends a relay command and returns the bridge's reply to it. */
const ask = async (socket: FakeSocket, message: Record<string, unknown>) => {
  await socket.onmessage?.({ data: JSON.stringify(message) });
  const reply = socket.sent.find(
    (m) => m.type === 'reply' && m.id === message.id,
  );
  if (!reply) throw new Error(`no reply to ${String(message.id)}`);
  return reply;
};

let seq = 0;
const cdp = (
  socket: FakeSocket,
  method: string,
  params: Record<string, unknown> = {},
  sessionId = 'tab-1',
) => ask(socket, { id: `c${++seq}`, type: 'cdp', sessionId, method, params });

const evaluate = (socket: FakeSocket) =>
  cdp(socket, 'Runtime.evaluate', { expression: '1' });

const stops = (socket: FakeSocket) =>
  socket.sent.filter((m) => m.type === 'stop').map((m) => m.reason);

/** Starts a run, opens its socket and attaches the tab. */
async function attachedRun(
  runId = 'run-1',
  allowedFrameOrigins: string[] = [],
) {
  const started = await bridge.handleMessage({
    type: 'startRun',
    relayUrl: RELAY,
    token: tokenFor(runId),
    tabId: 1,
    allowedFrameOrigins,
  });
  expect(started).toEqual({ ok: true });
  const socket = FakeSocket.all.at(-1) as FakeSocket;
  await socket.open();
  expect(
    (await ask(socket, { id: `a${++seq}`, type: 'attach', targetId: 'T' }))
      .error,
  ).toBeUndefined();
  return socket;
}

/** The main frame commits `url` (ordered event stream, then tabs.onUpdated). */
function commit(url: string) {
  fake.tabUrl = url;
  bridge.onDebuggerEvent({ tabId: 1 }, 'Page.frameNavigated', {
    frame: { id: 'T', url },
  });
  bridge.onTabUpdated(1, { url });
}

const PAUSED = 'run is paused';

const pause = (runId = 'run-1') =>
  bridge.handleMessage({ type: 'pauseAgent', runId });
const resume = (runId = 'run-1') =>
  bridge.handleMessage({ type: 'resumeAgent', runId });

describe('pause and resume', () => {
  it("refuses the agent's commands while paused and allows them after", async () => {
    const socket = await attachedRun();
    expect((await evaluate(socket)).error).toBeUndefined();

    expect(await pause()).toEqual({ ok: true });
    expect((await status()).paused).toBe(true);
    const refused = await evaluate(socket);
    expect(refused.error).toBe(PAUSED);
    // The run is alive and nothing ended it.
    expect((await status()).active).toBe(true);
    expect(stops(socket)).toEqual([]);

    expect(await resume()).toEqual({ ok: true });
    expect((await status()).paused).toBeUndefined();
    expect((await evaluate(socket)).error).toBeUndefined();
  });

  it('forwards no page event while paused, and forwards again after', async () => {
    const socket = await attachedRun();
    await pause();
    const before = socket.sent.length;
    bridge.onDebuggerEvent({ tabId: 1 }, 'Page.frameNavigated', {
      frame: { id: 'T', url: `${ORIGIN}/step-2` },
    });
    expect(socket.sent.slice(before).filter((m) => m.type !== 'reply')).toEqual(
      [],
    );
    await resume();
    bridge.onDebuggerEvent({ tabId: 1 }, 'Page.frameNavigated', {
      frame: { id: 'T', url: `${ORIGIN}/step-3` },
    });
    expect(
      socket.sent.slice(before).filter((m) => m.type === 'event').length,
    ).toBeGreaterThan(0);
  });

  it('answers a fixed error for no run and for another run id', async () => {
    expect(await pause()).toEqual({ ok: false, error: 'no active run' });
    await attachedRun('run-1');
    expect(await pause('run-2')).toEqual({ ok: false, error: 'wrong run' });
    expect(await resume('run-2')).toEqual({ ok: false, error: 'wrong run' });
    expect((await status()).paused).toBeUndefined();
  });

  it('keeps the pause across a same-run token refresh', async () => {
    await attachedRun('run-1');
    await pause();
    const refreshed = await bridge.handleMessage({
      type: 'startRun',
      relayUrl: RELAY,
      token: tokenFor('run-1'),
      tabId: 1,
      allowedFrameOrigins: [],
    });
    expect(refreshed).toEqual({ ok: true });
    expect((await status()).paused).toBe(true);
  });

  it('does not carry a pause to a different run', async () => {
    await attachedRun('run-1');
    await pause();
    await bridge.handleMessage({
      type: 'startRun',
      relayUrl: RELAY,
      token: tokenFor('run-2'),
      tabId: 1,
      allowedFrameOrigins: [],
    });
    expect((await status()).paused).toBeUndefined();
  });
});
