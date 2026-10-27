import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

// The origin hold (plan 4b, Task 10) against a fake chrome and relay socket:
// a tab that commits an https URL outside the run's origins is held (agent
// cut off, nothing forwarded) for ORIGIN_HOLD_MS instead of ending the run at
// once; `allowOrigin` for exactly the pending origin lets the run go on
// there; anything else ends the run with `left-origin`.

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
    tabs: { get: async () => ({ url: fake.tabUrl }) },
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

describe('an https commit outside the run origins', () => {
  it('holds the run: blocked, pendingOrigin set, run alive, commands refused', async () => {
    const socket = await attachedRun();
    const sentBefore = socket.sent.length;
    commit(`${B}/login?user=rosalind#step`);

    const s = await status();
    expect(s.active).toBe(true);
    // The origin only: never the path, query or fragment.
    expect(s.pendingOrigin).toBe(B);
    expect(JSON.stringify(s)).not.toContain('rosalind');
    expect(JSON.stringify(s)).not.toContain('login');
    expect((await evaluate(socket)).error).toBe(OUT_OF_SCOPE);
    // Nothing went up but the refusal: no stop, no event.
    expect(stops(socket)).toEqual([]);
    expect(
      socket.sent.slice(sentBefore).filter((m) => m.type !== 'reply'),
    ).toEqual([]);
    expect(JSON.stringify(socket.sent)).not.toContain(B);
  });

  it('holds on tabs.onUpdated alone', async () => {
    const socket = await attachedRun();
    fake.tabUrl = `${B}/x`;
    bridge.onTabUpdated(1, { url: `${B}/x` });
    expect((await status()).pendingOrigin).toBe(B);
    expect((await evaluate(socket)).error).toBe(OUT_OF_SCOPE);
    expect(stops(socket)).toEqual([]);
  });

  it('holds on a pre-commit signal and drops it', async () => {
    const socket = await attachedRun();
    bridge.onDebuggerEvent({ tabId: 1 }, 'Network.requestWillBeSent', {
      requestId: 'r1',
      frameId: 'T',
      type: 'Document',
      request: { url: `${B}/sso` },
    });
    expect((await status()).pendingOrigin).toBe(B);
    expect(JSON.stringify(socket.sent)).not.toContain(B);
    expect((await evaluate(socket)).error).toBe(OUT_OF_SCOPE);
  });

  it('holds when the command check finds the tab elsewhere', async () => {
    const socket = await attachedRun();
    fake.tabUrl = `${B}/x`;
    expect((await evaluate(socket)).error).toBe(OUT_OF_SCOPE);
    expect((await status()).pendingOrigin).toBe(B);
    expect(stops(socket)).toEqual([]);
  });

  it('a main-frame commit to an allowed FRAME origin still holds', async () => {
    const socket = await attachedRun('run-1', [FRAME]);
    commit(`${FRAME}/page`);
    expect((await status()).pendingOrigin).toBe(FRAME);
    expect((await evaluate(socket)).error).toBe(OUT_OF_SCOPE);
  });

  it('refuses the frame, dialog, screenshot, child and attach paths while held', async () => {
    const socket = await attachedRun('run-1', [FRAME]);
    bridge.onDebuggerEvent({ tabId: 1 }, 'Target.attachedToTarget', {
      sessionId: 'C',
      targetInfo: { targetId: 'F', type: 'iframe', url: `${FRAME}/x` },
      waitingForDebugger: false,
    });
    commit(`${B}/x`);
    expect((await cdp(socket, 'Page.getFrameTree')).error).toBe(OUT_OF_SCOPE);
    expect(
      (await cdp(socket, 'Page.handleJavaScriptDialog', { accept: false }))
        .error,
    ).toBe(OUT_OF_SCOPE);
    expect((await cdp(socket, 'Page.captureScreenshot')).error).toBe(
      OUT_OF_SCOPE,
    );
    expect(
      (await cdp(socket, 'Runtime.evaluate', { expression: '1' }, 'C')).error,
    ).toBe(OUT_OF_SCOPE);
    expect(
      (await ask(socket, { id: 'att', type: 'attach', targetId: 'T' })).error,
    ).toBe(OUT_OF_SCOPE);
    expect((await ask(socket, { id: 'lt', type: 'listTargets' })).error).toBe(
      OUT_OF_SCOPE,
    );
    expect(
      sendCommand.mock.calls.some(([, m]) => m === 'Page.captureScreenshot'),
    ).toBe(false);
  });

  it('keeps tracking while held: a paused child is resumed, nothing is forwarded', async () => {
    const socket = await attachedRun();
    commit(`${B}/x`);
    const sentBefore = socket.sent.length;
    bridge.onDebuggerEvent({ tabId: 1 }, 'Target.attachedToTarget', {
      sessionId: 'C2',
      targetInfo: { targetId: 'F2', type: 'iframe', url: `${B}/frame` },
      waitingForDebugger: true,
    });
    expect(
      sendCommand.mock.calls.some(
        ([d, m]) =>
          d.sessionId === 'C2' && m === 'Runtime.runIfWaitingForDebugger',
      ),
    ).toBe(true);
    expect(socket.sent.length).toBe(sentBefore);
  });
});

describe('a commit that ends the run at once', () => {
  it.each([
    ['http', 'http://b.gov/x'],
    ['about:blank', 'about:blank'],
    ['chrome:', 'chrome://newtab/'],
    ['data:', 'data:text/html,hi'],
    ['blob:', `blob:${B}/6f1c`],
    ['javascript:', 'javascript:alert(1)'],
    ['unparsable', 'not a url'],
  ])('%s ends with left-origin, no hold', async (_name, url) => {
    const socket = await attachedRun();
    fake.tabUrl = url;
    bridge.onTabUpdated(1, { url });
    expect(stops(socket)).toEqual(['left-origin']);
    expect((await status()).active).toBe(false);
  });

  it('a non-https commit while held ends the run', async () => {
    const socket = await attachedRun();
    commit(`${B}/x`);
    commit('http://b.gov/y');
    expect(stops(socket)).toEqual(['left-origin']);
    expect((await status()).active).toBe(false);
  });

  it('a non-https pre-commit signal ends the run', async () => {
    const socket = await attachedRun();
    bridge.onDebuggerEvent({ tabId: 1 }, 'Page.frameRequestedNavigation', {
      frameId: 'T',
      url: 'http://b.gov/x',
    });
    expect(stops(socket)).toEqual(['left-origin']);
  });
});

describe('allowOrigin', () => {
  it('with the pending origin unblocks; commands and later navigation on it work', async () => {
    const socket = await attachedRun();
    commit(`${B}/login`);
    // A context of B's document, created while held (tracked, not sent).
    bridge.onDebuggerEvent({ tabId: 1 }, 'Runtime.executionContextCreated', {
      context: { id: 41, origin: B, auxData: { frameId: 'T' } },
    });
    expect(await allow(B)).toEqual({ ok: true });
    const s = await status();
    expect(s.pendingOrigin).toBeUndefined();
    expect(s.active).toBe(true);

    expect((await evaluate(socket)).error).toBeUndefined();
    // The main frame (judged by frame scope) and B's context are in scope.
    expect((await cdp(socket, 'Page.getFrameTree')).error).toBeUndefined();
    expect(
      (
        await cdp(socket, 'Runtime.evaluate', {
          expression: '1',
          contextId: 41,
        })
      ).error,
    ).toBeUndefined();

    // Later navigation on B is in scope, and its events are forwarded.
    commit(`${B}/next`);
    expect((await status()).pendingOrigin).toBeUndefined();
    expect(
      socket.sent.some(
        (m) => m.type === 'event' && m.method === 'Page.frameNavigated',
      ),
    ).toBe(true);
    expect((await evaluate(socket)).error).toBeUndefined();
    expect(stops(socket)).toEqual([]);
  });

  it('a child on the allowed origin is in scope after the allow (same scope function)', async () => {
    const socket = await attachedRun();
    bridge.onDebuggerEvent({ tabId: 1 }, 'Target.attachedToTarget', {
      sessionId: 'C',
      targetInfo: { targetId: 'F', type: 'iframe', url: `${B}/x` },
      waitingForDebugger: false,
    });
    const child = () =>
      cdp(socket, 'Runtime.evaluate', { expression: '1' }, 'C');
    expect((await child()).error).toBe(FRAME_OUT_OF_SCOPE);
    commit(`${B}/login`);
    expect(await allow(B)).toEqual({ ok: true });
    commit(`${ORIGIN}/form`);
    expect((await child()).error).toBeUndefined();
  });

  it.each([
    ['a different origin', C],
    ['a non-https origin', 'http://b.gov'],
    ['a URL with a path', `${B}/login`],
    ['a trailing slash', `${B}/`],
    ['an uppercase host', 'https://B.gov'],
    ['a malformed string', 'b.gov'],
    ['a non-string', 42],
  ])('refuses %s and changes nothing', async (_name, origin) => {
    const socket = await attachedRun();
    commit(`${B}/login`);
    expect(await allow(origin)).toEqual({
      ok: false,
      error: 'origin not pending',
    });
    expect((await status()).pendingOrigin).toBe(B);
    expect((await evaluate(socket)).error).toBe(OUT_OF_SCOPE);
  });

  it('refuses with no pending origin', async () => {
    const socket = await attachedRun();
    expect(await allow(B)).toEqual({ ok: false, error: 'origin not pending' });
    commit(`${B}/x`);
    expect((await status()).pendingOrigin).toBe(B);
    expect((await evaluate(socket)).error).toBe(OUT_OF_SCOPE);
  });

  it('refuses with no active run', async () => {
    expect(await allow(B)).toEqual({ ok: false, error: 'no active run' });
  });

  it('refuses the old origin after the pending one moved on (B then C)', async () => {
    const socket = await attachedRun();
    commit(`${B}/x`);
    commit(`${C}/x`);
    expect(await allow(B)).toEqual({ ok: false, error: 'origin not pending' });
    expect((await status()).pendingOrigin).toBe(C);
    expect((await evaluate(socket)).error).toBe(OUT_OF_SCOPE);
  });

  it('refuses when the tab is no longer on the pending origin', async () => {
    const socket = await attachedRun();
    commit(`${B}/x`);
    // The tab moved on, and its commit has not been seen yet.
    fake.tabUrl = `${C}/x`;
    expect(await allow(B)).toEqual({ ok: false, error: 'origin not pending' });
    expect((await evaluate(socket)).error).toBe(OUT_OF_SCOPE);
  });

  it('allows at most 3 extra origins per run', async () => {
    const socket = await attachedRun();
    for (const origin of [B, C, 'https://d.gov']) {
      commit(`${origin}/x`);
      expect(await allow(origin)).toEqual({ ok: true });
    }
    commit('https://e.gov/x');
    expect(await allow('https://e.gov')).toEqual({
      ok: false,
      error: 'too many allowed origins',
    });
    expect((await status()).pendingOrigin).toBe('https://e.gov');
    expect((await evaluate(socket)).error).toBe(OUT_OF_SCOPE);
  });

  it('is cleared by stopRun', async () => {
    await attachedRun();
    commit(`${B}/x`);
    expect(await allow(B)).toEqual({ ok: true });
    commit(`${ORIGIN}/form`);
    expect(await bridge.handleMessage({ type: 'stopRun' })).toEqual({
      ok: true,
    });
    const socket = await attachedRun('run-2');
    commit(`${B}/x`);
    expect((await status()).pendingOrigin).toBe(B);
    expect((await evaluate(socket)).error).toBe(OUT_OF_SCOPE);
  });

  it('is cleared by a replacement run (same run id too)', async () => {
    await attachedRun('run-1');
    commit(`${B}/x`);
    expect(await allow(B)).toEqual({ ok: true });
    commit(`${ORIGIN}/form`);
    const socket = await attachedRun('run-1');
    commit(`${B}/x`);
    expect((await status()).pendingOrigin).toBe(B);
    expect((await evaluate(socket)).error).toBe(OUT_OF_SCOPE);
  });
});

describe('the hold timer', () => {
  it('ends the run with left-origin when no allowOrigin arrives within 60 s', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'Date'] });
    const socket = await attachedRun();
    commit(`${B}/x`);
    await vi.advanceTimersByTimeAsync(HOLD_MS - 1);
    expect((await status()).active).toBe(true);
    expect(stops(socket)).toEqual([]);
    await vi.advanceTimersByTimeAsync(1);
    expect(stops(socket)).toEqual(['left-origin']);
    expect((await status()).active).toBe(false);
  });

  it('navigating back into the origin before the timer clears the hold', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'Date'] });
    const socket = await attachedRun();
    commit(`${B}/x`);
    await vi.advanceTimersByTimeAsync(10_000);
    commit(`${ORIGIN}/callback`);
    expect((await status()).pendingOrigin).toBeUndefined();
    expect((await evaluate(socket)).error).toBeUndefined();
    await vi.advanceTimersByTimeAsync(CAP_MS);
    expect(stops(socket)).toEqual([]);
    expect((await status()).active).toBe(true);
    // B was never allowed: leaving again holds again.
    commit(`${B}/x`);
    expect((await status()).pendingOrigin).toBe(B);
  });

  it('another origin restarts the 60 s timer, up to the 120 s total cap', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'Date'] });
    const socket = await attachedRun();
    commit(`${B}/x`);
    await vi.advanceTimersByTimeAsync(50_000);
    commit(`${C}/x`);
    expect((await status()).pendingOrigin).toBe(C);
    // Past B's deadline (60 s), inside C's (110 s).
    await vi.advanceTimersByTimeAsync(50_000);
    expect((await status()).active).toBe(true);
    commit('https://d.gov/x');
    // D's own 60 s would end at 160 s; the cap ends the hold at 120 s.
    await vi.advanceTimersByTimeAsync(19_999);
    expect((await status()).active).toBe(true);
    await vi.advanceTimersByTimeAsync(1);
    expect(stops(socket)).toEqual(['left-origin']);
  });

  it('another page on the pending origin does not restart the timer', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'Date'] });
    const socket = await attachedRun();
    commit(`${B}/x`);
    await vi.advanceTimersByTimeAsync(50_000);
    commit(`${B}/y`);
    await vi.advanceTimersByTimeAsync(10_000);
    expect(stops(socket)).toEqual(['left-origin']);
  });

  it('an allowed origin leaves no timer behind', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'Date'] });
    const socket = await attachedRun();
    commit(`${B}/x`);
    expect(await allow(B)).toEqual({ ok: true });
    await vi.advanceTimersByTimeAsync(CAP_MS * 2);
    expect(stops(socket)).toEqual([]);
    expect((await status()).active).toBe(true);
  });
});

describe('release from the event stream', () => {
  it('tabs.onUpdated back in scope does not release while attached', async () => {
    const socket = await attachedRun();
    commit(`${B}/x`);
    fake.tabUrl = `${ORIGIN}/form`;
    bridge.onTabUpdated(1, { url: `${ORIGIN}/form` });
    expect((await status()).pendingOrigin).toBe(B);
    // The ordered commit releases it.
    bridge.onDebuggerEvent({ tabId: 1 }, 'Page.frameNavigated', {
      frame: { id: 'T', url: `${ORIGIN}/form` },
    });
    expect((await status()).pendingOrigin).toBeUndefined();
    expect((await evaluate(socket)).error).toBeUndefined();
  });

  it('a pre-commit signal back into scope does not release', async () => {
    await attachedRun();
    commit(`${B}/x`);
    bridge.onDebuggerEvent({ tabId: 1 }, 'Page.frameRequestedNavigation', {
      frameId: 'T',
      url: `${ORIGIN}/form`,
    });
    expect((await status()).pendingOrigin).toBe(B);
  });
});

describe('a command that spans a hold', () => {
  it('is refused even when the tab is back in scope when it finishes (A -> B -> A)', async () => {
    const socket = await attachedRun();
    let open = () => {};
    fake.axGate = new Promise<void>((resolve) => {
      open = resolve;
    });
    const pending = cdp(socket, 'Accessibility.getFullAXTree');
    // Let the command reach chrome.debugger.
    for (
      let i = 0;
      i < 200 &&
      !sendCommand.mock.calls.some(
        ([, m]) => m === 'Accessibility.getFullAXTree',
      );
      i++
    ) {
      await new Promise((resolve) => setTimeout(resolve, 1));
    }
    expect(
      sendCommand.mock.calls.some(
        ([, m]) => m === 'Accessibility.getFullAXTree',
      ),
    ).toBe(true);
    commit(`${B}/x`);
    commit(`${ORIGIN}/form`);
    open();
    expect((await pending).error).toBe(OUT_OF_SCOPE);
    // The run itself carries on.
    expect((await evaluate(socket)).error).toBeUndefined();
  });
});

describe('the relay socket during a hold', () => {
  it('a reconnect during a hold still registers', async () => {
    const socket = await attachedRun();
    commit(`${B}/x`);
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    socket.drop(1006);
    await vi.advanceTimersByTimeAsync(1_000);
    const next = FakeSocket.all.at(-1) as FakeSocket;
    expect(next).not.toBe(socket);
    await next.open();
    expect(next.sent.map((m) => m.type)).toEqual(['register']);
    expect(await allow(B)).toEqual({ ok: true });
    expect((await evaluate(next)).error).toBeUndefined();
  });

  it('a run resumed after a worker restart on another origin ends at once', async () => {
    storage.set('run', {
      relayUrl: RELAY,
      token: tokenFor('run-1'),
      tabId: 1,
      allowedFrameOrigins: [],
    });
    fake.tabUrl = `${B}/x`;
    await bridge.resumeRun();
    const socket = FakeSocket.all.at(-1) as FakeSocket;
    await socket.open();
    expect(stops(socket)).toEqual(['left-origin']);
    expect((await status()).active).toBe(false);
  });
});
