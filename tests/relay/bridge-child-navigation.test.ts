import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';

// The bridge against a fake chrome and relay socket: a child session's own
// navigation event moves it out of scope in the same turn, with no
// Target.targetInfoChanged ever sent.

const ORIGIN = 'https://a.gov';
const FRAME = 'https://frames.gov';
const AWAY = 'https://away.example/next';
const FRAME_OUT_OF_SCOPE = 'frame is outside the allowed origins';

interface Sent {
  type: string;
  id?: string;
  result?: unknown;
  error?: string;
  method?: string;
  params?: unknown;
}

class FakeSocket {
  static OPEN = 1;
  static last: FakeSocket;
  readyState = 1;
  sent: Sent[] = [];
  onopen?: () => Promise<void>;
  onmessage?: (event: { data: string }) => Promise<void>;
  onclose?: (event: { code: number }) => void;
  constructor(readonly url: string) {
    FakeSocket.last = this;
  }
  send(data: string) {
    this.sent.push(JSON.parse(data));
  }
  close() {
    this.readyState = 3;
  }
}

const sendCommand = vi.fn(
  async (
    debuggee: { tabId: number; sessionId?: string },
    method: string,
  ): Promise<unknown> => {
    if (method === 'Page.getFrameTree') {
      return debuggee.sessionId
        ? {
            frameTree: { frame: { id: 'F', parentId: 'T', url: `${FRAME}/x` } },
          }
        : { frameTree: { frame: { id: 'T', url: `${ORIGIN}/form` } } };
    }
    if (method === 'Runtime.evaluate') {
      return { result: { type: 'number', value: 1, description: '1' } };
    }
    return {};
  },
);

const b64url = (value: unknown) =>
  Buffer.from(JSON.stringify(value)).toString('base64url');

let bridge: typeof import('@/extension/src/bridge');

beforeAll(async () => {
  vi.stubGlobal('WebSocket', FakeSocket);
  vi.stubGlobal('chrome', {
    tabs: { get: async () => ({ url: `${ORIGIN}/form` }) },
    storage: {
      session: {
        set: async () => {},
        get: async () => ({}),
        remove: async () => {},
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
      attach: async () => {},
      detach: async () => {},
      sendCommand,
    },
  });
  bridge = await import('@/extension/src/bridge');
});

afterEach(async () => {
  await bridge.handleMessage({ type: 'stopRun' });
});

/** Sends a relay message and returns the bridge's reply to it. */
const ask = async (socket: FakeSocket, message: Record<string, unknown>) => {
  await socket.onmessage?.({ data: JSON.stringify(message) });
  const reply = socket.sent.find(
    (m) => m.type === 'reply' && m.id === message.id,
  );
  if (!reply) throw new Error(`no reply to ${String(message.id)}`);
  return reply;
};

const childEvaluate = (id: string) => ({
  id,
  type: 'cdp',
  sessionId: 'C',
  method: 'Runtime.evaluate',
  params: { expression: '1' },
});

async function attachedWithChild() {
  const token = `${b64url({
    runId: 'run-1',
    role: 'extension',
    origin: ORIGIN,
    exp: Math.floor(Date.now() / 1000) + 3600,
  })}.sig`;
  const started = await bridge.handleMessage({
    type: 'startRun',
    relayUrl: 'ws://127.0.0.1:1',
    token,
    tabId: 1,
    allowedFrameOrigins: [FRAME],
  });
  expect(started).toEqual({ ok: true });
  const socket = FakeSocket.last;
  await socket.onopen?.();
  expect(
    (await ask(socket, { id: 'a', type: 'attach', targetId: 'T' })).error,
  ).toBeUndefined();
  bridge.onDebuggerEvent({ tabId: 1 }, 'Target.attachedToTarget', {
    sessionId: 'C',
    targetInfo: { targetId: 'F', type: 'iframe', url: `${FRAME}/x` },
    waitingForDebugger: false,
  });
  return socket;
}

describe('child navigation ordering', () => {
  it('drives an allowed child', async () => {
    const socket = await attachedWithChild();
    const reply = await ask(socket, childEvaluate('1'));
    expect(reply.error).toBeUndefined();
  });

  it('refuses the next command after the child navigates away, without targetInfoChanged', async () => {
    const socket = await attachedWithChild();
    expect((await ask(socket, childEvaluate('1'))).error).toBeUndefined();
    const evaluations = () =>
      sendCommand.mock.calls.filter(
        ([d, m]) => d.sessionId === 'C' && m === 'Runtime.evaluate',
      ).length;
    const before = evaluations();

    // The child's own commit, then a command in the same turn.
    bridge.onDebuggerEvent(
      { tabId: 1, sessionId: 'C' },
      'Page.frameNavigated',
      { frame: { id: 'F', parentId: 'T', url: AWAY } },
    );
    const reply = await ask(socket, childEvaluate('2'));
    expect(reply.error).toBe(FRAME_OUT_OF_SCOPE);
    expect(evaluations()).toBe(before);
    // The event that moved it was not forwarded either.
    expect(JSON.stringify(socket.sent)).not.toContain(AWAY);
  });

  it('a same-document navigation keeps the child allowed', async () => {
    const socket = await attachedWithChild();
    bridge.onDebuggerEvent(
      { tabId: 1, sessionId: 'C' },
      'Page.navigatedWithinDocument',
      { frameId: 'F', url: `${FRAME}/x#step2` },
    );
    expect((await ask(socket, childEvaluate('3'))).error).toBeUndefined();
  });
});
