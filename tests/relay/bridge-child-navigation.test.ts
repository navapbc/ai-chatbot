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

const OTHER = 'https://other.a.gov';
const FOCUS_CONTEXT = 77;

/** Per-test knobs for the fake chrome.debugger. */
const fake: {
  /** Extra frames of the tab, under its main frame T. */
  tabFrames: Array<{ id: string; url: string }>;
  /** What DOM.describeNode says about the focused element. */
  focused: Record<string, unknown>;
  /** Runs while the bridge reads focus, before it gets the answer. */
  duringFocusRead?: () => void;
} = { tabFrames: [], focused: { localName: 'input' } };

const sendCommand = vi.fn(
  async (
    debuggee: { tabId: number; sessionId?: string },
    method: string,
    params?: Record<string, unknown>,
  ): Promise<unknown> => {
    if (method === 'Page.getFrameTree') {
      return debuggee.sessionId
        ? {
            frameTree: { frame: { id: 'F', parentId: 'T', url: `${FRAME}/x` } },
          }
        : {
            frameTree: {
              frame: { id: 'T', url: `${ORIGIN}/form` },
              childFrames: fake.tabFrames.map((f) => ({
                frame: { ...f, parentId: 'T' },
              })),
            },
          };
    }
    if (method === 'Page.createIsolatedWorld') {
      return { executionContextId: FOCUS_CONTEXT };
    }
    if (method === 'Runtime.evaluate') {
      if (params?.contextId === FOCUS_CONTEXT) {
        fake.duringFocusRead?.();
        return {
          result: { type: 'object', subtype: 'node', objectId: '1.77.1' },
        };
      }
      return { result: { type: 'number', value: 1, description: '1' } };
    }
    if (method === 'DOM.describeNode') return { node: fake.focused };
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
  fake.tabFrames = [];
  fake.focused = { localName: 'input' };
  fake.duringFocusRead = undefined;
  sendCommand.mockClear();
});

const sentTo = (method: string) =>
  sendCommand.mock.calls.filter(([, m]) => m === method);

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

const tabCommand = (id: string, method: string, params: object = {}) => ({
  id,
  type: 'cdp',
  sessionId: 'tab-1',
  method,
  params,
});

describe('input while a frame out of scope is present', () => {
  it('reaches an ordinary focused element', async () => {
    fake.tabFrames = [{ id: 'S', url: `${OTHER}/x` }];
    const socket = await attachedWithChild();
    const reply = await ask(
      socket,
      tabCommand('i1', 'Input.insertText', { text: 'a' }),
    );
    expect(reply.error).toBeUndefined();
    expect(sentTo('Input.insertText')).toHaveLength(1);
  });

  it('is refused when the out-of-scope frame set changes while focus is read', async () => {
    fake.tabFrames = [{ id: 'S', url: `${OTHER}/x` }];
    const socket = await attachedWithChild();
    fake.duringFocusRead = () => {
      bridge.onDebuggerEvent({ tabId: 1 }, 'Page.frameNavigated', {
        frame: { id: 'S2', parentId: 'T', url: `${OTHER}/late` },
      });
    };
    const reply = await ask(
      socket,
      tabCommand('i2', 'Input.insertText', { text: 'a' }),
    );
    expect(reply.error).toBe(FRAME_OUT_OF_SCOPE);
    expect(sentTo('Input.insertText')).toHaveLength(0);
  });

  it.each([
    [
      'a host with a closed shadow root',
      { localName: 'div', shadowRoots: [{ shadowRootType: 'closed' }] },
    ],
    ['a frame owner without a frame', { localName: 'object' }],
    [
      'a frame owner showing the out-of-scope frame',
      { localName: 'iframe', frameId: 'S' },
    ],
    [
      'a frame owner showing a disallowed child',
      { localName: 'iframe', frameId: 'X' },
    ],
  ])('is refused when focus is on %s', async (_what, node) => {
    fake.tabFrames = [{ id: 'S', url: `${OTHER}/x` }];
    fake.focused = node;
    const socket = await attachedWithChild();
    const reply = await ask(
      socket,
      tabCommand('i3', 'Input.dispatchKeyEvent', { type: 'char', text: 'a' }),
    );
    expect(reply.error).toBe(FRAME_OUT_OF_SCOPE);
  });

  it('allows an open shadow host and an in-scope frame owner', async () => {
    fake.tabFrames = [{ id: 'S', url: `${OTHER}/x` }];
    const socket = await attachedWithChild();
    for (const node of [
      { localName: 'div', shadowRoots: [{ shadowRootType: 'open' }] },
      { localName: 'iframe', frameId: 'F' },
    ]) {
      fake.focused = node;
      const reply = await ask(
        socket,
        tabCommand(`i4-${node.localName}`, 'Input.insertText', { text: 'a' }),
      );
      expect(reply.error).toBeUndefined();
    }
  });

  it('a disallowed child alone turns the check on; with nothing out of scope there is no lookup', async () => {
    const socket = await attachedWithChild();
    await ask(socket, tabCommand('i5', 'Input.insertText', { text: 'a' }));
    expect(sentTo('Page.createIsolatedWorld')).toHaveLength(0);
    bridge.onDebuggerEvent({ tabId: 1 }, 'Target.attachedToTarget', {
      sessionId: 'D',
      targetInfo: {
        targetId: 'X',
        type: 'iframe',
        url: 'https://ads.example/',
      },
      waitingForDebugger: false,
    });
    fake.focused = { localName: 'iframe', frameId: 'X' };
    const reply = await ask(
      socket,
      tabCommand('i6', 'Input.insertText', { text: 'a' }),
    );
    expect(reply.error).toBe(FRAME_OUT_OF_SCOPE);
  });
});

describe('dialogs raised by a frame out of scope', () => {
  const opening = (frameId: string, type = 'prompt') => ({
    frameId,
    type,
    message: 'question',
    url: '',
    defaultPrompt: '',
    hasBrowserHandler: false,
  });
  const handle = (id: string, params: object) =>
    tabCommand(id, 'Page.handleJavaScriptDialog', params);

  it('are answered by the bridge, hidden from the agent, and take no agent text until closed', async () => {
    fake.tabFrames = [{ id: 'S', url: `${OTHER}/x` }];
    const socket = await attachedWithChild();
    bridge.onDebuggerEvent(
      { tabId: 1 },
      'Page.javascriptDialogOpening',
      opening('S'),
    );
    expect(sentTo('Page.handleJavaScriptDialog')).toEqual([
      [{ tabId: 1 }, 'Page.handleJavaScriptDialog', { accept: false }],
    ]);
    expect(JSON.stringify(socket.sent)).not.toContain('javascriptDialog');
    expect(
      (await ask(socket, handle('d1', { accept: true, promptText: 'hello' })))
        .error,
    ).toBe(FRAME_OUT_OF_SCOPE);
    expect(
      (await ask(socket, handle('d2', { accept: false, promptText: '' })))
        .error,
    ).toBe(FRAME_OUT_OF_SCOPE);
    expect((await ask(socket, handle('d3', { accept: false }))).error).toBe(
      undefined,
    );
    bridge.onDebuggerEvent({ tabId: 1 }, 'Page.javascriptDialogClosed', {
      result: false,
      userInput: '',
    });
    expect(JSON.stringify(socket.sent)).not.toContain('javascriptDialog');
    // Cleared on close: the next (in-scope) dialog's answer is the agent's.
    expect(
      (await ask(socket, handle('d4', { accept: true, promptText: 'x' })))
        .error,
    ).toBeUndefined();
  });

  it('an alert is accepted (its only answer)', async () => {
    fake.tabFrames = [{ id: 'S', url: `${OTHER}/x` }];
    await attachedWithChild();
    bridge.onDebuggerEvent(
      { tabId: 1 },
      'Page.javascriptDialogOpening',
      opening('S', 'alert'),
    );
    expect(sentTo('Page.handleJavaScriptDialog')[0]?.[2]).toEqual({
      accept: true,
    });
  });

  it('an in-scope dialog reaches the agent (blank) and the agent answers it', async () => {
    const socket = await attachedWithChild();
    bridge.onDebuggerEvent(
      { tabId: 1 },
      'Page.javascriptDialogOpening',
      opening('T'),
    );
    expect(sentTo('Page.handleJavaScriptDialog')).toHaveLength(0);
    const event = socket.sent.find(
      (m) => m.method === 'Page.javascriptDialogOpening',
    );
    expect(event?.params).toMatchObject({ message: '', type: 'prompt' });
    expect(
      (await ask(socket, handle('d5', { accept: true, promptText: 'ok' })))
        .error,
    ).toBeUndefined();
  });
});
