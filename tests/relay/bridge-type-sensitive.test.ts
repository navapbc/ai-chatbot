import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

// The worker's typeSensitive message against a fake chrome.debugger and
// relay socket: it registers the value, checks the focused field (enabled
// text entry, in scope, masked by the redactor) and types it with exactly
// one call on the checked element (never Input.insertText, which types
// wherever focus is), or refuses with a fixed string and types nothing.
// Each test loads a fresh bridge module (no run, lock or world leaks).

const ORIGIN = 'https://a.gov';
const OTHER = 'https://other.a.gov';
const AWAY = 'https://away.example/next';
const FOCUS_CONTEXT = 77;
const FIELD = 42;
// The value typed in every test. It never appears in a test name.
const VALUE = 'Quill-7731-Zebra';

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

const deferred = <T = unknown>() => {
  let resolve!: (v: T) => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
};

/** Per-test knobs for the fake chrome.debugger. */
interface Fake {
  tabFrames: Array<{ id: string; url: string }>;
  /** Whether anything has focus (else document.activeElement is null). */
  focused: boolean;
  /** FIELD_KIND's answer for the focused element. */
  kind: string;
  /** The field's accessible name (its <label>). */
  label: string;
  /** The field's DOM attributes (DOM.describeNode). */
  attributes: string[];
  /** The context of the field's own main world (1: the tab's main frame). */
  fieldContext: number;
  axMissing: boolean;
  describeFails: boolean;
  /** What the agent's Runtime.evaluate gets back (a page echo). */
  pageText: string;
  /**
   * Answers the typing call on the checked element (TYPE_INTO): the page
   * side re-checks focus and types in one task.
   */
  typeInto: (params: Record<string, unknown>) => Promise<unknown>;
  /** Holds the screenshot mask's world creation, when set. */
  maskGate?: Promise<unknown>;
  maskStarted?: () => void;
  /** Holds the accessibility lookup of the field, when set. */
  axGate?: Promise<unknown>;
  axStarted?: () => void;
  /** The tab's URL (chrome.tabs.get). */
  tabUrl: string;
  /** chrome.debugger.attach fails (say, another client holds the tab). */
  attachFails: boolean;
}

let fake: Fake;

const freshFake = (): Fake => ({
  tabFrames: [],
  focused: true,
  kind: 'entry',
  label: 'Social Security Number',
  attributes: ['id', 'f17', 'name', 'field_17'],
  fieldContext: 1,
  axMissing: false,
  describeFails: false,
  pageText: '',
  typeInto: async () => ({ result: { type: 'string', value: 'typed' } }),
  tabUrl: `${ORIGIN}/form`,
  attachFails: false,
});

const attach = vi.fn(async () => {
  if (fake.attachFails) {
    throw new Error('Another debugger is already attached to the tab');
  }
});
const detach = vi.fn(async () => {});

const sendCommand = vi.fn(
  async (
    _debuggee: { tabId: number; sessionId?: string },
    method: string,
    params: Record<string, unknown> = {},
  ): Promise<unknown> => {
    switch (method) {
      case 'Page.getFrameTree':
        return {
          frameTree: {
            frame: { id: 'T', url: `${ORIGIN}/form` },
            childFrames: fake.tabFrames.map((f) => ({
              frame: { ...f, parentId: 'T' },
            })),
          },
        };
      case 'Page.createIsolatedWorld':
        if (params.worldName === 'relay-focus') {
          return { executionContextId: FOCUS_CONTEXT };
        }
        fake.maskStarted?.();
        if (fake.maskGate) await fake.maskGate;
        throw new Error('no mask world here');
      case 'Runtime.evaluate':
        if (params.contextId === FOCUS_CONTEXT) {
          return fake.focused
            ? {
                result: { type: 'object', subtype: 'node', objectId: '1.77.1' },
              }
            : { result: { type: 'object', subtype: 'null', value: null } };
        }
        if (params.expression === 'document') {
          return { result: { type: 'object', objectId: '1.1.2' } };
        }
        return { result: { type: 'string', value: fake.pageText } };
      case 'Runtime.callFunctionOn': {
        const body = String(params.functionDeclaration);
        if (body.includes('execCommand')) return fake.typeInto(params);
        if (body.includes(':enabled:read-write')) {
          return { result: { type: 'string', value: fake.kind } };
        }
        return {};
      }
      case 'DOM.describeNode':
        if (params.objectId === '1.77.1') {
          return {
            node: {
              backendNodeId: FIELD,
              nodeName: 'INPUT',
              localName: 'input',
              attributes: fake.attributes,
            },
          };
        }
        if (params.objectId === '1.1.2') {
          return { node: { documentURL: `${ORIGIN}/form` } };
        }
        if (params.backendNodeId === FIELD) {
          if (fake.describeFails) throw new Error('No node with given id');
          return { node: { nodeName: 'INPUT', attributes: fake.attributes } };
        }
        if (params.backendNodeId === 50) {
          return {
            node: {
              frameId: 'S',
              contentDocument: { documentURL: `${OTHER}/x`, backendNodeId: 51 },
            },
          };
        }
        return {};
      case 'DOM.getFrameOwner':
        return params.frameId === 'S' ? { backendNodeId: 50 } : {};
      case 'DOM.resolveNode':
        if (params.backendNodeId === FIELD) {
          return { object: { objectId: `1.${fake.fieldContext}.9` } };
        }
        if (params.backendNodeId === 51) {
          return { object: { objectId: '1.9.3' } };
        }
        return {};
      case 'Accessibility.getPartialAXTree':
        fake.axStarted?.();
        if (fake.axGate) await fake.axGate;
        return fake.axMissing
          ? { nodes: [] }
          : {
              nodes: [
                {
                  nodeId: '7',
                  ignored: false,
                  role: { type: 'role', value: 'textbox' },
                  name: { type: 'computedString', value: fake.label },
                  backendDOMNodeId: FIELD,
                },
              ],
            };
      default:
        return {};
    }
  },
);

const b64url = (value: unknown) =>
  Buffer.from(JSON.stringify(value)).toString('base64url');

const tokenFor = (runId: string) =>
  `${b64url({
    runId,
    role: 'extension',
    origin: ORIGIN,
    exp: Math.floor(Date.now() / 1000) + 3600,
  })}.sig`;

let bridge: typeof import('@/extension/src/bridge');

function stubChrome() {
  vi.stubGlobal('WebSocket', FakeSocket);
  const storage = new Map<string, unknown>();
  vi.stubGlobal('chrome', {
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
          url: '',
          attached: false,
        },
      ],
      attach,
      detach,
      sendCommand,
    },
  });
}

beforeEach(async () => {
  vi.resetModules();
  fake = freshFake();
  sendCommand.mockClear();
  attach.mockClear();
  detach.mockClear();
  stubChrome();
  bridge = await import('@/extension/src/bridge');
});

afterEach(async () => {
  await bridge.handleMessage({ type: 'stopRun' });
  vi.unstubAllGlobals();
  vi.doUnmock('@/extension/src/config');
  vi.doUnmock('@/extension/src/jev-classifier');
});

const sentTo = (method: string) =>
  sendCommand.mock.calls.filter(([, m]) => m === method);

/** The typing calls: each one types into the element it names. */
const inserts = () =>
  sentTo('Runtime.callFunctionOn').filter(([, , p]) =>
    String(p?.functionDeclaration).includes('execCommand'),
  );

/** Sends a relay message and returns the bridge's reply to it. */
const ask = async (socket: FakeSocket, message: Record<string, unknown>) => {
  await socket.onmessage?.({ data: JSON.stringify(message) });
  const reply = socket.sent.find(
    (m) => m.type === 'reply' && m.id === message.id,
  );
  if (!reply) throw new Error(`no reply to ${String(message.id)}`);
  return reply;
};

async function startedRun(runId = 'run-1') {
  expect(
    await bridge.handleMessage({
      type: 'startRun',
      relayUrl: 'ws://127.0.0.1:1',
      token: tokenFor(runId),
      tabId: 1,
    }),
  ).toEqual({ ok: true });
  const socket = FakeSocket.last;
  await socket.onopen?.();
  return socket;
}

async function attachedRun(runId = 'run-1') {
  const socket = await startedRun(runId);
  expect(
    (await ask(socket, { id: 'a', type: 'attach', targetId: 'T' })).error,
  ).toBeUndefined();
  return socket;
}

const type = (value: unknown = VALUE, runId = 'run-1') =>
  bridge.handleMessage({ type: 'typeSensitive', runId, value });

const tabCommand = (id: string, method: string, params: object = {}) => ({
  id,
  type: 'cdp',
  sessionId: 'tab-1',
  method,
  params,
});

describe('typeSensitive types into a masked focused field', () => {
  it('types the value once, into the element that was checked', async () => {
    await attachedRun();
    const reply = await type();
    expect(reply).toEqual({ ok: true });
    expect(inserts()).toHaveLength(1);
    expect(inserts()[0][0]).toEqual({ tabId: 1 });
    expect(inserts()[0][2]).toMatchObject({
      objectId: '1.77.1',
      arguments: [{ value: VALUE }],
      returnByValue: true,
    });
    // Never through focus.
    expect(sentTo('Input.insertText')).toHaveLength(0);
  });

  it('judges the field by its accessible name, not only its attributes', async () => {
    // field_17 says nothing; only the label makes it sensitive.
    await attachedRun();
    expect(await type()).toEqual({ ok: true });
    fake.label = 'Middle name';
    expect(await type()).toEqual({
      ok: false,
      error: 'field not sensitive',
    });
    expect(inserts()).toHaveLength(1);
  });

  it('accepts a field flagged by its attributes alone', async () => {
    fake.label = 'Number';
    fake.attributes = ['id', 'ssn', 'name', 'ssn'];
    await attachedRun();
    expect(await type()).toEqual({ ok: true });
  });

  it('a password input counts as masked', async () => {
    fake.label = 'Code';
    fake.attributes = ['type', 'password', 'name', 'code'];
    await attachedRun();
    expect(await type()).toEqual({ ok: true });
  });

  it('the reply and every event after typing carry no trace of the value', async () => {
    const socket = await attachedRun();
    const before = socket.sent.length;
    expect(await type()).toEqual({ ok: true });
    // The page echoes it: in a script result, a URL, a request body.
    fake.pageText = `You entered ${VALUE}`;
    const echoed = await ask(
      socket,
      tabCommand('e1', 'Runtime.evaluate', { expression: 'x' }),
    );
    expect(echoed.error).toBeUndefined();
    bridge.onDebuggerEvent({ tabId: 1 }, 'Page.navigatedWithinDocument', {
      frameId: 'T',
      url: `${ORIGIN}/form?id=${encodeURIComponent(VALUE)}`,
    });
    bridge.onDebuggerEvent({ tabId: 1 }, 'Network.requestWillBeSent', {
      requestId: 'r1',
      frameId: 'T',
      request: {
        url: `${ORIGIN}/save`,
        method: 'POST',
        postData: `field_17=${VALUE}`,
      },
    });
    const after = socket.sent.slice(before);
    expect(after.some((m) => m.type === 'event')).toBe(true);
    expect(JSON.stringify(after)).not.toContain(VALUE);
    expect(JSON.stringify(after)).not.toContain('7731');
  });

  it('a value Jev flags makes an otherwise ordinary field typeable', async () => {
    vi.resetModules();
    vi.doMock('@/extension/src/config', () => ({
      APP_ORIGIN: 'https://app.example',
    }));
    const classify = vi.fn(async () => true);
    vi.doMock('@/extension/src/jev-classifier', () => ({
      createJevClassifier: () => ({
        classify,
        failures: () => 0,
        requests: () => 0,
        stop: () => {},
      }),
    }));
    bridge = await import('@/extension/src/bridge');
    fake.label = 'Middle name';
    fake.attributes = ['id', 'middle', 'name', 'middle'];
    await attachedRun();
    expect(await type()).toEqual({ ok: true });
    expect(classify).toHaveBeenCalledTimes(1);
    // The hook sees naming strings only, never the value.
    expect(JSON.stringify(classify.mock.calls)).not.toContain(VALUE);
    classify.mockImplementation(async () => false);
    fake.label = 'Nickname';
    expect(await type()).toEqual({
      ok: false,
      error: 'field not sensitive',
    });
    expect(inserts()).toHaveLength(1);
  });
});

describe('typeSensitive refusals type nothing', () => {
  const refused = async (error: string, value: unknown = VALUE) => {
    const reply = await type(value);
    expect(reply).toEqual({ ok: false, error });
    expect(JSON.stringify(reply)).not.toContain(VALUE);
    expect(inserts()).toHaveLength(0);
  };

  it('no active run', async () => {
    await refused('no active run');
  });

  it('wrong run', async () => {
    await attachedRun();
    expect(await type(VALUE, 'run-2')).toEqual({
      ok: false,
      error: 'wrong run',
    });
    expect(inserts()).toHaveLength(0);
  });

  it('not attached: the attach for typing fails', async () => {
    await startedRun();
    fake.attachFails = true;
    await refused('not attached');
    expect(attach).toHaveBeenCalledTimes(1);
  });

  it('run is held', async () => {
    await attachedRun();
    bridge.onDebuggerEvent({ tabId: 1 }, 'Page.frameNavigated', {
      frame: { id: 'T', url: AWAY },
    });
    expect(
      (await bridge.handleMessage({ type: 'getRunStatus' })) as object,
    ).toMatchObject({ pendingOrigin: 'https://away.example' });
    await refused('run is held');
  });

  it.each([
    ['empty', ''],
    ['blank', '   \n'],
    ['600 characters', 'x'.repeat(600)],
    ['not a string', 1234567],
  ])('value not accepted: %s', async (_what, value) => {
    await attachedRun();
    await refused('value not accepted', value);
  });

  it('value not accepted: the 33rd distinct value', async () => {
    await attachedRun();
    for (let i = 0; i < 32; i++) {
      expect(await type(`${VALUE}-${i}`)).toEqual({ ok: true });
    }
    expect(inserts()).toHaveLength(32);
    expect(await type(`${VALUE}-32`)).toEqual({
      ok: false,
      error: 'value not accepted',
    });
    expect(inserts()).toHaveLength(32);
    // A value already registered is still typed.
    expect(await type(`${VALUE}-3`)).toEqual({ ok: true });
  });

  it('no focused field: nothing has focus', async () => {
    fake.focused = false;
    await attachedRun();
    await refused('no focused field');
  });

  it.each([['not text entry, read-only or disabled', { kind: 'other' }]])(
    'no focused field: %s',
    async (_what, knobs) => {
      Object.assign(fake, knobs);
      await attachedRun();
      await refused('no focused field');
    },
  );

  it('field out of scope: focus stopped at a frame it cannot see into', async () => {
    fake.kind = 'frame';
    await attachedRun();
    await refused('field out of scope');
  });

  it('field out of scope: the field is in a frame of another origin', async () => {
    fake.tabFrames = [{ id: 'S', url: `${OTHER}/x` }];
    fake.fieldContext = 9;
    await attachedRun();
    await refused('field out of scope');
  });

  it('field out of scope: the field is in a frame the bridge does not know', async () => {
    fake.fieldContext = 404;
    await attachedRun();
    await refused('field out of scope');
  });

  it('field not sensitive: an ordinary field', async () => {
    fake.label = 'Middle name';
    fake.attributes = ['id', 'middle', 'name', 'middle'];
    await attachedRun();
    await refused('field not sensitive');
  });

  it('field not sensitive: describe failed (fails closed, unlike redaction)', async () => {
    fake.label = 'Middle name';
    fake.describeFails = true;
    await attachedRun();
    await refused('field not sensitive');
  });

  it('type failed: no accessibility node for the field', async () => {
    fake.axMissing = true;
    await attachedRun();
    await refused('type failed');
  });

  it('a refusal after registering keeps the value registered', async () => {
    fake.label = 'Middle name';
    fake.attributes = ['id', 'middle', 'name', 'middle'];
    const socket = await attachedRun();
    await refused('field not sensitive');
    fake.pageText = VALUE;
    const echoed = await ask(
      socket,
      tabCommand('e2', 'Runtime.evaluate', { expression: 'x' }),
    );
    expect(JSON.stringify(echoed)).not.toContain(VALUE);
  });
});

describe('typeSensitive attaches on demand (the agent is not attached)', () => {
  /** A run the agent attached, then a same-run token refresh. */
  const refreshedRun = async () => {
    await attachedRun();
    const socket = await startedRun();
    attach.mockClear();
    detach.mockClear();
    return socket;
  };

  /** Starts typing and waits until the field is being judged. */
  const typingHeld = async () => {
    const gate = deferred();
    const axStarted = deferred<void>();
    fake.axGate = gate.promise;
    fake.axStarted = () => axStarted.resolve();
    const typing = type();
    await axStarted.promise;
    return { typing, open: () => gate.resolve(undefined) };
  };

  const status = async () =>
    (await bridge.handleMessage({ type: 'getRunStatus' })) as {
      attached?: boolean;
    };

  it('after a same-run refresh: attaches, types once, replies ok and lets go', async () => {
    const socket = await refreshedRun();
    expect(await type()).toEqual({ ok: true });
    expect(attach).toHaveBeenCalledTimes(1);
    expect(inserts()).toHaveLength(1);
    expect(inserts()[0][2]).toMatchObject({
      objectId: '1.77.1',
      arguments: [{ value: VALUE }],
    });
    // Let go again before answering: the agent's daemon is still stale.
    expect(detach.mock.calls.length).toBeGreaterThanOrEqual(2);
    expect(detach.mock.invocationCallOrder.at(-1)).toBeGreaterThan(
      attach.mock.invocationCallOrder[0],
    );
    // The agent-facing flag never moved: its commands still get
    // `not attached`, so the Eve tool rotates its stale daemon.
    expect((await status()).attached).toBe(false);
    const evaluated = await ask(
      socket,
      tabCommand('c1', 'Runtime.evaluate', { expression: 'x' }),
    );
    expect(evaluated.error).toBe('not attached');
    const listed = await ask(socket, { id: 'l1', type: 'listTargets' });
    expect(listed.result).toMatchObject({
      targetInfos: [{ targetId: 'T', attached: false }],
    });
  });

  it('a fresh run the agent never attached: typed the same way', async () => {
    const socket = await startedRun();
    expect(await type()).toEqual({ ok: true });
    expect(attach).toHaveBeenCalledTimes(1);
    expect(inserts()).toHaveLength(1);
    expect(
      (
        await ask(
          socket,
          tabCommand('c1', 'Runtime.evaluate', { expression: 'x' }),
        )
      ).error,
    ).toBe('not attached');
  });

  it('an agent command during a typing attach is refused before it reaches the page', async () => {
    const socket = await refreshedRun();
    const held = await typingHeld();
    expect((await status()).attached).toBe(false);
    const resolvesBefore = sentTo('DOM.resolveNode').length;
    const callsBefore = sendCommand.mock.calls.length;
    const reply = await ask(
      socket,
      tabCommand('c1', 'DOM.resolveNode', { backendNodeId: FIELD }),
    );
    expect(reply.error).toBe('not attached');
    expect(sentTo('DOM.resolveNode')).toHaveLength(resolvesBefore);
    expect(sendCommand.mock.calls.length).toBe(callsBefore);
    held.open();
    expect(await held.typing).toEqual({ ok: true });
  });

  it('events from a typing attach are not forwarded and start no child session', async () => {
    const socket = await refreshedRun();
    const held = await typingHeld();
    const before = socket.sent.length;
    bridge.onDebuggerEvent({ tabId: 1 }, 'Runtime.consoleAPICalled', {
      type: 'log',
      args: [{ type: 'string', value: 'hello' }],
      executionContextId: 1,
    });
    bridge.onDebuggerEvent({ tabId: 1 }, 'Page.frameNavigated', {
      frame: { id: 'T', url: `${ORIGIN}/form#2` },
    });
    bridge.onDebuggerEvent({ tabId: 1 }, 'Target.attachedToTarget', {
      sessionId: 'child-1',
      targetInfo: {
        targetId: 'C',
        type: 'iframe',
        url: `${ORIGIN}/child`,
        attached: true,
      },
      waitingForDebugger: false,
    });
    expect(socket.sent.slice(before)).toEqual([]);
    held.open();
    expect(await held.typing).toEqual({ ok: true });
    // The child was never tracked: a command naming it is refused.
    const reply = await ask(socket, {
      id: 'c1',
      type: 'cdp',
      sessionId: 'child-1',
      method: 'Runtime.evaluate',
      params: { expression: 'x' },
    });
    expect(reply.error).toBe('not attached');
    expect(socket.sent.filter((m) => m.type === 'event')).toEqual([]);
  });

  it('an out-of-scope tab refuses before any attach', async () => {
    await refreshedRun();
    fake.tabUrl = AWAY;
    expect(await type()).toEqual({ ok: false, error: 'run is held' });
    expect(attach).not.toHaveBeenCalled();
    expect(detach).not.toHaveBeenCalled();
    expect(inserts()).toHaveLength(0);
  });

  it('a failed attach answers not attached and the value stays registered', async () => {
    const socket = await refreshedRun();
    fake.attachFails = true;
    fake.label = 'Middle name';
    expect(await type()).toEqual({ ok: false, error: 'not attached' });
    expect(inserts()).toHaveLength(0);
    // The agent attaches later; the page echoes the value: scrubbed.
    fake.attachFails = false;
    expect(
      (await ask(socket, { id: 'a2', type: 'attach', targetId: 'T' })).error,
    ).toBeUndefined();
    fake.pageText = `You entered ${VALUE}`;
    const echoed = await ask(
      socket,
      tabCommand('e1', 'Runtime.evaluate', { expression: 'x' }),
    );
    expect(echoed.error).toBeUndefined();
    expect(JSON.stringify(echoed)).not.toContain(VALUE);
  });

  it('a later agent attach works and takes the tab over', async () => {
    const socket = await refreshedRun();
    expect(await type()).toEqual({ ok: true });
    expect(
      (await ask(socket, { id: 'a2', type: 'attach', targetId: 'T' })).error,
    ).toBeUndefined();
    expect((await status()).attached).toBe(true);
    expect(
      (
        await ask(
          socket,
          tabCommand('c1', 'Runtime.evaluate', { expression: 'x' }),
        )
      ).error,
    ).toBeUndefined();
    // Typing now uses the agent's attach: no attach or detach of its own.
    attach.mockClear();
    detach.mockClear();
    expect(await type()).toEqual({ ok: true });
    expect(attach).not.toHaveBeenCalled();
    expect(detach).not.toHaveBeenCalled();
    expect((await status()).attached).toBe(true);
  });

  it('an agent attach during typing takes over: the typing fails, the tab stays attached', async () => {
    const socket = await refreshedRun();
    const held = await typingHeld();
    const attached = ask(socket, { id: 'a2', type: 'attach', targetId: 'T' });
    // The agent's attach has let go of the typing one and attached again.
    await vi.waitFor(() => expect(attach).toHaveBeenCalledTimes(2));
    held.open();
    expect(await held.typing).toEqual({ ok: false, error: 'type failed' });
    expect((await attached).error).toBeUndefined();
    expect(inserts()).toHaveLength(0);
    expect((await status()).attached).toBe(true);
    // The typing's own let-go did not detach the agent's attach.
    const lastAttach = attach.mock.invocationCallOrder.at(-1) ?? 0;
    expect(
      detach.mock.invocationCallOrder.filter((n) => n > lastAttach),
    ).toEqual([]);
  });

  it('a debugger detach during typing clears the typing attach', async () => {
    const socket = await refreshedRun();
    const held = await typingHeld();
    bridge.onDebuggerDetach({ tabId: 1 }, 'target_closed');
    held.open();
    expect(await held.typing).toEqual({ ok: false, error: 'not attached' });
    expect(inserts()).toHaveLength(0);
    // Nothing announced to the agent, which was never given this session.
    expect(
      socket.sent.filter((m) => m.method === 'Target.detachedFromTarget'),
    ).toEqual([]);
    // The run waits for the agent now, not for typing.
    attach.mockClear();
    expect(await type()).toEqual({ ok: false, error: 'not attached' });
    expect(attach).not.toHaveBeenCalled();
  });

  it('a hold that starts during a typing attach refuses with run is held', async () => {
    const socket = await refreshedRun();
    const held = await typingHeld();
    const before = socket.sent.length;
    bridge.onDebuggerEvent({ tabId: 1 }, 'Page.frameNavigated', {
      frame: { id: 'T', url: AWAY },
    });
    expect(await status()).toMatchObject({
      pendingOrigin: 'https://away.example',
    });
    held.open();
    expect(await held.typing).toEqual({ ok: false, error: 'run is held' });
    expect(inserts()).toHaveLength(0);
    expect(socket.sent.slice(before).filter((m) => m.type === 'event')).toEqual(
      [],
    );
  });
});

describe('typeSensitive timing', () => {
  it('one at a time: a second call while one is typing is refused', async () => {
    await attachedRun();
    const gate = deferred();
    const axStarted = deferred<void>();
    fake.axGate = gate.promise;
    fake.axStarted = () => axStarted.resolve();
    const first = type();
    await axStarted.promise;
    const second = await type(`${VALUE}-2`);
    expect(second).toEqual({ ok: false, error: 'type failed' });
    gate.resolve(undefined);
    expect(await first).toEqual({ ok: true });
    expect(inserts()).toHaveLength(1);
    // Free again once the first has answered.
    fake.axGate = undefined;
    expect(await type()).toEqual({ ok: true });
  });

  it('focus moved to an ordinary field after the check: nothing is typed there', async () => {
    // A model of the page: the checked field (1.77.1) and an ordinary one.
    // The page script moves focus while the field is judged; the typing call
    // re-checks focus on the checked element and types in the same task.
    const page = { focus: '1.77.1', values: new Map<string, string>() };
    fake.typeInto = async (params) => {
      const target = String(params.objectId);
      if (page.focus !== target) {
        return { result: { type: 'string', value: 'moved' } };
      }
      const [arg] = params.arguments as Array<{ value: string }>;
      page.values.set(target, arg.value);
      return { result: { type: 'string', value: 'typed' } };
    };
    await attachedRun();
    const gate = deferred();
    const axStarted = deferred<void>();
    fake.axGate = gate.promise;
    fake.axStarted = () => axStarted.resolve();
    const typing = type();
    await axStarted.promise;
    page.focus = 'ordinary-field';
    gate.resolve(undefined);
    const reply = await typing;
    expect(reply).toEqual({ ok: false, error: 'no focused field' });
    expect(JSON.stringify(reply)).not.toContain(VALUE);
    expect(page.values.size).toBe(0);
    // The one typing call named the checked element, never the focus.
    expect(inserts().map(([, , p]) => p?.objectId)).toEqual(['1.77.1']);
    expect(sentTo('Input.insertText')).toHaveLength(0);
  });

  it('type failed: the page refused the input (value unchanged)', async () => {
    fake.typeInto = async () => ({
      result: { type: 'string', value: 'unchanged' },
    });
    await attachedRun();
    expect(await type()).toEqual({ ok: false, error: 'type failed' });
    expect(inserts()).toHaveLength(1);
  });

  it('type failed: the typing call fails, and it is never sent again', async () => {
    fake.typeInto = async () => {
      throw new Error(`Cannot insert ${VALUE}`);
    };
    await attachedRun();
    const reply = await type();
    expect(reply).toEqual({ ok: false, error: 'type failed' });
    expect(inserts()).toHaveLength(1);
  });

  it('type failed: the typing call hangs past its timeout', async () => {
    fake.typeInto = () => new Promise(() => {});
    await attachedRun();
    const reply = await type();
    expect(reply).toEqual({ ok: false, error: 'type failed' });
    expect(inserts()).toHaveLength(1);
  }, 10_000);

  it('a screenshot in flight delays typing until it lets go of the tab', async () => {
    const socket = await attachedRun();
    const gate = deferred();
    const maskStarted = deferred<void>();
    fake.maskGate = gate.promise;
    fake.maskStarted = () => maskStarted.resolve();
    const shot = ask(socket, tabCommand('s1', 'Page.captureScreenshot'));
    await maskStarted.promise;
    const typing = type();
    // The screenshot holds the tab: the focus is not even read yet.
    await new Promise((r) => setTimeout(r, 100));
    expect(
      sentTo('Runtime.evaluate').filter(
        ([, , p]) => p?.contextId === FOCUS_CONTEXT,
      ),
    ).toHaveLength(0);
    expect(inserts()).toHaveLength(0);
    gate.resolve(undefined);
    expect((await shot).error).toBeDefined();
    expect(await typing).toEqual({ ok: true });
    expect(inserts()).toHaveLength(1);
  });

  it('Stop while the field is judged types nothing and the reply has no value', async () => {
    await attachedRun();
    const gate = deferred();
    const axStarted = deferred<void>();
    fake.axGate = gate.promise;
    fake.axStarted = () => axStarted.resolve();
    const typing = type();
    await axStarted.promise;
    const stopped = bridge.handleMessage({ type: 'stopRun', runId: 'run-1' });
    gate.resolve(undefined);
    const reply = await typing;
    await stopped;
    expect(reply).toEqual({ ok: false, error: 'type failed' });
    expect(JSON.stringify(reply)).not.toContain(VALUE);
    expect(inserts()).toHaveLength(0);
  });

  it('Stop while the typing call is in flight: the reply has no value', async () => {
    const insert = deferred();
    fake.typeInto = () => insert.promise;
    await attachedRun();
    const typing = type();
    await vi.waitFor(() => expect(inserts()).toHaveLength(1));
    const stopped = bridge.handleMessage({ type: 'stopRun', runId: 'run-1' });
    // The detach fails the command.
    insert.reject(new Error(`Detached while handling ${VALUE}`));
    const reply = await typing;
    await stopped;
    expect(reply).toEqual({ ok: false, error: 'type failed' });
    expect(JSON.stringify(reply)).not.toContain(VALUE);
    expect(inserts()).toHaveLength(1);
  });

  it('a hold that starts while the field is judged refuses with run is held', async () => {
    await attachedRun();
    const gate = deferred();
    const axStarted = deferred<void>();
    fake.axGate = gate.promise;
    fake.axStarted = () => axStarted.resolve();
    const typing = type();
    await axStarted.promise;
    bridge.onDebuggerEvent({ tabId: 1 }, 'Page.frameNavigated', {
      frame: { id: 'T', url: AWAY },
    });
    gate.resolve(undefined);
    expect(await typing).toEqual({ ok: false, error: 'run is held' });
    expect(inserts()).toHaveLength(0);
  });
});
