import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

// The worker's showUnfilledField message against a fake chrome.debugger and
// relay socket: it counts the required fields still empty (one script in the
// bridge's own isolated world), classifies the first (any doubt counts as
// sensitive), highlights it with Overlay (or scrolls and focuses it), and
// answers counts and flags only. Refusals are fixed strings and make no CDP
// call. Each test loads a fresh bridge module (no run, lock or world leaks).

const ORIGIN = 'https://a.gov';
const OTHER = 'https://other.a.gov';
const AWAY = 'https://away.example/next';
const FOCUS_CONTEXT = 77;
const FIELD = 42;
const LIST = '1.77.50';
const CANDIDATE = '1.77.1';
// Later candidates: object 1.77.(OTHER_BASE + index), backend node the same.
const OTHER_BASE = 100;
// Page text and field naming that must never reach a reply.
const PAGE_TEXT = 'Quill-7731-Zebra';
const SECRET_NAME = 'applicant_secret_field';

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
  /** How many required empty fields the scan finds. */
  count: number;
  /** The first candidate's accessible role and name. */
  role: string;
  label: string;
  /** The first candidate's DOM attributes (undefined: none reported). */
  attributes: string[] | undefined;
  /** The context of the candidate's own main world (1: the main frame). */
  fieldContext: number;
  axMissing: boolean;
  axFails: boolean;
  /** Overlay.highlightNode fails (after Overlay.enable succeeded). */
  highlightFails: boolean;
  /** Overlay.enable fails (the domain is unavailable). */
  overlayFails: boolean;
  /** What the agent's Runtime.evaluate gets back. */
  pageText: string;
  /** Holds the screenshot mask's world creation, when set. */
  maskGate?: Promise<unknown>;
  maskStarted?: () => void;
  /** Holds the accessibility lookup of the candidate, when set. */
  axGate?: Promise<unknown>;
  axStarted?: () => void;
  tabUrl: string;
  attachFails: boolean;
}

let fake: Fake;

const freshFake = (): Fake => ({
  tabFrames: [],
  count: 3,
  role: 'textbox',
  label: 'Middle name',
  attributes: ['id', 'f17', 'name', 'field_17'],
  fieldContext: 1,
  axMissing: false,
  axFails: false,
  highlightFails: false,
  overlayFails: false,
  pageText: '',
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
          return { result: { type: 'object', objectId: LIST } };
        }
        if (params.expression === 'document') {
          return { result: { type: 'object', objectId: '1.1.2' } };
        }
        return { result: { type: 'string', value: fake.pageText } };
      case 'Runtime.callFunctionOn': {
        const body = String(params.functionDeclaration);
        if (params.objectId === LIST && body.includes('this.els.length')) {
          return { result: { type: 'object', value: { count: fake.count } } };
        }
        if (params.objectId === LIST && body.includes('this.els[')) {
          const index = (params.arguments as Array<{ value: number }>)[0].value;
          const objectId =
            index === 0 ? CANDIDATE : `1.77.${OTHER_BASE + index}`;
          return { result: { type: 'object', subtype: 'node', objectId } };
        }
        return { result: { type: 'undefined' } };
      }
      case 'DOM.describeNode':
        if (params.objectId === CANDIDATE) {
          return {
            node: {
              backendNodeId: FIELD,
              nodeName: 'INPUT',
              localName: 'input',
              ...(fake.attributes ? { attributes: fake.attributes } : {}),
            },
          };
        }
        if (params.objectId === '1.1.2') {
          return { node: { documentURL: `${ORIGIN}/form` } };
        }
        const other = /^1\.77\.(\d+)$/.exec(String(params.objectId));
        if (other && Number(other[1]) > OTHER_BASE) {
          const id = Number(other[1]);
          return {
            node: { backendNodeId: id, attributes: ['id', `other-${id}`] },
          };
        }
        if (params.backendNodeId === FIELD) {
          return {
            node: { nodeName: 'INPUT', attributes: fake.attributes ?? [] },
          };
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
        if (params.backendNodeId !== FIELD) {
          // A later candidate's lookup never answers.
          await new Promise(() => {});
          return {
            nodes: [
              {
                nodeId: '8',
                role: { type: 'role', value: 'textbox' },
                name: { type: 'computedString', value: 'Other field' },
                backendDOMNodeId: params.backendNodeId,
              },
            ],
          };
        }
        fake.axStarted?.();
        if (fake.axGate) await fake.axGate;
        if (fake.axFails) throw new Error('No node with given id');
        return fake.axMissing
          ? { nodes: [] }
          : {
              nodes: [
                {
                  nodeId: '7',
                  ignored: false,
                  role: { type: 'role', value: fake.role },
                  name: { type: 'computedString', value: fake.label },
                  backendDOMNodeId: FIELD,
                },
              ],
            };
      case 'Overlay.highlightNode':
        if (fake.highlightFails)
          throw new Error('Could not compute box model.');
        return {};
      case 'Overlay.enable':
        if (fake.overlayFails) throw new Error("'Overlay.enable' wasn't found");
        return {};
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
  vi.useRealTimers();
  await bridge.handleMessage({ type: 'stopRun' });
  vi.unstubAllGlobals();
});

const sentTo = (method: string) =>
  sendCommand.mock.calls.filter(([, m]) => m === method);

/** The fallback: the candidate scrolled into view and focused. */
const focusCalls = () =>
  sentTo('Runtime.callFunctionOn').filter(([, , p]) =>
    String(p?.functionDeclaration).includes('focus('),
  );

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

const show = (runId = 'run-1') =>
  bridge.handleMessage({ type: 'showUnfilledField', runId });

const tabCommand = (id: string, method: string, params: object = {}) => ({
  id,
  type: 'cdp',
  sessionId: 'tab-1',
  method,
  params,
});

const status = async () =>
  (await bridge.handleMessage({ type: 'getRunStatus' })) as {
    attached?: boolean;
    needsRegistration?: boolean;
  };

describe('showUnfilledField finds and highlights the first empty required field', () => {
  it('counts the candidates and highlights the first, once', async () => {
    await attachedRun();
    const reply = await show();
    expect(reply).toEqual({
      ok: true,
      found: true,
      remaining: 3,
      sensitive: false,
    });
    // One scan, in the bridge's own world, kept by reference.
    const scans = sentTo('Runtime.evaluate').filter(
      ([, , p]) => p?.contextId === FOCUS_CONTEXT,
    );
    expect(scans).toHaveLength(1);
    expect(scans[0][2]).not.toHaveProperty('returnByValue');
    expect(sentTo('DOM.scrollIntoViewIfNeeded')).toHaveLength(1);
    expect(sentTo('Overlay.highlightNode')).toHaveLength(1);
    expect(sentTo('Overlay.highlightNode')[0][2]).toMatchObject({
      backendNodeId: FIELD,
      highlightConfig: { showInfo: false },
    });
    // Overlay needs DOM on; DOM is never turned off (the agent uses it).
    expect(sentTo('DOM.enable').length).toBeGreaterThanOrEqual(1);
    expect(sentTo('DOM.disable')).toHaveLength(0);
    expect(focusCalls()).toHaveLength(0);
    // Read-only: nothing typed.
    expect(sentTo('Input.insertText')).toHaveLength(0);
    expect(sentTo('Input.dispatchKeyEvent')).toHaveLength(0);
  });

  it('no empty required field: found false, remaining 0, no highlight', async () => {
    fake.count = 0;
    await attachedRun();
    expect(await show()).toEqual({ ok: true, found: false, remaining: 0 });
    expect(sentTo('Overlay.highlightNode')).toHaveLength(0);
    expect(sentTo('Accessibility.getPartialAXTree')).toHaveLength(0);
    expect(focusCalls()).toHaveLength(0);
  });

  it('more candidates than the cap: capped, remaining is the full count', async () => {
    fake.count = 13;
    await attachedRun();
    expect(await show()).toEqual({
      ok: true,
      found: true,
      remaining: 13,
      sensitive: false,
      capped: true,
    });
  });

  it('a first candidate with a sensitive label is reported sensitive', async () => {
    fake.label = 'Social Security Number';
    await attachedRun();
    expect(await show()).toMatchObject({ found: true, sensitive: true });
    // Still highlighted: it is the caseworker's own page.
    expect(sentTo('Overlay.highlightNode')).toHaveLength(1);
  });

  it('a checkbox with a sensitive label is sensitive (masksFilledField says no by role)', async () => {
    fake.role = 'checkbox';
    fake.label = 'Social Security Number';
    await attachedRun();
    expect(await show()).toMatchObject({ found: true, sensitive: true });
  });

  it('a field flagged by its attributes alone is sensitive', async () => {
    fake.role = 'checkbox';
    fake.label = 'Number';
    fake.attributes = ['id', 'ssn', 'name', 'ssn'];
    await attachedRun();
    expect(await show()).toMatchObject({ found: true, sensitive: true });
  });

  it.each([
    ['no accessibility node', { axMissing: true }],
    ['the accessibility lookup fails', { axFails: true }],
    ['no attributes reported', { attributes: undefined }],
  ])(
    'a classification failure counts as sensitive: %s',
    async (_what, knobs) => {
      Object.assign(fake, knobs);
      await attachedRun();
      expect(await show()).toMatchObject({ found: true, sensitive: true });
    },
  );

  it('classifies only the highlighted candidate; the others never delay it', async () => {
    // Any lookup of a later candidate would hang (see the fake).
    fake.count = 13;
    await attachedRun();
    expect(await show()).toEqual({
      ok: true,
      found: true,
      remaining: 13,
      sensitive: false,
      capped: true,
    });
    const looked = sentTo('Accessibility.getPartialAXTree').map(
      ([, , p]) => p?.backendNodeId,
    );
    expect(looked).toEqual([FIELD]);
    const fetched = sentTo('Runtime.callFunctionOn')
      .filter(([, , p]) => String(p?.functionDeclaration).includes('this.els['))
      .map(([, , p]) => (p?.arguments as Array<{ value: number }>)[0].value);
    expect(fetched).toEqual([0]);
    expect(sentTo('Overlay.highlightNode')).toHaveLength(1);
  });

  it('a first candidate whose lookup hangs is sensitive and still highlighted', async () => {
    await attachedRun();
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'Date'] });
    fake.axGate = new Promise(() => {});
    const axStarted = deferred<void>();
    fake.axStarted = () => axStarted.resolve();
    const scanning = show();
    await axStarted.promise;
    await vi.advanceTimersByTimeAsync(5_100);
    expect(await scanning).toEqual({
      ok: true,
      found: true,
      remaining: 3,
      sensitive: true,
    });
    expect(sentTo('Overlay.highlightNode')).toHaveLength(1);
  });

  it('the reply carries counts and flags only, never a value, name, label or page text', async () => {
    fake.label = `Applicant ${PAGE_TEXT}`;
    fake.attributes = [
      'id',
      SECRET_NAME,
      'name',
      SECRET_NAME,
      'value',
      PAGE_TEXT,
    ];
    fake.pageText = PAGE_TEXT;
    await attachedRun();
    const reply = await show();
    expect(Object.keys(reply).sort()).toEqual(
      ['found', 'ok', 'remaining', 'sensitive'].sort(),
    );
    const text = JSON.stringify(reply);
    for (const leak of [PAGE_TEXT, SECRET_NAME, 'Applicant', 'Middle']) {
      expect(text).not.toContain(leak);
    }
  });

  it('Overlay unavailable: the field is scrolled into view and focused instead', async () => {
    fake.overlayFails = true;
    await attachedRun();
    expect(await show()).toEqual({
      ok: true,
      found: true,
      remaining: 3,
      sensitive: false,
    });
    expect(sentTo('Overlay.highlightNode')).toHaveLength(0);
    expect(focusCalls()).toHaveLength(1);
    expect(focusCalls()[0][2]).toMatchObject({ objectId: CANDIDATE });
  });

  it('a highlight that fails after Overlay.enable turns Overlay off, then scrolls and focuses', async () => {
    fake.highlightFails = true;
    await attachedRun();
    expect(await show()).toMatchObject({ ok: true, found: true });
    expect(sentTo('Overlay.disable')).toHaveLength(1);
    const order = sendCommand.mock.calls.map(([, m, p]) =>
      m === 'Runtime.callFunctionOn' &&
      String(p?.functionDeclaration).includes('focus(')
        ? 'focus'
        : m,
    );
    expect(order.indexOf('Overlay.disable')).toBeLessThan(
      order.indexOf('focus'),
    );
    expect(focusCalls()).toHaveLength(1);
  });

  it('Overlay events are never forwarded to the agent', async () => {
    const socket = await attachedRun();
    await show();
    const before = socket.sent.length;
    for (const method of [
      'Overlay.nodeHighlightRequested',
      'Overlay.inspectNodeRequested',
      'Overlay.screenshotRequested',
      'Overlay.inspectModeCanceled',
    ]) {
      bridge.onDebuggerEvent({ tabId: 1 }, method, { backendNodeId: FIELD });
    }
    expect(socket.sent.slice(before)).toEqual([]);
  });
});

describe('showUnfilledField refusals make no CDP call', () => {
  /** Refused with `error` and nothing sent to the page for it. */
  const refusedWithoutCdp = async (error: string, runId = 'run-1') => {
    const before = sendCommand.mock.calls.length;
    const attaches = attach.mock.calls.length;
    const reply = await show(runId);
    expect(reply).toEqual({ ok: false, error });
    expect(sendCommand.mock.calls.length).toBe(before);
    expect(attach.mock.calls.length).toBe(attaches);
  };

  it('no active run', async () => {
    await refusedWithoutCdp('no active run');
  });

  it('wrong run', async () => {
    await attachedRun();
    await refusedWithoutCdp('wrong run', 'run-2');
  });

  it('registration pending: a run whose known values were lost', async () => {
    // A startRun for a run id this worker already began, with no live run
    // of that id, needs re-registration.
    await startedRun('run-1');
    await bridge.handleMessage({ type: 'stopRun' });
    await startedRun('run-1');
    expect((await status()).needsRegistration).toBe(true);
    await refusedWithoutCdp('registration pending');
    expect(
      await bridge.handleMessage({
        type: 'registrationComplete',
        runId: 'run-1',
        count: 0,
      }),
    ).toEqual({ ok: true });
    expect(await show()).toMatchObject({ ok: true, found: true });
  });

  it('registration pending even with the agent attached', async () => {
    await startedRun('run-1');
    await bridge.handleMessage({ type: 'stopRun' });
    const socket = await startedRun('run-1');
    expect(
      (await ask(socket, { id: 'a', type: 'attach', targetId: 'T' })).error,
    ).toBeUndefined();
    await refusedWithoutCdp('registration pending');
  });

  it('run is held', async () => {
    await attachedRun();
    bridge.onDebuggerEvent({ tabId: 1 }, 'Page.frameNavigated', {
      frame: { id: 'T', url: AWAY },
    });
    expect((await status()) as object).toMatchObject({
      pendingOrigin: 'https://away.example',
    });
    await refusedWithoutCdp('run is held');
  });

  it('scan failed: one at a time, a second call while one scans is refused', async () => {
    await attachedRun();
    const gate = deferred();
    const axStarted = deferred<void>();
    fake.axGate = gate.promise;
    fake.axStarted = () => axStarted.resolve();
    const first = show();
    await axStarted.promise;
    await refusedWithoutCdp('scan failed');
    // Nor typing meanwhile (the scan may let go of its attach).
    expect(
      await bridge.handleMessage({
        type: 'typeSensitive',
        runId: 'run-1',
        value: PAGE_TEXT,
      }),
    ).toEqual({ ok: false, error: 'type failed' });
    gate.resolve(undefined);
    expect(await first).toMatchObject({ ok: true, found: true });
  });

  it('not attached: the attach for the scan fails', async () => {
    await startedRun();
    fake.attachFails = true;
    expect(await show()).toEqual({ ok: false, error: 'not attached' });
    expect(attach).toHaveBeenCalledTimes(1);
    expect(sentTo('Runtime.evaluate')).toHaveLength(0);
  });

  it('field out of scope: the candidate is in a frame of another origin', async () => {
    fake.tabFrames = [{ id: 'S', url: `${OTHER}/x` }];
    fake.fieldContext = 9;
    await attachedRun();
    expect(await show()).toEqual({ ok: false, error: 'field out of scope' });
    expect(sentTo('Overlay.highlightNode')).toHaveLength(0);
    expect(focusCalls()).toHaveLength(0);
    expect(sentTo('Accessibility.getPartialAXTree')).toHaveLength(0);
  });

  it('a hold that starts during the scan refuses with run is held, no highlight', async () => {
    await attachedRun();
    const gate = deferred();
    const axStarted = deferred<void>();
    fake.axGate = gate.promise;
    fake.axStarted = () => axStarted.resolve();
    const scanning = show();
    await axStarted.promise;
    bridge.onDebuggerEvent({ tabId: 1 }, 'Page.frameNavigated', {
      frame: { id: 'T', url: AWAY },
    });
    gate.resolve(undefined);
    expect(await scanning).toEqual({ ok: false, error: 'run is held' });
    expect(sentTo('Overlay.highlightNode')).toHaveLength(0);
  });

  it('Stop during the scan: scan failed, no highlight', async () => {
    await attachedRun();
    const gate = deferred();
    const axStarted = deferred<void>();
    fake.axGate = gate.promise;
    fake.axStarted = () => axStarted.resolve();
    const scanning = show();
    await axStarted.promise;
    const stopped = bridge.handleMessage({ type: 'stopRun', runId: 'run-1' });
    gate.resolve(undefined);
    expect(await scanning).toEqual({ ok: false, error: 'scan failed' });
    await stopped;
    expect(sentTo('Overlay.highlightNode')).toHaveLength(0);
  });
});

describe('showUnfilledField timing and the highlight lifetime', () => {
  it('a screenshot in flight delays the scan until it lets go of the tab', async () => {
    const socket = await attachedRun();
    const gate = deferred();
    const maskStarted = deferred<void>();
    fake.maskGate = gate.promise;
    fake.maskStarted = () => maskStarted.resolve();
    const shot = ask(socket, tabCommand('s1', 'Page.captureScreenshot'));
    await maskStarted.promise;
    const scanning = show();
    await new Promise((r) => setTimeout(r, 100));
    expect(
      sentTo('Runtime.evaluate').filter(
        ([, , p]) => p?.contextId === FOCUS_CONTEXT,
      ),
    ).toHaveLength(0);
    gate.resolve(undefined);
    expect((await shot).error).toBeDefined();
    expect(await scanning).toMatchObject({ ok: true, found: true });
  });

  it('the highlight is hidden after about 6 s', async () => {
    await attachedRun();
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    expect(await show()).toMatchObject({ found: true });
    await vi.advanceTimersByTimeAsync(5_800);
    expect(sentTo('Overlay.hideHighlight')).toHaveLength(0);
    await vi.advanceTimersByTimeAsync(400);
    expect(sentTo('Overlay.hideHighlight')).toHaveLength(1);
    expect(sentTo('Overlay.disable')).toHaveLength(1);
  });

  it('a main-frame navigation hides the highlight at once', async () => {
    await attachedRun();
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    expect(await show()).toMatchObject({ found: true });
    bridge.onDebuggerEvent({ tabId: 1 }, 'Page.frameNavigated', {
      frame: { id: 'T', url: `${ORIGIN}/next` },
    });
    await vi.advanceTimersByTimeAsync(0);
    expect(sentTo('Overlay.hideHighlight')).toHaveLength(1);
    // The timer is gone: nothing more later.
    await vi.advanceTimersByTimeAsync(7_000);
    expect(sentTo('Overlay.hideHighlight')).toHaveLength(1);
  });

  it('a same-document navigation of the main frame hides it too; a child frame does not', async () => {
    await attachedRun();
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    expect(await show()).toMatchObject({ found: true });
    bridge.onDebuggerEvent({ tabId: 1 }, 'Page.frameNavigated', {
      frame: { id: 'C', parentId: 'T', url: `${ORIGIN}/inner` },
    });
    await vi.advanceTimersByTimeAsync(0);
    expect(sentTo('Overlay.hideHighlight')).toHaveLength(0);
    bridge.onDebuggerEvent({ tabId: 1 }, 'Page.navigatedWithinDocument', {
      frameId: 'T',
      url: `${ORIGIN}/form#step2`,
    });
    await vi.advanceTimersByTimeAsync(0);
    expect(sentTo('Overlay.hideHighlight')).toHaveLength(1);
  });

  it('Stop ends the highlight with the detach: no hide is sent later', async () => {
    await attachedRun();
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    expect(await show()).toMatchObject({ found: true });
    const stopped = bridge.handleMessage({ type: 'stopRun', runId: 'run-1' });
    await vi.advanceTimersByTimeAsync(7_000);
    await stopped;
    expect(detach).toHaveBeenCalled();
    expect(sentTo('Overlay.hideHighlight')).toHaveLength(0);
  });

  it('a debugger detach ends the highlight: no hide is sent later', async () => {
    await attachedRun();
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    expect(await show()).toMatchObject({ found: true });
    bridge.onDebuggerDetach({ tabId: 1 }, 'target_closed');
    await vi.advanceTimersByTimeAsync(4_000);
    expect(sentTo('Overlay.hideHighlight')).toHaveLength(0);
  });

  it('a new call hides the last highlight first and draws one new one', async () => {
    await attachedRun();
    expect(await show()).toMatchObject({ found: true });
    expect(sentTo('Overlay.hideHighlight')).toHaveLength(0);
    expect(await show()).toMatchObject({ found: true });
    expect(sentTo('Overlay.hideHighlight')).toHaveLength(1);
    expect(sentTo('Overlay.highlightNode')).toHaveLength(2);
    const order = sendCommand.mock.calls.map(([, m]) => m);
    expect(order.lastIndexOf('Overlay.hideHighlight')).toBeLessThan(
      order.lastIndexOf('Overlay.highlightNode'),
    );
  });
});

describe('showUnfilledField attaches on demand (the agent is not attached)', () => {
  it('after a same-run refresh: scrolls and focuses, lets go, never sets attached', async () => {
    await attachedRun();
    const socket = await startedRun();
    attach.mockClear();
    detach.mockClear();
    sendCommand.mockClear();
    expect(await show()).toEqual({
      ok: true,
      found: true,
      remaining: 3,
      sensitive: false,
    });
    expect(attach).toHaveBeenCalledTimes(1);
    // A detach takes an Overlay highlight with it: the fallback is used.
    expect(sentTo('Overlay.enable')).toHaveLength(0);
    expect(sentTo('Overlay.highlightNode')).toHaveLength(0);
    expect(focusCalls()).toHaveLength(1);
    // Let go before answering.
    expect(detach.mock.invocationCallOrder.at(-1)).toBeGreaterThan(
      attach.mock.invocationCallOrder[0],
    );
    expect((await status()).attached).toBe(false);
    const evaluated = await ask(
      socket,
      tabCommand('c1', 'Runtime.evaluate', { expression: 'x' }),
    );
    expect(evaluated.error).toBe('not attached');
  });
});
