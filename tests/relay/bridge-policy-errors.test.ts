import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

// Chrome 155 policy errors (Host access / Screenshot capture is restricted by
// policy) on attach, a CDP send, a screenshot and the worker-initiated paths:
// each becomes one fixed refusal and Chrome's text goes nowhere.
// The harness below is the typeSensitive test's.
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
  /** When set, chrome.debugger.attach rejects with this message. */
  attachError?: string;
  /** When set, these CDP methods reject with this message. */
  methodError?: { methods: string[]; message: string };
  /** Holds the next chrome.debugger.attach until it settles (once). */
  attachGate?: Promise<unknown>;
  /** Holds the next Page.getFrameTree until it settles (once). */
  frameTreeGate?: Promise<unknown>;
}

let fake: Fake;
/** Set once the mask's world creation has failed (the restore phase). */
let worldFailed = false;

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
  const gate = fake.attachGate;
  fake.attachGate = undefined;
  if (gate) await gate;
  if (fake.attachError) throw new Error(fake.attachError);
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
    if (
      fake.methodError &&
      (fake.methodError.methods.includes(method) ||
        (fake.methodError.methods.includes('*') && worldFailed))
    ) {
      throw new Error(fake.methodError.message);
    }
    switch (method) {
      case 'Page.getFrameTree': {
        const gate = fake.frameTreeGate;
        fake.frameTreeGate = undefined;
        if (gate) await gate;
        return {
          frameTree: {
            frame: { id: 'T', url: `${ORIGIN}/form` },
            childFrames: fake.tabFrames.map((f) => ({
              frame: { ...f, parentId: 'T' },
            })),
          },
        };
      }
      case 'Page.createIsolatedWorld':
        if (params.worldName === 'relay-focus') {
          return { executionContextId: FOCUS_CONTEXT };
        }
        fake.maskStarted?.();
        if (fake.maskGate) await fake.maskGate;
        worldFailed = true;
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
  worldFailed = false;
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

const HOST = 'Host access is restricted by policy';
const SHOT = 'Screenshot capture is restricted by policy';
const CHROME_TEXT = /restricted by policy|Host access|Screenshot capture/;

const status = () => bridge.handleMessage({ type: 'getRunStatus' });

describe('isPolicyRestriction', () => {
  it('matches only the two Chrome phrases, case-sensitively', () => {
    expect(bridge.isPolicyRestriction(new Error(HOST))).toBe(true);
    expect(bridge.isPolicyRestriction(new Error(`x: ${SHOT}.`))).toBe(true);
    expect(bridge.isPolicyRestriction(new Error(HOST.toLowerCase()))).toBe(
      false,
    );
    expect(bridge.isPolicyRestriction(new Error('Cannot attach'))).toBe(false);
    expect(bridge.isPolicyRestriction(undefined)).toBe(false);
    expect(bridge.isPolicyRestriction({ message: HOST })).toBe(true);
  });
});

describe.each([HOST, SHOT])('Chrome says: %s', (phrase) => {
  it('the agent attach answers a fixed message and flags the run', async () => {
    const log = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const err = vi.spyOn(console, 'error').mockImplementation(() => {});
    const info = vi.spyOn(console, 'log').mockImplementation(() => {});
    fake.attachError = phrase;
    const socket = await startedRun();
    const reply = await ask(socket, {
      id: 'a',
      type: 'attach',
      targetId: 'T',
    });
    expect(reply.error).toBe(bridge.POLICY_AGENT_MESSAGE);
    expect(reply.error).not.toMatch(CHROME_TEXT);
    expect(await status()).toMatchObject({
      active: true,
      policyRestricted: true,
    });
    const everything = JSON.stringify([
      socket.sent,
      await status(),
      log.mock.calls,
      err.mock.calls,
      info.mock.calls,
    ]);
    expect(everything).not.toContain(phrase);
    expect(everything).not.toMatch(/Host access|Screenshot capture/);
    log.mockRestore();
    err.mockRestore();
    info.mockRestore();
  });

  it('a CDP send rejected by policy answers the fixed message', async () => {
    const socket = await attachedRun();
    fake.methodError = { methods: ['Runtime.evaluate'], message: phrase };
    const reply = await ask(
      socket,
      tabCommand('e1', 'Runtime.evaluate', { expression: 'x' }),
    );
    expect(reply.error).toBe(bridge.POLICY_AGENT_MESSAGE);
    expect(JSON.stringify(socket.sent)).not.toContain(phrase);
    // A send-level failure does not flag the run.
    expect(
      (await status()) as { policyRestricted?: boolean },
    ).not.toHaveProperty('policyRestricted');
  });

  it('a screenshot step rejected by policy answers the fixed message', async () => {
    const socket = await attachedRun();
    fake.methodError = {
      methods: ['Page.createIsolatedWorld'],
      message: phrase,
    };
    const reply = await ask(socket, tabCommand('s1', 'Page.captureScreenshot'));
    expect(reply.error).toBe(bridge.POLICY_AGENT_MESSAGE);
    expect(reply.result).toBeUndefined();
    expect(JSON.stringify(socket.sent)).not.toContain(phrase);
  });

  it('typeSensitive answers restricted by policy when its attach is refused', async () => {
    const socket = await startedRun();
    fake.attachError = phrase;
    const reply = await type();
    expect(reply).toEqual({ ok: false, error: 'restricted by policy' });
    expect(JSON.stringify(reply)).not.toContain(phrase);
    expect(JSON.stringify(reply)).not.toContain(VALUE);
    expect(await status()).toMatchObject({ policyRestricted: true });
    expect(socket.sent.some((m) => m.type === 'reply')).toBe(false);
  });

  it('typeSensitive answers restricted by policy when the typing call is refused', async () => {
    await attachedRun();
    fake.typeInto = async () => {
      throw new Error(phrase);
    };
    expect(await type()).toEqual({ ok: false, error: 'restricted by policy' });
    expect(await status()).not.toHaveProperty('policyRestricted');
  });

  it('showUnfilledField and the submit watch answer restricted by policy', async () => {
    await startedRun();
    fake.attachError = phrase;
    expect(
      await bridge.handleMessage({ type: 'showUnfilledField', runId: 'run-1' }),
    ).toEqual({ ok: false, error: 'restricted by policy' });
    expect(
      await bridge.handleMessage({ type: 'armSubmitWatch', runId: 'run-1' }),
    ).toEqual({ ok: false, error: 'restricted by policy' });
  });
});

describe('a failing restore', () => {
  it('with the policy phrase never turns a non-policy refusal into the policy message', async () => {
    const socket = await attachedRun();
    // Apply fails (no mask world); every restore step then fails with the phrase.
    fake.methodError = { methods: ['*'], message: HOST };
    const reply = await ask(socket, tabCommand('s2', 'Page.captureScreenshot'));
    expect(reply.error).toBeDefined();
    expect(reply.error).not.toBe(bridge.POLICY_AGENT_MESSAGE);
    expect(reply.result).toBeUndefined();
  });
});

describe('policy flag and unrelated errors', () => {
  it('an unrelated attach error is not mapped and does not flag the run', async () => {
    fake.attachError = 'Another debugger is already attached to the tab';
    const socket = await startedRun();
    const reply = await ask(socket, {
      id: 'a',
      type: 'attach',
      targetId: 'T',
    });
    expect(reply.error).toBe('Another debugger is already attached to the tab');
    expect(await status()).not.toHaveProperty('policyRestricted');
    expect(await type()).toEqual({ ok: false, error: 'not attached' });
  });

  it('the flag clears on a successful attach and on a new run', async () => {
    fake.attachError = HOST;
    const socket = await startedRun();
    await ask(socket, { id: 'a', type: 'attach', targetId: 'T' });
    expect(await status()).toMatchObject({ policyRestricted: true });
    fake.attachError = undefined;
    expect(
      (await ask(socket, { id: 'b', type: 'attach', targetId: 'T' })).error,
    ).toBeUndefined();
    expect(await status()).not.toHaveProperty('policyRestricted');
    // A new run starts clean even after a flagged one.
    fake.attachError = HOST;
    await bridge.handleMessage({ type: 'stopRun' });
    const next = await startedRun('run-2');
    await ask(next, { id: 'c', type: 'attach', targetId: 'T' });
    expect(await status()).toMatchObject({ policyRestricted: true });
    await bridge.handleMessage({ type: 'stopRun' });
    fake.attachError = undefined;
    await startedRun('run-3');
    expect(await status()).not.toHaveProperty('policyRestricted');
  });
});
