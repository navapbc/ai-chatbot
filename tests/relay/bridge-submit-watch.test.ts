import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

// The worker's submit watch (armSubmitWatch / disarmSubmitWatch) against a
// fake chrome.debugger and relay socket: while armed, a main-frame form
// submission request followed by a main-frame navigation in scope sets
// `submitted` in the run status; nothing else does. When the agent is not
// attached the watch attaches for itself (`reviewAttached`), never setting
// the agent-facing `attached`. Each test loads a fresh bridge module.

const ORIGIN = 'https://a.gov';
const AWAY = 'https://away.example/next';
const FORM = `${ORIGIN}/form`;
const DONE = `${ORIGIN}/done`;

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
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
};

interface Fake {
  tabUrl: string;
  attachFails: boolean;
  /** Holds the next chrome.debugger.attach until it settles (once). */
  attachGate?: Promise<unknown>;
}

let fake: Fake;

const attach = vi.fn(async () => {
  const gate = fake.attachGate;
  fake.attachGate = undefined;
  if (gate) await gate;
  if (fake.attachFails) {
    throw new Error('Another debugger is already attached to the tab');
  }
});
const detach = vi.fn(async () => {});
const sendCommand = vi.fn(
  async (
    _debuggee: { tabId: number; sessionId?: string },
    method: string,
  ): Promise<unknown> => {
    if (method === 'Page.getFrameTree') {
      return { frameTree: { frame: { id: 'T', url: FORM } } };
    }
    if (method === 'Runtime.evaluate') {
      return { result: { type: 'string', value: 'page text' } };
    }
    return {};
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

beforeEach(async () => {
  vi.resetModules();
  fake = { tabUrl: FORM, attachFails: false };
  sendCommand.mockClear();
  attach.mockClear();
  detach.mockClear();
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
  bridge = await import('@/extension/src/bridge');
});

afterEach(async () => {
  vi.useRealTimers();
  await bridge.handleMessage({ type: 'stopRun' });
  vi.unstubAllGlobals();
});

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

async function agentAttach(socket: FakeSocket, id = 'a') {
  expect(
    (await ask(socket, { id, type: 'attach', targetId: 'T' })).error,
  ).toBeUndefined();
}

async function attachedRun(runId = 'run-1') {
  const socket = await startedRun(runId);
  await agentAttach(socket);
  return socket;
}

const arm = (runId = 'run-1') =>
  bridge.handleMessage({ type: 'armSubmitWatch', runId });
const disarm = (runId = 'run-1') =>
  bridge.handleMessage({ type: 'disarmSubmitWatch', runId });

type Status = {
  active: boolean;
  attached?: boolean;
  pendingOrigin?: string;
  needsRegistration?: boolean;
  submitted?: { at: number };
};
const status = async () =>
  (await bridge.handleMessage({ type: 'getRunStatus' })) as Status;

const event = (
  method: string,
  params: Record<string, unknown>,
  sessionId?: string,
) =>
  bridge.onDebuggerEvent(
    sessionId === undefined ? { tabId: 1 } : { tabId: 1, sessionId },
    method,
    params,
  );

/** A main-frame navigation request, as Chrome sends it. */
const requested = (
  reason: string,
  url = DONE,
  extra: Record<string, unknown> = {},
) =>
  event('Page.frameRequestedNavigation', {
    frameId: 'T',
    reason,
    url,
    disposition: 'currentTab',
    ...extra,
  });

/** A main-frame commit. */
const committed = (url = DONE, extra: Record<string, unknown> = {}) =>
  event('Page.frameNavigated', {
    frame: { id: 'T', loaderId: 'L', url, ...extra },
    type: 'Navigation',
  });

const submitPost = (url = DONE) => {
  requested('formSubmissionPost', url);
  committed(url);
};

const tabCommand = (id: string, method: string, params: object = {}) => ({
  id,
  type: 'cdp',
  sessionId: 'tab-1',
  method,
  params,
});

describe('the submit watch reports a form submission while armed', () => {
  it('a POST request then a main-frame navigation sets submitted: a time, nothing else', async () => {
    await attachedRun();
    expect(await arm()).toEqual({ ok: true });
    const before = Date.now();
    requested('formSubmissionPost');
    expect((await status()).submitted).toBeUndefined();
    committed();
    const seen = await status();
    expect(Object.keys(seen.submitted ?? {})).toEqual(['at']);
    expect(seen.submitted?.at).toBeGreaterThanOrEqual(before);
    expect(JSON.stringify(seen)).not.toContain('/done');
  });

  it('a GET form submission to another URL does the same', async () => {
    await attachedRun();
    await arm();
    requested('formSubmissionGet', `${FORM}?q=1`);
    committed(`${FORM}?q=1`);
    expect((await status()).submitted).toBeDefined();
  });

  it('a POST answered on the same URL counts; a GET to the same URL does not', async () => {
    await attachedRun();
    await arm();
    requested('formSubmissionGet', FORM);
    committed(FORM);
    expect((await status()).submitted).toBeUndefined();
    requested('formSubmissionPost', FORM);
    committed(FORM);
    expect((await status()).submitted).toBeDefined();
  });

  it('only the first submission is kept', async () => {
    await attachedRun();
    await arm();
    submitPost();
    const first = (await status()).submitted?.at;
    await new Promise((r) => setTimeout(r, 5));
    submitPost(`${ORIGIN}/again`);
    expect((await status()).submitted?.at).toBe(first);
  });

  it('works on a run that needs registration: no page text is read', async () => {
    await startedRun('run-1');
    await bridge.handleMessage({ type: 'stopRun' });
    const socket = await startedRun('run-1');
    await agentAttach(socket);
    expect((await status()).needsRegistration).toBe(true);
    const sends = sendCommand.mock.calls.length;
    expect(await arm()).toEqual({ ok: true });
    expect(sendCommand.mock.calls.length).toBe(sends);
    submitPost();
    expect((await status()).submitted).toBeDefined();
  });
});

describe('what does not count as a submission', () => {
  it('not armed: nothing is recorded, and arming later reports nothing', async () => {
    await attachedRun();
    submitPost();
    expect(await status()).not.toHaveProperty('submitted');
    requested('formSubmissionPost');
    expect(await arm()).toEqual({ ok: true });
    committed();
    expect((await status()).submitted).toBeUndefined();
  });

  it('a navigation with no submission request before it', async () => {
    await attachedRun();
    await arm();
    committed();
    expect((await status()).submitted).toBeUndefined();
  });

  it.each([
    'anchorClick',
    'scriptInitiated',
    'reload',
    'metaTagRefresh',
    'httpHeaderRefresh',
  ])('a request with reason %s', async (reason) => {
    await attachedRun();
    await arm();
    requested(reason);
    committed();
    expect((await status()).submitted).toBeUndefined();
  });

  it('a form request replaced by another kind of request before the commit', async () => {
    await attachedRun();
    await arm();
    requested('formSubmissionPost');
    requested('anchorClick', `${ORIGIN}/elsewhere`);
    committed(`${ORIGIN}/elsewhere`);
    expect((await status()).submitted).toBeUndefined();
  });

  it('a form submission inside an iframe, then a main-frame navigation', async () => {
    await attachedRun();
    await arm();
    event('Page.frameRequestedNavigation', {
      frameId: 'C',
      reason: 'formSubmissionPost',
      url: `${ORIGIN}/inner-done`,
      disposition: 'currentTab',
    });
    committed();
    expect((await status()).submitted).toBeUndefined();
  });

  it('a main-frame form request followed by an iframe navigation', async () => {
    await attachedRun();
    await arm();
    requested('formSubmissionPost');
    event('Page.frameNavigated', {
      frame: { id: 'C', parentId: 'T', url: `${ORIGIN}/inner-done` },
      type: 'Navigation',
    });
    expect((await status()).submitted).toBeUndefined();
    // A frame without a parent is still not the main frame unless it is it.
    event('Page.frameNavigated', {
      frame: { id: 'X', url: DONE },
      type: 'Navigation',
    });
    expect((await status()).submitted).toBeUndefined();
  });

  it('a child session (an out-of-process frame) requesting and navigating', async () => {
    await attachedRun();
    await arm();
    event(
      'Page.frameRequestedNavigation',
      {
        frameId: 'T',
        reason: 'formSubmissionPost',
        url: DONE,
        disposition: 'currentTab',
      },
      'child-1',
    );
    event(
      'Page.frameNavigated',
      { frame: { id: 'T', url: DONE }, type: 'Navigation' },
      'child-1',
    );
    expect((await status()).submitted).toBeUndefined();
  });

  it('a same-document navigation after a form request', async () => {
    await attachedRun();
    await arm();
    requested('formSubmissionGet', `${FORM}#sent`);
    event('Page.navigatedWithinDocument', {
      frameId: 'T',
      url: `${FORM}#sent`,
    });
    expect((await status()).submitted).toBeUndefined();
  });

  it('a submission opening another tab or window', async () => {
    await attachedRun();
    await arm();
    requested('formSubmissionPost', DONE, { disposition: 'newTab' });
    committed();
    expect((await status()).submitted).toBeUndefined();
  });

  it('a javascript: form action', async () => {
    await attachedRun();
    await arm();
    requested('formSubmissionPost', 'javascript:void(0)');
    committed(FORM);
    expect((await status()).submitted).toBeUndefined();
  });

  it('a request whose load stops before any commit (a 204), then a reload', async () => {
    await attachedRun();
    await arm();
    requested('formSubmissionPost', `${ORIGIN}/no-content`);
    event('Page.frameStoppedLoading', { frameId: 'T' });
    committed(FORM);
    expect((await status()).submitted).toBeUndefined();
  });

  it('a back/forward cache restore or an error page', async () => {
    await attachedRun();
    await arm();
    requested('formSubmissionPost');
    event('Page.frameNavigated', {
      frame: { id: 'T', url: DONE },
      type: 'BackForwardCacheRestore',
    });
    requested('formSubmissionPost');
    committed('chrome-error://chromewebdata/', { unreachableUrl: DONE });
    requested('formSubmissionPost');
    committed(DONE, { unreachableUrl: DONE });
    expect((await status()).submitted).toBeUndefined();
  });

  it('a navigation more than 30 s after the request', async () => {
    await attachedRun();
    await arm();
    vi.useFakeTimers({ toFake: ['Date'] });
    requested('formSubmissionPost');
    vi.setSystemTime(Date.now() + 30_500);
    committed();
    expect((await status()).submitted).toBeUndefined();
    // Within the window it counts.
    requested('formSubmissionPost');
    vi.setSystemTime(Date.now() + 29_000);
    committed();
    expect((await status()).submitted).toBeDefined();
  });

  it('a navigation out of scope holds the run as before and sets nothing', async () => {
    await attachedRun();
    await arm();
    requested('formSubmissionPost', AWAY);
    committed(AWAY);
    const held = await status();
    expect(held.pendingOrigin).toBe('https://away.example');
    expect(held.submitted).toBeUndefined();
    // Back in scope with no new request: still nothing.
    committed(DONE);
    expect((await status()).pendingOrigin).toBeUndefined();
    expect((await status()).submitted).toBeUndefined();
  });

  it('a request made while held, landing back in scope, is not a submission', async () => {
    await attachedRun();
    await arm();
    committed(AWAY);
    expect((await status()).pendingOrigin).toBe('https://away.example');
    requested('formSubmissionPost', DONE);
    committed(DONE);
    expect((await status()).submitted).toBeUndefined();
  });
});

describe('arming and disarming', () => {
  it('refuses with no active run, a wrong run, or a held run, with no CDP call', async () => {
    expect(await arm()).toEqual({ ok: false, error: 'no active run' });
    expect(await disarm()).toEqual({ ok: false, error: 'no active run' });
    await startedRun();
    expect(await arm('run-2')).toEqual({ ok: false, error: 'wrong run' });
    expect(await disarm('run-2')).toEqual({ ok: false, error: 'wrong run' });
    fake.tabUrl = AWAY;
    bridge.onTabUpdated(1, { url: AWAY });
    expect((await status()).pendingOrigin).toBe('https://away.example');
    const sends = sendCommand.mock.calls.length;
    expect(await arm()).toEqual({ ok: false, error: 'run is held' });
    expect(attach).not.toHaveBeenCalled();
    expect(sendCommand.mock.calls.length).toBe(sends);
  });

  it('disarm clears submitted; arming again does not bring it back', async () => {
    await attachedRun();
    await arm();
    submitPost();
    expect((await status()).submitted).toBeDefined();
    expect(await disarm()).toEqual({ ok: true });
    expect(await status()).not.toHaveProperty('submitted');
    expect(await arm()).toEqual({ ok: true });
    expect((await status()).submitted).toBeUndefined();
  });

  it('disarm clears a pending request too', async () => {
    await attachedRun();
    await arm();
    requested('formSubmissionPost');
    await disarm();
    await arm();
    committed();
    expect((await status()).submitted).toBeUndefined();
  });

  it('arming twice is idempotent and keeps what was seen', async () => {
    await attachedRun();
    expect(await arm()).toEqual({ ok: true });
    submitPost();
    const at = (await status()).submitted?.at;
    expect(await arm()).toEqual({ ok: true });
    expect((await status()).submitted?.at).toBe(at);
  });

  it('under the agent attach: no attach of its own; disarm leaves the agent attached', async () => {
    const socket = await attachedRun();
    const attaches = attach.mock.calls.length;
    const detaches = detach.mock.calls.length;
    expect(await arm()).toEqual({ ok: true });
    expect(await disarm()).toEqual({ ok: true });
    expect(attach.mock.calls.length).toBe(attaches);
    expect(detach.mock.calls.length).toBe(detaches);
    expect((await status()).attached).toBe(true);
    const reply = await ask(
      socket,
      tabCommand('c1', 'Runtime.evaluate', { expression: 'x' }),
    );
    expect(reply.error).toBeUndefined();
  });

  it('a token refresh loses the watch: the new run is not armed', async () => {
    await attachedRun();
    await arm();
    submitPost();
    await startedRun();
    expect(await status()).not.toHaveProperty('submitted');
    submitPost();
    expect(await status()).not.toHaveProperty('submitted');
  });

  it('stop ends the watch', async () => {
    await startedRun();
    await arm();
    await bridge.handleMessage({ type: 'stopRun', runId: 'run-1' });
    expect(await status()).toEqual({ ok: true, active: false });
    expect(detach).toHaveBeenCalled();
  });
});

describe('the watch attaches on demand when the agent is not attached', () => {
  it('attaches once, never sets attached, and an agent command still gets not attached', async () => {
    const socket = await startedRun();
    expect(await arm()).toEqual({ ok: true });
    expect(attach).toHaveBeenCalledTimes(1);
    expect(await status()).toMatchObject({ active: true, attached: false });
    const refused = await ask(
      socket,
      tabCommand('c1', 'Runtime.evaluate', { expression: 'x' }),
    );
    expect(refused.error).toBe('not attached');
    // The agent's listTargets says the same.
    const targets = await ask(socket, { id: 't1', type: 'listTargets' });
    expect(JSON.stringify(targets.result)).toContain('"attached":false');
    // Arming again neither attaches nor detaches.
    const detaches = detach.mock.calls.length;
    expect(await arm()).toEqual({ ok: true });
    expect(attach).toHaveBeenCalledTimes(1);
    expect(detach.mock.calls.length).toBe(detaches);
    // The watch works under its own attach.
    submitPost();
    expect((await status()).submitted).toBeDefined();
  });

  it('events under the review attach are not forwarded', async () => {
    const socket = await startedRun();
    await arm();
    const before = socket.sent.length;
    submitPost();
    event('Runtime.consoleAPICalled', { type: 'log', args: [] });
    expect(socket.sent.slice(before)).toEqual([]);
  });

  it('a navigation out of scope under the review attach holds the run', async () => {
    await startedRun();
    await arm();
    requested('formSubmissionPost', AWAY);
    committed(AWAY);
    expect((await status()).pendingOrigin).toBe('https://away.example');
    expect((await status()).submitted).toBeUndefined();
  });

  it('a failed attach answers not attached and leaves the watch unarmed', async () => {
    await startedRun();
    fake.attachFails = true;
    expect(await arm()).toEqual({ ok: false, error: 'not attached' });
    expect((await status()).attached).toBe(false);
    // Had it been armed, these would count.
    submitPost();
    expect(await status()).not.toHaveProperty('submitted');
  });

  it('disarm lets go of the review attach', async () => {
    await startedRun();
    await arm();
    const detaches = detach.mock.calls.length;
    expect(await disarm()).toEqual({ ok: true });
    expect(detach.mock.calls.length).toBe(detaches + 1);
    // Arming again attaches again.
    expect(await arm()).toEqual({ ok: true });
    expect(attach).toHaveBeenCalledTimes(2);
  });

  it('a disarm while the arm is attaching ends disarmed and let go', async () => {
    await startedRun();
    const gate = deferred();
    fake.attachGate = gate.promise;
    const arming = arm();
    await new Promise((r) => setTimeout(r, 0));
    expect(attach).toHaveBeenCalledTimes(1);
    const disarming = disarm();
    gate.resolve(undefined);
    expect(await arming).toEqual({ ok: false, error: 'not attached' });
    expect(await disarming).toEqual({ ok: true });
    // Let go after the attach landed.
    expect(detach.mock.invocationCallOrder.at(-1)).toBeGreaterThan(
      attach.mock.invocationCallOrder[0],
    );
    submitPost();
    expect(await status()).not.toHaveProperty('submitted');
  });

  it('a later agent attach takes over: the watch goes on, disarm does not detach the agent', async () => {
    const socket = await startedRun();
    await arm();
    await agentAttach(socket);
    expect((await status()).attached).toBe(true);
    submitPost();
    expect((await status()).submitted).toBeDefined();
    const detaches = detach.mock.calls.length;
    expect(await disarm()).toEqual({ ok: true });
    expect(detach.mock.calls.length).toBe(detaches);
    const reply = await ask(
      socket,
      tabCommand('c1', 'Runtime.evaluate', { expression: 'x' }),
    );
    expect(reply.error).toBeUndefined();
  });

  it('a debugger detach ends the watch; arming again uses the agent attach once it returns', async () => {
    const socket = await startedRun();
    await arm();
    submitPost();
    expect((await status()).submitted).toBeDefined();
    bridge.onDebuggerDetach({ tabId: 1 }, 'target_closed');
    expect(await status()).not.toHaveProperty('submitted');
    // Not armed any more: nothing counts.
    submitPost();
    expect(await status()).not.toHaveProperty('submitted');
    // During the detach grace the watch does not attach for itself.
    const attaches = attach.mock.calls.length;
    expect(await arm()).toEqual({ ok: false, error: 'not attached' });
    expect(attach.mock.calls.length).toBe(attaches);
    // The agent re-attaches within the grace: arming uses its attach.
    await agentAttach(socket, 'a2');
    expect(await arm()).toEqual({ ok: true });
    submitPost();
    expect((await status()).submitted).toBeDefined();
  });

  it('a debugger detach under the agent attach ends the watch too', async () => {
    const socket = await attachedRun();
    await arm();
    bridge.onDebuggerDetach({ tabId: 1 }, 'target_closed');
    await agentAttach(socket, 'a2');
    submitPost();
    expect(await status()).not.toHaveProperty('submitted');
  });
});
