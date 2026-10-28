import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { isSensitiveField } from '@/lib/relay/sensitive-fields';
import {
  CONFIRMATION_LABEL_KINDS,
  CONFIRMATION_READ,
} from '@/extension/src/typing-scripts';

// The worker's confirmation read (readConfirmation, P4c4-f) against a fake
// chrome.debugger and relay socket. After the submit watch sees a
// submission, the worker waits for the page to load and reads it up to 3
// times, 1 s apart, in its isolated world. The fake runs the real in-page
// script (CONFIRMATION_READ) against `fake.pageText`, so the script and the
// worker's vetting are tested together; `fake.answer` replaces the script's
// answer to test one worker drop at a time. Each test loads a fresh bridge.

const ORIGIN = 'https://a.gov';
const AWAY = 'https://away.example/next';
const FORM = `${ORIGIN}/form`;
const DONE = `${ORIGIN}/done`;
const SSN = '123-45-6789';

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
  /** The page's visible text (document.body.innerText). */
  pageText: string;
  /** Replaces the in-page script's answer when set (null: found nothing). */
  answer?: unknown;
  /** Holds the next confirmation evaluation until it settles (once). */
  evaluateGate?: Promise<unknown>;
  /** Confirmation evaluations so far. */
  reads: number;
}

let fake: Fake;

const isConfirmationRead = (params: unknown) =>
  typeof (params as { expression?: unknown })?.expression === 'string' &&
  (params as { expression: string }).expression.includes(CONFIRMATION_READ);

const runScript = (expression: string, text: string): unknown =>
  new Function('document', `return ${expression}`)({
    body: { innerText: text },
  });

const attach = vi.fn(async () => {});
const detach = vi.fn(async () => {});
const getTargets = vi.fn(async () => [
  { tabId: 1, type: 'page', id: 'T', title: '', url: '', attached: false },
]);
const sendCommand = vi.fn(
  async (
    _debuggee: { tabId: number; sessionId?: string },
    method: string,
    params?: Record<string, unknown>,
  ): Promise<unknown> => {
    if (method === 'Page.getFrameTree') {
      return { frameTree: { frame: { id: 'T', url: FORM } } };
    }
    if (method === 'Page.createIsolatedWorld') {
      return { executionContextId: 7 };
    }
    if (method === 'Runtime.evaluate' && isConfirmationRead(params)) {
      fake.reads += 1;
      const gate = fake.evaluateGate;
      fake.evaluateGate = undefined;
      if (gate) await gate;
      const value =
        'answer' in fake
          ? fake.answer
          : runScript(params?.expression as string, fake.pageText);
      return value === null
        ? { result: { type: 'object', subtype: 'null', value: null } }
        : { result: { type: 'object', value } };
    }
    if (method === 'Runtime.evaluate') {
      // The agent's own read of the page.
      return { result: { type: 'string', value: fake.pageText } };
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
const consoleCalls: unknown[][] = [];

beforeEach(async () => {
  vi.resetModules();
  fake = {
    tabUrl: FORM,
    pageText: 'Thank you, Jane Doe.\nConfirmation number: AB-1234\nKeep it.',
    reads: 0,
  };
  sendCommand.mockClear();
  attach.mockClear();
  detach.mockClear();
  getTargets.mockClear();
  consoleCalls.length = 0;
  for (const level of ['log', 'info', 'warn', 'error', 'debug'] as const) {
    vi.spyOn(console, level).mockImplementation((...args: unknown[]) => {
      consoleCalls.push(args);
    });
  }
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
    debugger: { getTargets, attach, detach, sendCommand },
  });
  bridge = await import('@/extension/src/bridge');
});

afterEach(async () => {
  await bridge.handleMessage({ type: 'stopRun' });
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
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

async function attachedRun(runId = 'run-1') {
  const socket = await startedRun(runId);
  expect(
    (await ask(socket, { id: 'a', type: 'attach', targetId: 'T' })).error,
  ).toBeUndefined();
  return socket;
}

const arm = (runId = 'run-1', reset = false) =>
  bridge.handleMessage({ type: 'armSubmitWatch', runId, reset });
const disarm = (runId = 'run-1') =>
  bridge.handleMessage({ type: 'disarmSubmitWatch', runId });

type Status = {
  active: boolean;
  needsRegistration?: boolean;
  pendingOrigin?: string;
  submitted?: {
    at: number;
    confirmation?: { token: string; kind: string };
  };
};
const status = async () =>
  (await bridge.handleMessage({ type: 'getRunStatus' })) as Status;

const event = (method: string, params: Record<string, unknown>) =>
  bridge.onDebuggerEvent({ tabId: 1 }, method, params);

const committed = (url = DONE) =>
  event('Page.frameNavigated', {
    frame: { id: 'T', loaderId: 'L', url },
    type: 'Navigation',
  });

/** A form POST that lands on `url`; with `load`, the page then loads. */
const submit = (url = DONE, load = true) => {
  event('Page.frameRequestedNavigation', {
    frameId: 'T',
    reason: 'formSubmissionPost',
    url,
    disposition: 'currentTab',
  });
  committed(url);
  if (load) event('Page.loadEventFired', { timestamp: 1 });
};

/** Waits until the read has made `n` page reads. */
const reads = (n: number) =>
  vi.waitFor(() => expect(fake.reads).toBe(n), { timeout: 4_000 });

/** Waits for a confirmation in the status. */
const confirmed = () =>
  vi.waitFor(
    async () => {
      const seen = await status();
      expect(seen.submitted?.confirmation).toBeDefined();
      return seen;
    },
    { timeout: 4_000 },
  );

/** Waits for the read to finish (its last read and a moment after). */
const settled = async (n = 1) => {
  await reads(n);
  await new Promise((r) => setTimeout(r, 50));
};

/** An armed run (agent attached) that saw a submission and read the page. */
async function readOnce(text: string) {
  fake.pageText = text;
  const socket = await attachedRun();
  await arm();
  submit();
  await settled();
  return { socket, seen: await status() };
}

describe('a vetted confirmation number reaches getRunStatus, and only it', () => {
  it('a labelled alphanumeric token: { token, kind: confirmation }, no other page text', async () => {
    const { socket } = await readOnce(
      'Thank you, Jane Doe.\nConfirmation number: AB-1234\nWe emailed jane@x.org.',
    );
    const seen = await confirmed();
    expect(seen.submitted).toEqual({
      at: expect.any(Number),
      confirmation: { token: 'AB-1234', kind: 'confirmation' },
    });
    const json = JSON.stringify(seen);
    expect(json).not.toContain('Jane');
    expect(json).not.toContain('jane@x.org');
    expect(json).not.toContain('Confirmation number');
    // Never sent up the relay (nor anything else from the page).
    expect(JSON.stringify(socket.sent)).not.toContain('AB-1234');
    expect(consoleCalls).toEqual([]);
  });

  it.each([
    ['Application number: 20240017', '20240017', 'other'],
    ['Reference number: REF-77', 'REF-77', 'other'],
    ['Tracking number: 1Z999', '1Z999', 'other'],
    ['Confirmation #: C-55/2024.', 'C-55/2024', 'confirmation'],
    ['Confirmation #AB12', 'AB12', 'confirmation'],
    ['Your confirmation code:\n  X9Y8Z7', 'X9Y8Z7', 'confirmation'],
    ['CONFIRMATION NUMBER  ab_12.c', 'ab_12.c', 'confirmation'],
  ])('%j gives %s (%s)', async (text, token, kind) => {
    await readOnce(text);
    expect((await confirmed()).submitted?.confirmation).toEqual({
      token,
      kind,
    });
  });

  it('a confirmation label wins over an earlier application number', async () => {
    await readOnce(
      'Application number: 20240017\nStatus: received\nConfirmation number: ZX-42',
    );
    expect((await confirmed()).submitted?.confirmation).toEqual({
      token: 'ZX-42',
      kind: 'confirmation',
    });
  });

  it('the token is not registered as a known value: the agent still reads it unchanged', async () => {
    const { socket } = await readOnce('Confirmation number: AB-1234');
    await confirmed();
    const reply = await ask(socket, {
      id: 'c1',
      type: 'cdp',
      sessionId: 'tab-1',
      method: 'Runtime.evaluate',
      params: { expression: 'document.body.innerText' },
    });
    expect(JSON.stringify(reply.result)).toContain('AB-1234');
  });
});

describe('a token is dropped (submitted stays { at })', () => {
  it.each([
    ['nine digits', 'Confirmation number: 123456789'],
    [
      'ten digits (not SSN-shaped: only the shape rule drops it)',
      'Confirmation number: 1234567890',
    ],
    ['a long digit run with separators', 'Confirmation number: 1234-5678-90'],
    ['an SSN-shaped token', `Confirmation number: ${SSN}`],
    ['over 32 characters', `Confirmation number: ${'A1'.repeat(17)}`],
    ['under 4 characters', 'Confirmation number: A12'],
    ['no digit', 'Confirmation number: PENDING'],
    ['the case number label', 'Case number: 4455'],
    [
      'a label that is not a label',
      'Confirmation numbers are emailed within 2 days. 12345',
    ],
    [
      'a label followed by a word',
      'Your confirmation number will be emailed: AB-1234',
    ],
    ['no label at all', 'Thank you. AB-1234 is on its way.'],
  ])('%s', async (_name, text) => {
    const { seen } = await readOnce(text);
    expect(seen.submitted).toEqual({ at: expect.any(Number) });
    // A found-but-dropped token ends the read: no more reads after it.
  });

  it('the hasSensitiveShape rule alone: a script answer of nine digits with nearSensitive false', async () => {
    fake.answer = {
      token: '123456789',
      label: 'confirmation number',
      nearSensitive: false,
    };
    const { seen } = await readOnce('');
    expect(seen.submitted).toEqual({ at: expect.any(Number) });
  });

  it('the nearSensitive rule alone: a clean token with nearSensitive true', async () => {
    fake.answer = {
      token: 'AB-1234',
      label: 'confirmation number',
      nearSensitive: true,
    };
    const { seen } = await readOnce('');
    expect(seen.submitted).toEqual({ at: expect.any(Number) });
  });

  it('a page echoing an SSN near the token: dropped, and the SSN goes nowhere', async () => {
    const { socket, seen } = await readOnce(
      `Applicant SSN ${SSN}\nConfirmation number: AB-1234`,
    );
    expect(seen.submitted).toEqual({ at: expect.any(Number) });
    expect(JSON.stringify(seen)).not.toContain(SSN);
    expect(JSON.stringify(socket.sent)).not.toContain(SSN);
    expect(JSON.stringify(socket.sent)).not.toContain('AB-1234');
    expect(consoleCalls).toEqual([]);
  });

  it.each([
    ['spaces', '123 45 6789'],
    ['no separators', '123456789'],
    ['after the token', 'tail'],
  ])('an SSN echo (%s) near the token drops it', async (_name, ssn) => {
    const text =
      ssn === 'tail'
        ? `Confirmation number: AB-1234\nSSN on file: ${SSN}`
        : `SSN ${ssn}\nConfirmation number: AB-1234`;
    const { seen } = await readOnce(text);
    expect(seen.submitted).toEqual({ at: expect.any(Number) });
  });

  it('an SSN far from the label (over 200 characters) does not drop it', async () => {
    await readOnce(
      `SSN ${SSN}\n${'x'.repeat(300)}\nConfirmation number: AB-1234`,
    );
    expect((await confirmed()).submitted?.confirmation?.token).toBe('AB-1234');
  });

  it('a token equal to a registered value is dropped', async () => {
    fake.pageText = 'Confirmation number: AB-1234';
    const socket = await attachedRun();
    expect(
      await bridge.handleMessage({
        type: 'addKnownValue',
        runId: 'run-1',
        value: 'AB-1234',
      }),
    ).toEqual({ ok: true });
    await arm();
    submit();
    await settled();
    expect((await status()).submitted).toEqual({ at: expect.any(Number) });
    expect(JSON.stringify(socket.sent)).not.toContain('AB-1234');
  });

  it('a label outside the fixed list from the script is dropped', async () => {
    fake.answer = { token: '4455', label: 'case number', nearSensitive: false };
    const { seen } = await readOnce('');
    expect(seen.submitted).toEqual({ at: expect.any(Number) });
  });

  it('a malformed answer is dropped', async () => {
    fake.answer = { token: 1234, label: 'confirmation number' };
    const { seen } = await readOnce('');
    expect(seen.submitted).toEqual({ at: expect.any(Number) });
  });

  it('none of the six labels is itself a sensitive label', () => {
    expect(Object.keys(CONFIRMATION_LABEL_KINDS)).toHaveLength(6);
    for (const label of Object.keys(CONFIRMATION_LABEL_KINDS)) {
      expect(isSensitiveField({ label })).toBe(false);
    }
    expect(isSensitiveField({ label: 'case number' })).toBe(true);
  });
});

describe('when and how often the page is read', () => {
  it('waits for the page to load: no read before the load event', async () => {
    await attachedRun();
    await arm();
    submit(DONE, false);
    await new Promise((r) => setTimeout(r, 300));
    expect(fake.reads).toBe(0);
    event('Page.frameStoppedLoading', { frameId: 'T' });
    await confirmed();
    expect(fake.reads).toBe(1);
  });

  it('nothing found: reads 3 times, about 1 s apart, then stops', async () => {
    fake.pageText = 'Thank you.';
    await attachedRun();
    await arm();
    const started = Date.now();
    submit();
    await reads(3);
    expect(Date.now() - started).toBeGreaterThanOrEqual(1_900);
    await new Promise((r) => setTimeout(r, 1_300));
    expect(fake.reads).toBe(3);
    expect((await status()).submitted).toEqual({ at: expect.any(Number) });
  }, 10_000);

  it('a number the page renders late is found on a later read', async () => {
    fake.pageText = 'Loading...';
    await attachedRun();
    await arm();
    submit();
    await reads(1);
    fake.pageText = 'Confirmation number: LATE-7';
    expect((await confirmed()).submitted?.confirmation?.token).toBe('LATE-7');
    expect(fake.reads).toBe(2);
  });

  it.each([
    ['kept', 'Application number: 20240017'],
    // Ten digits: dropped by its shape, yet not SSN-shaped (a bare nine
    // digit run nearby would drop the confirmation number as well).
    ['dropped (ten digits)', 'Application number: 1234567890'],
  ])(
    'an application number (%s) does not end the read: a confirmation number rendered later wins',
    async (_name, header) => {
      fake.pageText = header;
      await attachedRun();
      await arm();
      submit();
      await reads(1);
      fake.pageText = `${header}\nConfirmation number: LATE-7`;
      expect((await confirmed()).submitted?.confirmation).toEqual({
        token: 'LATE-7',
        kind: 'confirmation',
      });
      expect(fake.reads).toBe(2);
    },
  );

  it('only an application number: kept after the third read, no fourth', async () => {
    fake.pageText = 'Application number: 20240017';
    await attachedRun();
    await arm();
    submit();
    await reads(3);
    expect((await confirmed()).submitted?.confirmation).toEqual({
      token: '20240017',
      kind: 'other',
    });
    await new Promise((r) => setTimeout(r, 1_300));
    expect(fake.reads).toBe(3);
  }, 10_000);

  it('a found but dropped token ends the read', async () => {
    fake.pageText = 'Confirmation number: 123456789';
    await attachedRun();
    await arm();
    submit();
    await reads(1);
    await new Promise((r) => setTimeout(r, 1_300));
    expect(fake.reads).toBe(1);
  });

  it("works on the watch's own attach (agent not attached)", async () => {
    await startedRun();
    expect(await arm()).toEqual({ ok: true });
    submit();
    expect((await confirmed()).submitted?.confirmation?.token).toBe('AB-1234');
  });

  it('a run that needs registration never reads the page', async () => {
    await startedRun('run-1');
    await bridge.handleMessage({ type: 'stopRun' });
    await attachedRun('run-1');
    expect((await status()).needsRegistration).toBe(true);
    await arm();
    submit();
    await new Promise((r) => setTimeout(r, 300));
    expect(fake.reads).toBe(0);
    expect((await status()).submitted).toEqual({ at: expect.any(Number) });
  });

  it('a run held before the load does not read', async () => {
    await attachedRun();
    await arm();
    submit(DONE, false);
    fake.tabUrl = AWAY;
    bridge.onTabUpdated(1, { url: AWAY });
    expect((await status()).pendingOrigin).toBe('https://away.example');
    event('Page.loadEventFired', { timestamp: 2 });
    await new Promise((r) => setTimeout(r, 300));
    expect(fake.reads).toBe(0);
  });
});

describe('ending the read: nothing is set and the state is cleared', () => {
  /** Starts a read whose page read hangs until `gate` resolves. */
  async function hungRead() {
    const gate = deferred();
    fake.evaluateGate = gate.promise;
    const socket = await attachedRun();
    await arm();
    submit();
    await reads(1);
    return { gate, socket };
  }

  it('stop during the read', async () => {
    const { gate } = await hungRead();
    await bridge.handleMessage({ type: 'stopRun' });
    gate.resolve(undefined);
    await new Promise((r) => setTimeout(r, 50));
    expect(await status()).toEqual({ ok: true, active: false });
  });

  it('disarm during the read', async () => {
    const { gate } = await hungRead();
    expect(await disarm()).toEqual({ ok: true });
    gate.resolve(undefined);
    await new Promise((r) => setTimeout(r, 50));
    expect((await status()).submitted).toBeUndefined();
    await arm();
    expect((await status()).submitted).toBeUndefined();
  });

  it('a reset during the read: a later submission does not get the old token', async () => {
    const { gate } = await hungRead();
    expect(await arm('run-1', true)).toEqual({ ok: true });
    fake.answer = null;
    gate.resolve(undefined);
    await new Promise((r) => setTimeout(r, 50));
    expect((await status()).submitted).toBeUndefined();
    // A new submission is read on its own (nothing found on it).
    submit(`${ORIGIN}/again`);
    await reads(4);
    expect((await status()).submitted).toEqual({ at: expect.any(Number) });
  }, 10_000);

  it('a debugger detach during the read', async () => {
    const { gate } = await hungRead();
    bridge.onDebuggerDetach({ tabId: 1 }, 'target_closed');
    gate.resolve(undefined);
    await new Promise((r) => setTimeout(r, 50));
    expect((await status()).submitted).toBeUndefined();
  });

  it('a navigation away during the read', async () => {
    const { gate } = await hungRead();
    committed(`${ORIGIN}/elsewhere`);
    gate.resolve(undefined);
    await new Promise((r) => setTimeout(r, 1_300));
    expect((await status()).submitted).toEqual({ at: expect.any(Number) });
    expect(fake.reads).toBe(1);
  });

  it('a hold during the read', async () => {
    const { gate } = await hungRead();
    fake.tabUrl = AWAY;
    bridge.onTabUpdated(1, { url: AWAY });
    gate.resolve(undefined);
    await new Promise((r) => setTimeout(r, 50));
    const seen = await status();
    expect(seen.pendingOrigin).toBe('https://away.example');
    expect(seen.submitted?.confirmation).toBeUndefined();
  });
});

describe('the in-page script (CONFIRMATION_READ) on its own', () => {
  const read = (text: string) =>
    runScript(`(${CONFIRMATION_READ})(document)`, text);

  it('answers the token, the canonical label and nearSensitive, nothing else', () => {
    expect(read('Thanks Jane!\nConfirmation   Number :  AB-1234.')).toEqual({
      token: 'AB-1234',
      label: 'confirmation number',
      nearSensitive: false,
    });
  });

  it('answers null on no match, no body, or a throwing body', () => {
    expect(read('nothing here')).toBeNull();
    expect(
      new Function('document', `return (${CONFIRMATION_READ})(document)`)({}),
    ).toBeNull();
    expect(
      new Function('document', `return (${CONFIRMATION_READ})(document)`)({
        get body() {
          throw new Error('boom');
        },
      }),
    ).toBeNull();
  });

  it('reads only the first 20 000 characters', () => {
    expect(
      read(`${'x'.repeat(20_000)}Confirmation number: AB-1234`),
    ).toBeNull();
  });
});
