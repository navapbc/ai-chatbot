import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('server-only', () => ({}));

const authMock = vi.hoisted(() => vi.fn());
vi.mock('@/app/(auth)/auth', () => ({ auth: authMock }));

const eve = vi.hoisted(() => ({
  createEveSession: vi.fn(),
  continueEveSession: vi.fn(),
  openEveStream: vi.fn(),
  parseNdjson: vi.fn(),
}));
vi.mock('@/lib/ai/eve/eve-client', () => eve);

import { POST } from '@/app/(chat)/api/eve-chat/route';
import {
  createMemoryRunStore,
  EveRunExistsError,
  type EveRunStore,
  setRunStoreForTests,
} from '@/lib/ai/eve/run-store';

const USER = 'user-1';
const CHAT = 'chat-1';

const turn = (text = 'hello', id = CHAT) =>
  POST(
    new Request('http://localhost/api/eve-chat', {
      method: 'POST',
      body: JSON.stringify({
        id,
        message: { role: 'user', parts: [{ type: 'text', text }] },
      }),
    }),
  );

/** Reads the response to the end so the stream loop actually runs. */
const drain = async (res: Response) => {
  if (!res.body) return;
  const reader = res.body.getReader();
  while (!(await reader.read()).done);
};

/** Makes the next stream read yield these events (one array per read). */
const streamEvents = (...events: unknown[]) => {
  eve.parseNdjson.mockImplementationOnce(async function* () {
    for (const e of events) yield e;
  });
};
const waiting = (token: string) => ({
  type: 'session.waiting',
  data: { continuationToken: token },
});

let store: EveRunStore;

beforeEach(() => {
  vi.clearAllMocks();
  for (const fn of Object.values(eve)) fn.mockReset();
  vi.spyOn(console, 'log').mockImplementation(() => {});
  vi.spyOn(console, 'error').mockImplementation(() => {});
  authMock.mockResolvedValue({ user: { id: USER } });
  store = createMemoryRunStore();
  setRunStoreForTests(store);
  eve.createEveSession.mockResolvedValue({
    sessionId: 'sess-1',
    continuationToken: 'tok-0',
  });
  eve.openEveStream.mockResolvedValue({ body: {} });
});

describe('POST /api/eve-chat run continuity', () => {
  it('creates a row with the session id and token on a first turn', async () => {
    streamEvents({ type: 'turn.started' });
    const res = await turn();
    await drain(res);

    expect(eve.createEveSession).toHaveBeenCalledTimes(1);
    const row = await store.get(USER, CHAT);
    expect(row?.eveSessionId).toBe('sess-1');
    expect(row?.continuationToken).toBe('tok-0');
    expect(row?.relayOrigin).toBeNull();
  });

  it('continues a second turn and stores the returned token', async () => {
    await store.create(USER, CHAT, {
      eveSessionId: 'sess-1',
      continuationToken: 'tok-0',
    });
    eve.continueEveSession.mockResolvedValue({ continuationToken: 'tok-1' });
    streamEvents({ type: 'turn.started' });

    const res = await turn('again');
    expect(res.status).toBe(200);
    await drain(res);

    expect(eve.continueEveSession).toHaveBeenCalledWith(
      'sess-1',
      'tok-0',
      'again',
    );
    expect(eve.createEveSession).not.toHaveBeenCalled();
    expect((await store.get(USER, CHAT))?.continuationToken).toBe('tok-1');
  });

  it('sends exactly one continue for two concurrent second turns', async () => {
    await store.create(USER, CHAT, {
      eveSessionId: 'sess-1',
      continuationToken: 'tok-0',
    });
    eve.continueEveSession.mockImplementation(async () => {
      await new Promise((r) => setTimeout(r, 10));
      return { continuationToken: 'tok-1' };
    });
    streamEvents({ type: 'turn.started' });
    streamEvents({ type: 'turn.started' });

    const [a, b] = await Promise.all([turn('one'), turn('two')]);

    expect(eve.continueEveSession).toHaveBeenCalledTimes(1);
    expect([a.status, b.status].sort()).toEqual([200, 503]);
    const loser = a.status === 200 ? b : a;
    expect((await loser.json()).code).toBe('offline:chat');
    await drain(a.status === 200 ? a : b);
    expect((await store.get(USER, CHAT))?.continuationToken).toBe('tok-1');
  });

  it('errors without a fresh session when a relay run cannot continue', async () => {
    await store.create(USER, CHAT, {
      eveSessionId: 'sess-1',
      continuationToken: 'tok-0',
      relayOrigin: 'https://relay.example.com',
    });
    eve.continueEveSession.mockRejectedValue(new Error('eve down'));

    const res = await turn();

    expect(res.status).toBe(503);
    expect(eve.createEveSession).not.toHaveBeenCalled();
    const row = await store.get(USER, CHAT);
    expect(row?.eveSessionId).toBe('sess-1');
    expect(row?.continuationToken).toBe('tok-0');
    expect(row?.relayOrigin).toBe('https://relay.example.com');
  });

  it('creates a session for a web chat that has no row', async () => {
    streamEvents({ type: 'turn.started' });
    const res = await turn('hi', 'fresh-chat');
    await drain(res);

    expect(res.status).toBe(200);
    expect(eve.createEveSession).toHaveBeenCalledTimes(1);
    expect(await store.get(USER, 'fresh-chat')).toBeDefined();
  });

  it('treats a duplicate create as the conflict error', async () => {
    store.create = vi.fn().mockRejectedValue(new EveRunExistsError());

    const res = await turn();

    expect(res.status).toBe(503);
    expect((await res.json()).code).toBe('offline:chat');
    // The orphaned Eve session is reported, not silently dropped.
    expect(console.error).toHaveBeenCalledWith(
      expect.stringContaining('orphan'),
      expect.objectContaining({ sessionId: 'sess-1' }),
    );
  });

  it('does not mask other create failures as the conflict', async () => {
    const boom = new Error('connection refused');
    store.create = vi.fn().mockRejectedValue(boom);

    const res = await turn();

    expect(res.status).toBe(400);
    expect((await res.json()).code).not.toBe('offline:chat');
  });

  it('advances the cursor at each session.waiting, never backward', async () => {
    streamEvents({ type: 'turn.started' }, waiting('tok-a'));
    await drain(await turn('one'));
    let row = await store.get(USER, CHAT);
    expect(row?.streamIndex).toBe(2);
    expect(row?.continuationToken).toBe('tok-a');

    eve.continueEveSession.mockResolvedValue({ continuationToken: 'tok-b' });
    streamEvents({ type: 'turn.started' }, waiting('tok-c'));
    await drain(await turn('two'));
    expect(eve.openEveStream).toHaveBeenLastCalledWith('sess-1', 2);
    row = await store.get(USER, CHAT);
    expect(row?.streamIndex).toBe(4);
    expect(row?.continuationToken).toBe('tok-c');

    // A replayed boundary (a lower index than what is stored) is ignored.
    eve.continueEveSession.mockResolvedValue({ continuationToken: 'tok-d' });
    eve.parseNdjson.mockImplementationOnce(async function* () {
      await store.advance(USER, CHAT, {
        continuationToken: 'tok-new',
        streamIndex: 50,
      });
      yield waiting('tok-stale');
    });
    await drain(await turn('three'));
    row = await store.get(USER, CHAT);
    expect(row?.streamIndex).toBe(50);
    expect(row?.continuationToken).toBe('tok-new');
  });

  describe('claim handling', () => {
    const seed = () =>
      store.create(USER, CHAT, {
        eveSessionId: 'sess-1',
        continuationToken: 'tok-0',
      });

    it('streams and does not roll back when the final swap throws', async () => {
      await seed();
      eve.continueEveSession.mockResolvedValue({ continuationToken: 'tok-1' });
      const real = store.swapToken.bind(store);
      let calls = 0;
      store.swapToken = vi.fn(
        async (...args: Parameters<EveRunStore['swapToken']>) => {
          if (++calls === 2) throw new Error('db blip');
          return real(...args);
        },
      );
      streamEvents(waiting('tok-2'));

      const res = await turn();
      expect(res.status).toBe(200);
      await drain(res);

      // Only the claim and the failed final swap ran: no rollback to tok-0.
      expect(calls).toBe(2);
      const row = await store.get(USER, CHAT);
      expect(row?.continuationToken).toBe('tok-2');
      expect(console.error).toHaveBeenCalledWith(
        expect.stringContaining('final token swap'),
        expect.objectContaining({ chatId: CHAT }),
      );
    });

    it('logs and streams when the final swap returns false', async () => {
      await seed();
      eve.continueEveSession.mockResolvedValue({ continuationToken: 'tok-1' });
      const real = store.swapToken.bind(store);
      let calls = 0;
      store.swapToken = vi.fn(
        async (...args: Parameters<EveRunStore['swapToken']>) =>
          ++calls === 2 ? false : real(...args),
      );
      streamEvents(waiting('tok-2'));

      const res = await turn();
      expect(res.status).toBe(200);
      await drain(res);

      expect(calls).toBe(2);
      expect(console.error).toHaveBeenCalledWith(
        expect.stringContaining('final token swap'),
        expect.objectContaining({ chatId: CHAT }),
      );
      expect((await store.get(USER, CHAT))?.continuationToken).toBe('tok-2');
    });

    it('treats a stored claim as a turn in flight and never sends it', async () => {
      await seed();
      await store.swapToken(USER, CHAT, 'tok-0', 'claim:abc');

      const res = await turn();

      expect(res.status).toBe(503);
      expect(eve.continueEveSession).not.toHaveBeenCalled();
      expect(console.error).toHaveBeenCalledWith(
        expect.stringContaining('turn in flight'),
        expect.objectContaining({ chatId: CHAT }),
      );
      expect((await store.get(USER, CHAT))?.continuationToken).toBe(
        'claim:abc',
      );
    });

    it('logs a failing rollback distinctly and still fails the turn', async () => {
      await seed();
      eve.continueEveSession.mockRejectedValue(new Error('eve down'));
      const real = store.swapToken.bind(store);
      let calls = 0;
      store.swapToken = vi.fn(
        async (...args: Parameters<EveRunStore['swapToken']>) => {
          if (++calls === 2) throw new Error('db blip');
          return real(...args);
        },
      );

      const res = await turn();

      expect(res.status).toBe(503);
      expect(console.error).toHaveBeenCalledWith(
        expect.stringContaining('rollback'),
        expect.objectContaining({ chatId: CHAT }),
      );
    });
  });
});

describe('POST /api/eve-chat relay runs', () => {
  const RELAY_CHAT = '3f2b8c1e-5d4a-4b6f-9a7e-1c2d3e4f5a6b';
  const ORIGIN = 'https://apply.example.gov';

  const relayTurn = (relay: unknown, id = RELAY_CHAT) =>
    POST(
      new Request('http://localhost/api/eve-chat', {
        method: 'POST',
        body: JSON.stringify({
          id,
          message: { role: 'user', parts: [{ type: 'text', text: 'hi' }] },
          relay,
        }),
      }),
    );

  beforeEach(() => {
    vi.stubEnv('RELAY_RUNS_ENABLED', '1');
    vi.stubEnv('RELAY_PUBLIC_URL', 'https://relay.example.gov');
    vi.stubEnv('EVE_INTERNAL_SECRET', 'internal-secret-for-tests');
  });

  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it('creates the run with the origin and sends the run reference to Eve', async () => {
    streamEvents({ type: 'turn.started' });
    const res = await relayTurn({ origin: ORIGIN });
    expect(res.status).toBe(200);
    await drain(res);

    expect(eve.createEveSession).toHaveBeenCalledWith('hi', undefined, {
      url: 'wss://relay.example.gov',
      runId: RELAY_CHAT,
    });
    expect((await store.get(USER, RELAY_CHAT))?.relayOrigin).toBe(ORIGIN);
  });

  it('rejects a relay field with 400 when the flag is off', async () => {
    vi.stubEnv('RELAY_RUNS_ENABLED', undefined);
    const res = await relayTurn({ origin: ORIGIN });
    expect(res.status).toBe(400);
    expect(eve.createEveSession).not.toHaveBeenCalled();
    expect(await store.get(USER, RELAY_CHAT)).toBeUndefined();
  });

  it.each([
    ['a path', 'https://a.gov/x'],
    ['http', 'http://a.gov'],
    ['a trailing slash', 'https://a.gov/'],
    ['userinfo', 'https://u@a.gov'],
    ['an uppercase host', 'https://A.gov'],
    ['an explicit default port', 'https://a.gov:443'],
    ['an IDN host', 'https://bücher.de'],
    ['a non-string', 5],
  ])('rejects an origin with %s', async (_n, origin) => {
    const res = await relayTurn({ origin });
    expect(res.status).toBe(400);
    expect(eve.createEveSession).not.toHaveBeenCalled();
  });

  it('rejects a malformed relay field and a non-UUID chat id', async () => {
    expect((await relayTurn('https://a.gov')).status).toBe(400);
    expect((await relayTurn({ origin: ORIGIN }, 'chat-1')).status).toBe(400);
    expect(eve.createEveSession).not.toHaveBeenCalled();
  });

  it('rejects with 400 when RELAY_PUBLIC_URL is unusable', async () => {
    vi.stubEnv('RELAY_PUBLIC_URL', 'http://relay.example.gov');
    expect((await relayTurn({ origin: ORIGIN })).status).toBe(400);
    expect(eve.createEveSession).not.toHaveBeenCalled();
  });

  it('ignores a relay field on a follow-up turn', async () => {
    await store.create(USER, RELAY_CHAT, {
      eveSessionId: 'sess-1',
      continuationToken: 'tok-0',
      relayOrigin: ORIGIN,
    });
    eve.continueEveSession.mockResolvedValue({ continuationToken: 'tok-1' });
    streamEvents({ type: 'turn.started' });

    // Even an invalid origin is not looked at: the stored origin governs.
    const res = await relayTurn({ origin: 'https://other.example.gov/x' });
    expect(res.status).toBe(200);
    await drain(res);

    expect(eve.createEveSession).not.toHaveBeenCalled();
    expect(eve.continueEveSession).toHaveBeenCalledTimes(1);
    expect((await store.get(USER, RELAY_CHAT))?.relayOrigin).toBe(ORIGIN);
  });

  it('does not touch relay for a plain turn with the flag on', async () => {
    streamEvents({ type: 'turn.started' });
    await drain(await turn());
    expect(eve.createEveSession).toHaveBeenCalledWith('hello', undefined);
  });
});

describe('POST /api/eve-chat structured questions (relay runs)', () => {
  const RELAY_CHAT = '3f2b8c1e-5d4a-4b6f-9a7e-1c2d3e4f5a6b';
  const ORIGIN = 'https://apply.example.gov';

  const nameReq = {
    requestId: 'call-a',
    prompt: 'Name?',
    display: 'text' as const,
    options: [],
    allowFreeform: true,
  };
  const colorReq = {
    requestId: 'call-b',
    prompt: 'Color?',
    display: 'select' as const,
    options: [
      { id: 'red', label: 'Red' },
      { id: 'green', label: 'Green' },
    ],
    allowFreeform: false,
  };
  const spikedEvent = {
    type: 'input.requested',
    data: {
      requests: [
        { ...nameReq, action: { kind: 'tool-call', callId: 'call-a' } },
        {
          ...colorReq,
          action: { kind: 'tool-call', callId: 'call-b' },
          options: [
            { id: 'red', label: 'Red', description: 'warm' },
            { id: 'green', label: 'Green' },
          ],
        },
      ],
    },
  };
  const validAnswers = [
    { requestId: 'call-a', text: 'Ada' },
    { requestId: 'call-b', optionId: 'red' },
  ];

  const post = (body: Record<string, unknown>, id = RELAY_CHAT) =>
    POST(
      new Request('http://localhost/api/eve-chat', {
        method: 'POST',
        body: JSON.stringify({ id, ...body }),
      }),
    );
  const answer = (inputResponses: unknown, extra = {}) =>
    post({ inputResponses, ...extra });
  const say = (text: string) =>
    post({ message: { role: 'user', parts: [{ type: 'text', text }] } });

  const seedRelay = async (pending = [nameReq, colorReq]) => {
    await store.create(USER, RELAY_CHAT, {
      eveSessionId: 'sess-1',
      continuationToken: 'tok-0',
      relayOrigin: ORIGIN,
    });
    await store.setPending(USER, RELAY_CHAT, pending);
  };

  /** Reads the SSE body into parsed chunks. */
  const readChunks = async (res: Response) => {
    // The route pipes JSON through JsonToSseTransformStream, which yields
    // strings rather than bytes, so Response.text() cannot read it.
    let text = '';
    const reader = (res.body as unknown as ReadableStream<string>).getReader();
    for (let r = await reader.read(); !r.done; r = await reader.read()) {
      text += r.value;
    }
    return text
      .split('\n')
      .filter((l) => l.startsWith('data: ') && l !== 'data: [DONE]')
      .map((l) => JSON.parse(l.slice(6)));
  };

  beforeEach(() => {
    vi.stubEnv('RELAY_RUNS_ENABLED', '1');
    vi.stubEnv('RELAY_PUBLIC_URL', 'https://relay.example.gov');
    vi.stubEnv('EVE_INTERNAL_SECRET', 'internal-secret-for-tests');
  });
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it('streams a structured chunk and persists two pending requests on a first relay turn', async () => {
    streamEvents(spikedEvent, waiting('tok-1'));
    const res = await post({
      message: { role: 'user', parts: [{ type: 'text', text: 'go' }] },
      relay: { origin: ORIGIN },
    });
    expect(res.status).toBe(200);
    const chunks = await readChunks(res);

    expect(chunks.filter((c) => c.type === 'data-input-request')).toEqual([
      { type: 'data-input-request', data: { requests: [nameReq, colorReq] } },
    ]);
    expect(chunks.some((c) => c.type === 'text-delta')).toBe(false);
    expect((await store.get(USER, RELAY_CHAT))?.pendingRequests).toEqual([
      nameReq,
      colorReq,
    ]);
  });

  it('persists pending on a follow-up relay turn and survives a fresh read', async () => {
    await seedRelay([]);
    eve.continueEveSession.mockResolvedValue({ continuationToken: 'tok-1' });
    streamEvents(spikedEvent, waiting('tok-2'));
    await drain(await say('next'));

    // A restarted process reads the row back from the same store.
    const fresh = await store.get(USER, RELAY_CHAT);
    expect(fresh?.pendingRequests).toEqual([nameReq, colorReq]);
    expect(fresh?.continuationToken).toBe('tok-2');
  });

  it('forwards only the validated answers and clears pending after Eve accepts', async () => {
    await seedRelay();
    const order: string[] = [];
    eve.continueEveSession.mockImplementation(async () => {
      order.push('continue');
      return { continuationToken: 'tok-1' };
    });
    const realSetPending = store.setPending.bind(store);
    const setPending = vi.fn(
      async (...args: Parameters<EveRunStore['setPending']>) => {
        order.push(`setPending:${args[2].length}`);
        return realSetPending(...args);
      },
    );
    store.setPending = setPending;
    streamEvents({ type: 'turn.started' }, waiting('tok-2'));

    // Extra keys (labels, prompts) and a message are never forwarded.
    const res = await answer(
      [
        { requestId: 'call-a', text: '  Ada  ', label: 'x' },
        { requestId: 'call-b', optionId: 'red', label: 'Red', text: null },
      ],
      { message: { role: 'user', parts: [{ type: 'text', text: 'secret' }] } },
    );
    expect(res.status).toBe(200);
    await drain(res);

    expect(eve.continueEveSession).toHaveBeenCalledTimes(1);
    const [sid, tok, input] = eve.continueEveSession.mock.calls[0];
    expect(sid).toBe('sess-1');
    expect(tok).toBe('tok-0');
    expect(input).toStrictEqual({
      inputResponses: [
        { requestId: 'call-a', text: 'Ada' },
        { requestId: 'call-b', optionId: 'red' },
      ],
    });
    expect(Object.keys(input)).toEqual(['inputResponses']);
    expect(order).toEqual(['continue', 'setPending:0']);
    const row = await store.get(USER, RELAY_CHAT);
    expect(row?.pendingRequests).toEqual([]);
    expect(row?.continuationToken).toBe('tok-2');

    // Replaying the same answers finds nothing pending.
    const replay = await answer(validAnswers);
    expect(replay.status).toBe(400);
    expect((await replay.json()).cause).toBe('unknown_request');
    expect(eve.continueEveSession).toHaveBeenCalledTimes(1);
  });

  describe('rejections never touch the token or Eve', () => {
    it.each([
      ['an unknown requestId', [{ requestId: 'nope', text: 'x' }, ...validAnswers], 'unknown_request'],
      ['text on an option-only request', [validAnswers[0], { requestId: 'call-b', text: 'red' }], 'text_not_allowed'],
      ['an unknown option', [validAnswers[0], { requestId: 'call-b', optionId: 'blue' }], 'unknown_option'],
      ['a missing answer', [validAnswers[0]], 'missing_answer'],
      ['a duplicate answer', [...validAnswers, validAnswers[0]], 'duplicate_answer'],
      ['a blank answer', [{ requestId: 'call-a', text: '   ' }, validAnswers[1]], 'empty_answer'],
      ['a non-array body', { requestId: 'call-a' }, 'missing_answer'],
      ['an empty answer list', [], 'missing_answer'],
      ['oversized text', [{ requestId: 'call-a', text: 'x'.repeat(2001) }, validAnswers[1]], 'empty_answer'],
      ['a numeric optionId', [validAnswers[0], { requestId: 'call-b', optionId: 1 }], 'unknown_option'],
    ])('rejects %s', async (_n, inputResponses, cause) => {
      await seedRelay();
      const swap = vi.spyOn(store, 'swapToken');

      const res = await answer(inputResponses);

      expect(res.status).toBe(400);
      expect(await res.json()).toEqual({
        code: 'bad_request:api',
        message: expect.any(String),
        cause,
      });
      expect(eve.continueEveSession).not.toHaveBeenCalled();
      expect(swap).not.toHaveBeenCalled();
      const row = await store.get(USER, RELAY_CHAT);
      expect(row?.continuationToken).toBe('tok-0');
      expect(row?.pendingRequests).toEqual([nameReq, colorReq]);
    });

    it('rejects a stale requestId from an earlier batch', async () => {
      await seedRelay([colorReq]);
      const swap = vi.spyOn(store, 'swapToken');
      const res = await answer(validAnswers);
      expect(res.status).toBe(400);
      expect((await res.json()).cause).toBe('unknown_request');
      expect(eve.continueEveSession).not.toHaveBeenCalled();
      expect(swap).not.toHaveBeenCalled();
    });

    it('rejects a plain message while questions are pending', async () => {
      await seedRelay();
      const swap = vi.spyOn(store, 'swapToken');
      const res = await say('Ada and red');
      expect(res.status).toBe(400);
      expect(await res.json()).toEqual({
        code: 'bad_request:api',
        message: expect.any(String),
        cause: 'answers_pending',
      });
      expect(eve.continueEveSession).not.toHaveBeenCalled();
      expect(swap).not.toHaveBeenCalled();
    });

    it('rejects answers when nothing is pending', async () => {
      await seedRelay([]);
      const swap = vi.spyOn(store, 'swapToken');
      const res = await answer(validAnswers);
      expect(res.status).toBe(400);
      expect((await res.json()).cause).toBe('unknown_request');
      expect(eve.continueEveSession).not.toHaveBeenCalled();
      expect(swap).not.toHaveBeenCalled();
    });
  });

  it('keeps pending and rolls the token back when Eve rejects the forward', async () => {
    await seedRelay();
    eve.continueEveSession.mockRejectedValue(
      new Error('Eve server responded 500'),
    );
    const setPending = vi.spyOn(store, 'setPending');

    const res = await answer(validAnswers);

    expect(res.status).toBe(503);
    expect((await res.json()).code).toBe('offline:chat');
    expect(setPending).not.toHaveBeenCalled();
    const row = await store.get(USER, RELAY_CHAT);
    expect(row?.pendingRequests).toEqual([nameReq, colorReq]);
    expect(row?.continuationToken).toBe('tok-0');
    expect(row?.eveSessionId).toBe('sess-1');
    expect(row?.relayOrigin).toBe(ORIGIN);
    expect(eve.createEveSession).not.toHaveBeenCalled();
  });

  // Eve's ask_question omits allowFreeform by default; parsed as option-only,
  // such a request could never be answered and the run would deadlock.
  it('accepts a text answer to a realistic no-options request', async () => {
    await seedRelay([]);
    eve.continueEveSession.mockResolvedValue({ continuationToken: 'tok-1' });
    streamEvents(
      {
        type: 'input.requested',
        data: {
          requests: [{ requestId: 'call-n', prompt: 'Name?', display: 'text' }],
        },
      },
      waiting('tok-2'),
    );
    await drain(await say('next'));
    expect((await store.get(USER, RELAY_CHAT))?.pendingRequests).toEqual([
      {
        requestId: 'call-n',
        prompt: 'Name?',
        display: 'text',
        options: [],
        allowFreeform: true,
      },
    ]);

    eve.continueEveSession.mockResolvedValue({ continuationToken: 'tok-3' });
    streamEvents({ type: 'turn.started' }, waiting('tok-4'));
    const res = await answer([{ requestId: 'call-n', text: 'Ada' }]);
    expect(res.status).toBe(200);
    await drain(res);

    expect(eve.continueEveSession).toHaveBeenLastCalledWith('sess-1', 'tok-2', {
      inputResponses: [{ requestId: 'call-n', text: 'Ada' }],
    });
    expect((await store.get(USER, RELAY_CHAT))?.pendingRequests).toEqual([]);
  });

  it('heals stale pending at session.waiting when the clear failed', async () => {
    await seedRelay();
    eve.continueEveSession.mockResolvedValue({ continuationToken: 'tok-1' });
    const realSetPending = store.setPending.bind(store);
    let calls = 0;
    store.setPending = vi.fn(
      async (...args: Parameters<EveRunStore['setPending']>) => {
        if (++calls === 1) throw new Error('db blip');
        return realSetPending(...args);
      },
    );
    streamEvents({ type: 'turn.started' }, waiting('tok-2'));

    const res = await answer(validAnswers);
    expect(res.status).toBe(200);
    await drain(res);

    expect(calls).toBe(2);
    const row = await store.get(USER, RELAY_CHAT);
    expect(row?.pendingRequests).toEqual([]);
    expect(row?.continuationToken).toBe('tok-2');
  });

  it('keeps new questions at session.waiting rather than healing them away', async () => {
    await seedRelay();
    eve.continueEveSession.mockResolvedValue({ continuationToken: 'tok-1' });
    streamEvents(spikedEvent, waiting('tok-2'));
    await drain(await answer(validAnswers));
    expect((await store.get(USER, RELAY_CHAT))?.pendingRequests).toEqual([
      nameReq,
      colorReq,
    ]);
  });

  it('forwards exactly once and clears once for two concurrent answer sets', async () => {
    await seedRelay();
    eve.continueEveSession.mockImplementation(async () => {
      await new Promise((r) => setTimeout(r, 10));
      return { continuationToken: 'tok-1' };
    });
    const setPending = vi.spyOn(store, 'setPending');
    streamEvents({ type: 'turn.started' });
    streamEvents({ type: 'turn.started' });

    const [a, b] = await Promise.all([
      answer(validAnswers),
      answer(validAnswers),
    ]);

    expect(eve.continueEveSession).toHaveBeenCalledTimes(1);
    expect([a.status, b.status].sort()).toEqual([200, 503]);
    await drain(a.status === 200 ? a : b);
    expect(setPending.mock.calls.filter((c) => c[2].length === 0)).toHaveLength(
      1,
    );
    expect((await store.get(USER, RELAY_CHAT))?.pendingRequests).toEqual([]);
  });

  it('clears pending before releasing the claim', async () => {
    await seedRelay();
    eve.continueEveSession.mockResolvedValue({ continuationToken: 'tok-1' });
    streamEvents({ type: 'turn.started' });
    const realSetPending = store.setPending.bind(store);
    let racer: Response | undefined;
    store.setPending = vi.fn(
      async (...args: Parameters<EveRunStore['setPending']>) => {
        // A second valid answer set lands while the clear is in flight.
        if (args[2].length === 0 && !racer) racer = await answer(validAnswers);
        return realSetPending(...args);
      },
    );

    const res = await answer(validAnswers);
    expect(res.status).toBe(200);
    await drain(res);

    expect(racer?.status).toBe(503);
    expect(eve.continueEveSession).toHaveBeenCalledTimes(1);
    expect(store.setPending).toHaveBeenCalledTimes(1);
  });

  it('never logs answer text, labels or prompts', async () => {
    await seedRelay();
    eve.continueEveSession.mockResolvedValue({ continuationToken: 'tok-1' });
    streamEvents(spikedEvent, waiting('tok-2'));
    await drain(
      await answer([
        { requestId: 'call-a', text: 'Ada-Lovelace-123' },
        { requestId: 'call-b', optionId: 'red' },
      ]),
    );
    const logged = JSON.stringify([
      ...vi.mocked(console.log).mock.calls,
      ...vi.mocked(console.error).mock.calls,
    ]);
    for (const s of ['Ada-Lovelace-123', 'Name?', 'Color?', 'Red']) {
      expect(logged).not.toContain(s);
    }
  });

  it('keeps prose and ignores inputResponses on a web chat', async () => {
    await store.create(USER, CHAT, {
      eveSessionId: 'sess-1',
      continuationToken: 'tok-0',
    });
    eve.continueEveSession.mockResolvedValue({ continuationToken: 'tok-1' });
    streamEvents(spikedEvent, waiting('tok-2'));
    const setPending = vi.spyOn(store, 'setPending');

    const res = await post(
      {
        message: { role: 'user', parts: [{ type: 'text', text: 'again' }] },
        inputResponses: validAnswers,
      },
      CHAT,
    );
    expect(res.status).toBe(200);
    const chunks = await readChunks(res);

    expect(eve.continueEveSession).toHaveBeenCalledWith(
      'sess-1',
      'tok-0',
      'again',
    );
    expect(chunks.some((c) => c.type === 'data-input-request')).toBe(false);
    expect(chunks.some((c) => c.type === 'text-delta')).toBe(true);
    expect(setPending).not.toHaveBeenCalled();
  });

  it('rejects inputResponses without a message on a web chat', async () => {
    await store.create(USER, CHAT, {
      eveSessionId: 'sess-1',
      continuationToken: 'tok-0',
    });
    const res = await post({ inputResponses: validAnswers }, CHAT);
    expect(res.status).toBe(400);
    expect(eve.continueEveSession).not.toHaveBeenCalled();
  });

  describe('progress', () => {
    const progressEvent = (input: unknown) => ({
      type: 'actions.requested',
      data: {
        actions: [
          {
            kind: 'tool-call',
            callId: 'p1',
            toolName: 'report_progress',
            input,
          },
        ],
      },
    });
    const good = {
      sections: [{ name: 'Name', status: 'done', filled: 2, ssn: '123' }],
    };
    const first = {
      message: { role: 'user', parts: [{ type: 'text', text: 'go' }] },
      relay: { origin: ORIGIN },
    };

    it('stores sanitized progress on a relay run', async () => {
      const setProgress = vi.spyOn(store, 'setProgress');
      streamEvents(progressEvent(good), waiting('tok-1'));
      await drain(await post(first));
      expect(setProgress).toHaveBeenCalledWith(USER, RELAY_CHAT, {
        sections: [{ name: 'Name', status: 'done', filled: 2 }],
      });
    });

    it('does not store invalid progress', async () => {
      const setProgress = vi.spyOn(store, 'setProgress');
      streamEvents(progressEvent({ sections: 'nope' }), waiting('tok-1'));
      await drain(await post(first));
      expect(setProgress).not.toHaveBeenCalled();
    });

    it('does not store progress for a web run', async () => {
      const setProgress = vi.spyOn(store, 'setProgress');
      streamEvents(progressEvent(good), waiting('tok-1'));
      await drain(await turn('hi', 'web-chat'));
      expect(setProgress).not.toHaveBeenCalled();
    });
  });
});
