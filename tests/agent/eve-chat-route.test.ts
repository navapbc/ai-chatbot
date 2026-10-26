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
