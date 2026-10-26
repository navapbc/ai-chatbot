import { beforeEach, describe, expect, it, vi } from 'vitest';

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
});
