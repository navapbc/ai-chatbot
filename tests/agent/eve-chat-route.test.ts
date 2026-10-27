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
import { PARTICIPANTS, buildApplicationPrompt } from '@/lib/data/participants';
import { MAX_RELAY_TEXT_CHARS, WITHHELD } from '@/lib/data/sensitive-record';
import { extractParticipantFromText } from '@/lib/jev/participant';
import {
  buildExtensionRunMessage,
  EXTENSION_RUN_MARKER,
} from '@/lib/relay/run-prompt';
import { buildRelayOpeningMessage } from '@/lib/relay/run-input';
import {
  createStaticParticipantSource,
  setParticipantSourceForTests,
} from '@/lib/data/participant-source';
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
    eve.continueEveSession.mockResolvedValue({
      continuationToken: 'tok-1',
      sessionId: 'sess-1',
    });
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
      return { continuationToken: 'tok-1', sessionId: 'sess-1' };
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
    streamEvents({ type: 'turn.started' }, waiting('tok-0'));
    await drain(await turn('one'));
    let row = await store.get(USER, CHAT);
    expect(row?.streamIndex).toBe(2);
    expect(row?.continuationToken).toBe('tok-0');

    eve.continueEveSession.mockResolvedValue({
      continuationToken: 'tok-b',
      sessionId: 'sess-1',
    });
    streamEvents({ type: 'turn.started' }, waiting('tok-b'));
    await drain(await turn('two'));
    expect(eve.openEveStream).toHaveBeenLastCalledWith('sess-1', 2);
    row = await store.get(USER, CHAT);
    expect(row?.streamIndex).toBe(4);
    expect(row?.continuationToken).toBe('tok-b');

    // A replayed boundary (a lower index than what is stored) is ignored.
    eve.continueEveSession.mockResolvedValue({
      continuationToken: 'tok-d',
      sessionId: 'sess-1',
    });
    eve.parseNdjson.mockImplementationOnce(async function* () {
      await store.advance(USER, CHAT, { streamIndex: 50 });
      yield waiting('tok-stale');
    });
    await drain(await turn('three'));
    row = await store.get(USER, CHAT);
    expect(row?.streamIndex).toBe(50);
    expect(row?.continuationToken).toBe('tok-d');
  });

  it('never writes a consumed token back at a replayed turn boundary', async () => {
    // Rotating tokens, as the route is written to tolerate: each continue
    // returns the token for the next turn, and a turn's session.waiting
    // carries the token its own create/continue returned.
    streamEvents({ type: 'turn.started' }, waiting('T1'));
    eve.createEveSession.mockResolvedValue({
      sessionId: 'sess-1',
      continuationToken: 'T1',
    });
    await drain(await turn('turn 1'));
    expect(await store.get(USER, CHAT)).toMatchObject({
      continuationToken: 'T1',
      streamIndex: 2,
    });

    // 1. Turn N: Eve accepts it and returns T2, but the read dies before
    //    turn N's session.waiting, so the cursor stays at 2.
    eve.continueEveSession.mockResolvedValueOnce({
      continuationToken: 'T2',
      sessionId: 'sess-1',
    });
    streamEvents({ type: 'turn.started' });
    await drain(await turn('turn N'));
    expect(await store.get(USER, CHAT)).toMatchObject({
      continuationToken: 'T2',
      streamIndex: 2,
    });

    // 2-3. Turn N+1 claims T2; Eve returns T3; the final swap stores T3.
    eve.continueEveSession.mockResolvedValueOnce({
      continuationToken: 'T3',
      sessionId: 'sess-1',
    });
    // 4. Its read starts at the stale cursor and replays turn N's events,
    //    reaching turn N's session.waiting, which carries the consumed T2.
    streamEvents({ type: 'turn.started' }, waiting('T2'));
    await drain(await turn('turn N+1'));
    expect(eve.continueEveSession).toHaveBeenLastCalledWith(
      'sess-1',
      'T2',
      'turn N+1',
    );
    expect(eve.openEveStream).toHaveBeenLastCalledWith('sess-1', 2);

    // 5. The stored token is still T3, so the chat can continue with it.
    expect(await store.get(USER, CHAT)).toMatchObject({
      continuationToken: 'T3',
      streamIndex: 4,
    });
    eve.continueEveSession.mockResolvedValueOnce({
      continuationToken: 'T4',
      sessionId: 'sess-1',
    });
    streamEvents({ type: 'turn.started' }, waiting('T3'));
    const res = await turn('turn N+2');
    expect(res.status).toBe(200);
    await drain(res);
    expect(eve.continueEveSession).toHaveBeenLastCalledWith(
      'sess-1',
      'T3',
      'turn N+2',
    );
  });

  it('keeps the sent token when Eve returns none on continue', async () => {
    // Eve 0.27's continue route answers `{ ok, sessionId }`: no token. The
    // session keeps one token, so the one just sent stays valid.
    await store.create(USER, CHAT, {
      eveSessionId: 'sess-1',
      continuationToken: 'eve:tok',
    });
    eve.continueEveSession.mockResolvedValue({
      continuationToken: '',
      sessionId: 'sess-1',
    });
    // The read dies before session.waiting.
    streamEvents({ type: 'turn.started' });
    await drain(await turn('one'));
    expect((await store.get(USER, CHAT))?.continuationToken).toBe('eve:tok');

    streamEvents({ type: 'turn.started' }, waiting('eve:tok'));
    const res = await turn('two');
    expect(res.status).toBe(200);
    await drain(res);
    expect(eve.continueEveSession).toHaveBeenLastCalledWith(
      'sess-1',
      'eve:tok',
      'two',
    );
  });

  describe('claim handling', () => {
    const seed = () =>
      store.create(USER, CHAT, {
        eveSessionId: 'sess-1',
        continuationToken: 'tok-0',
      });

    it('streams and does not roll back when the final swap throws', async () => {
      await seed();
      eve.continueEveSession.mockResolvedValue({
        continuationToken: 'tok-1',
        sessionId: 'sess-1',
      });
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
      // session.waiting heals the claim with the token Eve returned for this
      // turn, never with the event's own token.
      expect(calls).toBe(2);
      const row = await store.get(USER, CHAT);
      expect(row?.continuationToken).toBe('tok-1');
      expect(console.error).toHaveBeenCalledWith(
        expect.stringContaining('final token swap'),
        expect.objectContaining({ chatId: CHAT }),
      );
    });

    it('logs and streams when the final swap returns false', async () => {
      await seed();
      eve.continueEveSession.mockResolvedValue({
        continuationToken: 'tok-1',
        sessionId: 'sess-1',
      });
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
      expect((await store.get(USER, CHAT))?.continuationToken).toBe('tok-1');
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

  /** A follow-up (or web) turn that also carries a relay field. */
  const relayTurn = (relay: unknown, id = RELAY_CHAT, text = 'hi') =>
    POST(
      new Request('http://localhost/api/eve-chat', {
        method: 'POST',
        body: JSON.stringify({
          id,
          message: { role: 'user', parts: [{ type: 'text', text }] },
          relay,
        }),
      }),
    );
  /** A relay first turn: no message, the server builds it. */
  const startRun = (
    relay: unknown,
    id = RELAY_CHAT,
    extra: Record<string, unknown> = {},
  ) =>
    POST(
      new Request('http://localhost/api/eve-chat', {
        method: 'POST',
        body: JSON.stringify({ id, relay, ...extra }),
      }),
    );
  const RECORD = PARTICIPANTS[0];
  const start = (extra: Record<string, unknown> = {}) => ({
    origin: ORIGIN,
    participant: { recordId: RECORD.record_id },
    ...extra,
  });
  const REF = { url: 'wss://relay.example.gov', runId: RELAY_CHAT };

  beforeEach(() => {
    vi.stubEnv('RELAY_RUNS_ENABLED', '1');
    vi.stubEnv('RELAY_PUBLIC_URL', 'https://relay.example.gov');
    vi.stubEnv(
      'EVE_INTERNAL_SECRET',
      'internal-secret-for-tests-padded-to-be-over-thirty-two-chars',
    );
    setParticipantSourceForTests(createStaticParticipantSource(PARTICIPANTS));
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    setParticipantSourceForTests(undefined);
  });

  describe('extension origin', () => {
    const EXT = `chrome-extension://${'a'.repeat(32)}`;
    const withOrigin = (body: unknown, origin?: string) =>
      POST(
        new Request('http://localhost/api/eve-chat', {
          method: 'POST',
          headers: origin === undefined ? {} : { Origin: origin },
          body: JSON.stringify(body),
        }),
      );
    const webBody = (id: string) => ({
      id,
      message: { role: 'user', parts: [{ type: 'text', text: 'hi' }] },
    });
    const expectUntouched = async () => {
      expect(eve.createEveSession).not.toHaveBeenCalled();
      expect(eve.continueEveSession).not.toHaveBeenCalled();
      expect(eve.openEveStream).not.toHaveBeenCalled();
    };

    beforeEach(() => {
      vi.stubEnv('RELAY_EXTENSION_ORIGIN', EXT);
    });

    it.each([undefined, 'https://evil.example'])(
      'refuses a relay first turn with Origin %j before the store or Eve',
      async (origin) => {
        const res = await withOrigin(
          { id: RELAY_CHAT, relay: start() },
          origin,
        );
        expect(res.status).toBe(403);
        expect(await store.get(USER, RELAY_CHAT)).toBeUndefined();
        await expectUntouched();
      },
    );

    it.each([undefined, 'https://evil.example'])(
      'refuses a relay follow-up with Origin %j before any claim or Eve call',
      async (origin) => {
        await store.create(USER, RELAY_CHAT, {
          eveSessionId: 'sess-1',
          continuationToken: 'tok-0',
          relayOrigin: ORIGIN,
        });
        const res = await withOrigin(webBody(RELAY_CHAT), origin);
        expect(res.status).toBe(403);
        expect((await store.get(USER, RELAY_CHAT))?.continuationToken).toBe(
          'tok-0',
        );
        await expectUntouched();
      },
    );

    it('serves a relay first turn from the exact origin', async () => {
      streamEvents({ type: 'turn.started' });
      const res = await withOrigin({ id: RELAY_CHAT, relay: start() }, EXT);
      expect(res.status).toBe(200);
      await drain(res);
      expect(eve.createEveSession).toHaveBeenCalledTimes(1);
    });

    it('leaves web chats alone, with any Origin or none', async () => {
      streamEvents({ type: 'turn.started' });
      await drain(
        await withOrigin(webBody('web-1'), 'https://app.example.gov'),
      );
      expect(eve.createEveSession).toHaveBeenCalledTimes(1);

      eve.continueEveSession.mockResolvedValue({
        continuationToken: 'tok-1',
        sessionId: 'sess-1',
      });
      streamEvents({ type: 'turn.started' });
      const res = await withOrigin(webBody('web-1'));
      expect(res.status).toBe(200);
      await drain(res);
      expect(eve.continueEveSession).toHaveBeenCalledTimes(1);
    });

    it('checks nothing with the env unset', async () => {
      vi.stubEnv('RELAY_EXTENSION_ORIGIN', undefined);
      streamEvents({ type: 'turn.started' });
      const res = await withOrigin(
        { id: RELAY_CHAT, relay: start() },
        'https://evil.example',
      );
      expect(res.status).toBe(200);
      await drain(res);
    });
  });

  it('creates the run with the origin and sends the run reference to Eve', async () => {
    streamEvents({ type: 'turn.started' });
    const res = await startRun(start());
    expect(res.status).toBe(200);
    await drain(res);

    expect(eve.createEveSession).toHaveBeenCalledWith(
      buildExtensionRunMessage(
        buildRelayOpeningMessage({ origin: ORIGIN, participant: RECORD }),
      ),
      'claude-opus-5',
      REF,
    );
    expect((await store.get(USER, RELAY_CHAT))?.relayOrigin).toBe(ORIGIN);
  });

  it('builds the opening message on the server, marker first, record after', async () => {
    streamEvents({ type: 'turn.started' });
    await drain(await startRun(start()));
    const sent = eve.createEveSession.mock.calls[0][0] as string;
    expect(sent.startsWith(EXTENSION_RUN_MARKER)).toBe(true);
    expect(sent).toContain(`Fill out ${ORIGIN} for the participant below.`);
    expect(sent).toContain(RECORD.participant.name.first);
    expect(sent).toContain(RECORD.participant.name.last);
    expect(extractParticipantFromText(sent)?.record_id).toBe(RECORD.record_id);
  });

  it('never logs the record id, origin or participant name', async () => {
    streamEvents({ type: 'turn.started' });
    await drain(await startRun(start()));
    const logged = JSON.stringify([
      ...vi.mocked(console.log).mock.calls,
      ...vi.mocked(console.error).mock.calls,
    ]);
    for (const s of [
      RECORD.participant.name.first,
      RECORD.participant.name.last,
      ORIGIN,
    ]) {
      expect(logged).not.toContain(s);
    }
  });

  describe('first-turn refusals happen before any Eve session', () => {
    afterEach(() => {
      expect(eve.createEveSession).not.toHaveBeenCalled();
      expect(eve.continueEveSession).not.toHaveBeenCalled();
    });

    const expectCause = async (
      res: Response,
      status: number,
      cause: string,
    ) => {
      expect(res.status).toBe(status);
      expect((await res.json()).cause).toBe(cause);
      expect(await store.get(USER, RELAY_CHAT)).toBeUndefined();
    };

    it('refuses a relay first turn that carries a message', async () => {
      await expectCause(
        await startRun(start(), RELAY_CHAT, {
          message: { role: 'user', parts: [{ type: 'text', text: 'hi' }] },
        }),
        400,
        'message_not_allowed',
      );
    });

    it('refuses a record id the source does not have', async () => {
      await expectCause(
        await startRun(start({ participant: { recordId: 'no-such-id' } })),
        400,
        'unknown_participant',
      );
    });

    it('refuses a model outside the default allowlist', async () => {
      await expectCause(
        await startRun(start({ model: 'gpt-5.4' })),
        400,
        'model_not_allowed',
      );
    });

    it('refuses a model the env list does not name', async () => {
      vi.stubEnv('RELAY_ALLOWED_MODELS', 'claude-opus-5,claude-sonnet-5');
      await expectCause(
        await startRun(start({ model: 'claude-haiku-4-5' })),
        400,
        'model_not_allowed',
      );
    });

    it('refuses a sensitive details key rather than stripping it', async () => {
      await expectCause(
        await startRun(
          start({ participant: { details: { Name: 'Ada', SSN: 'x' } } }),
        ),
        400,
        'sensitive_detail',
      );
    });

    it('refuses too many details', async () => {
      const details = Object.fromEntries(
        Array.from({ length: 41 }, (_, i) => [`Field ${i}`, 'v']),
      );
      await expectCause(
        await startRun(start({ participant: { details } })),
        400,
        'details_too_large',
      );
    });

    it('refuses a nested details value', async () => {
      await expectCause(
        await startRun(
          start({ participant: { details: { Address: { city: 'X' } } } }),
        ),
        400,
        'details_invalid',
      );
    });

    it('refuses a sensitive value typed into a details field', async () => {
      const res = await startRun(
        start({ participant: { details: { Note: 'SSN 123-45-6789' } } }),
      );
      expect(res.status).toBe(400);
      const body = await res.json();
      expect(body.cause).toBe('sensitive_value_in_message');
      expect(JSON.stringify(body)).not.toContain('123-45-6789');
    });

    it('refuses a chat id another user already runs relay on, with the conflict error', async () => {
      await store.create('user-2', RELAY_CHAT, {
        eveSessionId: 'sess-other',
        continuationToken: 'tok-other',
        relayOrigin: ORIGIN,
      });
      const res = await startRun(start());
      expect(res.status).toBe(503);
      expect((await res.json()).code).toBe('offline:chat');
      // The other user's run is untouched.
      expect(await store.getByChatId(RELAY_CHAT)).toMatchObject({
        userId: 'user-2',
        eveSessionId: 'sess-other',
      });
    });

    it('refuses the run over the hourly limit with rate_limit:chat', async () => {
      for (let i = 0; i < 20; i++) {
        await store.create(USER, `seed-${i}`, {
          eveSessionId: `s-${i}`,
          continuationToken: 't',
          relayOrigin: ORIGIN,
        });
      }
      const res = await startRun(start());
      expect(res.status).toBe(429);
      expect((await res.json()).code).toBe('rate_limit:chat');
      expect(await store.get(USER, RELAY_CHAT)).toBeUndefined();
    });

    it('honors a lower RELAY_MAX_RUNS_PER_HOUR', async () => {
      vi.stubEnv('RELAY_MAX_RUNS_PER_HOUR', '1');
      await store.create(USER, 'seed-0', {
        eveSessionId: 's-0',
        continuationToken: 't',
        relayOrigin: ORIGIN,
      });
      expect((await startRun(start())).status).toBe(429);
    });
  });

  describe('guests (P4a-a)', () => {
    const GUEST = { id: 'guest-user', email: 'guest-0a1b2c3d@preview.local' };

    it('refuses a guest relay first turn with 403 before any lookup or Eve call', async () => {
      authMock.mockResolvedValue({ user: GUEST });
      const get = vi.fn();
      setParticipantSourceForTests({ search: vi.fn(), get });
      const count = vi.spyOn(store, 'countRecentRelayRuns');
      const byChat = vi.spyOn(store, 'getByChatId');

      const res = await startRun(start());

      expect(res.status).toBe(403);
      expect((await res.json()).code).toBe('forbidden:chat');
      expect(get).not.toHaveBeenCalled();
      expect(count).not.toHaveBeenCalled();
      expect(byChat).not.toHaveBeenCalled();
      expect(eve.createEveSession).not.toHaveBeenCalled();
      expect(await store.get(GUEST.id, RELAY_CHAT)).toBeUndefined();
    });

    it('still lets a guest start a web chat', async () => {
      authMock.mockResolvedValue({ user: GUEST });
      streamEvents({ type: 'turn.started' });
      const res = await turn('hi', 'guest-web-chat');
      expect(res.status).toBe(200);
      await drain(res);
      expect(eve.createEveSession).toHaveBeenCalledWith('hi', undefined);
    });

    it('does not refuse a regular user whose email merely mentions guest', async () => {
      authMock.mockResolvedValue({
        user: { id: USER, email: 'guest-relations@example.gov' },
      });
      streamEvents({ type: 'turn.started' });
      const res = await startRun(start());
      expect(res.status).toBe(200);
      await drain(res);
      expect(eve.createEveSession).toHaveBeenCalledTimes(1);
    });
  });

  it('allows the 20th run in an hour', async () => {
    for (let i = 0; i < 19; i++) {
      await store.create(USER, `seed-${i}`, {
        eveSessionId: `s-${i}`,
        continuationToken: 't',
        relayOrigin: ORIGIN,
      });
    }
    // Web chats do not count toward the limit.
    for (let i = 0; i < 5; i++) {
      await store.create(USER, `web-${i}`, {
        eveSessionId: `w-${i}`,
        continuationToken: 't',
      });
    }
    streamEvents({ type: 'turn.started' });
    const res = await startRun(start());
    expect(res.status).toBe(200);
    await drain(res);
    expect(eve.createEveSession).toHaveBeenCalledTimes(1);
  });

  it('sends an allowed model to Eve and ignores modelOverride', async () => {
    vi.stubEnv('RELAY_ALLOWED_MODELS', 'claude-opus-5,claude-sonnet-5');
    streamEvents({ type: 'turn.started' });
    const res = await startRun(
      start({ model: 'claude-sonnet-5' }),
      RELAY_CHAT,
      {
        modelOverride: 'gpt-5.4',
      },
    );
    expect(res.status).toBe(200);
    await drain(res);
    expect(eve.createEveSession.mock.calls[0][1]).toBe('claude-sonnet-5');
  });

  it('uses the default allowed model when the start names none, whatever modelOverride says', async () => {
    streamEvents({ type: 'turn.started' });
    await drain(
      await startRun(start(), RELAY_CHAT, { modelOverride: 'gpt-5.4' }),
    );
    expect(eve.createEveSession.mock.calls[0][1]).toBe('claude-opus-5');
  });

  it('uses the first env-allowed model when the start names none', async () => {
    vi.stubEnv('RELAY_ALLOWED_MODELS', 'claude-sonnet-5');
    streamEvents({ type: 'turn.started' });
    await drain(await startRun(start()));
    expect(eve.createEveSession.mock.calls[0][1]).toBe('claude-sonnet-5');
  });

  it('builds the opening message from entered details', async () => {
    const details = { 'First name': 'Ada', City: 'Riverside' };
    streamEvents({ type: 'turn.started' });
    const res = await startRun(start({ participant: { details } }));
    expect(res.status).toBe(200);
    await drain(res);
    expect(eve.createEveSession).toHaveBeenCalledWith(
      buildExtensionRunMessage(
        buildRelayOpeningMessage({ origin: ORIGIN, participant: details }),
      ),
      'claude-opus-5',
      REF,
    );
  });

  it('refuses a follow-up on a relay run that has no message', async () => {
    await store.create(USER, RELAY_CHAT, {
      eveSessionId: 'sess-1',
      continuationToken: 'tok-0',
      relayOrigin: ORIGIN,
    });
    const res = await startRun(start());
    expect(res.status).toBe(400);
    expect(eve.continueEveSession).not.toHaveBeenCalled();
    expect(eve.createEveSession).not.toHaveBeenCalled();
    expect((await store.get(USER, RELAY_CHAT))?.continuationToken).toBe(
      'tok-0',
    );
  });

  it('still refuses a web first turn with no message', async () => {
    const res = await POST(
      new Request('http://localhost/api/eve-chat', {
        method: 'POST',
        body: JSON.stringify({ id: 'web-x' }),
      }),
    );
    expect(res.status).toBe(400);
    expect(eve.createEveSession).not.toHaveBeenCalled();
  });

  describe('sensitive record values', () => {
    const SECRET = PARTICIPANTS[0].participant.calworks_id;
    const opening = () =>
      buildApplicationPrompt({
        participant: PARTICIPANTS[0],
        target: 'a WIC application',
      });

    it('a relay first turn sends Eve a message whose record has no calworks_id value', async () => {
      streamEvents({ type: 'turn.started' });
      await drain(await startRun(start()));
      const sent = eve.createEveSession.mock.calls[0][0] as string;
      expect(sent).toContain(RECORD.participant.name.first);
      expect(sent).not.toContain(SECRET);
      expect(sent).toContain(WITHHELD);
      expect(extractParticipantFromText(sent)?.participant.calworks_id).toBe(
        WITHHELD,
      );
    });

    it('a web chat first turn is unchanged', async () => {
      streamEvents({ type: 'turn.started' });
      await drain(await turn(opening(), 'web-record-chat'));
      expect(eve.createEveSession).toHaveBeenCalledWith(opening(), undefined);
    });

    it('a relay follow-up without a record is not rewritten', async () => {
      await store.create(USER, RELAY_CHAT, {
        eveSessionId: 'sess-1',
        continuationToken: 'tok-0',
        relayOrigin: ORIGIN,
      });
      eve.continueEveSession.mockResolvedValue({
        continuationToken: 'tok-1',
        sessionId: 'sess-1',
      });
      streamEvents({ type: 'turn.started' });
      await drain(
        await relayTurn(undefined, RELAY_CHAT, 'use 123 Main St instead'),
      );
      expect(eve.continueEveSession.mock.calls[0][2]).toBe(
        'use 123 Main St instead',
      );
    });

    it('a record pasted into a relay follow-up is stripped too', async () => {
      await store.create(USER, RELAY_CHAT, {
        eveSessionId: 'sess-1',
        continuationToken: 'tok-0',
        relayOrigin: ORIGIN,
      });
      eve.continueEveSession.mockResolvedValue({
        continuationToken: 'tok-1',
        sessionId: 'sess-1',
      });
      streamEvents({ type: 'turn.started' });
      await drain(await relayTurn(undefined, RELAY_CHAT, opening()));
      const sent = eve.continueEveSession.mock.calls[0][2] as string;
      expect(sent).not.toContain(SECRET);
      expect(sent).toContain(WITHHELD);
    });
  });

  it('does not add the marker to a follow-up turn', async () => {
    await store.create(USER, RELAY_CHAT, {
      eveSessionId: 'sess-1',
      continuationToken: 'tok-0',
      relayOrigin: ORIGIN,
    });
    eve.continueEveSession.mockResolvedValue({
      continuationToken: 'tok-1',
      sessionId: 'sess-1',
    });
    streamEvents({ type: 'turn.started' });
    await drain(await relayTurn(undefined));
    expect(JSON.stringify(eve.continueEveSession.mock.calls)).not.toContain(
      EXTENSION_RUN_MARKER,
    );
  });

  it('rejects a relay field with 400 when the flag is off', async () => {
    vi.stubEnv('RELAY_RUNS_ENABLED', undefined);
    const res = await startRun(start());
    expect(res.status).toBe(400);
    expect(eve.createEveSession).not.toHaveBeenCalled();
    expect(await store.get(USER, RELAY_CHAT)).toBeUndefined();
    // The old body shape is refused the same way.
    expect((await relayTurn({ origin: ORIGIN })).status).toBe(400);
    expect(eve.createEveSession).not.toHaveBeenCalled();
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
    const res = await startRun(start({ origin }));
    expect(res.status).toBe(400);
    expect((await res.json()).cause).toBe('invalid_origin');
    expect(eve.createEveSession).not.toHaveBeenCalled();
  });

  it('rejects a malformed relay field and a non-UUID chat id', async () => {
    expect((await startRun('https://a.gov')).status).toBe(400);
    // The old body shape (no participant) is refused.
    const old = await relayTurn({ origin: ORIGIN });
    expect(old.status).toBe(400);
    expect((await old.json()).cause).toBe('message_not_allowed');
    const nonUuid = await startRun(start(), 'chat-1');
    expect(nonUuid.status).toBe(400);
    expect(eve.createEveSession).not.toHaveBeenCalled();
    expect(await store.get(USER, 'chat-1')).toBeUndefined();
  });

  it('rejects with 400 when RELAY_PUBLIC_URL is unusable', async () => {
    vi.stubEnv('RELAY_PUBLIC_URL', 'http://relay.example.gov');
    expect((await startRun(start())).status).toBe(400);
    expect(eve.createEveSession).not.toHaveBeenCalled();
  });

  it('ignores a relay field on a follow-up turn', async () => {
    await store.create(USER, RELAY_CHAT, {
      eveSessionId: 'sess-1',
      continuationToken: 'tok-0',
      relayOrigin: ORIGIN,
    });
    eve.continueEveSession.mockResolvedValue({
      continuationToken: 'tok-1',
      sessionId: 'sess-1',
    });
    streamEvents({ type: 'turn.started' });

    // Even an invalid origin is not looked at: the stored origin governs.
    const res = await relayTurn({ origin: 'https://other.example.gov/x' });
    expect(res.status).toBe(200);
    await drain(res);

    expect(eve.createEveSession).not.toHaveBeenCalled();
    expect(eve.continueEveSession).toHaveBeenCalledTimes(1);
    expect((await store.get(USER, RELAY_CHAT))?.relayOrigin).toBe(ORIGIN);
  });

  it('fails a follow-up that Eve delivered to a different session', async () => {
    // Eve's send path starts a fresh session (with the continue request's
    // auth, no relay attributes) when the old run has ended.
    await store.create(USER, RELAY_CHAT, {
      eveSessionId: 'sess-1',
      continuationToken: 'tok-0',
      relayOrigin: ORIGIN,
    });
    eve.continueEveSession.mockResolvedValue({
      continuationToken: '',
      sessionId: 'wrun_fresh',
    });
    const errors = vi.mocked(console.error);

    const res = await relayTurn(undefined);
    expect(res.status).toBe(503);
    expect((await res.json()).code).toBe('offline:chat');
    expect(eve.openEveStream).not.toHaveBeenCalled();
    expect(eve.createEveSession).not.toHaveBeenCalled();
    // The claim is rolled back and the row still names the old session.
    expect(await store.get(USER, RELAY_CHAT)).toMatchObject({
      eveSessionId: 'sess-1',
      continuationToken: 'tok-0',
      relayOrigin: ORIGIN,
    });
    const line = errors.mock.calls.find((c) =>
      String(c[0]).includes('landed on another session'),
    );
    expect(line).toBeDefined();
    expect(JSON.stringify(line)).not.toContain('wrun_fresh');
    expect(JSON.stringify(line)).not.toContain('tok-0');
  });

  it('keeps the old behavior for a web chat whose continue lands elsewhere', async () => {
    await store.create(USER, CHAT, {
      eveSessionId: 'sess-1',
      continuationToken: 'tok-0',
    });
    eve.continueEveSession.mockResolvedValue({
      continuationToken: '',
      sessionId: 'wrun_fresh',
    });
    streamEvents({ type: 'turn.started' });
    const res = await turn('again');
    expect(res.status).toBe(200);
    await drain(res);
  });

  it('rejects a relay field on a follow-up of a web chat', async () => {
    await store.create(USER, RELAY_CHAT, {
      eveSessionId: 'sess-1',
      continuationToken: 'tok-0',
    });

    const res = await relayTurn({ origin: ORIGIN });
    expect(res.status).toBe(400);
    expect(eve.continueEveSession).not.toHaveBeenCalled();
    expect(eve.createEveSession).not.toHaveBeenCalled();
    // The row is untouched: still a web chat, token not claimed.
    expect(await store.get(USER, RELAY_CHAT)).toMatchObject({
      relayOrigin: null,
      continuationToken: 'tok-0',
    });
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
    vi.stubEnv(
      'EVE_INTERNAL_SECRET',
      'internal-secret-for-tests-padded-to-be-over-thirty-two-chars',
    );
    setParticipantSourceForTests(createStaticParticipantSource(PARTICIPANTS));
  });
  afterEach(() => {
    vi.unstubAllEnvs();
    setParticipantSourceForTests(undefined);
  });

  it('streams a structured chunk and persists two pending requests on a first relay turn', async () => {
    streamEvents(spikedEvent, waiting('tok-1'));
    const res = await post({
      relay: {
        origin: ORIGIN,
        participant: { recordId: PARTICIPANTS[0].record_id },
      },
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
    eve.continueEveSession.mockResolvedValue({
      continuationToken: 'tok-1',
      sessionId: 'sess-1',
    });
    streamEvents(spikedEvent, waiting('tok-1'));
    await drain(await say('next'));

    // A restarted process reads the row back from the same store.
    const fresh = await store.get(USER, RELAY_CHAT);
    expect(fresh?.pendingRequests).toEqual([nameReq, colorReq]);
    expect(fresh?.continuationToken).toBe('tok-1');
  });

  it('forwards only the validated answers and clears pending after Eve accepts', async () => {
    await seedRelay();
    const order: string[] = [];
    eve.continueEveSession.mockImplementation(async () => {
      order.push('continue');
      return { continuationToken: 'tok-1', sessionId: 'sess-1' };
    });
    const realSetPending = store.setPending.bind(store);
    const setPending = vi.fn(
      async (...args: Parameters<EveRunStore['setPending']>) => {
        order.push(`setPending:${args[2].length}`);
        return realSetPending(...args);
      },
    );
    store.setPending = setPending;
    streamEvents({ type: 'turn.started' }, waiting('tok-1'));

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
    expect(row?.continuationToken).toBe('tok-1');

    // Replaying the same answers finds nothing pending.
    const replay = await answer(validAnswers);
    expect(replay.status).toBe(400);
    expect((await replay.json()).cause).toBe('unknown_request');
    expect(eve.continueEveSession).toHaveBeenCalledTimes(1);
  });

  describe('rejections never touch the token or Eve', () => {
    it.each([
      [
        'an unknown requestId',
        [{ requestId: 'nope', text: 'x' }, ...validAnswers],
        'unknown_request',
      ],
      [
        'text on an option-only request',
        [validAnswers[0], { requestId: 'call-b', text: 'red' }],
        'text_not_allowed',
      ],
      [
        'an unknown option',
        [validAnswers[0], { requestId: 'call-b', optionId: 'blue' }],
        'unknown_option',
      ],
      ['a missing answer', [validAnswers[0]], 'missing_answer'],
      [
        'a duplicate answer',
        [...validAnswers, validAnswers[0]],
        'duplicate_answer',
      ],
      [
        'a blank answer',
        [{ requestId: 'call-a', text: '   ' }, validAnswers[1]],
        'empty_answer',
      ],
      ['a non-array body', { requestId: 'call-a' }, 'missing_answer'],
      ['an empty answer list', [], 'missing_answer'],
      [
        'oversized text',
        [{ requestId: 'call-a', text: 'x'.repeat(2001) }, validAnswers[1]],
        'empty_answer',
      ],
      [
        'a numeric optionId',
        [validAnswers[0], { requestId: 'call-b', optionId: 1 }],
        'unknown_option',
      ],
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

  describe('free-text answers get the prose guard', () => {
    const phoneReq = {
      requestId: 'call-p',
      prompt: 'Phone number?',
      display: 'text' as const,
      options: [],
      allowFreeform: true,
    };
    const refRequest = {
      requestId: 'call-r',
      prompt: 'Reference',
      display: 'text' as const,
      options: [],
      allowFreeform: true,
    };

    it.each([
      ['an SSN-shaped answer', 'call-a', '123-45-6789'],
      ['a label and value answer', 'call-r', 'case number 1234567890'],
      [
        'a pasted record that stripping would change',
        'call-r',
        '{"record_id":"R1","participant":{"name":{"first":"A","last":"B"},"calworks_id":"zz-900"}}',
      ],
    ])(
      'refuses %s without touching Eve, the token or pending',
      async (_n, requestId, text) => {
        await seedRelay([nameReq, refRequest]);
        const swap = vi.spyOn(store, 'swapToken');
        const res = await answer([
          { requestId: 'call-a', text: requestId === 'call-a' ? text : 'Ada' },
          { requestId: 'call-r', text: requestId === 'call-r' ? text : 'x' },
        ]);
        expect(res.status).toBe(400);
        const body = await res.json();
        expect(body.cause).toBe('sensitive_value_in_message');
        expect(JSON.stringify(body)).not.toContain(text);
        expect(eve.continueEveSession).not.toHaveBeenCalled();
        expect(swap).not.toHaveBeenCalled();
        const row = await store.get(USER, RELAY_CHAT);
        expect(row?.continuationToken).toBe('tok-0');
        expect(row?.pendingRequests).toEqual([nameReq, refRequest]);
      },
    );

    it('refuses answers whose total length is over the cap', async () => {
      const many = Array.from({ length: 200 }, (_, i) => ({
        ...nameReq,
        requestId: `call-${i}`,
      }));
      await seedRelay(many);
      const res = await answer(
        many.map((m) => ({ requestId: m.requestId, text: 'a'.repeat(1900) })),
      );
      expect(res.status).toBe(400);
      expect((await res.json()).cause).toBe('message_too_large');
      expect(eve.continueEveSession).not.toHaveBeenCalled();
    });

    it.each([
      ['yes', 'Yes'],
      ['a name', 'Ada Lovelace'],
      ['a phone number', '555-123-4567'],
      ['a ZIP+4', '94110-1234'],
    ])('accepts %s', async (_n, text) => {
      await seedRelay([phoneReq, colorReq]);
      eve.continueEveSession.mockResolvedValue({
        continuationToken: 'tok-1',
        sessionId: 'sess-1',
      });
      streamEvents({ type: 'turn.started' }, waiting('tok-1'));
      const res = await answer([
        { requestId: 'call-p', text },
        { requestId: 'call-b', optionId: 'red' },
      ]);
      expect(res.status).toBe(200);
      await drain(res);
      expect(eve.continueEveSession).toHaveBeenCalledTimes(1);
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
    eve.continueEveSession.mockResolvedValue({
      continuationToken: 'tok-1',
      sessionId: 'sess-1',
    });
    streamEvents(
      {
        type: 'input.requested',
        data: {
          requests: [{ requestId: 'call-n', prompt: 'Name?', display: 'text' }],
        },
      },
      waiting('tok-1'),
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

    eve.continueEveSession.mockResolvedValue({
      continuationToken: 'tok-3',
      sessionId: 'sess-1',
    });
    streamEvents({ type: 'turn.started' }, waiting('tok-3'));
    const res = await answer([{ requestId: 'call-n', text: 'Ada' }]);
    expect(res.status).toBe(200);
    await drain(res);

    expect(eve.continueEveSession).toHaveBeenLastCalledWith('sess-1', 'tok-1', {
      inputResponses: [{ requestId: 'call-n', text: 'Ada' }],
    });
    expect((await store.get(USER, RELAY_CHAT))?.pendingRequests).toEqual([]);
  });

  it('heals stale pending at session.waiting when the clear failed', async () => {
    await seedRelay();
    eve.continueEveSession.mockResolvedValue({
      continuationToken: 'tok-1',
      sessionId: 'sess-1',
    });
    const realSetPending = store.setPending.bind(store);
    let calls = 0;
    store.setPending = vi.fn(
      async (...args: Parameters<EveRunStore['setPending']>) => {
        if (++calls === 1) throw new Error('db blip');
        return realSetPending(...args);
      },
    );
    streamEvents({ type: 'turn.started' }, waiting('tok-1'));

    const res = await answer(validAnswers);
    expect(res.status).toBe(200);
    await drain(res);

    expect(calls).toBe(2);
    const row = await store.get(USER, RELAY_CHAT);
    expect(row?.pendingRequests).toEqual([]);
    expect(row?.continuationToken).toBe('tok-1');
  });

  it('keeps new questions at session.waiting rather than healing them away', async () => {
    await seedRelay();
    eve.continueEveSession.mockResolvedValue({
      continuationToken: 'tok-1',
      sessionId: 'sess-1',
    });
    streamEvents(spikedEvent, waiting('tok-1'));
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
      return { continuationToken: 'tok-1', sessionId: 'sess-1' };
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
    eve.continueEveSession.mockResolvedValue({
      continuationToken: 'tok-1',
      sessionId: 'sess-1',
    });
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
    eve.continueEveSession.mockResolvedValue({
      continuationToken: 'tok-1',
      sessionId: 'sess-1',
    });
    streamEvents(spikedEvent, waiting('tok-1'));
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
    eve.continueEveSession.mockResolvedValue({
      continuationToken: 'tok-1',
      sessionId: 'sess-1',
    });
    streamEvents(spikedEvent, waiting('tok-1'));
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
      relay: {
        origin: ORIGIN,
        participant: { recordId: PARTICIPANTS[0].record_id },
      },
    };

    it('stores sanitized progress on a relay run', async () => {
      const setProgress = vi.spyOn(store, 'setProgress');
      streamEvents(progressEvent(good), waiting('tok-1'));
      await drain(await post(first));
      expect(setProgress).toHaveBeenCalledWith(USER, RELAY_CHAT, {
        sections: [{ name: 'Name', status: 'done', filled: 2 }],
      });
    });

    it('survives a failing progress write and never logs the report', async () => {
      const errSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
      const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
      const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
      const SENTINEL = 'SENTINEL-4f9c';
      vi.spyOn(store, 'setProgress').mockRejectedValue(new Error('db down'));
      streamEvents(
        progressEvent({
          sections: [{ name: 'Name', status: 'done', filled: 1, v: SENTINEL }],
          current: 'x',
        }),
        waiting('tok-0'),
      );
      const chunks = await readChunks(await post(first));
      expect(chunks.some((c) => c.type === 'error')).toBe(false);
      expect((await store.get(USER, RELAY_CHAT))?.continuationToken).toBe(
        'tok-0',
      );
      for (const spy of [errSpy, logSpy, warnSpy]) {
        expect(JSON.stringify(spy.mock.calls)).not.toContain(SENTINEL);
      }
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

describe('POST /api/eve-chat relay message bounds', () => {
  const RELAY_CHAT = '4a2b8c1e-5d4a-4b6f-9a7e-1c2d3e4f5a6b';
  const ORIGIN = 'https://apply.example.gov';
  const startRun = (participant: unknown) =>
    POST(
      new Request('http://localhost/api/eve-chat', {
        method: 'POST',
        body: JSON.stringify({
          id: RELAY_CHAT,
          relay: { origin: ORIGIN, participant },
        }),
      }),
    );
  // A record whose free-text note is longer than the whole message cap.
  const HUGE = {
    ...PARTICIPANTS[0],
    record_id: 'huge-1',
    family_profile: { linked: false, notes: 'x'.repeat(MAX_RELAY_TEXT_CHARS) },
  };
  const followUp = (text: string) =>
    POST(
      new Request('http://localhost/api/eve-chat', {
        method: 'POST',
        body: JSON.stringify({
          id: RELAY_CHAT,
          message: { role: 'user', parts: [{ type: 'text', text }] },
        }),
      }),
    );

  beforeEach(() => {
    vi.stubEnv('RELAY_RUNS_ENABLED', '1');
    vi.stubEnv('RELAY_PUBLIC_URL', 'https://relay.example.gov');
    vi.stubEnv(
      'EVE_INTERNAL_SECRET',
      'internal-secret-for-tests-padded-to-be-over-thirty-two-chars',
    );
    setParticipantSourceForTests(
      createStaticParticipantSource([...PARTICIPANTS, HUGE]),
    );
  });
  afterEach(() => {
    vi.unstubAllEnvs();
    setParticipantSourceForTests(undefined);
  });

  it('rejects an oversized first message before Eve or the store', async () => {
    const res = await startRun({ recordId: HUGE.record_id });
    expect(res.status).toBe(400);
    expect((await res.json()).cause).toBe('message_too_large');
    expect(eve.createEveSession).not.toHaveBeenCalled();
    expect(await store.get(USER, RELAY_CHAT)).toBeUndefined();
  });

  it('rejects an oversized follow-up without touching the token', async () => {
    await store.create(USER, RELAY_CHAT, {
      eveSessionId: 'sess-1',
      continuationToken: 'tok-0',
      relayOrigin: ORIGIN,
    });
    const res = await followUp('x'.repeat(MAX_RELAY_TEXT_CHARS + 1));
    expect(res.status).toBe(400);
    expect(eve.continueEveSession).not.toHaveBeenCalled();
    expect((await store.get(USER, RELAY_CHAT))?.continuationToken).toBe(
      'tok-0',
    );
  });

  it('refuses a first message with a sensitive value in prose', async () => {
    const res = await startRun({
      details: { Notes: 'CalWORKs ID: 12345 SSN 123-45-6789' },
    });
    expect(res.status).toBe(400);
    expect(JSON.stringify(await res.json())).not.toContain('123-45-6789');
    expect(eve.createEveSession).not.toHaveBeenCalled();
    expect(await store.get(USER, RELAY_CHAT)).toBeUndefined();
  });

  it('refuses a follow-up with a sensitive value without touching the token', async () => {
    await store.create(USER, RELAY_CHAT, {
      eveSessionId: 'sess-1',
      continuationToken: 'tok-0',
      relayOrigin: ORIGIN,
    });
    const res = await followUp('my SSN is 123-45-6789');
    expect(res.status).toBe(400);
    expect(eve.continueEveSession).not.toHaveBeenCalled();
    expect((await store.get(USER, RELAY_CHAT))?.continuationToken).toBe(
      'tok-0',
    );
  });

  it('accepts the normal opening message', async () => {
    streamEvents({ type: 'turn.started' });
    const res = await startRun({ recordId: PARTICIPANTS[0].record_id });
    expect(res.status).toBe(200);
    await drain(res);
  });
});

describe('POST /api/eve-chat ended relay runs', () => {
  const RELAY_CHAT = '3f2b8c1e-5d4a-4b6f-9a7e-1c2d3e4f5a6b';
  const ORIGIN = 'https://apply.example.gov';
  const EXT = `chrome-extension://${'a'.repeat(32)}`;
  const nameReq = {
    requestId: 'call-a',
    prompt: 'Name?',
    display: 'text' as const,
    options: [],
    allowFreeform: true,
  };

  const post = (body: Record<string, unknown>, id = RELAY_CHAT) =>
    POST(
      new Request('http://localhost/api/eve-chat', {
        method: 'POST',
        headers: { Origin: EXT },
        body: JSON.stringify({ id, ...body }),
      }),
    );
  const say = (text: string, id = RELAY_CHAT) =>
    post({ message: { role: 'user', parts: [{ type: 'text', text }] } }, id);

  const seedEnded = async (pending: (typeof nameReq)[] = []) => {
    await store.create(USER, RELAY_CHAT, {
      eveSessionId: 'sess-1',
      continuationToken: 'tok-0',
      relayOrigin: ORIGIN,
    });
    await store.setPending(USER, RELAY_CHAT, pending);
    await store.end(USER, RELAY_CHAT);
  };

  const expectRunEnded = async (res: Response) => {
    expect(res.status).toBe(400);
    expect((await res.json()).cause).toBe('run_ended');
    expect(eve.continueEveSession).not.toHaveBeenCalled();
    expect(eve.createEveSession).not.toHaveBeenCalled();
    expect(eve.openEveStream).not.toHaveBeenCalled();
    expect((await store.get(USER, RELAY_CHAT))?.continuationToken).toBe(
      'tok-0',
    );
  };

  beforeEach(() => {
    vi.stubEnv('RELAY_RUNS_ENABLED', '1');
    vi.stubEnv('RELAY_EXTENSION_ORIGIN', EXT);
    vi.stubEnv('RELAY_PUBLIC_URL', 'https://relay.example.gov');
    vi.stubEnv(
      'EVE_INTERNAL_SECRET',
      'internal-secret-for-tests-padded-to-be-over-thirty-two-chars',
    );
    setParticipantSourceForTests(createStaticParticipantSource(PARTICIPANTS));
  });
  afterEach(() => {
    vi.unstubAllEnvs();
    setParticipantSourceForTests(undefined);
  });

  it('refuses a follow-up with run_ended before the token claim', async () => {
    await seedEnded();
    const swap = vi.spyOn(store, 'swapToken');
    await expectRunEnded(await say('keep going'));
    expect(swap).not.toHaveBeenCalled();
  });

  it('refuses an answer turn with run_ended, keeping pending', async () => {
    await seedEnded([nameReq]);
    const swap = vi.spyOn(store, 'swapToken');
    const setPending = vi.spyOn(store, 'setPending');
    await expectRunEnded(
      await post({ inputResponses: [{ requestId: 'call-a', text: 'Ada' }] }),
    );
    expect(swap).not.toHaveBeenCalled();
    expect(setPending).not.toHaveBeenCalled();
    expect((await store.get(USER, RELAY_CHAT))?.pendingRequests).toEqual([
      nameReq,
    ]);
  });

  it('refuses run_ended ahead of the answers_pending check', async () => {
    await seedEnded([nameReq]);
    await expectRunEnded(await say('hello'));
  });

  it('leaves relay first turns alone', async () => {
    streamEvents({ type: 'turn.started' });
    const res = await post({
      relay: {
        origin: ORIGIN,
        participant: { recordId: PARTICIPANTS[0].record_id },
      },
    });
    expect(res.status).toBe(200);
    await drain(res);
    expect(eve.createEveSession).toHaveBeenCalledTimes(1);
  });

  it('leaves web chats alone', async () => {
    await store.create(USER, CHAT, {
      eveSessionId: 'sess-w',
      continuationToken: 'tok-w',
    });
    // end() on a web row is not something the stop route does; even so, the
    // ended check is scoped to relay runs.
    await store.end(USER, CHAT);
    eve.continueEveSession.mockResolvedValue({
      continuationToken: 'tok-w2',
      sessionId: 'sess-w',
    });
    streamEvents({ type: 'turn.started' });
    const res = await say('hi', CHAT);
    expect(res.status).toBe(200);
    await drain(res);
    expect(eve.continueEveSession).toHaveBeenCalledTimes(1);
  });
});
