import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('server-only', () => ({}));

import {
  createMemoryRunStore,
  setRunStoreForTests,
} from '@/lib/ai/eve/run-store';
import {
  _resetAgentTokenCache,
  fetchAgentToken,
} from '@/lib/relay/agent-token';
import { verifyRelayToken } from '@/lib/relay/token';

const SECRET = 'test-secret-that-is-at-least-32-characters-long';
const INTERNAL = 'internal-secret-value-for-tests';
const RUN_ID = '3f2b8c1e-5d4a-4b6f-9a7e-1c2d3e4f5a6b';
const ORIGIN = 'https://apply.example.gov';
const SESSION = 'wrun_01WINNERSESSION';

const route = await import('@/app/(chat)/api/relay/agent-token/route');

const post = (
  body: unknown,
  headers: Record<string, string> = { 'x-eve-internal-secret': INTERNAL },
  raw = false,
) =>
  route.POST(
    new Request('http://127.0.0.1/api/relay/agent-token', {
      method: 'POST',
      headers,
      body: raw ? (body as string) : JSON.stringify(body),
    }),
  );

let store: ReturnType<typeof createMemoryRunStore>;
let logged: string[];

const seed = (relayOrigin: string | null = ORIGIN) =>
  store.create('u1', RUN_ID, {
    eveSessionId: SESSION,
    continuationToken: 't',
    relayOrigin,
  });

beforeEach(() => {
  vi.stubEnv('RELAY_RUNS_ENABLED', '1');
  vi.stubEnv('EVE_INTERNAL_SECRET', INTERNAL);
  vi.stubEnv('RELAY_TOKEN_SECRET', SECRET);
  store = createMemoryRunStore();
  setRunStoreForTests(store);
  _resetAgentTokenCache();
  logged = [];
  for (const m of ['log', 'info', 'warn', 'error', 'debug'] as const) {
    vi.spyOn(console, m).mockImplementation((...a: unknown[]) => {
      logged.push(JSON.stringify(a));
    });
  }
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllEnvs();
  setRunStoreForTests(null);
});

const expectBodylessNoStore404 = async (res: Response) => {
  expect(res.status).toBe(404);
  expect(res.headers.get('cache-control')).toBe('no-store');
  expect(await res.text()).toBe('');
};

describe('POST /api/relay/agent-token', () => {
  it.each([undefined, '0', 'true', ''])(
    'returns a bodyless 404 without touching the store when the flag is %j',
    async (flag) => {
      vi.stubEnv('RELAY_RUNS_ENABLED', flag);
      await seed();
      const spy = vi.spyOn(store, 'getByChatId');
      await expectBodylessNoStore404(
        await post({ runId: RUN_ID, sessionId: SESSION }),
      );
      expect(spy).not.toHaveBeenCalled();
    },
  );

  it.each([
    ['missing', {}],
    ['wrong', { 'x-eve-internal-secret': 'nope' }],
    ['empty', { 'x-eve-internal-secret': '' }],
  ])('returns 404 for a %s secret', async (_n, headers) => {
    await seed();
    const spy = vi.spyOn(store, 'getByChatId');
    await expectBodylessNoStore404(
      await post({ runId: RUN_ID, sessionId: SESSION }, headers),
    );
    expect(spy).not.toHaveBeenCalled();
  });

  it('returns 404 when the server has no EVE_INTERNAL_SECRET', async () => {
    vi.stubEnv('EVE_INTERNAL_SECRET', undefined);
    await seed();
    await expectBodylessNoStore404(
      await post({ runId: RUN_ID, sessionId: SESSION }),
    );
  });

  it('returns 400 for a non-UUID or malformed body', async () => {
    for (const body of [
      { runId: 'nope', sessionId: SESSION },
      {},
      { runId: 5, sessionId: SESSION },
      { runId: RUN_ID },
      { runId: RUN_ID, sessionId: '' },
      { runId: RUN_ID, sessionId: 5 },
      { runId: RUN_ID, sessionId: 'x'.repeat(201) },
    ]) {
      const res = await post(body);
      expect(res.status).toBe(400);
      expect(res.headers.get('cache-control')).toBe('no-store');
    }
    expect((await post('{bad', undefined, true)).status).toBe(400);
  });

  it('returns 404 for an unknown run', async () => {
    await expectBodylessNoStore404(
      await post({ runId: RUN_ID, sessionId: SESSION }),
    );
  });

  it('returns 404 for a run with no relay origin', async () => {
    await seed(null);
    await expectBodylessNoStore404(
      await post({ runId: RUN_ID, sessionId: SESSION }),
    );
  });

  it.each([
    ['another', 'wrun_01ORPHANSESSION'],
    ['a prefix of the', 'wrun_01WINNER'],
    ['a case variant of the', SESSION.toLowerCase()],
  ])(
    'returns 404 for %s Eve session than the run was created with',
    async (_n, sessionId) => {
      await seed();
      await expectBodylessNoStore404(await post({ runId: RUN_ID, sessionId }));
    },
  );

  it('refuses the orphan session of a lost first-turn race', async () => {
    // Two first turns raced: both created an Eve session carrying this run
    // id, the winner stored its session, the loser's store.create failed.
    // The orphan still holds the run reference, but not the stored session.
    await seed();
    await expectBodylessNoStore404(
      await post({ runId: RUN_ID, sessionId: 'wrun_01LOSERSESSION' }),
    );
    const winner = await post({ runId: RUN_ID, sessionId: SESSION });
    expect(winner.status).toBe(200);
  });

  it('ignores a web row another user holds on the chat id', async () => {
    await seed();
    await store.create('u2', RUN_ID, {
      eveSessionId: 's2',
      continuationToken: 't2',
    });
    const res = await post({ runId: RUN_ID, sessionId: SESSION });
    expect(res.status).toBe(200);
  });

  it.each([undefined, 'short'])(
    'returns a credential-free 500 when RELAY_TOKEN_SECRET is %j',
    async (value) => {
      vi.stubEnv('RELAY_TOKEN_SECRET', value);
      await seed();
      const res = await post({ runId: RUN_ID, sessionId: SESSION });
      expect(res.status).toBe(500);
      expect(res.headers.get('cache-control')).toBe('no-store');
      const text = await res.text();
      for (const leak of [SECRET, INTERNAL, ORIGIN, 'short']) {
        expect(text).not.toContain(leak);
      }
    },
  );

  it('mints an agent token for the stored origin only', async () => {
    await seed();
    const res = await post({ runId: RUN_ID, sessionId: SESSION });
    expect(res.status).toBe(200);
    expect(res.headers.get('cache-control')).toBe('no-store');
    const body = await res.json();
    expect(Object.keys(body).sort()).toEqual(['expiresAt', 'token']);
    const claims = verifyRelayToken(body.token, SECRET);
    expect(claims).toMatchObject({
      runId: RUN_ID,
      role: 'agent',
      origin: ORIGIN,
      sub: 'u1',
    });
    expect(body.expiresAt).toBeLessThan(1e11);
    expect(JSON.stringify(body)).not.toContain(INTERNAL);
    expect(logged.join('')).not.toContain(body.token);
  });

  it('round-trips through fetchAgentToken into a verifiable token', async () => {
    await seed();
    const result = await fetchAgentToken(RUN_ID, SESSION, {
      fetchImpl: ((url: string, init: RequestInit) =>
        route.POST(new Request(url, init))) as unknown as typeof fetch,
      baseUrl: 'http://127.0.0.1:3000',
    });
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(verifyRelayToken(result.token, SECRET)).toMatchObject({
      runId: RUN_ID,
      role: 'agent',
      origin: ORIGIN,
      sub: 'u1',
    });
    const refused = await fetchAgentToken(RUN_ID, 'wrun_01LOSERSESSION', {
      fetchImpl: ((url: string, init: RequestInit) =>
        route.POST(new Request(url, init))) as unknown as typeof fetch,
      baseUrl: 'http://127.0.0.1:3000',
    });
    expect(refused).toEqual({
      ok: false,
      error: 'The relay token request was refused.',
    });
  });
});
