import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('server-only', () => ({}));
const authMock = vi.hoisted(() => vi.fn());
vi.mock('@/app/(auth)/auth', () => ({ auth: authMock }));

const eve = vi.hoisted(() => ({
  createEveSession: vi.fn(),
  continueEveSession: vi.fn(),
  openEveStream: vi.fn(),
  parseNdjson: vi.fn(),
  cancelEveTurn: vi.fn(),
}));
vi.mock('@/lib/ai/eve/eve-client', () => eve);

const { createMemoryRunStore, setRunStoreForTests } = await import(
  '@/lib/ai/eve/run-store'
);
const stop = await import('@/app/(chat)/api/relay/runs/[runId]/stop/route');
const state = await import('@/app/(chat)/api/relay/runs/[runId]/route');
const token = await import('@/app/(chat)/api/relay/token/route');
const agentToken = await import('@/app/(chat)/api/relay/agent-token/route');
const eveChat = await import('@/app/(chat)/api/eve-chat/route');

const USER = 'user-1';
const RUN = '11111111-1111-4111-8111-111111111111';
const WEB_RUN = '22222222-2222-4222-8222-222222222222';
const OTHER_RUN = '33333333-3333-4333-8333-333333333333';
const UNKNOWN_RUN = '44444444-4444-4444-8444-444444444444';
const ORIGIN = 'https://apply.example.gov';
const SESSION = 'wrun_SECRETSESSION01';
const TOKEN = 'eve:SECRET-CONTINUATION-TOKEN';
const EXT = `chrome-extension://${'a'.repeat(32)}`;
const INTERNAL =
  'internal-secret-value-for-tests-padded-to-be-over-thirty-two-chars';

const postStop = (runId: string, origin: string | null = EXT) =>
  stop.POST(
    new Request(`http://localhost/api/relay/runs/${runId}/stop`, {
      method: 'POST',
      headers: origin === null ? {} : { Origin: origin },
    }),
    { params: Promise.resolve({ runId }) },
  );

let store: ReturnType<typeof createMemoryRunStore>;
let logged: string[];

const expectBodylessNoStore404 = async (res: Response) => {
  expect(res.status).toBe(404);
  expect(res.headers.get('Cache-Control')).toBe('no-store');
  expect(await res.text()).toBe('');
};

beforeEach(async () => {
  vi.stubEnv('RELAY_RUNS_ENABLED', '1');
  vi.stubEnv('RELAY_EXTENSION_ORIGIN', EXT);
  authMock.mockReset();
  authMock.mockResolvedValue({
    user: { id: USER, email: 'caseworker@example.gov' },
  });
  for (const fn of Object.values(eve)) fn.mockReset();
  eve.cancelEveTurn.mockResolvedValue('accepted');
  store = createMemoryRunStore();
  setRunStoreForTests(store);
  await store.create(USER, RUN, {
    eveSessionId: SESSION,
    continuationToken: TOKEN,
    relayOrigin: ORIGIN,
  });
  await store.create(USER, WEB_RUN, {
    eveSessionId: 'wrun_web',
    continuationToken: 't2',
  });
  await store.create('user-2', OTHER_RUN, {
    eveSessionId: 'wrun_other',
    continuationToken: 't3',
    relayOrigin: ORIGIN,
  });
  logged = [];
  for (const m of ['log', 'info', 'warn', 'error', 'debug'] as const) {
    vi.spyOn(console, m).mockImplementation((...a: unknown[]) => {
      logged.push(
        a.map((x) => (x instanceof Error ? x.message : String(x))).join(' '),
      );
    });
  }
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllEnvs();
  setRunStoreForTests(null);
});

describe('POST /api/relay/runs/[runId]/stop', () => {
  it('exports only POST', () => {
    expect((stop as Record<string, unknown>).GET).toBeUndefined();
  });

  it('ends the run, cancels the Eve turn once, and leaks nothing', async () => {
    const res = await postStop(RUN);
    expect(res.status).toBe(200);
    expect(res.headers.get('Cache-Control')).toBe('no-store');
    const text = await res.text();
    expect(text).not.toContain(SESSION);
    expect(text).not.toContain(TOKEN);
    expect(JSON.parse(text)).toEqual({ ended: true, alreadyEnded: false });
    expect(eve.cancelEveTurn).toHaveBeenCalledTimes(1);
    expect(eve.cancelEveTurn).toHaveBeenCalledWith(SESSION);
    expect((await store.get(USER, RUN))?.endedAt).toBeInstanceOf(Date);
  });

  it('a later stop reports alreadyEnded and still attempts the cancel', async () => {
    await postStop(RUN);
    const endedAt = (await store.get(USER, RUN))?.endedAt;
    for (let i = 0; i < 2; i++) {
      const res = await postStop(RUN);
      expect(res.status).toBe(200);
      expect(await res.json()).toEqual({ ended: true, alreadyEnded: true });
    }
    // Eve's cancel is idempotent; a first cancel may have failed transiently.
    expect(eve.cancelEveTurn).toHaveBeenCalledTimes(3);
    for (const call of eve.cancelEveTurn.mock.calls) {
      expect(call).toEqual([SESSION]);
    }
    expect((await store.get(USER, RUN))?.endedAt).toEqual(endedAt);
  });

  it('a later stop answers alreadyEnded even when its cancel fails', async () => {
    await postStop(RUN);
    eve.cancelEveTurn.mockRejectedValue(new Error(`Eve ${SESSION}`));
    const res = await postStop(RUN);
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ ended: true, alreadyEnded: true });
    expect(logged).toContain('[relay-stop] cancel failed');
    expect(logged.join('\n')).not.toContain(SESSION);
  });

  it('two concurrent stops: exactly one ends the run', async () => {
    const bodies = await Promise.all(
      [postStop(RUN), postStop(RUN)].map(async (p) => (await p).json()),
    );
    expect(bodies.map((b) => b.alreadyEnded).sort()).toEqual([false, true]);
    expect(bodies.every((b) => b.ended === true)).toBe(true);
    expect((await store.getByChatId(OTHER_RUN))?.endedAt).toBeNull();
  });

  it('still answers ended when the cancel fails, and logs no values', async () => {
    eve.cancelEveTurn.mockRejectedValue(
      new Error(`Eve cancel ${SESSION} failed`),
    );
    const res = await postStop(RUN);
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ ended: true, alreadyEnded: false });
    expect((await store.get(USER, RUN))?.endedAt).toBeInstanceOf(Date);
    expect(logged).toContain('[relay-stop] cancel failed');
    expect(logged.join('\n')).not.toContain(SESSION);
    expect(logged.join('\n')).not.toContain(TOKEN);
  });

  it("returns 404 for another user's run, cancels nothing, and leaves it live", async () => {
    const end = vi.spyOn(store, 'end');
    await expectBodylessNoStore404(await postStop(OTHER_RUN));
    expect(eve.cancelEveTurn).not.toHaveBeenCalled();
    expect(end).not.toHaveBeenCalled();
    const other = await store.getByChatId(OTHER_RUN);
    expect(other?.endedAt).toBeNull();
  });

  it.each([
    ['a web (non-relay) run', WEB_RUN],
    ['an unknown run', UNKNOWN_RUN],
  ])('returns a bodyless 404 for %s and cancels nothing', async (_n, runId) => {
    const end = vi.spyOn(store, 'end');
    await expectBodylessNoStore404(await postStop(runId));
    expect(end).not.toHaveBeenCalled();
    expect(eve.cancelEveTurn).not.toHaveBeenCalled();
  });

  it('leaves a web run un-ended', async () => {
    await postStop(WEB_RUN);
    expect((await store.get(USER, WEB_RUN))?.endedAt).toBeNull();
  });

  it.each(['not-a-uuid', '../x', ''])(
    'returns 400 for malformed id %j before the store',
    async (runId) => {
      const get = vi.spyOn(store, 'get');
      const res = await postStop(runId);
      expect(res.status).toBe(400);
      expect(res.headers.get('Cache-Control')).toBe('no-store');
      expect(get).not.toHaveBeenCalled();
    },
  );

  it.each([null, 'https://evil.example'])(
    'refuses Origin %j with 403 before auth or any store call',
    async (origin) => {
      const get = vi.spyOn(store, 'get');
      const end = vi.spyOn(store, 'end');
      const res = await postStop(RUN, origin);
      expect(res.status).toBe(403);
      expect(res.headers.get('Cache-Control')).toBe('no-store');
      expect(authMock).not.toHaveBeenCalled();
      expect(get).not.toHaveBeenCalled();
      expect(end).not.toHaveBeenCalled();
      expect(eve.cancelEveTurn).not.toHaveBeenCalled();
    },
  );

  it('needs no Origin when RELAY_EXTENSION_ORIGIN is unset', async () => {
    vi.stubEnv('RELAY_EXTENSION_ORIGIN', undefined);
    expect((await postStop(RUN, null)).status).toBe(200);
  });

  it.each([undefined, '0'])(
    'returns a bodyless 404 before auth when the flag is %j',
    async (flag) => {
      vi.stubEnv('RELAY_RUNS_ENABLED', flag);
      await expectBodylessNoStore404(await postStop(RUN));
      expect(authMock).not.toHaveBeenCalled();
      expect(eve.cancelEveTurn).not.toHaveBeenCalled();
    },
  );

  it('returns 403 for a guest and ends nothing', async () => {
    authMock.mockResolvedValue({
      user: { id: USER, email: 'guest-0a1b2c3d@preview.local' },
    });
    const res = await postStop(RUN);
    expect(res.status).toBe(403);
    expect(res.headers.get('Cache-Control')).toBe('no-store');
    expect((await store.get(USER, RUN))?.endedAt).toBeNull();
  });

  it('returns a value-free no-store 500 when the store throws', async () => {
    vi.spyOn(store, 'end').mockRejectedValue(new Error(`db ${SESSION}`));
    const res = await postStop(RUN);
    expect(res.status).toBe(500);
    expect(res.headers.get('Cache-Control')).toBe('no-store');
    expect(await res.text()).not.toContain(SESSION);
    expect(eve.cancelEveTurn).not.toHaveBeenCalled();
  });
});

describe('after stop', () => {
  beforeEach(async () => {
    vi.stubEnv('RELAY_TOKEN_ENDPOINT_ENABLED', '1');
    vi.stubEnv('RELAY_TOKEN_SECRET', 'x'.repeat(48));
    vi.stubEnv('RELAY_PUBLIC_URL', 'https://relay.example.gov');
    vi.stubEnv('EVE_INTERNAL_SECRET', INTERNAL);
    expect((await postStop(RUN)).status).toBe(200);
  });

  it('the state route reports ended without secrets', async () => {
    const res = await state.GET(
      new Request(`http://localhost/api/relay/runs/${RUN}`),
      { params: Promise.resolve({ runId: RUN }) },
    );
    const text = await res.text();
    expect(JSON.parse(text).ended).toBe(true);
    expect(text).not.toContain(SESSION);
    expect(text).not.toContain(TOKEN);
  });

  it('the token route answers a bodyless 404', async () => {
    const res = await token.POST(
      new Request('http://localhost/api/relay/token', {
        method: 'POST',
        headers: { Origin: EXT },
        body: JSON.stringify({ runId: RUN }),
      }),
    );
    await expectBodylessNoStore404(res);
  });

  it('the agent-token route answers a bodyless 404', async () => {
    const res = await agentToken.POST(
      new Request('http://127.0.0.1/api/relay/agent-token', {
        method: 'POST',
        headers: { 'x-eve-internal-secret': INTERNAL },
        body: JSON.stringify({ runId: RUN, sessionId: SESSION }),
      }),
    );
    expect(res.status).toBe(404);
    expect(res.headers.get('cache-control')).toBe('no-store');
    expect(await res.text()).toBe('');
  });

  it('eve-chat refuses a follow-up with run_ended before the claim', async () => {
    const swap = vi.spyOn(store, 'swapToken');
    const res = await eveChat.POST(
      new Request('http://localhost/api/eve-chat', {
        method: 'POST',
        headers: { Origin: EXT },
        body: JSON.stringify({
          id: RUN,
          message: { role: 'user', parts: [{ type: 'text', text: 'go on' }] },
        }),
      }),
    );
    expect(res.status).toBe(400);
    expect((await res.json()).cause).toBe('run_ended');
    expect(swap).not.toHaveBeenCalled();
    expect(eve.continueEveSession).not.toHaveBeenCalled();
    expect((await store.get(USER, RUN))?.continuationToken).toBe(TOKEN);
  });
});
