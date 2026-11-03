import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('server-only', () => ({}));
const authMock = vi.hoisted(() => vi.fn());
vi.mock('@/app/(auth)/auth', () => ({ auth: authMock }));

const { createMemoryRunStore, setRunStoreForTests } = await import(
  '@/lib/ai/eve/run-store'
);
const route = await import('@/app/(chat)/api/relay/runs/[runId]/route');

const USER = 'user-1';
const RUN = '11111111-1111-4111-8111-111111111111';
const WEB_RUN = '22222222-2222-4222-8222-222222222222';
const OTHER_RUN = '33333333-3333-4333-8333-333333333333';
const UNKNOWN_RUN = '44444444-4444-4444-8444-444444444444';
const ORIGIN = 'https://apply.example.gov';
const SESSION = 'wrun_SECRETSESSION01';
const TOKEN = 'eve:SECRET-CONTINUATION-TOKEN';
const EXT = `chrome-extension://${'a'.repeat(32)}`;

const PENDING = [
  {
    requestId: 'call-a',
    prompt: 'Name?',
    display: 'text' as const,
    options: [],
    allowFreeform: true,
  },
];
const PROGRESS = {
  sections: [{ name: 'Household', status: 'filling' as const, filled: 2 }],
};

const get = (runId: string, origin?: string) =>
  route.GET(
    new Request(`http://localhost/api/relay/runs/${runId}`, {
      headers: origin === undefined ? {} : { Origin: origin },
    }),
    { params: Promise.resolve({ runId }) },
  );

let store: ReturnType<typeof createMemoryRunStore>;

const expectBodylessNoStore404 = async (res: Response) => {
  expect(res.status).toBe(404);
  expect(res.headers.get('Cache-Control')).toBe('no-store');
  expect(await res.text()).toBe('');
};

beforeEach(async () => {
  vi.stubEnv('RELAY_RUNS_ENABLED', '1');
  authMock.mockReset();
  authMock.mockResolvedValue({
    user: { id: USER, email: 'caseworker@example.gov' },
  });
  store = createMemoryRunStore();
  setRunStoreForTests(store);
  await store.create(USER, RUN, {
    eveSessionId: SESSION,
    continuationToken: TOKEN,
    relayOrigin: ORIGIN,
  });
  await store.setPending(USER, RUN, PENDING);
  await store.setProgress(USER, RUN, PROGRESS);
  await store.create(USER, WEB_RUN, {
    eveSessionId: 'wrun_web',
    continuationToken: 't2',
  });
  await store.create('user-2', OTHER_RUN, {
    eveSessionId: 'wrun_other',
    continuationToken: 't3',
    relayOrigin: ORIGIN,
  });
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllEnvs();
  setRunStoreForTests(null);
});

describe('GET /api/relay/runs/[runId]', () => {
  it('exports only GET', () => {
    expect((route as Record<string, unknown>).POST).toBeUndefined();
  });

  it('returns the run state to its owner, with no secrets', async () => {
    const res = await get(RUN);
    expect(res.status).toBe(200);
    expect(res.headers.get('Cache-Control')).toBe('no-store');
    const text = await res.text();
    expect(text).not.toContain(SESSION);
    expect(text).not.toContain(TOKEN);
    const body = JSON.parse(text);
    expect(Object.keys(body).sort()).toEqual([
      'ended',
      'endedAt',
      'origin',
      'pending',
      'phase',
      'progress',
      'runId',
    ]);
    expect(body).toEqual({
      runId: RUN,
      origin: ORIGIN,
      ended: false,
      endedAt: null,
      progress: PROGRESS,
      pending: PENDING,
      phase: 'needs_you',
    });
  });

  describe('phase', () => {
    const phaseOf = async () => (await (await get(RUN)).json()).phase;

    it('is working while the turn runs and nothing is pending', async () => {
      await store.setPending(USER, RUN, []);
      expect(await phaseOf()).toBe('working');
    });

    it('is review when the turn is idle and nothing is pending', async () => {
      await store.setPending(USER, RUN, []);
      await store.setTurnIdle(USER, RUN, true);
      expect(await phaseOf()).toBe('review');
    });

    it('is needs_you when questions are pending, even with an idle turn', async () => {
      await store.setTurnIdle(USER, RUN, true);
      expect(await phaseOf()).toBe('needs_you');
    });

    it('is ended for an ended run, whatever else the row holds', async () => {
      await store.setTurnIdle(USER, RUN, true);
      await store.end(USER, RUN);
      expect(await phaseOf()).toBe('ended');
    });

    it('never exposes turnIdle itself', async () => {
      await store.setTurnIdle(USER, RUN, true);
      const text = await (await get(RUN)).text();
      expect(text).not.toContain('turnIdle');
    });
  });

  it('reports a stopped run as ended', async () => {
    await store.end(USER, RUN);
    const res = await get(RUN);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.ended).toBe(true);
    expect(typeof body.endedAt).toBe('string');
    expect(Number.isNaN(Date.parse(body.endedAt))).toBe(false);
  });

  it('reports no pending questions for an ended run, progress unchanged', async () => {
    // A stream loop may still persist questions after Stop; an ended run
    // takes no answers, so the panel must never be shown them.
    await store.end(USER, RUN);
    expect((await store.get(USER, RUN))?.pendingRequests).toEqual(PENDING);
    const body = await (await get(RUN)).json();
    expect(body.ended).toBe(true);
    expect(body.pending).toEqual([]);
    expect(body.progress).toEqual(PROGRESS);
  });

  it.each([
    ["another user's relay run", OTHER_RUN],
    ['a web (non-relay) run', WEB_RUN],
    ['an unknown run', UNKNOWN_RUN],
  ])('returns a bodyless 404 for %s', async (_n, runId) => {
    await expectBodylessNoStore404(await get(runId));
  });

  it.each(['not-a-uuid', '../x', `${RUN}x`, ''])(
    'returns 400 for malformed id %j before the store',
    async (runId) => {
      const spy = vi.spyOn(store, 'get');
      const res = await get(runId);
      expect(res.status).toBe(400);
      expect(res.headers.get('Cache-Control')).toBe('no-store');
      expect(spy).not.toHaveBeenCalled();
    },
  );

  it.each([undefined, '0'])(
    'returns a bodyless 404 before auth when the flag is %j',
    async (flag) => {
      vi.stubEnv('RELAY_RUNS_ENABLED', flag);
      await expectBodylessNoStore404(await get(RUN));
      await expectBodylessNoStore404(await get('not-a-uuid'));
      expect(authMock).not.toHaveBeenCalled();
    },
  );

  it('returns 401 without a session user', async () => {
    authMock.mockResolvedValue(null);
    const res = await get(RUN);
    expect(res.status).toBe(401);
    expect(res.headers.get('Cache-Control')).toBe('no-store');
  });

  it('returns 403 for a guest before the store', async () => {
    authMock.mockResolvedValue({
      user: { id: USER, email: 'guest-0a1b2c3d@preview.local' },
    });
    const spy = vi.spyOn(store, 'get');
    const res = await get(RUN);
    expect(res.status).toBe(403);
    expect(res.headers.get('Cache-Control')).toBe('no-store');
    expect(spy).not.toHaveBeenCalled();
  });

  describe('with RELAY_EXTENSION_ORIGIN set', () => {
    beforeEach(() => vi.stubEnv('RELAY_EXTENSION_ORIGIN', EXT));

    it('refuses a foreign Origin with 403 before auth or the store', async () => {
      const spy = vi.spyOn(store, 'get');
      const res = await get(RUN, 'https://evil.example');
      expect(res.status).toBe(403);
      expect(res.headers.get('Cache-Control')).toBe('no-store');
      expect(authMock).not.toHaveBeenCalled();
      expect(spy).not.toHaveBeenCalled();
    });

    it.each([undefined, EXT])('serves Origin %j', async (origin) => {
      expect((await get(RUN, origin)).status).toBe(200);
    });
  });

  it('returns a value-free no-store 500 when the store throws', async () => {
    vi.spyOn(store, 'get').mockRejectedValue(new Error(`db ${SESSION}`));
    const res = await get(RUN);
    expect(res.status).toBe(500);
    expect(res.headers.get('Cache-Control')).toBe('no-store');
    expect(await res.text()).not.toContain(SESSION);
  });
});
