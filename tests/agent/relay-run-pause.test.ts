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
const pause = await import('@/app/(chat)/api/relay/runs/[runId]/pause/route');

const USER = 'user-1';
const RUN = '11111111-1111-4111-8111-111111111111';
const WEB_RUN = '22222222-2222-4222-8222-222222222222';
const OTHER_RUN = '33333333-3333-4333-8333-333333333333';
const UNKNOWN_RUN = '44444444-4444-4444-8444-444444444444';
const ORIGIN = 'https://apply.example.gov';
const SESSION = 'wrun_SECRETSESSION01';
const EXT = `chrome-extension://${'a'.repeat(32)}`;

const postPause = (runId: string, origin: string | null = EXT) =>
  pause.POST(
    new Request(`http://localhost/api/relay/runs/${runId}/pause`, {
      method: 'POST',
      headers: origin === null ? {} : { Origin: origin },
    }),
    { params: Promise.resolve({ runId }) },
  );

let store: ReturnType<typeof createMemoryRunStore>;

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
    continuationToken: 'tok',
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
  vi.spyOn(console, 'error').mockImplementation(() => {});
});

afterEach(() => {
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

describe('POST /api/relay/runs/[runId]/pause', () => {
  it('cancels the Eve turn and leaves the run live', async () => {
    const res = await postPause(RUN);
    expect(res.status).toBe(200);
    expect(res.headers.get('Cache-Control')).toBe('no-store');
    expect(await res.json()).toEqual({ paused: true });
    expect(eve.cancelEveTurn).toHaveBeenCalledWith(SESSION);
    expect((await store.get(USER, RUN))?.endedAt).toBeNull();
  });

  it.each([
    ['an unknown run', UNKNOWN_RUN],
    ["another user's run", OTHER_RUN],
    ['a web chat', WEB_RUN],
  ])('answers a bodyless 404 for %s and cancels nothing', async (_n, id) => {
    const res = await postPause(id);
    expect(res.status).toBe(404);
    expect(await res.text()).toBe('');
    expect(eve.cancelEveTurn).not.toHaveBeenCalled();
  });

  it('refuses an ended run with run_ended and cancels nothing', async () => {
    await store.end(USER, RUN);
    const res = await postPause(RUN);
    expect(res.status).toBe(400);
    expect(JSON.stringify(await res.json())).toContain('run_ended');
    expect(eve.cancelEveTurn).not.toHaveBeenCalled();
  });

  it('answers 500 without values when the cancel fails, run still live', async () => {
    eve.cancelEveTurn.mockRejectedValue(new Error(`boom ${SESSION}`));
    const res = await postPause(RUN);
    expect(res.status).toBe(500);
    expect(await res.text()).not.toContain(SESSION);
    expect((await store.get(USER, RUN))?.endedAt).toBeNull();
  });

  it('requires the extension origin and a signed-in user', async () => {
    expect((await postPause(RUN, null)).status).toBeGreaterThanOrEqual(400);
    expect(eve.cancelEveTurn).not.toHaveBeenCalled();
    authMock.mockResolvedValue(null);
    expect((await postPause(RUN)).status).toBeGreaterThanOrEqual(400);
    expect(eve.cancelEveTurn).not.toHaveBeenCalled();
  });
});
