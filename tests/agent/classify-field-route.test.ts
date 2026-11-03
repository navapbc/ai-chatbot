import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('server-only', () => ({}));
const authMock = vi.hoisted(() => vi.fn());
vi.mock('@/app/(auth)/auth', () => ({ auth: authMock }));

const jev = vi.hoisted(() => ({
  askJev: vi.fn(),
  isJevEnabled: vi.fn(),
}));
vi.mock('@/lib/jev/client', () => jev);

const { createMemoryRunStore, setRunStoreForTests } = await import(
  '@/lib/ai/eve/run-store'
);
const { POST } = await import('@/app/(chat)/api/relay/classify-field/route');

const USER = 'user-1';
const RUN = '11111111-1111-4111-8111-111111111111';
const WEB_RUN = '22222222-2222-4222-8222-222222222222';
const OTHER_RUN = '33333333-3333-4333-8333-333333333333';

const noul = (n: number) => ({
  ok: true,
  answers: { sensitive: { type: 'noul', noul: n } },
  model: 'm',
  usage: { inputTokens: 1, outputTokens: 1 },
});

const call = (body: unknown, raw?: string) =>
  POST(
    new Request('http://localhost/api/relay/classify-field', {
      method: 'POST',
      body: raw ?? JSON.stringify(body),
    }),
  );

const fieldsFor = (n: number) =>
  Array.from({ length: n }, (_, i) => ({ label: `Field ${i}` }));

beforeEach(async () => {
  vi.clearAllMocks();
  vi.stubEnv('RELAY_RUNS_ENABLED', '1');
  authMock.mockResolvedValue({ user: { id: USER, email: 'a@example.gov' } });
  jev.isJevEnabled.mockReturnValue(true);
  jev.askJev.mockResolvedValue(noul(0));
  const store = createMemoryRunStore();
  setRunStoreForTests(store);
  await store.create(USER, RUN, {
    eveSessionId: 's',
    continuationToken: 't',
    relayOrigin: 'https://example.gov',
  });
  await store.create(USER, WEB_RUN, {
    eveSessionId: 's2',
    continuationToken: 't2',
  });
  await store.create('user-2', OTHER_RUN, {
    eveSessionId: 's3',
    continuationToken: 't3',
    relayOrigin: 'https://example.gov',
  });
});

describe('POST /api/relay/classify-field', () => {
  it('returns one verdict per field in order, skipping Jev for deterministic hits', async () => {
    jev.askJev.mockImplementation(async ({ state }) =>
      noul(state.label === 'Mystery' ? 0.99 : 0),
    );
    const res = await call({
      runId: RUN,
      fields: [
        { label: 'Mystery' },
        { label: 'Favorite color' },
        { label: 'Social Security Number' },
      ],
    });
    expect(res.status).toBe(200);
    expect(res.headers.get('Cache-Control')).toBe('no-store');
    expect(await res.json()).toEqual({ sensitive: [true, false, true] });
    expect(jev.askJev).toHaveBeenCalledTimes(2);
  });

  it('sends only sanitized descriptor keys to Jev', async () => {
    await call({
      runId: RUN,
      fields: [
        { label: 'Reference', value: '123-45-6789', url: 'https://x.test' },
      ],
    });
    const args = jev.askJev.mock.calls[0][0];
    expect(args.state).toEqual({ label: 'Reference' });
    expect(args.feature).toBe('sensitive-field');
    expect(JSON.stringify(args)).not.toContain('6789');
  });

  it('turns digits into 9 before the call', async () => {
    await call({ runId: RUN, fields: [{ label: 'Ref 4821' }] });
    expect(jev.askJev.mock.calls[0][0].state).toEqual({ label: 'Ref 9999' });
  });

  it('answers null (no verdict) when Jev reports failure', async () => {
    jev.askJev.mockResolvedValue({ ok: false, reason: 'timeout' });
    const res = await call({ runId: RUN, fields: fieldsFor(2) });
    expect((await res.json()).sensitive).toEqual([null, null]);
  });

  it('answers null when the noul is not a finite number', async () => {
    for (const bad of [
      Number.NaN,
      undefined,
      '0.1',
      Number.POSITIVE_INFINITY,
    ]) {
      jev.askJev.mockResolvedValue({
        ok: true,
        answers: { sensitive: { noul: bad } },
      });
      const res = await call({ runId: RUN, fields: fieldsFor(1) });
      expect((await res.json()).sensitive).toEqual([null]);
    }
  });

  it('answers null when the answer is missing', async () => {
    jev.askJev.mockResolvedValue({ ok: true, answers: {} });
    const res = await call({ runId: RUN, fields: fieldsFor(1) });
    expect((await res.json()).sensitive).toEqual([null]);
  });

  it('answers null when askJev throws', async () => {
    jev.askJev.mockRejectedValue(new Error('boom'));
    const res = await call({ runId: RUN, fields: fieldsFor(2) });
    expect((await res.json()).sensitive).toEqual([null, null]);
  });

  it('answers true at the threshold and false just under it', async () => {
    jev.askJev.mockImplementation(async ({ state }) =>
      noul(state.label === 'Nickname' ? 0.5 : 0.49),
    );
    const res = await call({
      runId: RUN,
      fields: [{ label: 'Nickname' }, { label: 'Favorite color' }],
    });
    expect((await res.json()).sensitive).toEqual([true, false]);
  });

  it('answers null for an empty descriptor without calling Jev', async () => {
    const res = await call({ runId: RUN, fields: [{ value: 'x' }] });
    expect((await res.json()).sensitive).toEqual([null]);
    expect(jev.askJev).not.toHaveBeenCalled();
  });

  it('with the Jev feature off: true for deterministic hits, null otherwise', async () => {
    jev.isJevEnabled.mockReturnValue(false);
    const res = await call({
      runId: RUN,
      fields: [{ label: 'Social Security Number' }, ...fieldsFor(2)],
    });
    expect((await res.json()).sensitive).toEqual([true, null, null]);
    expect(jev.askJev).not.toHaveBeenCalled();
  });

  it.each([0, 51])('refuses %i fields with 400', async (n) => {
    const res = await call({ runId: RUN, fields: fieldsFor(n) });
    expect(res.status).toBe(400);
    expect(res.headers.get('Cache-Control')).toBe('no-store');
  });

  it('accepts exactly 50 fields', async () => {
    const res = await call({ runId: RUN, fields: fieldsFor(50) });
    expect(res.status).toBe(200);
    expect((await res.json()).sensitive).toHaveLength(50);
  });

  it('refuses a non-object field with 400', async () => {
    for (const bad of ['SSN', null, 5, ['x']]) {
      const res = await call({ runId: RUN, fields: [{ label: 'a' }, bad] });
      expect(res.status).toBe(400);
    }
  });

  it('refuses malformed bodies and bad run ids with 400', async () => {
    expect((await call(null, 'not json')).status).toBe(400);
    expect((await call({ runId: 'nope', fields: fieldsFor(1) })).status).toBe(
      400,
    );
    expect((await call({ runId: RUN })).status).toBe(400);
  });

  it('refuses an oversized body with 400', async () => {
    const res = await call({
      runId: RUN,
      fields: [{ label: 'a'.repeat(70 * 1024) }],
    });
    expect(res.status).toBe(400);
    expect(res.headers.get('Cache-Control')).toBe('no-store');
    expect(jev.askJev).not.toHaveBeenCalled();
  });

  it.each([
    ['another user run', OTHER_RUN],
    ['a web run', WEB_RUN],
    ['an unknown run', '44444444-4444-4444-8444-444444444444'],
  ])('answers a bodyless 404 for %s', async (_n, runId) => {
    const res = await call({ runId, fields: fieldsFor(1) });
    expect(res.status).toBe(404);
    expect(await res.text()).toBe('');
    expect(res.headers.get('Cache-Control')).toBe('no-store');
    expect(jev.askJev).not.toHaveBeenCalled();
  });

  it('answers a bodyless no-store 404 when the flag is off', async () => {
    vi.stubEnv('RELAY_RUNS_ENABLED', undefined);
    const res = await call({ runId: RUN, fields: fieldsFor(1) });
    expect(res.status).toBe(404);
    expect(await res.text()).toBe('');
    expect(res.headers.get('Cache-Control')).toBe('no-store');
  });

  it('refuses unauthenticated and guest callers', async () => {
    authMock.mockResolvedValue(null);
    const unauth = await call({ runId: RUN, fields: fieldsFor(1) });
    expect(unauth.status).toBe(401);
    expect(unauth.headers.get('Cache-Control')).toBe('no-store');
    authMock.mockResolvedValue({
      user: { id: USER, email: 'guest-0a1b2c3d@preview.local' },
    });
    const guest = await call({ runId: RUN, fields: fieldsFor(1) });
    expect(guest.status).toBe(403);
    expect(guest.headers.get('Cache-Control')).toBe('no-store');
  });

  it('returns a bodyless no-store 404 for an ended run without asking Jev', async () => {
    const { getRunStore } = await import('@/lib/ai/eve/run-store');
    expect(await getRunStore().end(USER, RUN)).toBe(true);
    const res = await call({ runId: RUN, fields: [{ label: 'Mystery' }] });
    expect(res.status).toBe(404);
    expect(res.headers.get('Cache-Control')).toBe('no-store');
    expect(await res.text()).toBe('');
    expect(jev.askJev).not.toHaveBeenCalled();
  });

  it('still classifies for a live run', async () => {
    const res = await call({ runId: RUN, fields: [{ label: 'Mystery' }] });
    expect(res.status).toBe(200);
    expect(jev.askJev).toHaveBeenCalledTimes(1);
  });

  it('never runs more than 8 Jev calls at once', async () => {
    let active = 0;
    let peak = 0;
    jev.askJev.mockImplementation(async () => {
      active++;
      peak = Math.max(peak, active);
      await new Promise((r) => setTimeout(r, 5));
      active--;
      return noul(0);
    });
    const res = await call({ runId: RUN, fields: fieldsFor(30) });
    expect(res.status).toBe(200);
    expect(jev.askJev).toHaveBeenCalledTimes(30);
    expect(peak).toBe(8);
  });
});
