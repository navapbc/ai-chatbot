import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const authMock = vi.fn();
vi.mock('@/app/(auth)/auth', () => ({ auth: () => authMock() }));

const { PARTICIPANTS } = await import('@/lib/data/participants');
const { createStaticParticipantSource, setParticipantSourceForTests } =
  await import('@/lib/data/participant-source');
const list = await import('@/app/(chat)/api/participants/route');
const detail = await import('@/app/(chat)/api/participants/[recordId]/route');

const searchRequest = (body: unknown, raw = false) =>
  new Request('http://localhost/api/participants', {
    method: 'POST',
    body: raw ? (body as string) : JSON.stringify(body),
  });
const detailCall = (recordId: string) =>
  detail.GET(new Request(`http://localhost/api/participants/${recordId}`), {
    params: Promise.resolve({ recordId }),
  });

describe('participant routes', () => {
  beforeEach(() => {
    vi.stubEnv('RELAY_RUNS_ENABLED', '1');
    authMock.mockReset();
    authMock.mockResolvedValue({
      user: { id: 'u1', email: 'person@example.gov' },
    });
    setParticipantSourceForTests(createStaticParticipantSource(PARTICIPANTS));
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    setParticipantSourceForTests(undefined);
  });

  it('passes guard refusals through with no-store', async () => {
    vi.stubEnv('RELAY_RUNS_ENABLED', undefined);
    const res = await list.POST(searchRequest({ q: 'a' }));
    expect(res.status).toBe(404);
    expect(res.headers.get('Cache-Control')).toBe('no-store');

    vi.stubEnv('RELAY_RUNS_ENABLED', '1');
    authMock.mockResolvedValue(null);
    const unauth = await detailCall('339637');
    expect(unauth.status).toBe(401);
    expect(unauth.headers.get('Cache-Control')).toBe('no-store');
  });

  it('searches and returns summaries only', async () => {
    const res = await list.POST(searchRequest({ q: 'sawyer' }));
    expect(res.status).toBe(200);
    expect(res.headers.get('Cache-Control')).toBe('no-store');
    const body = await res.json();
    expect(body.participants.length).toBeGreaterThan(0);
    for (const p of body.participants) {
      expect(Object.keys(p).sort()).toEqual(['county', 'name', 'recordId']);
    }
  });

  it('has no GET handler, so Next answers 405 and no term rides a URL', () => {
    expect((list as Record<string, unknown>).GET).toBeUndefined();
  });

  it('answers an empty list for a missing q', async () => {
    const res = await list.POST(searchRequest({}));
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ participants: [] });
  });

  it('rejects q over 100 characters', async () => {
    const res = await list.POST(searchRequest({ q: 'a'.repeat(101) }));
    expect(res.status).toBe(400);
    expect(res.headers.get('Cache-Control')).toBe('no-store');
    const ok = await list.POST(searchRequest({ q: 'a'.repeat(100) }));
    expect(ok.status).toBe(200);
  });

  it('rejects bad JSON, a non-object, a non-string q and an oversized body', async () => {
    for (const raw of ['{', '[]', 'null', '"a"', '{"q":5}']) {
      const res = await list.POST(searchRequest(raw, true));
      expect(res.status).toBe(400);
      expect(res.headers.get('Cache-Control')).toBe('no-store');
    }
    const big = await list.POST(
      searchRequest({ q: 'a', pad: 'x'.repeat(5000) }),
    );
    expect(big.status).toBe(400);
  });

  it('rejects a non-integer limit and clamps a large one', async () => {
    for (const limit of ['abc', '5', -1, 1.5, null]) {
      const bad = await list.POST(searchRequest({ q: 'a', limit }));
      expect(bad.status).toBe(400);
    }

    const search = vi.fn().mockResolvedValue([]);
    setParticipantSourceForTests({ search, get: vi.fn() });
    await list.POST(searchRequest({ q: 'a', limit: 500 }));
    expect(search).toHaveBeenLastCalledWith('a', 20);
    await list.POST(searchRequest({ q: ' a ', limit: 3 }));
    expect(search).toHaveBeenLastCalledWith('a', 3);
    await list.POST(searchRequest({ q: 'a' }));
    expect(search).toHaveBeenLastCalledWith('a', 10);
  });

  it('refuses a guest', async () => {
    authMock.mockResolvedValue({
      user: { id: 'g', email: 'guest-0123abcd@preview.local' },
    });
    const res = await list.POST(searchRequest({ q: 'a' }));
    expect(res.status).toBe(403);
  });

  it('returns the three summary keys and leaks nothing else', async () => {
    const res = await detailCall('339637');
    expect(res.status).toBe(200);
    expect(res.headers.get('Cache-Control')).toBe('no-store');
    const text = await res.text();
    expect(Object.keys(JSON.parse(text)).sort()).toEqual([
      'county',
      'name',
      'recordId',
    ]);
    expect(text).not.toContain('A100121');
    expect(text).not.toContain('1954-01-10');
  });

  it('answers 404 with a body for an unknown id', async () => {
    const res = await detailCall('nope');
    expect(res.status).toBe(404);
    expect(res.headers.get('Cache-Control')).toBe('no-store');
    expect(await res.json()).toEqual({
      code: 'not_found:api',
      message: 'Participant not found.',
    });
  });
});
