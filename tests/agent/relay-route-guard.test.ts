import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const authMock = vi.fn();
vi.mock('@/app/(auth)/auth', () => ({ auth: () => authMock() }));

const { guardRelayRoute, isGuestEmail } = await import(
  '@/lib/relay/route-guard'
);

describe('isGuestEmail', () => {
  it.each([
    ['guest-1a2b3c4d@preview.local', true],
    ['Guest-1a2b3c4d@preview.local', true],
    ['guest-1a2b3c4d@example.com', false],
    ['someone@preview.local', false],
    [null, false],
    [undefined, false],
    ['', false],
  ])('%j -> %s', (email, expected) => {
    expect(isGuestEmail(email)).toBe(expected);
  });
});

describe('guardRelayRoute', () => {
  beforeEach(() => {
    vi.stubEnv('RELAY_RUNS_ENABLED', '1');
    authMock.mockReset();
    authMock.mockResolvedValue({
      user: { id: 'u1', email: 'person@example.gov' },
    });
  });

  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it.each([undefined, '0', 'true', ''])(
    'answers a bodyless 404 without calling auth when the flag is %j',
    async (flag) => {
      vi.stubEnv('RELAY_RUNS_ENABLED', flag);
      const result = await guardRelayRoute();
      expect(result.ok).toBe(false);
      if (result.ok) return;
      expect(result.response.status).toBe(404);
      expect(result.response.body).toBeNull();
      expect(authMock).not.toHaveBeenCalled();
    },
  );

  it('answers 401 without a session user id', async () => {
    authMock.mockResolvedValue(null);
    const result = await guardRelayRoute();
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.response.status).toBe(401);

    authMock.mockResolvedValue({ user: {} });
    const second = await guardRelayRoute();
    expect(second.ok).toBe(false);
    if (second.ok) return;
    expect(second.response.status).toBe(401);
  });

  it('answers 403 for a guest', async () => {
    authMock.mockResolvedValue({
      user: { id: 'g1', email: 'guest-1a2b3c4d@preview.local' },
    });
    const result = await guardRelayRoute();
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.response.status).toBe(403);
  });

  it('returns the user id for a regular session', async () => {
    expect(await guardRelayRoute()).toEqual({ ok: true, userId: 'u1' });
  });
});

describe('guardRelayRoute extension origin', () => {
  const ORIGIN = `chrome-extension://${'a'.repeat(32)}`;
  const req = (origin?: string) =>
    new Request('http://localhost/x', {
      headers: origin === undefined ? {} : { Origin: origin },
    });

  beforeEach(() => {
    vi.stubEnv('RELAY_RUNS_ENABLED', '1');
    vi.stubEnv('RELAY_EXTENSION_ORIGIN', ORIGIN);
    authMock.mockReset();
    authMock.mockResolvedValue({
      user: { id: 'u1', email: 'person@example.gov' },
    });
  });

  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it('checks the flag before the origin', async () => {
    vi.stubEnv('RELAY_RUNS_ENABLED', undefined);
    const result = await guardRelayRoute(req('https://evil.example'));
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.response.status).toBe(404);
    expect(result.response.body).toBeNull();
  });

  it('answers 403 to a foreign origin before auth, even unauthenticated', async () => {
    authMock.mockResolvedValue(null);
    const result = await guardRelayRoute(req('https://evil.example'));
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.response.status).toBe(403);
    expect(authMock).not.toHaveBeenCalled();
    expect(await result.response.text()).not.toContain(ORIGIN);
  });

  it('refuses a missing origin only when required', async () => {
    const required = await guardRelayRoute(req(), { requireOrigin: true });
    expect(required.ok).toBe(false);
    if (required.ok) return;
    expect(required.response.status).toBe(403);
    expect(await guardRelayRoute(req())).toEqual({ ok: true, userId: 'u1' });
  });

  it('passes the exact origin on to auth', async () => {
    expect(await guardRelayRoute(req(ORIGIN), { requireOrigin: true })).toEqual(
      {
        ok: true,
        userId: 'u1',
      },
    );
    authMock.mockResolvedValue(null);
    const result = await guardRelayRoute(req(ORIGIN));
    if (result.ok) throw new Error('expected a refusal');
    expect(result.response.status).toBe(401);
  });

  it('ignores the env when no request is passed or the env is unset', async () => {
    expect(await guardRelayRoute()).toEqual({ ok: true, userId: 'u1' });
    vi.stubEnv('RELAY_EXTENSION_ORIGIN', undefined);
    expect(
      await guardRelayRoute(req('https://evil.example'), {
        requireOrigin: true,
      }),
    ).toEqual({ ok: true, userId: 'u1' });
  });
});
