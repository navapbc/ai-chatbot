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
