import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  _resetAgentTokenCache,
  fetchAgentToken,
} from '@/lib/relay/agent-token';

const RUN = '0b6f7c4e-5d1a-4c55-9a53-3f7d2f0c9a11';
const TOKEN = 'AGENT-TOKEN-do-not-leak-abcdef0123456789';
const SECRET = 'internal-secret-do-not-leak';
const BASE = 'http://127.0.0.1:3999';

const ok = (token = TOKEN, expiresAt = 2_000) =>
  new Response(JSON.stringify({ token, expiresAt }), { status: 200 });

describe('fetchAgentToken', () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  beforeEach(() => {
    _resetAgentTokenCache();
    vi.stubEnv('EVE_INTERNAL_SECRET', SECRET);
  });

  it('returns the token from a 200 and posts the run id with the secret', async () => {
    const fetchImpl = vi.fn().mockResolvedValue(ok());
    const result = await fetchAgentToken(RUN, {
      fetchImpl,
      baseUrl: BASE,
      now: () => 1_000,
    });
    expect(result).toEqual({ ok: true, token: TOKEN, expiresAt: 2_000 });
    const [url, init] = fetchImpl.mock.calls[0] as [string, RequestInit];
    expect(url).toBe(`${BASE}/api/relay/agent-token`);
    expect(init.method).toBe('POST');
    expect(init.headers).toMatchObject({ 'x-eve-internal-secret': SECRET });
    expect(init.body).toBe(JSON.stringify({ runId: RUN }));
  });

  it('caches per run id until 60 s before expiry, then refetches', async () => {
    let t = 1_000;
    const fetchImpl = vi
      .fn()
      .mockImplementation(async () => ok(TOKEN, t + 300));
    const opts = { fetchImpl, baseUrl: BASE, now: () => t };
    await fetchAgentToken(RUN, opts);
    await fetchAgentToken(RUN, opts);
    expect(fetchImpl).toHaveBeenCalledTimes(1);
    await fetchAgentToken('11111111-1111-4111-8111-111111111111', opts);
    expect(fetchImpl).toHaveBeenCalledTimes(2);
    t = 1_000 + 241; // inside the 60 s window before expiry (1_300)
    await fetchAgentToken(RUN, opts);
    expect(fetchImpl).toHaveBeenCalledTimes(3);
  });

  it('returns fixed errors without credentials for each failure', async () => {
    const cases: Array<[string, () => Promise<Response>]> = [
      [
        'The relay token request was refused.',
        async () => new Response('x', { status: 404 }),
      ],
      [
        'The relay token service is unavailable.',
        async () => {
          throw new Error(`boom ${SECRET} ${TOKEN}`);
        },
      ],
      [
        'The relay token response was invalid.',
        async () => new Response(JSON.stringify({ token: 5 }), { status: 200 }),
      ],
      [
        'The relay token response was invalid.',
        async () => new Response('not json', { status: 200 }),
      ],
    ];
    for (const [expected, impl] of cases) {
      _resetAgentTokenCache();
      const result = await fetchAgentToken(RUN, {
        fetchImpl: vi.fn().mockImplementation(impl),
        baseUrl: BASE,
        now: () => 1_000,
      });
      expect(result).toEqual({ ok: false, error: expected });
      expect(JSON.stringify(result)).not.toContain(SECRET);
      expect(JSON.stringify(result)).not.toContain(TOKEN);
    }
  });

  it('fails without a request when there is no internal secret', async () => {
    vi.stubEnv('EVE_INTERNAL_SECRET', '');
    const fetchImpl = vi.fn();
    const result = await fetchAgentToken(RUN, { fetchImpl, baseUrl: BASE });
    expect(result.ok).toBe(false);
    expect(fetchImpl).not.toHaveBeenCalled();
  });

  it('does not cache failures', async () => {
    const fetchImpl = vi
      .fn()
      .mockResolvedValueOnce(new Response('x', { status: 500 }))
      .mockResolvedValueOnce(ok());
    const opts = { fetchImpl, baseUrl: BASE, now: () => 1_000 };
    expect((await fetchAgentToken(RUN, opts)).ok).toBe(false);
    expect((await fetchAgentToken(RUN, opts)).ok).toBe(true);
  });

  it('passes an abort signal, refuses redirects, and the 5 s timeout aborts', async () => {
    vi.useFakeTimers();
    try {
      let signal: AbortSignal | undefined;
      const fetchImpl = vi.fn().mockImplementation(
        (_url: string, init: RequestInit) =>
          new Promise((_resolve, reject) => {
            signal = init.signal as AbortSignal;
            signal.addEventListener('abort', () => reject(signal?.reason));
          }),
      );
      const pending = fetchAgentToken(RUN, { fetchImpl, baseUrl: BASE });
      expect(signal).toBeInstanceOf(AbortSignal);
      expect((fetchImpl.mock.calls[0][1] as RequestInit).redirect).toBe(
        'error',
      );
      expect(signal?.aborted).toBe(false);
      await vi.advanceTimersByTimeAsync(5_000);
      expect(signal?.aborted).toBe(true);
      await expect(pending).resolves.toEqual({
        ok: false,
        error: 'The relay token service is unavailable.',
      });
    } finally {
      vi.useRealTimers();
    }
  });

  it.each([
    ['a non-string token', { token: 5, expiresAt: 2_000 }],
    ['an empty token', { token: '', expiresAt: 2_000 }],
    ['a missing token', { expiresAt: 2_000 }],
    ['a string expiresAt', { token: TOKEN, expiresAt: '2000' }],
    ['a missing expiresAt', { token: TOKEN }],
    ['a null expiresAt', { token: TOKEN, expiresAt: null }],
    ['a millisecond expiresAt', { token: TOKEN, expiresAt: 1.8e12 }],
    ['a non-object body', 'nope'],
    ['a null body', null],
  ])(
    'rejects %s as an invalid response and does not cache it',
    async (_n, body) => {
      const fetchImpl = vi
        .fn()
        .mockImplementation(
          async () => new Response(JSON.stringify(body), { status: 200 }),
        );
      const opts = { fetchImpl, baseUrl: BASE, now: () => 1_000 };
      const result = await fetchAgentToken(RUN, opts);
      expect(result).toEqual({
        ok: false,
        error: 'The relay token response was invalid.',
      });
      expect(JSON.stringify(result)).not.toContain(TOKEN);
      await fetchAgentToken(RUN, opts);
      expect(fetchImpl).toHaveBeenCalledTimes(2);
    },
  );

  it('treats a non-finite expiresAt as invalid', async () => {
    // JSON cannot carry Infinity; a raw body that parses to a huge number is
    // the closest wire form, and a stubbed json() covers NaN.
    const res = new Response('{}', { status: 200 });
    vi.spyOn(res, 'json').mockResolvedValue({
      token: TOKEN,
      expiresAt: Number.NaN,
    });
    const result = await fetchAgentToken(RUN, {
      fetchImpl: vi.fn().mockResolvedValue(res),
      baseUrl: BASE,
    });
    expect(result).toEqual({
      ok: false,
      error: 'The relay token response was invalid.',
    });
  });

  it('survives a fetchImpl that throws synchronously, without caching', async () => {
    const fetchImpl = vi
      .fn()
      .mockImplementationOnce(() => {
        throw new Error(`sync ${SECRET} ${TOKEN}`);
      })
      .mockImplementationOnce(async () => ok());
    const opts = { fetchImpl, baseUrl: BASE, now: () => 1_000 };
    const first = await fetchAgentToken(RUN, opts);
    expect(first).toEqual({
      ok: false,
      error: 'The relay token service is unavailable.',
    });
    expect(JSON.stringify(first)).not.toContain(SECRET);
    expect((await fetchAgentToken(RUN, opts)).ok).toBe(true);
  });
});
