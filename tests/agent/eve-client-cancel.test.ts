import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cancelEveTurn } from '@/lib/ai/eve/eve-client';

const SESSION = 'wrun_01JABCDEF-ghij_klm';

describe('cancelEveTurn', () => {
  const fetchMock = vi.fn();
  beforeEach(() => vi.stubGlobal('fetch', fetchMock));
  afterEach(() => {
    vi.unstubAllGlobals();
    fetchMock.mockReset();
  });

  const respond = (status: number, body: unknown) =>
    fetchMock.mockResolvedValueOnce(
      new Response(typeof body === 'string' ? body : JSON.stringify(body), {
        status,
      }),
    );

  it.each(['accepted', 'no_active_turn'] as const)(
    'posts to the session cancel route and returns %s',
    async (status) => {
      respond(202, { ok: true, sessionId: SESSION, status });
      await expect(cancelEveTurn(SESSION)).resolves.toBe(status);
      expect(fetchMock).toHaveBeenCalledTimes(1);
      const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
      expect(url).toMatch(new RegExp(`/eve/v1/session/${SESSION}/cancel$`));
      expect(init.method).toBe('POST');
      expect(init.body).toBe('{}');
      // Same conventions as a follow-up send: no relay grant headers.
      const headers = new Headers(init.headers);
      expect(headers.get('x-eve-internal-secret')).toBeNull();
      expect(headers.get('x-eve-relay-run')).toBeNull();
    },
  );

  it.each([400, 401, 404, 500])(
    'throws on HTTP %d without the session id or body in the message',
    async (status) => {
      respond(status, `nope ${SESSION}`);
      const err = await cancelEveTurn(SESSION).catch((e: unknown) => e);
      expect(err).toBeInstanceOf(Error);
      expect((err as Error).message).not.toContain(SESSION);
      expect((err as Error).message).not.toContain('nope');
    },
  );

  it('gives up after 5 seconds when Eve never answers', async () => {
    // Stands in for the 5 s timer, so the test controls when it fires.
    const timer = new AbortController();
    const timeout = vi
      .spyOn(AbortSignal, 'timeout')
      .mockReturnValue(timer.signal);
    // A fetch that only settles when its signal aborts.
    fetchMock.mockImplementationOnce(
      (_url: string, init: RequestInit) =>
        new Promise((_resolve, reject) => {
          init.signal?.addEventListener('abort', () =>
            reject(new Error(`aborted ${SESSION}`)),
          );
        }),
    );
    const pending = cancelEveTurn(SESSION).catch((e: unknown) => e);
    expect(timeout).toHaveBeenCalledWith(5000);
    expect((fetchMock.mock.calls[0][1] as RequestInit).signal).toBe(
      timer.signal,
    );
    timer.abort(new DOMException('timed out', 'TimeoutError'));
    const err = await pending;
    expect(err).toBeInstanceOf(Error);
    expect((err as Error).message).not.toContain(SESSION);
    timeout.mockRestore();
  });

  it('throws on a 2xx with an unknown status', async () => {
    respond(202, { ok: true, status: 'something_else' });
    await expect(cancelEveTurn(SESSION)).rejects.toThrow();
  });

  it('throws on a 2xx with a non-JSON body', async () => {
    respond(200, 'not json');
    await expect(cancelEveTurn(SESSION)).rejects.toThrow();
  });

  it.each(['', '../x', 'a/b', 'a?b=1', 'a#b', 'a b', 'a%2Fb', 'x'.repeat(201)])(
    'rejects session id %j without a fetch',
    async (id) => {
      await expect(cancelEveTurn(id)).rejects.toThrow();
      expect(fetchMock).not.toHaveBeenCalled();
    },
  );
});
