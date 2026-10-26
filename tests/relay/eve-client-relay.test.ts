import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { createEveSession } from '@/lib/ai/eve/eve-client';

const relay = {
  url: 'wss://relay.example',
  runId: '0b6f7c4e-5d1a-4c55-9a53-3f7d2f0c9a11',
};

describe('createEveSession with relay credentials', () => {
  const fetchMock = vi.fn();
  beforeEach(() => {
    fetchMock.mockResolvedValue(
      new Response('{}', {
        status: 200,
        headers: { 'x-eve-session-id': 's1' },
      }),
    );
    vi.stubGlobal('fetch', fetchMock);
  });
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.unstubAllEnvs();
    fetchMock.mockReset();
  });

  it('throws a credential-free error without EVE_INTERNAL_SECRET', async () => {
    vi.stubEnv('EVE_INTERNAL_SECRET', '');
    const error = await createEveSession('hi', undefined, relay).catch(
      (e) => e,
    );
    expect(error).toBeInstanceOf(Error);
    expect(error.message).toBe('relay sessions require EVE_INTERNAL_SECRET');
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('sends the secret and relay headers, never in the body', async () => {
    vi.stubEnv('EVE_INTERNAL_SECRET', 'sekret');
    await createEveSession('hi', undefined, relay);
    const init = fetchMock.mock.calls[0][1] as RequestInit;
    expect(init.headers).toMatchObject({
      'x-eve-internal-secret': 'sekret',
      'x-eve-relay-url': relay.url,
      'x-eve-relay-run': relay.runId,
    });
    expect(init.headers).not.toHaveProperty('x-eve-relay-token');
    expect(init.body).toBe(JSON.stringify({ message: 'hi' }));
  });

  it('sends no secret when there are no relay credentials', async () => {
    vi.stubEnv('EVE_INTERNAL_SECRET', 'sekret');
    await createEveSession('hi', 'model-x');
    const init = fetchMock.mock.calls[0][1] as RequestInit;
    expect(init.headers).not.toHaveProperty('x-eve-internal-secret');
  });
});
