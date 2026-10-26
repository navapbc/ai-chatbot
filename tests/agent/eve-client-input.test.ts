import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { continueEveSession } from '@/lib/ai/eve/eve-client';

describe('continueEveSession input forms', () => {
  const fetchMock = vi.fn();
  beforeEach(() => {
    fetchMock.mockResolvedValue(
      new Response(JSON.stringify({ continuationToken: 'tok-1' }), {
        status: 200,
      }),
    );
    vi.stubGlobal('fetch', fetchMock);
  });
  afterEach(() => {
    vi.unstubAllGlobals();
    fetchMock.mockReset();
  });

  const sentBody = () => (fetchMock.mock.calls[0][1] as RequestInit).body;

  it('keeps the string form as a plain message', async () => {
    const r = await continueEveSession('s1', 'tok-0', 'hello');
    expect(r).toEqual({ continuationToken: 'tok-1' });
    expect(fetchMock.mock.calls[0][0]).toMatch(/\/eve\/v1\/session\/s1$/);
    expect(sentBody()).toBe(
      JSON.stringify({ continuationToken: 'tok-0', message: 'hello' }),
    );
  });

  it('sends inputResponses with no message key', async () => {
    const inputResponses = [
      { requestId: 'call-a', text: 'Ada' },
      { requestId: 'call-b', optionId: 'red' },
    ];
    await continueEveSession('s1', 'tok-0', { inputResponses });
    expect(sentBody()).toBe(
      JSON.stringify({ continuationToken: 'tok-0', inputResponses }),
    );
  });

  it('sends a message given in the object form', async () => {
    await continueEveSession('s1', 'tok-0', { message: 'hi' });
    expect(sentBody()).toBe(
      JSON.stringify({ continuationToken: 'tok-0', message: 'hi' }),
    );
  });
});
