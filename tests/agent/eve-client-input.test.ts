import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { continueEveSession, createEveSession } from '@/lib/ai/eve/eve-client';

describe('continueEveSession input forms', () => {
  const fetchMock = vi.fn();
  beforeEach(() => {
    fetchMock.mockResolvedValue(
      new Response(JSON.stringify({ ok: true, sessionId: 's1' }), {
        status: 200,
        headers: { 'x-eve-session-id': 's1' },
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
    // Eve 0.27's continue response carries no token; the session id comes
    // from the `x-eve-session-id` header.
    expect(r).toEqual({ continuationToken: '', sessionId: 's1' });
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

describe('eve-client session id and token checks', () => {
  const fetchMock = vi.fn();
  beforeEach(() => vi.stubGlobal('fetch', fetchMock));
  afterEach(() => {
    vi.unstubAllGlobals();
    fetchMock.mockReset();
  });
  const respond = (body: unknown, sessionId?: string) =>
    fetchMock.mockResolvedValueOnce(
      new Response(JSON.stringify(body), {
        status: 200,
        headers: sessionId ? { 'x-eve-session-id': sessionId } : {},
      }),
    );

  it('returns the session id a continue landed on', async () => {
    respond({ ok: true }, 'wrun_other');
    await expect(continueEveSession('s1', 'tok-0', 'hi')).resolves.toEqual({
      continuationToken: '',
      sessionId: 'wrun_other',
    });
  });

  it('throws when a continue response names no session', async () => {
    respond({ ok: true });
    await expect(continueEveSession('s1', 'tok-0', 'hi')).rejects.toThrow(
      'Eve continue returned no session id',
    );
  });

  it('throws when create returns no session id', async () => {
    respond({ continuationToken: 'eve:tok' });
    await expect(createEveSession('hi')).rejects.toThrow(
      'Eve create returned no session id or token',
    );
  });

  it('throws when create returns no token', async () => {
    respond({ ok: true }, 'wrun_1');
    await expect(createEveSession('hi')).rejects.toThrow(
      'Eve create returned no session id or token',
    );
  });

  it('returns both on a complete create response', async () => {
    respond({ continuationToken: 'eve:tok' }, 'wrun_1');
    await expect(createEveSession('hi')).resolves.toEqual({
      sessionId: 'wrun_1',
      continuationToken: 'eve:tok',
    });
  });
});
