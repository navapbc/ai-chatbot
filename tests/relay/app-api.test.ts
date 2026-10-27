import { afterEach, describe, expect, it, vi } from 'vitest';
import { DESCRIPTOR_KEYS } from '@/lib/jev/questions';
import {
  AppApiError,
  DESCRIPTOR_KEYS as CLIENT_KEYS,
  createAppApi,
} from '../../extension/src/app-api';

const ORIGIN = 'https://app.example.gov';
const RUN = '123e4567-e89b-42d3-a456-426614174000';
const MARKER = 'SECRET-BODY-MARKER-9f2';

type Call = { url: string; init: RequestInit };

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });

const setup = (
  respond: (call: Call, n: number) => Response | Promise<Response>,
) => {
  const calls: Call[] = [];
  const fetchFn = vi.fn(async (url: RequestInfo | URL, init?: RequestInit) => {
    const call = { url: String(url), init: init ?? {} };
    calls.push(call);
    return respond(call, calls.length - 1);
  }) as unknown as typeof fetch;
  return { calls, api: createAppApi({ origin: ORIGIN, fetch: fetchFn }) };
};

const headersOf = (call: Call) => new Headers(call.init.headers as HeadersInit);

const caught = async (p: Promise<unknown>): Promise<AppApiError> => {
  try {
    await p;
  } catch (e) {
    return e as AppApiError;
  }
  throw new Error('expected rejection');
};

afterEach(() => vi.useRealTimers());

describe('request shape', () => {
  it('searchParticipants GETs with an encoded query', async () => {
    const { api, calls } = setup(() => json({ participants: [] }));
    await api.searchParticipants('a b&c=d', undefined);
    expect(calls[0].url).toBe(`${ORIGIN}/api/participants?q=a%20b%26c%3Dd`);
    expect(calls[0].init.method ?? 'GET').toBe('GET');
    expect(calls[0].init.body).toBeUndefined();
  });

  it('every request carries the fixed options and headers', async () => {
    const { api, calls } = setup((c) =>
      c.url.includes('/classify-field')
        ? json({ sensitive: [true] })
        : c.url.endsWith('/api/relay/config')
          ? json({ sites: [], models: [], defaultModel: 'x' })
          : c.url.includes('/participants/')
            ? json({ recordId: 'r', name: 'n', county: 'c' })
            : c.url.includes('/stop')
              ? json({ ended: true, alreadyEnded: false })
              : c.url.endsWith('/token')
                ? json({
                    extensionToken: 't',
                    relayUrl: 'wss://x',
                    expiresAt: 1,
                  })
                : c.url.includes('/runs/')
                  ? json({
                      runId: RUN,
                      origin: 'o',
                      ended: false,
                      endedAt: null,
                      progress: null,
                      pending: [],
                    })
                  : json({ participants: [] }),
    );
    await api.searchParticipants('x');
    await api.getParticipant('rec-1');
    await api.getConfig();
    await api.getRunState(RUN);
    await api.stopRun(RUN);
    await api.getExtensionToken(RUN);
    await api.classifyFields(RUN, [{ label: 'a' }]);
    expect(calls).toHaveLength(7);
    for (const c of calls) {
      expect(c.init.credentials).toBe('include');
      expect(c.init.cache).toBe('no-store');
      expect(c.init.redirect).toBe('error');
      expect(c.init.signal).toBeInstanceOf(AbortSignal);
      const h = headersOf(c);
      expect(h.get('Accept')).toBe('application/json');
      expect(h.has('Authorization')).toBe(false);
      expect(h.has('Cookie')).toBe(false);
      expect(c.url.startsWith(`${ORIGIN}/`)).toBe(true);
    }
  });

  it('routes and methods', async () => {
    const { api, calls } = setup((c) =>
      c.url.endsWith('/stop')
        ? json({ ended: true, alreadyEnded: false })
        : c.url.endsWith('/token')
          ? json({ extensionToken: 't', relayUrl: 'wss://x', expiresAt: 1 })
          : c.url.endsWith('/api/relay/config')
            ? json({ sites: [], models: [], defaultModel: 'x' })
            : c.url.includes('/participants/')
              ? json({ recordId: 'r', name: 'n', county: 'c' })
              : json({
                  runId: RUN,
                  origin: 'o',
                  ended: false,
                  endedAt: null,
                  progress: null,
                  pending: [],
                }),
    );
    await api.getParticipant('rec 1');
    await api.getConfig();
    await api.getRunState(RUN);
    await api.stopRun(RUN);
    await api.getExtensionToken(RUN);
    expect(calls.map((c) => [c.init.method ?? 'GET', c.url])).toEqual([
      ['GET', `${ORIGIN}/api/participants/rec%201`],
      ['GET', `${ORIGIN}/api/relay/config`],
      ['GET', `${ORIGIN}/api/relay/runs/${RUN}`],
      ['POST', `${ORIGIN}/api/relay/runs/${RUN}/stop`],
      ['POST', `${ORIGIN}/api/relay/token`],
    ]);
    expect(JSON.parse(String(calls[4].init.body))).toEqual({ runId: RUN });
    expect(headersOf(calls[4]).get('Content-Type')).toBe('application/json');
    expect(headersOf(calls[3]).get('Content-Type')).toBe('application/json');
    expect(headersOf(calls[0]).has('Content-Type')).toBe(false);
  });

  it('returns parsed shapes', async () => {
    const { api } = setup(() =>
      json({ participants: [{ recordId: 'a', name: 'b', county: 'c' }] }),
    );
    expect(await api.searchParticipants('a')).toEqual([
      { recordId: 'a', name: 'b', county: 'c' },
    ]);
  });
});

describe('classifyFields', () => {
  it('sends only descriptor keys and drops extras', async () => {
    const { api, calls } = setup(() => json({ sensitive: [false] }));
    const field = {
      label: 'Name',
      role: 'textbox',
      value: 'secret',
      url: 'https://x',
      text: 'hello',
      extra: 1,
    } as never;
    expect(await api.classifyFields(RUN, [field])).toEqual([false]);
    const body = JSON.parse(String(calls[0].init.body));
    expect(body).toEqual({
      runId: RUN,
      fields: [{ role: 'textbox', label: 'Name' }],
    });
    expect(
      Object.keys(body.fields[0]).every((k) =>
        (DESCRIPTOR_KEYS as readonly string[]).includes(k),
      ),
    ).toBe(true);
  });

  it('drops non-string descriptor values', async () => {
    const { api, calls } = setup(() => json({ sensitive: [true] }));
    await api.classifyFields(RUN, [{ label: 5, name: 'n' } as never]);
    expect(JSON.parse(String(calls[0].init.body)).fields).toEqual([
      { name: 'n' },
    ]);
  });

  it('splits more than 50 fields in order', async () => {
    const { api, calls } = setup((c) => {
      const { fields } = JSON.parse(String(c.init.body));
      return json({
        sensitive: fields.map(
          (f: { name: string }) => Number(f.name) % 2 === 0,
        ),
      });
    });
    const fields = Array.from({ length: 120 }, (_, i) => ({ name: String(i) }));
    const result = await api.classifyFields(RUN, fields);
    expect(
      calls.map((c) => JSON.parse(String(c.init.body)).fields.length),
    ).toEqual([50, 50, 20]);
    expect(result).toEqual(fields.map((_, i) => i % 2 === 0));
  });

  it('empty input makes no request', async () => {
    const { api, calls } = setup(() => json({ sensitive: [] }));
    expect(await api.classifyFields(RUN, [])).toEqual([]);
    expect(calls).toHaveLength(0);
  });

  it('a length mismatch is server_error', async () => {
    const { api } = setup(() => json({ sensitive: [true, true] }));
    const e = await caught(api.classifyFields(RUN, [{ label: 'a' }]));
    expect(e).toBeInstanceOf(AppApiError);
    expect(e.code).toBe('server_error');
  });

  it('non-boolean entries are server_error', async () => {
    const { api } = setup(() => json({ sensitive: ['yes'] }));
    expect((await caught(api.classifyFields(RUN, [{ label: 'a' }]))).code).toBe(
      'server_error',
    );
  });

  it('keeps the descriptor key list in step with the server', () => {
    expect([...CLIENT_KEYS]).toEqual([...DESCRIPTOR_KEYS]);
  });
});

describe('errors', () => {
  it.each([
    [401, 'signed_out'],
    [403, 'forbidden'],
    [404, 'not_found'],
    [400, 'bad_request'],
    [429, 'rate_limited'],
    [500, 'server_error'],
    [503, 'server_error'],
    [418, 'server_error'],
  ])('maps %i to %s without body text', async (status, code) => {
    const { api } = setup(() =>
      json({ message: MARKER, cause: MARKER }, status),
    );
    const e = await caught(api.getConfig());
    expect(e).toBeInstanceOf(AppApiError);
    expect(e.status).toBe(status);
    expect(e.code).toBe(code);
    expect(e.message).not.toContain(MARKER);
    expect(JSON.stringify(e)).not.toContain(MARKER);
    expect(String(e.stack)).not.toContain(MARKER);
    expect(Object.values(e).join(' ')).not.toContain(MARKER);
  });

  it('a non-JSON 2xx is server_error without the body', async () => {
    const { api } = setup(
      () => new Response(`<html>${MARKER}</html>`, { status: 200 }),
    );
    const e = await caught(api.getConfig());
    expect(e.code).toBe('server_error');
    expect(e.message).not.toContain(MARKER);
  });

  it('a wrong-shaped 2xx is server_error', async () => {
    const { api } = setup(() => json({ participants: 'nope' }));
    expect((await caught(api.searchParticipants('x'))).code).toBe(
      'server_error',
    );
    const b = setup(() => json({ nothing: true }));
    expect((await caught(b.api.getExtensionToken(RUN))).code).toBe(
      'server_error',
    );
    expect((await caught(b.api.getRunState(RUN))).code).toBe('server_error');
    expect((await caught(b.api.getParticipant('r'))).code).toBe('server_error');
    expect((await caught(b.api.stopRun(RUN))).code).toBe('server_error');
    expect((await caught(b.api.getConfig())).code).toBe('server_error');
  });

  it('a fetch rejection is network with status 0 and no cause text', async () => {
    const api = createAppApi({
      origin: ORIGIN,
      fetch: (async () => {
        throw new TypeError(MARKER);
      }) as unknown as typeof fetch,
    });
    const e = await caught(api.getConfig());
    expect(e.code).toBe('network');
    expect(e.status).toBe(0);
    expect(e.message).not.toContain(MARKER);
  });

  it('the timeout fires as network', async () => {
    vi.useFakeTimers();
    const fetchFn = ((_u: string, init: RequestInit) =>
      new Promise((_res, rej) => {
        init.signal?.addEventListener('abort', () => rej(new Error('aborted')));
      })) as unknown as typeof fetch;
    const api = createAppApi({ origin: ORIGIN, fetch: fetchFn });
    const p = caught(api.getConfig());
    await vi.advanceTimersByTimeAsync(9_999);
    let settled = false;
    p.then(() => {
      settled = true;
    });
    await vi.advanceTimersByTimeAsync(0);
    expect(settled).toBe(false);
    await vi.advanceTimersByTimeAsync(2);
    const e = await p;
    expect(e.code).toBe('network');
    expect(e.status).toBe(0);
  });

  it('timeoutMs is configurable', async () => {
    vi.useFakeTimers();
    const fetchFn = ((_u: string, init: RequestInit) =>
      new Promise((_res, rej) => {
        init.signal?.addEventListener('abort', () => rej(new Error('aborted')));
      })) as unknown as typeof fetch;
    const api = createAppApi({ origin: ORIGIN, fetch: fetchFn, timeoutMs: 50 });
    const p = caught(api.getConfig());
    await vi.advanceTimersByTimeAsync(51);
    expect((await p).code).toBe('network');
  });

  it("the caller's signal aborts the request as network", async () => {
    const fetchFn = ((_u: string, init: RequestInit) =>
      new Promise((_res, rej) => {
        init.signal?.addEventListener('abort', () => rej(new Error('aborted')));
      })) as unknown as typeof fetch;
    const api = createAppApi({ origin: ORIGIN, fetch: fetchFn });
    const ctl = new AbortController();
    const p = caught(api.searchParticipants('x', ctl.signal));
    ctl.abort();
    expect((await p).code).toBe('network');
  });

  it('an already-aborted caller signal never calls fetch', async () => {
    const { api, calls } = setup(() => json({ participants: [] }));
    const ctl = new AbortController();
    ctl.abort();
    expect((await caught(api.searchParticipants('x', ctl.signal))).code).toBe(
      'network',
    );
    expect(calls).toHaveLength(0);
  });
});

describe('inputs', () => {
  it.each([
    'https://app.example.gov/',
    'https://app.example.gov/path',
    'https://app.example.gov?x=1',
    'http://app.example.gov',
    'https://user@app.example.gov',
    'app.example.gov',
    '',
    'chrome-extension://abc',
  ])('rejects the origin %j', (origin) => {
    expect(() => createAppApi({ origin })).toThrow();
  });

  it('accepts an https origin with a port and loopback http', () => {
    expect(() =>
      createAppApi({ origin: 'https://app.example.gov:8443' }),
    ).not.toThrow();
    expect(() =>
      createAppApi({ origin: 'http://localhost:3000' }),
    ).not.toThrow();
  });

  it.each(['../x', 'a/b', '..', '.', '', '%2e%2e', 'a?b', 'a#b'])(
    'recordId %j never changes the path',
    async (recordId) => {
      const { api, calls } = setup(() =>
        json({ recordId: 'r', name: 'n', county: 'c' }),
      );
      try {
        await api.getParticipant(recordId);
      } catch {
        // rejected is fine
      }
      for (const c of calls) {
        const u = new URL(c.url);
        expect(u.pathname.startsWith('/api/participants/')).toBe(true);
        expect(u.pathname.split('/').length).toBe(4);
        expect(['', '.', '..']).not.toContain(u.pathname.split('/')[3]);
        expect(u.search).toBe('');
        expect(u.hash).toBe('');
      }
    },
  );

  it.each(['../x', 'a/b', '..', '', 'not-a-uuid', `${RUN}/stop`])(
    'runId %j is rejected before any request',
    async (runId) => {
      const { api, calls } = setup(() => json({}));
      for (const call of [
        () => api.getRunState(runId),
        () => api.stopRun(runId),
        () => api.getExtensionToken(runId),
        () => api.classifyFields(runId, [{ label: 'a' }]),
      ]) {
        expect((await caught(call())).code).toBe('bad_request');
      }
      expect(calls).toHaveLength(0);
    },
  );
});
