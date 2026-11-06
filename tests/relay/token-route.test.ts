import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('server-only', () => ({}));

import {
  createMemoryRunStore,
  setRunStoreForTests,
} from '@/lib/ai/eve/run-store';
import { verifyRelayToken } from '@/lib/relay/token';

const authMock = vi.fn();
vi.mock('@/app/(auth)/auth', () => ({ auth: () => authMock() }));

const SECRET = 'test-secret-that-is-at-least-32-characters-long';
const PUBLIC_URL = 'https://relay.example.gov';
const RUN_ID = '3f2b8c1e-5d4a-4b6f-9a7e-1c2d3e4f5a6b';
const OTHER_RUN_ID = '9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d';
const WEB_RUN_ID = '0b1c2d3e-4f5a-4b6c-9d7e-8f9a0b1c2d3e';
const ORIGIN = 'https://apply.example.gov';
const USER = 'u1';

const route = await import('@/app/(chat)/api/relay/token/route');

const post = (body: unknown, raw = false) =>
  route.POST(
    new Request('http://localhost/api/relay/token', {
      method: 'POST',
      body: raw ? (body as string) : JSON.stringify(body),
    }),
  );

const good = { runId: RUN_ID };

let store: ReturnType<typeof createMemoryRunStore>;

describe('POST /api/relay/token', () => {
  let logged: string[];

  beforeEach(async () => {
    vi.stubEnv('RELAY_TOKEN_ENDPOINT_ENABLED', '1');
    vi.stubEnv('RELAY_RUNS_ENABLED', '1');
    vi.stubEnv('RELAY_TOKEN_SECRET', SECRET);
    vi.stubEnv('RELAY_PUBLIC_URL', PUBLIC_URL);
    authMock.mockResolvedValue({
      user: { id: USER, email: 'caseworker@example.gov' },
    });
    store = createMemoryRunStore();
    setRunStoreForTests(store);
    await store.create(USER, RUN_ID, {
      eveSessionId: 'wrun_A',
      continuationToken: 't',
      relayOrigin: ORIGIN,
    });
    // Another user's relay run, and the caller's own web (non-relay) chat.
    await store.create('u2', OTHER_RUN_ID, {
      eveSessionId: 'wrun_B',
      continuationToken: 't',
      relayOrigin: 'https://other.example.gov',
    });
    await store.create(USER, WEB_RUN_ID, {
      eveSessionId: 'wrun_C',
      continuationToken: 't',
      relayOrigin: null,
    });
    logged = [];
    for (const method of ['log', 'info', 'warn', 'error', 'debug'] as const) {
      vi.spyOn(console, method).mockImplementation((...args: unknown[]) => {
        logged.push(JSON.stringify(args));
      });
    }
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllEnvs();
    setRunStoreForTests(null);
  });

  const expectBodylessNoStore404 = async (response: Response) => {
    expect(response.status).toBe(404);
    expect(response.headers.get('Cache-Control')).toBe('no-store');
    expect(await response.text()).toBe('');
  };

  it.each([undefined, '0', 'true', ''])(
    'returns 404 and does nothing when the flag is %j',
    async (flag) => {
      // `undefined` removes the variable for the test.
      vi.stubEnv('RELAY_TOKEN_ENDPOINT_ENABLED', flag);
      authMock.mockClear();
      const spy = vi.spyOn(store, 'get');
      await expectBodylessNoStore404(await post(good));
      expect(authMock).not.toHaveBeenCalled();
      expect(spy).not.toHaveBeenCalled();
    },
  );

  it.each([undefined, '0'])(
    'returns a bodyless 404 before auth when RELAY_RUNS_ENABLED is %j',
    async (flag) => {
      vi.stubEnv('RELAY_RUNS_ENABLED', flag);
      authMock.mockClear();
      await expectBodylessNoStore404(await post(good));
      expect(authMock).not.toHaveBeenCalled();
    },
  );

  it.each([
    ['https://user:pw@relay.example.com', null],
    ['https://relay.example.com?x=1', null],
    ['https://relay.example.com#f', null],
    ['https://relay.example.com/', 'wss://relay.example.com'],
    ['https://relay.example.com/base/', 'wss://relay.example.com/base'],
    ['wss://relay.example.com', 'wss://relay.example.com'],
    ['ws://relay.example.com', null],
    ['http://relay.example.com', null],
  ])('relayUrl for %s', async (value, expected) => {
    vi.stubEnv('RELAY_PUBLIC_URL', value);
    const response = await post(good);
    if (expected === null) {
      expect(response.status).toBe(500);
    } else {
      expect((await response.json()).relayUrl).toBe(expected);
    }
  });

  it('exports only POST', () => {
    expect((route as Record<string, unknown>).GET).toBeUndefined();
  });

  it('returns 401 without a session user', async () => {
    authMock.mockResolvedValue(null);
    const response = await post(good);
    expect(response.status).toBe(401);
    expect(response.headers.get('Cache-Control')).toBe('no-store');
  });

  it('returns 403 for a guest', async () => {
    authMock.mockResolvedValue({
      user: { id: USER, email: 'guest-0a1b2c3d@preview.local' },
    });
    const spy = vi.spyOn(store, 'get');
    const response = await post(good);
    expect(response.status).toBe(403);
    expect(response.headers.get('Cache-Control')).toBe('no-store');
    expect(spy).not.toHaveBeenCalled();
  });

  it.each([
    ['non-uuid runId', { runId: 'not-a-uuid' }],
    ['missing runId', {}],
    ['a non-object body', [RUN_ID]],
  ])('returns 400 for %s', async (_name, body) => {
    const response = await post(body);
    expect(response.status).toBe(400);
    expect(response.headers.get('Cache-Control')).toBe('no-store');
  });

  it('returns 400 for malformed JSON', async () => {
    const response = await post('{not json', true);
    expect(response.status).toBe(400);
    expect(response.headers.get('Cache-Control')).toBe('no-store');
  });

  it.each([
    ["another user's relay run", OTHER_RUN_ID],
    ['a web (non-relay) run', WEB_RUN_ID],
    ['an unknown run', '11111111-2222-4333-8444-555555555555'],
  ])('returns a bodyless 404 for %s', async (_name, runId) => {
    await expectBodylessNoStore404(await post({ runId }));
  });

  it('returns a bodyless 404 for an ended run, minting nothing', async () => {
    expect(await store.end(USER, RUN_ID)).toBe(true);
    await expectBodylessNoStore404(await post(good));
    expect(logged).toEqual([]);
  });

  it('returns a credential-free 500 when the store throws', async () => {
    vi.spyOn(store, 'get').mockRejectedValue(new Error(`boom ${ORIGIN}`));
    const response = await post(good);
    expect(response.status).toBe(500);
    expect(response.headers.get('Cache-Control')).toBe('no-store');
    expect(await response.text()).not.toContain(ORIGIN);
  });

  it('mints only an extension token, bound to the stored origin and the user', async () => {
    const response = await post(good);
    expect(response.status).toBe(200);
    expect(response.headers.get('Cache-Control')).toBe('no-store');
    const body = await response.json();
    expect(Object.keys(body).sort()).toEqual([
      'expiresAt',
      'extensionToken',
      'relayUrl',
    ]);

    const ext = verifyRelayToken(body.extensionToken, SECRET);
    expect(ext).toMatchObject({
      runId: RUN_ID,
      origin: ORIGIN,
      role: 'extension',
      sub: USER,
    });
    expect(body.expiresAt).toBe(ext?.exp);
    expect(body.relayUrl).toBe('wss://relay.example.gov');

    // Nothing sensitive reaches the console on success.
    expect(logged.join('\n')).not.toContain(SECRET);
    expect(logged.join('\n')).not.toContain(body.extensionToken);
  });

  it('ignores an origin in the body: the token carries the stored origin', async () => {
    const response = await post({
      runId: RUN_ID,
      origin: 'https://attacker.example.com',
      role: 'agent',
      sub: 'u2',
    });
    expect(response.status).toBe(200);
    const body = await response.json();
    expect(verifyRelayToken(body.extensionToken, SECRET)).toMatchObject({
      origin: ORIGIN,
      role: 'extension',
      sub: USER,
    });
  });

  it('uses ws for a loopback http dev URL', async () => {
    vi.stubEnv('RELAY_PUBLIC_URL', 'http://localhost:8787');
    const body = await (await post(good)).json();
    expect(body.relayUrl).toBe('ws://localhost:8787');
  });

  it.each([
    ['secret unset', 'RELAY_TOKEN_SECRET', undefined],
    ['secret too short', 'RELAY_TOKEN_SECRET', 'short'],
    ['public url unset', 'RELAY_PUBLIC_URL', undefined],
    ['public url not a url', 'RELAY_PUBLIC_URL', 'not a url'],
    ['public url plain http, non-loopback', 'RELAY_PUBLIC_URL', 'http://a.gov'],
  ])('returns 500 without echoing env when %s', async (_n, name, value) => {
    vi.stubEnv(name, value);

    const response = await post(good);
    expect(response.status).toBe(500);
    expect(response.headers.get('Cache-Control')).toBe('no-store');
    // Exactly one fixed line: no values, lengths or variable names.
    expect(logged).toEqual([JSON.stringify(['[relay-token] misconfigured'])]);
    const text = await response.text();
    expect(text).not.toContain(SECRET);
    expect(text).not.toContain(PUBLIC_URL);
    if (value) expect(text).not.toContain(value);
    expect(logged.join('\n')).not.toContain(SECRET);
  });
});
