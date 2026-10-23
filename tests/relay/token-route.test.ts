import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { verifyRelayToken } from '@/lib/relay/token';

const authMock = vi.fn();
vi.mock('@/app/(auth)/auth', () => ({ auth: () => authMock() }));

const SECRET = 'test-secret-that-is-at-least-32-characters-long';
const PUBLIC_URL = 'https://relay.example.gov';
const RUN_ID = '3f2b8c1e-5d4a-4b6f-9a7e-1c2d3e4f5a6b';

const route = await import('@/app/(chat)/api/relay/token/route');

const post = (body: unknown, raw = false) =>
  route.POST(
    new Request('http://localhost/api/relay/token', {
      method: 'POST',
      body: raw ? (body as string) : JSON.stringify(body),
    }),
  );

const good = { runId: RUN_ID, origin: 'https://apply.example.gov' };

describe('POST /api/relay/token', () => {
  let logged: string[];

  beforeEach(() => {
    process.env.RELAY_TOKEN_ENDPOINT_ENABLED = '1';
    process.env.RELAY_TOKEN_SECRET = SECRET;
    process.env.RELAY_PUBLIC_URL = PUBLIC_URL;
    authMock.mockResolvedValue({ user: { id: 'u1' } });
    logged = [];
    for (const method of ['log', 'info', 'warn', 'error', 'debug'] as const) {
      vi.spyOn(console, method).mockImplementation((...args: unknown[]) => {
        logged.push(JSON.stringify(args));
      });
    }
  });

  afterEach(() => {
    vi.restoreAllMocks();
    delete process.env.RELAY_TOKEN_ENDPOINT_ENABLED;
    delete process.env.RELAY_TOKEN_SECRET;
    delete process.env.RELAY_PUBLIC_URL;
  });

  it.each([undefined, '0', 'true', ''])(
    'returns 404 and does nothing when the flag is %j',
    async (flag) => {
      if (flag === undefined) delete process.env.RELAY_TOKEN_ENDPOINT_ENABLED;
      else process.env.RELAY_TOKEN_ENDPOINT_ENABLED = flag;
      authMock.mockClear();
      const response = await post(good);
      expect(response.status).toBe(404);
      expect(await response.text()).toBe('');
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
    process.env.RELAY_PUBLIC_URL = value;
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
    expect((await post(good)).status).toBe(401);
  });

  it.each([
    ['http origin', { ...good, origin: 'http://a.gov' }],
    ['origin with a path', { ...good, origin: 'https://a.gov/path' }],
    ['origin with trailing slash', { ...good, origin: 'https://a.gov/' }],
    ['non-uuid runId', { ...good, runId: 'not-a-uuid' }],
    ['missing origin', { runId: RUN_ID }],
  ])('returns 400 for %s', async (_name, body) => {
    expect((await post(body)).status).toBe(400);
  });

  it('returns 400 for malformed JSON', async () => {
    expect((await post('{not json', true)).status).toBe(400);
  });

  it('mints extension and agent tokens', async () => {
    const response = await post(good);
    expect(response.status).toBe(200);
    expect(response.headers.get('Cache-Control')).toBe('no-store');
    const body = await response.json();

    const ext = verifyRelayToken(body.extensionToken, SECRET);
    const agent = verifyRelayToken(body.agentToken, SECRET);
    expect(ext).toMatchObject({
      runId: RUN_ID,
      origin: good.origin,
      role: 'extension',
    });
    expect(agent).toMatchObject({
      runId: RUN_ID,
      origin: good.origin,
      role: 'agent',
    });
    expect(body.expiresAt).toBe(ext?.exp);
    expect(body.expiresAt).toBe(agent?.exp);
    expect(body.relayUrl).toBe('wss://relay.example.gov');

    // Nothing sensitive reaches the console on success.
    expect(logged.join('\n')).not.toContain(SECRET);
    expect(logged.join('\n')).not.toContain(body.extensionToken);
    expect(logged.join('\n')).not.toContain(body.agentToken);
  });

  it('uses ws for a loopback http dev URL', async () => {
    process.env.RELAY_PUBLIC_URL = 'http://localhost:8787';
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
    if (value === undefined) delete process.env[name];
    else process.env[name] = value;

    const response = await post(good);
    expect(response.status).toBe(500);
    const text = await response.text();
    expect(text).not.toContain(SECRET);
    expect(text).not.toContain(PUBLIC_URL);
    if (value) expect(text).not.toContain(value);
    expect(logged.join('\n')).not.toContain(SECRET);
  });
});
