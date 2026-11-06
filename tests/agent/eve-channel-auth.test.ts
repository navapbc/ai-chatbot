import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import channel from '@/agent/channels/eve';

// Drives the channel's real auth walk through a route handler. GET
// /eve/v1/info answers 500 ("requires internal channel dispatch context")
// once auth has passed, so the status tells the auth outcome apart: 403 is
// the relay refusal, 401 is no principal at all, 500 is accepted.

const SECRET =
  'internal-secret-for-channel-tests-padded-to-be-over-thirty-two-chars';
const RUN = '0b6f7c4e-5d1a-4c55-9a53-3f7d2f0c9a11';
const relay = {
  'x-eve-relay-url': 'wss://relay.example',
  'x-eve-relay-run': RUN,
};

type Route = {
  method: string;
  path: string;
  handler: (request: Request, ctx: unknown) => Promise<Response>;
};
const info = (channel as unknown as { routes: Route[] }).routes.find(
  (r) => r.method === 'GET' && r.path === '/eve/v1/info',
);

const call = (host: string, headers: Record<string, string>) => {
  if (!info) throw new Error('info route not found');
  return info.handler(
    new Request(`http://${host}/eve/v1/info`, { headers }),
    {},
  );
};

beforeEach(() => {
  vi.stubEnv('EVE_INTERNAL_SECRET', SECRET);
  vi.spyOn(console, 'error').mockImplementation(() => {});
  vi.spyOn(console, 'warn').mockImplementation(() => {});
});
afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllEnvs();
});

describe('eve channel auth: relay headers', () => {
  it('accepts a loopback request with no relay headers, as before', async () => {
    expect((await call('127.0.0.1', {})).status).toBe(500);
  });

  it('accepts granted relay headers', async () => {
    const res = await call('127.0.0.1', {
      ...relay,
      'x-eve-internal-secret': SECRET,
    });
    expect(res.status).toBe(500);
  });

  it('ignores the retired token header alone', async () => {
    const res = await call('127.0.0.1', { 'x-eve-relay-token': 'legacy' });
    expect(res.status).toBe(500);
  });

  it.each([
    ['a wrong secret', '127.0.0.1', { ...relay, 'x-eve-internal-secret': 'x' }],
    ['no secret', '127.0.0.1', relay],
    [
      'a malformed run id',
      '127.0.0.1',
      { ...relay, 'x-eve-relay-run': 'nope', 'x-eve-internal-secret': SECRET },
    ],
    [
      'a lone relay URL',
      '127.0.0.1',
      {
        'x-eve-relay-url': 'wss://relay.example',
        'x-eve-internal-secret': SECRET,
      },
    ],
    [
      'a non-loopback request',
      'agent.example.com',
      { ...relay, 'x-eve-internal-secret': SECRET },
    ],
  ])(
    'refuses relay headers with %s (403, no Kernel fallback)',
    async (_n, host, headers) => {
      const res = await call(host, headers as Record<string, string>);
      expect(res.status).toBe(403);
      const text = await res.text();
      expect(text).not.toContain(SECRET);
      expect(text).not.toContain(RUN);
    },
  );

  it('treats a 31-character secret as unset: the grant is refused', async () => {
    const short = 'x'.repeat(31);
    vi.stubEnv('EVE_INTERNAL_SECRET', short);
    const res = await call('127.0.0.1', {
      ...relay,
      'x-eve-internal-secret': short,
    });
    expect(res.status).toBe(403);
    expect(await res.text()).not.toContain(short);
  });

  it('accepts a 32-character secret', async () => {
    const ok = 'x'.repeat(32);
    vi.stubEnv('EVE_INTERNAL_SECRET', ok);
    const res = await call('127.0.0.1', {
      ...relay,
      'x-eve-internal-secret': ok,
    });
    expect(res.status).toBe(500);
  });

  it('leaves a non-loopback request without relay headers to the other strategies', async () => {
    expect((await call('agent.example.com', {})).status).toBe(401);
  });
});
