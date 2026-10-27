import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('server-only', () => ({}));
const authMock = vi.hoisted(() => vi.fn());
vi.mock('@/app/(auth)/auth', () => ({ auth: authMock }));

const { createMemoryRunStore, setRunStoreForTests } = await import(
  '@/lib/ai/eve/run-store'
);
const { PARTICIPANTS } = await import('@/lib/data/participants');
const { createStaticParticipantSource, setParticipantSourceForTests } =
  await import('@/lib/data/participant-source');
const list = await import('@/app/(chat)/api/participants/route');
const detail = await import('@/app/(chat)/api/participants/[recordId]/route');
const config = await import('@/app/(chat)/api/relay/config/route');
const token = await import('@/app/(chat)/api/relay/token/route');
const classify = await import('@/app/(chat)/api/relay/classify-field/route');

const ORIGIN = `chrome-extension://${'a'.repeat(32)}`;
const FOREIGN = 'https://evil.example';
const RUN = '11111111-1111-4111-8111-111111111111';

const headers = (origin?: string) =>
  origin === undefined ? {} : { Origin: origin };
const get = (path: string, origin?: string) =>
  new Request(`http://localhost${path}`, { headers: headers(origin) });
const post = (path: string, origin?: string) =>
  new Request(`http://localhost${path}`, {
    method: 'POST',
    headers: headers(origin),
    body: JSON.stringify({ runId: RUN, fields: [{ label: 'x' }] }),
  });
const recordId = PARTICIPANTS[0].record_id;

type Call = (origin?: string) => Promise<Response>;
const gets: [string, Call][] = [
  ['participants search', (o) => list.GET(get('/api/participants?q=a', o))],
  [
    'participants detail',
    (o) =>
      detail.GET(get(`/api/participants/${recordId}`, o), {
        params: Promise.resolve({ recordId }),
      }),
  ],
  ['relay config', (o) => config.GET(get('/api/relay/config', o))],
];
const posts: [string, Call][] = [
  ['relay token', (o) => token.POST(post('/api/relay/token', o))],
  [
    'classify-field',
    (o) => classify.POST(post('/api/relay/classify-field', o)),
  ],
];

describe('relay routes extension origin', () => {
  beforeEach(async () => {
    vi.stubEnv('RELAY_RUNS_ENABLED', '1');
    vi.stubEnv('RELAY_TOKEN_ENDPOINT_ENABLED', '1');
    vi.stubEnv('RELAY_EXTENSION_ORIGIN', ORIGIN);
    authMock.mockReset();
    authMock.mockResolvedValue({
      user: { id: 'u1', email: 'person@example.gov' },
    });
    setParticipantSourceForTests(createStaticParticipantSource(PARTICIPANTS));
    setRunStoreForTests(createMemoryRunStore());
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    setParticipantSourceForTests(undefined);
  });

  describe.each([...gets, ...posts])('%s', (_name, call) => {
    it('refuses a foreign Origin with 403 and no-store, before auth', async () => {
      authMock.mockResolvedValue(null);
      const res = await call(FOREIGN);
      expect(res.status).toBe(403);
      expect(res.headers.get('Cache-Control')).toBe('no-store');
      expect(await res.text()).not.toContain(ORIGIN);
      expect(authMock).not.toHaveBeenCalled();
    });

    it('with the env unset does not check the origin', async () => {
      vi.stubEnv('RELAY_EXTENSION_ORIGIN', undefined);
      expect((await call(FOREIGN)).status).not.toBe(403);
    });
  });

  describe.each(gets)('GET %s', (_name, call) => {
    it('passes with no Origin and with the exact origin', async () => {
      expect((await call()).status).toBe(200);
      expect((await call(ORIGIN)).status).toBe(200);
    });
  });

  describe.each(posts)('POST %s', (_name, call) => {
    it('refuses a missing Origin with 403 and no-store', async () => {
      const res = await call();
      expect(res.status).toBe(403);
      expect(res.headers.get('Cache-Control')).toBe('no-store');
    });

    it('lets the exact origin through to the route', async () => {
      expect((await call(ORIGIN)).status).not.toBe(403);
    });
  });
});
