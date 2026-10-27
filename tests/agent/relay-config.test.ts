import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const authMock = vi.fn();
vi.mock('@/app/(auth)/auth', () => ({ auth: () => authMock() }));

const { getRelayConfig } = await import('@/lib/relay/config');
const route = await import('@/app/(chat)/api/relay/config/route');

const site = (over: Record<string, unknown> = {}) => ({
  id: 'home-site',
  name: 'Home Site',
  url: 'https://example.gov/apply',
  ...over,
});
const sitesOf = (raw: unknown) => getRelayConfig({}, raw).sites;

describe('getRelayConfig sites', () => {
  it('keeps a valid site as given', () => {
    expect(sitesOf([site()])).toEqual([site()]);
  });

  it('drops http, userinfo, unparseable and non-url sites', () => {
    expect(sitesOf([site({ url: 'http://example.gov' })])).toEqual([]);
    expect(sitesOf([site({ url: 'https://u:p@example.gov' })])).toEqual([]);
    expect(sitesOf([site({ url: 'https://u@example.gov' })])).toEqual([]);
    expect(sitesOf([site({ url: 'not a url' })])).toEqual([]);
  });

  it('drops a duplicate id and keeps the first', () => {
    const first = site();
    const out = sitesOf([first, site({ name: 'Other' })]);
    expect(out).toEqual([first]);
  });

  it('drops bad ids and names', () => {
    expect(sitesOf([site({ id: 'Bad_ID' })])).toEqual([]);
    expect(sitesOf([site({ id: '' })])).toEqual([]);
    expect(sitesOf([site({ id: 'a'.repeat(65) })])).toEqual([]);
    expect(sitesOf([site({ name: '' })])).toEqual([]);
    expect(sitesOf([site({ name: 'n'.repeat(101) })])).toEqual([]);
  });

  it('drops missing fields and non-object entries without throwing', () => {
    const { url: _u, ...noUrl } = site();
    expect(sitesOf([noUrl, null, 'x', 4, [], site({ id: 3 })])).toEqual([]);
  });

  it('answers [] for non-array json', () => {
    expect(sitesOf({ id: 'x' })).toEqual([]);
    expect(sitesOf('nope')).toEqual([]);
    expect(sitesOf(null)).toEqual([]);
  });

  it('ignores extra keys on an entry', () => {
    expect(sitesOf([{ ...site(), extra: 'x' }])).toEqual([site()]);
  });

  it('defaults to the checked-in file (empty)', () => {
    expect(getRelayConfig({}).sites).toEqual([]);
  });
});

describe('getRelayConfig models', () => {
  it('uses the default for unset and empty env', () => {
    for (const env of [{}, { RELAY_ALLOWED_MODELS: '' }]) {
      const cfg = getRelayConfig(env);
      expect(cfg.models).toEqual([{ id: 'claude-opus-5' }]);
      expect(cfg.defaultModel).toBe('claude-opus-5');
    }
  });

  it('drops unknown model ids', () => {
    const cfg = getRelayConfig({
      RELAY_ALLOWED_MODELS: 'claude-opus-5,not-a-model',
    });
    expect(cfg.models).toEqual([{ id: 'claude-opus-5' }]);
  });

  it('falls back to the default when only unknown models are named', () => {
    const cfg = getRelayConfig({ RELAY_ALLOWED_MODELS: 'nope,nada' });
    expect(cfg.models).toEqual([{ id: 'claude-opus-5' }]);
    expect(cfg.defaultModel).toBe('claude-opus-5');
  });

  it('keeps order for two known models and defaults to the first', async () => {
    const { allowedRelayModels } = await import('@/lib/ai/eve/model-map');
    const env = { RELAY_ALLOWED_MODELS: 'claude-opus-5,claude-opus-4-7' };
    const ids = allowedRelayModels(env);
    expect(ids.length).toBe(2);
    const reversed = {
      RELAY_ALLOWED_MODELS: [...ids].reverse().join(','),
    };
    const cfg = getRelayConfig(env);
    expect(cfg.models.map((m) => m.id)).toEqual(ids);
    expect(cfg.defaultModel).toBe(ids[0]);
    const cfg2 = getRelayConfig(reversed);
    expect(cfg2.models.map((m) => m.id)).toEqual([...ids].reverse());
    expect(cfg2.defaultModel).toBe(ids[1]);
  });

  it('delegates to allowedRelayModels (env change flows through)', async () => {
    const { allowedRelayModels } = await import('@/lib/ai/eve/model-map');
    const env = { RELAY_ALLOWED_MODELS: 'claude-opus-4-7' };
    expect(getRelayConfig(env).models.map((m) => m.id)).toEqual(
      allowedRelayModels(env),
    );
  });
});

const configRequest = () => new Request('http://localhost/api/relay/config');

describe('GET /api/relay/config', () => {
  beforeEach(() => {
    vi.stubEnv('RELAY_RUNS_ENABLED', '1');
    authMock.mockReset();
    authMock.mockResolvedValue({
      user: { id: 'u1', email: 'person@example.gov' },
    });
  });
  afterEach(() => vi.unstubAllEnvs());

  it('refuses with no-store: flag off bodyless 404, 401, guest 403', async () => {
    vi.stubEnv('RELAY_RUNS_ENABLED', undefined);
    const off = await route.GET(configRequest());
    expect(off.status).toBe(404);
    expect(await off.text()).toBe('');
    expect(off.headers.get('Cache-Control')).toBe('no-store');

    vi.stubEnv('RELAY_RUNS_ENABLED', '1');
    authMock.mockResolvedValue(null);
    const unauth = await route.GET(configRequest());
    expect(unauth.status).toBe(401);
    expect(unauth.headers.get('Cache-Control')).toBe('no-store');

    authMock.mockResolvedValue({
      user: { id: 'g', email: 'guest-0123abcd@preview.local' },
    });
    const guest = await route.GET(configRequest());
    expect(guest.status).toBe(403);
    expect(guest.headers.get('Cache-Control')).toBe('no-store');
  });

  it('returns getRelayConfig with no-store', async () => {
    vi.stubEnv('RELAY_ALLOWED_MODELS', 'claude-opus-4-7');
    const res = await route.GET(configRequest());
    expect(res.status).toBe(200);
    expect(res.headers.get('Cache-Control')).toBe('no-store');
    expect(await res.json()).toEqual(getRelayConfig());
  });

  it('does not read RELAY_ALLOWED_MODELS itself', async () => {
    const { readFile } = await import('node:fs/promises');
    for (const p of [
      'app/(chat)/api/relay/config/route.ts',
      'lib/relay/config.ts',
    ]) {
      expect(await readFile(p, 'utf8')).not.toContain('RELAY_ALLOWED_MODELS');
    }
  });
});
