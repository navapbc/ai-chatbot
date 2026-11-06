import relaySites from '@/config/relay-sites.json';
import { allowedRelayModels } from '@/lib/ai/eve/model-map';

export type RelaySite = { id: string; name: string; url: string };
export type RelayConfig = {
  sites: RelaySite[];
  models: { id: string }[];
  defaultModel: string;
};

const ID_PATTERN = /^[a-z0-9-]{1,64}$/;
const MAX_NAME_LENGTH = 100;

const isHttpsWithoutUserinfo = (value: string): boolean => {
  try {
    const url = new URL(value);
    return url.protocol === 'https:' && !url.username && !url.password;
  } catch {
    return false;
  }
};

const parseSite = (entry: unknown): RelaySite | null => {
  if (typeof entry !== 'object' || entry === null || Array.isArray(entry)) {
    return null;
  }
  const { id, name, url } = entry as Record<string, unknown>;
  if (typeof id !== 'string' || !ID_PATTERN.test(id)) return null;
  if (typeof name !== 'string') return null;
  if (name.length < 1 || name.length > MAX_NAME_LENGTH) return null;
  if (typeof url !== 'string' || !isHttpsWithoutUserinfo(url)) return null;
  return { id, name, url };
};

const parseSites = (raw: unknown): RelaySite[] => {
  if (!Array.isArray(raw)) return [];
  const seen = new Set<string>();
  const sites: RelaySite[] = [];
  for (const entry of raw) {
    const site = parseSite(entry);
    if (!site || seen.has(site.id)) continue;
    seen.add(site.id);
    sites.push(site);
  }
  return sites;
};

// `rawSites` is injectable for tests; it defaults to the checked-in file.
export const getRelayConfig = (
  env: Record<string, string | undefined> = process.env,
  rawSites: unknown = relaySites,
): RelayConfig => {
  const ids = allowedRelayModels(env);
  return {
    sites: parseSites(rawSites),
    models: ids.map((id) => ({ id })),
    defaultModel: ids[0],
  };
};
