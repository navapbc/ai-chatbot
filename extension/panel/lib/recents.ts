// Recent runs shown on Home. Stored in chrome.storage.local with ONLY these
// four keys per entry: never a participant name, id or detail, a URL or an
// origin. The stored value is untrusted on read: anything with other keys or
// the wrong types is dropped, never rendered.

export type Recent = {
  runId: string;
  siteName: string;
  startedAt: number;
  endedReason?: string;
};

const KEY = 'recentRuns';
const MAX_RECENTS = 10;
const MAX_AGE_MS = 24 * 60 * 60 * 1000;
const MAX_NAME_LENGTH = 100;
const MAX_REASON_LENGTH = 40;
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const REASON = /^[a-z-]+$/;
const ALLOWED_KEYS = new Set(['runId', 'siteName', 'startedAt', 'endedReason']);

const isValidReason = (value: unknown): value is string =>
  typeof value === 'string' &&
  value.length >= 1 &&
  value.length <= MAX_REASON_LENGTH &&
  REASON.test(value);

const parseRecent = (value: unknown): Recent | null => {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    return null;
  }
  const entry = value as Record<string, unknown>;
  if (!Object.keys(entry).every((key) => ALLOWED_KEYS.has(key))) return null;
  const { runId, siteName, startedAt, endedReason } = entry;
  if (typeof runId !== 'string' || !UUID.test(runId)) return null;
  if (
    typeof siteName !== 'string' ||
    siteName.length < 1 ||
    siteName.length > MAX_NAME_LENGTH
  ) {
    return null;
  }
  if (typeof startedAt !== 'number' || !Number.isFinite(startedAt)) {
    return null;
  }
  if ('endedReason' in entry && !isValidReason(endedReason)) return null;
  // Rebuilt from the four fields so the stored shape is exact.
  return isValidReason(endedReason)
    ? { runId, siteName, startedAt, endedReason }
    : { runId, siteName, startedAt };
};

const read = async (
  now: number,
): Promise<{ list: Recent[]; dirty: boolean }> => {
  const stored = (await chrome.storage.local.get(KEY))[KEY];
  if (stored === undefined) return { list: [], dirty: false };
  if (!Array.isArray(stored)) return { list: [], dirty: true };
  const list = stored
    .map(parseRecent)
    .filter((r): r is Recent => r !== null && now - r.startedAt <= MAX_AGE_MS)
    .sort((a, b) => b.startedAt - a.startedAt)
    .slice(0, MAX_RECENTS);
  return { list, dirty: list.length !== stored.length };
};

export const loadRecents = async (
  now: number = Date.now(),
): Promise<Recent[]> => {
  try {
    const { list, dirty } = await read(now);
    if (dirty) await chrome.storage.local.set({ [KEY]: list });
    return list;
  } catch {
    return [];
  }
};

export const addRecent = async (
  recent: Recent,
  now: number = Date.now(),
): Promise<void> => {
  const valid = parseRecent(recent);
  if (!valid) throw new Error('invalid recent');
  const { list } = await read(now);
  const next = [valid, ...list.filter((r) => r.runId !== valid.runId)].slice(
    0,
    MAX_RECENTS,
  );
  await chrome.storage.local.set({ [KEY]: next });
};

export const markRecentEnded = async (
  runId: string,
  reason: string,
): Promise<void> => {
  try {
    if (!isValidReason(reason)) return;
    const { list } = await read(Date.now());
    if (!list.some((r) => r.runId === runId)) return;
    await chrome.storage.local.set({
      [KEY]: list.map((r) =>
        r.runId === runId ? { ...r, endedReason: reason } : r,
      ),
    });
  } catch {
    // Value-free: the recents list is a convenience.
  }
};

const MINUTE = 60_000;
const HOUR = 60 * MINUTE;

const plural = (n: number, unit: string) =>
  `${n} ${unit}${n === 1 ? '' : 's'} ago`;

/** Relative start time for the Recent list, from a fixed set of phrasings. */
export const formatAgo = (now: number, then: number): string => {
  const diff = now - then;
  if (diff < MINUTE) return 'just now';
  if (diff < HOUR) return plural(Math.floor(diff / MINUTE), 'minute');
  if (diff < 24 * HOUR) return plural(Math.floor(diff / HOUR), 'hour');
  return 'over a day ago';
};
