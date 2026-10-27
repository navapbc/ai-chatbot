import { beforeEach, describe, expect, it } from 'vitest';
import {
  addRecent,
  formatAgo,
  loadRecents,
  markRecentEnded,
  type Recent,
} from '../../extension/panel/lib/recents';

const KEY = 'recentRuns';
const HOUR = 3_600_000;
const NOW = 1_800_000_000_000;

let store: Record<string, unknown>;
let failing = false;

beforeEach(() => {
  store = {};
  failing = false;
  (globalThis as unknown as { chrome: unknown }).chrome = {
    storage: {
      local: {
        get: async (key: string) => {
          if (failing) throw new Error('boom');
          return key in store ? { [key]: store[key] } : {};
        },
        set: async (items: Record<string, unknown>) => {
          if (failing) throw new Error('boom');
          Object.assign(store, JSON.parse(JSON.stringify(items)));
        },
      },
    },
  };
});

const id = (n: number) =>
  `00000000-0000-4000-8000-${String(n).padStart(12, '0')}`;
const rec = (n: number, over: Partial<Recent> = {}): Recent => ({
  runId: id(n),
  siteName: `Site ${n}`,
  startedAt: NOW - n * 1000,
  ...over,
});

describe('recents', () => {
  it('stores exactly the four keys', async () => {
    await addRecent(rec(1, { endedReason: 'stopped' }), NOW);
    const stored = store[KEY] as Record<string, unknown>[];
    expect(Object.keys(stored[0]).sort()).toEqual([
      'endedReason',
      'runId',
      'siteName',
      'startedAt',
    ]);
    await addRecent(rec(2), NOW);
    const again = store[KEY] as Record<string, unknown>[];
    expect(Object.keys(again[0]).sort()).toEqual([
      'runId',
      'siteName',
      'startedAt',
    ]);
  });

  it('caps at 10, newest first', async () => {
    for (let n = 12; n >= 1; n--) await addRecent(rec(n), NOW);
    const list = await loadRecents(NOW);
    expect(list).toHaveLength(10);
    expect(list[0].runId).toBe(id(1));
  });

  it('prepends and de-duplicates by runId', async () => {
    await addRecent(rec(1), NOW);
    await addRecent(rec(2), NOW);
    await addRecent(rec(1, { siteName: 'Again' }), NOW);
    const list = await loadRecents(NOW);
    expect(list.map((r) => r.runId)).toEqual([id(1), id(2)]);
    expect(list[0].siteName).toBe('Again');
  });

  it('drops entries older than 24 hours', async () => {
    store[KEY] = [
      rec(1),
      { ...rec(2), startedAt: NOW - 24 * HOUR - 1 },
      { ...rec(3), startedAt: NOW - 24 * HOUR + 1000 },
    ];
    const list = await loadRecents(NOW);
    expect(list.map((r) => r.runId)).toEqual([id(1), id(3)]);
    expect((store[KEY] as unknown[]).length).toBe(2);
  });

  it('sorts newest first on read', async () => {
    store[KEY] = [rec(3), rec(1), rec(2)];
    const list = await loadRecents(NOW);
    expect(list.map((r) => r.runId)).toEqual([id(1), id(2), id(3)]);
  });

  it('drops planted and malformed entries and writes the cleaned list back', async () => {
    store[KEY] = [
      rec(1),
      { ...rec(2), participant: 'Jane Doe' },
      { ...rec(3), startedAt: '123' },
      { ...rec(4), siteName: 'x'.repeat(200) },
      { ...rec(5), siteName: '' },
      { ...rec(6), runId: 'not-a-uuid' },
      { ...rec(7), endedReason: 'Has Spaces' },
      { ...rec(8), endedReason: 'a'.repeat(41) },
      { ...rec(9), startedAt: Number.POSITIVE_INFINITY },
      'string',
      null,
    ];
    const list = await loadRecents(NOW);
    expect(list.map((r) => r.runId)).toEqual([id(1)]);
    expect(store[KEY]).toEqual([rec(1)]);
  });

  it('treats a non-array as empty', async () => {
    store[KEY] = { runId: id(1) };
    expect(await loadRecents(NOW)).toEqual([]);
  });

  it('returns [] when storage throws', async () => {
    failing = true;
    expect(await loadRecents(NOW)).toEqual([]);
  });

  it('addRecent throws on an invalid recent and writes nothing', async () => {
    await expect(addRecent({ ...rec(1), siteName: '' }, NOW)).rejects.toThrow();
    await expect(
      addRecent({ ...rec(1), participant: 'x' } as unknown as Recent, NOW),
    ).rejects.toThrow();
    expect(store[KEY]).toBeUndefined();
  });

  it('markRecentEnded sets the reason, and is a no-op when absent', async () => {
    await addRecent(rec(1), NOW);
    await markRecentEnded(id(1), 'stopped');
    expect((await loadRecents(NOW))[0].endedReason).toBe('stopped');
    const before = JSON.stringify(store[KEY]);
    await markRecentEnded(id(9), 'stopped');
    expect(JSON.stringify(store[KEY])).toBe(before);
  });

  it('markRecentEnded swallows storage errors', async () => {
    failing = true;
    await expect(markRecentEnded(id(1), 'stopped')).resolves.toBeUndefined();
  });

  it('markRecentEnded ignores an invalid reason', async () => {
    await addRecent(rec(1), NOW);
    await markRecentEnded(id(1), 'Not Valid!');
    expect((await loadRecents(NOW))[0].endedReason).toBeUndefined();
  });
});

describe('recents: read-modify-write is serialized', () => {
  // Slow storage, so two unserialized calls would both read the old list and
  // the second write would drop the first one's change.
  const slow = () => {
    const local = (
      globalThis as unknown as {
        chrome: { storage: { local: Record<string, unknown> } };
      }
    ).chrome.storage.local;
    const get = local.get as (key: string) => Promise<unknown>;
    const set = local.set as (items: Record<string, unknown>) => Promise<void>;
    local.get = async (key: string) => {
      const value = await get(key);
      await new Promise((r) => setTimeout(r, 5));
      return value;
    };
    local.set = async (items: Record<string, unknown>) => {
      await new Promise((r) => setTimeout(r, 5));
      return set(items);
    };
  };

  it('two near-simultaneous addRecent calls both land', async () => {
    slow();
    await Promise.all([addRecent(rec(1), NOW), addRecent(rec(2), NOW)]);
    const ids = (await loadRecents(NOW)).map((r) => r.runId).sort();
    expect(ids).toEqual([id(1), id(2)]);
  });

  it('addRecent and markRecentEnded both land', async () => {
    await addRecent(rec(1), NOW);
    slow();
    await Promise.all([
      markRecentEnded(id(1), 'stopped'),
      addRecent(rec(2), NOW),
    ]);
    const list = await loadRecents(NOW);
    expect(list.map((r) => r.runId).sort()).toEqual([id(1), id(2)]);
    expect(list.find((r) => r.runId === id(1))?.endedReason).toBe('stopped');
  });

  it('a failed call does not block the next one', async () => {
    slow();
    const settled = await Promise.allSettled([
      addRecent({ ...rec(1), siteName: '' }, NOW),
      addRecent(rec(2), NOW),
    ]);
    expect(settled.map((s) => s.status)).toEqual(['rejected', 'fulfilled']);
    failing = true;
    await addRecent(rec(3), NOW).catch(() => {});
    failing = false;
    await addRecent(rec(4), NOW);
    const ids = (await loadRecents(NOW)).map((r) => r.runId).sort();
    expect(ids).toEqual([id(2), id(4)]);
  });
});

describe('formatAgo', () => {
  it('formats relative times', () => {
    expect(formatAgo(NOW, NOW - 5_000)).toBe('just now');
    expect(formatAgo(NOW, NOW + 5_000)).toBe('just now');
    expect(formatAgo(NOW, NOW - 60_000)).toBe('1 minute ago');
    expect(formatAgo(NOW, NOW - 5 * 60_000)).toBe('5 minutes ago');
    expect(formatAgo(NOW, NOW - HOUR)).toBe('1 hour ago');
    expect(formatAgo(NOW, NOW - 3 * HOUR)).toBe('3 hours ago');
    expect(formatAgo(NOW, NOW - 30 * HOUR)).toBe('over a day ago');
  });
});
