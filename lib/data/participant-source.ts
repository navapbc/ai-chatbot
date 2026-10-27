import type { Participant } from './participants';
import { getParticipantName } from './participants';

export type ParticipantSummary = {
  recordId: string;
  name: string;
  county: string;
};

export interface ParticipantSource {
  search(query: string, limit: number): Promise<ParticipantSummary[]>;
  get(recordId: string): Promise<Participant | undefined>;
}

const MAX_LIMIT = 20;

const clampLimit = (limit: number): number =>
  Math.min(MAX_LIMIT, Math.max(1, Math.trunc(limit) || 1));

export const summarize = (p: Participant): ParticipantSummary => ({
  recordId: p.record_id,
  name: getParticipantName(p),
  county: p.address.residential.county,
});

/** Lowercase "first middle last recordId", the text prefix search matches. */
export const buildSearchText = (p: Participant): string =>
  `${getParticipantName(p)} ${p.record_id}`.toLowerCase();

/** Escapes LIKE wildcards (and the escape character) so input matches literally. */
export const escapeLike = (q: string): string =>
  q.replace(/[\\%_]/g, (c) => `\\${c}`);

export const createStaticParticipantSource = (
  records: Participant[],
): ParticipantSource => ({
  async search(query, limit) {
    const q = query.trim().toLowerCase();
    if (!q) return [];
    return records
      .filter(
        (p) => buildSearchText(p).startsWith(q) || p.record_id.startsWith(q),
      )
      .map(summarize)
      .sort((a, b) => a.name.localeCompare(b.name))
      .slice(0, clampLimit(limit));
  },
  async get(recordId) {
    const id = recordId.trim();
    return records.find((p) => p.record_id === id);
  },
});

// queries.ts is server-only and opens a database client at import, so it is
// loaded on first use; tests that only need the static source never touch it.
export const createDbParticipantSource = (): ParticipantSource => ({
  async search(query, limit) {
    const q = query.trim().toLowerCase();
    if (!q) return [];
    const { searchParticipantRows } = await import('@/lib/db/queries');
    const rows = await searchParticipantRows({
      pattern: escapeLike(q),
      limit: clampLimit(limit),
    });
    return rows.map((row) => ({
      recordId: row.recordId,
      name: row.displayName,
      county: row.county,
    }));
  },
  async get(recordId) {
    const id = recordId.trim();
    if (!id) return undefined;
    const { getParticipantRow } = await import('@/lib/db/queries');
    return (await getParticipantRow(id))?.record;
  },
});

let current: ParticipantSource | undefined;

export const getParticipantSource = (): ParticipantSource => {
  current ??= createDbParticipantSource();
  return current;
};

export const setParticipantSourceForTests = (
  source: ParticipantSource | undefined,
): void => {
  current = source;
};
