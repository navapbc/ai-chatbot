import { describe, expect, it } from 'vitest';
import {
  createStaticParticipantSource,
  escapeLike,
  summarize,
} from '@/lib/data/participant-source';
import { PARTICIPANTS } from '@/lib/data/participants';

const source = createStaticParticipantSource(PARTICIPANTS);

describe('static participant source', () => {
  it('finds a record by name, case-insensitively', async () => {
    const results = await source.search('celeste', 5);
    expect(results.map((r) => r.recordId)).toContain('339619');
  });

  it('matches by record id prefix', async () => {
    const results = await source.search('339', 5);
    expect(results.map((r) => r.recordId)).toContain('339619');
  });

  it('finds a record by last name alone', async () => {
    const results = await source.search('Thomas', 5);
    expect(results.map((r) => r.recordId)).toContain('339619');
  });

  it('finds a record with a middle name when the query skips it', async () => {
    const results = await source.search('celeste thomas', 5);
    expect(results.map((r) => r.recordId)).toContain('339619');
    expect(
      (await source.search('celeste nava', 5)).map((r) => r.recordId),
    ).toContain('339619');
  });

  it('does not match a mid-word fragment', async () => {
    expect(await source.search('eleste', 5)).toEqual([]);
  });

  it('returns nothing for an empty query', async () => {
    expect(await source.search('', 5)).toEqual([]);
    expect(await source.search('   ', 5)).toEqual([]);
  });

  it('clamps the limit to 1..20', async () => {
    const many = createStaticParticipantSource(
      Array.from({ length: 30 }, (_, i) => ({
        ...PARTICIPANTS[0],
        record_id: `9${String(i).padStart(3, '0')}`,
      })),
    );
    expect((await many.search('9', 100)).length).toBe(20);
    expect((await many.search('9', 0)).length).toBe(1);
    expect((await many.search('9', -5)).length).toBe(1);
  });

  it('treats wildcard characters literally', async () => {
    expect(await source.search('%', 5)).toEqual([]);
    expect(await source.search('_', 5)).toEqual([]);
    expect(escapeLike('50%_x')).toBe('50\\%\\_x');
    expect(escapeLike('a\\b')).toBe('a\\\\b');
  });

  it('trims the record id on get', async () => {
    expect((await source.get(' 339619 '))?.record_id).toBe('339619');
    expect(await source.get('nope')).toBeUndefined();
  });

  it('summarize exposes only recordId, name and county', () => {
    const summary = summarize(PARTICIPANTS[0]);
    expect(Object.keys(summary).sort()).toEqual(['county', 'name', 'recordId']);
  });
});
