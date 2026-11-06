import { describe, expect, it } from 'vitest';
import { PARTICIPANTS } from '@/lib/data/participants';
import { isSensitiveField } from '@/lib/relay/sensitive-fields';
import { DETAIL_FIELDS } from '@/extension/panel/lib/saved-details';
import {
  detailsFromParticipant,
  SAMPLE_PARTICIPANTS,
} from '@/extension/panel/lib/sample-participants';

describe('sample participants', () => {
  it('offers one sample per fixture record', () => {
    expect(SAMPLE_PARTICIPANTS.map((s) => s.recordId)).toEqual(
      PARTICIPANTS.map((p) => p.record_id),
    );
  });

  it('uses only the fixed non-sensitive detail fields', () => {
    const allowed = new Set<string>(DETAIL_FIELDS.map((f) => f.key));
    for (const sample of SAMPLE_PARTICIPANTS) {
      for (const [key, value] of Object.entries(sample.details)) {
        expect(allowed.has(key)).toBe(true);
        expect(isSensitiveField({ name: key, label: key })).toBe(false);
        expect(typeof value).toBe('string');
      }
    }
  });

  it('carries the name, birth date and address but not the record or calworks id', () => {
    const [first] = PARTICIPANTS;
    const details = detailsFromParticipant(first);
    expect(details.first_name).toBe(first.participant.name.first);
    expect(details.date_of_birth).toBe(first.participant.date_of_birth);
    expect(details.zip).toBe(first.address.residential.zip);
    const text = JSON.stringify(details);
    expect(text).not.toContain(first.participant.calworks_id);
    expect(text).not.toContain(first.record_id);
  });

  it('drops empty values', () => {
    const [first] = PARTICIPANTS;
    const details = detailsFromParticipant({
      ...first,
      address: {
        ...first.address,
        residential: { ...first.address.residential, unit: null },
      },
    });
    expect('unit' in details).toBe(false);
  });
});
