import { describe, expect, it } from 'vitest';
import { PARTICIPANTS, buildApplicationPrompt } from '@/lib/data/participants';
import {
  SENSITIVE_RECORD_PATHS,
  WITHHELD,
  stripSensitiveFields,
  stripSensitiveText,
} from '@/lib/data/sensitive-record';
import { flattenRecord } from '@/lib/jev/flatten';
import { extractParticipantFromText } from '@/lib/jev/participant';

const record = PARTICIPANTS[0];
const opening = (p = record) =>
  buildApplicationPrompt({ participant: p, target: 'a WIC application' });

const leafPaths = (
  v: unknown,
  prefix = '',
  out: Record<string, unknown> = {},
) => {
  if (v && typeof v === 'object') {
    for (const [k, c] of Object.entries(v)) {
      leafPaths(c, prefix ? `${prefix}.${k}` : k, out);
    }
  } else {
    out[prefix] = v;
  }
  return out;
};

describe('stripSensitiveFields', () => {
  it('lists calworks_id as a sensitive path', () => {
    expect(SENSITIVE_RECORD_PATHS).toContain('participant.calworks_id');
  });

  it('calworks_id becomes the placeholder', () => {
    expect(stripSensitiveFields(record).participant.calworks_id).toBe(WITHHELD);
  });

  it('a future ssn key at depth 3 is stripped', () => {
    const out = stripSensitiveFields({
      a: { b: { ssn: '123-45-6789', social_security_number: 123456789 } },
      list: [{ bank: { account_number: '000111222', routing: '999' } }],
    });
    expect(out.a.b.ssn).toBe(WITHHELD);
    expect(out.a.b.social_security_number).toBe(WITHHELD);
    expect(out.list[0].bank.account_number).toBe(WITHHELD);
    expect(out.list[0].bank.routing).toBe(WITHHELD);
  });

  it('a sensitive key holding an object or array is withheld whole', () => {
    const out = stripSensitiveFields({
      password: { current: 'hunter2' },
      tax_id: ['1', '2'],
    });
    expect(out.password).toBe(WITHHELD);
    expect(out.tax_id).toBe(WITHHELD);
  });

  it('a null sensitive value stays null', () => {
    expect(stripSensitiveFields({ ssn: null }).ssn).toBeNull();
  });

  it.each(PARTICIPANTS.map((p, i) => [i, p] as const))(
    'record %i: only calworks_id changes, every other leaf is byte for byte',
    (_i, p) => {
      const before = leafPaths(p);
      const after = leafPaths(stripSensitiveFields(p));
      const changed = Object.keys(before).filter((k) => before[k] !== after[k]);
      expect(changed).toEqual(['participant.calworks_id']);
      expect(Object.keys(after)).toEqual(Object.keys(before));
    },
  );

  it('the input record is not mutated', () => {
    const copy = structuredClone(record);
    const out = stripSensitiveFields(record);
    expect(record).toEqual(copy);
    expect(out).not.toBe(record);
  });
});

describe('stripSensitiveText', () => {
  it('round-trips: the stripped message still parses', () => {
    const out = stripSensitiveText(opening());
    const parsed = extractParticipantFromText(out);
    expect(parsed?.record_id).toBe(record.record_id);
    expect(parsed?.participant.calworks_id).toBe(WITHHELD);
    expect(out).not.toContain(record.participant.calworks_id);
  });

  it('keeps the surrounding words and the buildApplicationPrompt format', () => {
    const out = stripSensitiveText(`Hello.\n${opening()}\nBye.`);
    expect(out.startsWith('Hello.\nFill out a WIC application')).toBe(true);
    expect(out.endsWith('\nBye.')).toBe(true);
    expect(out).toContain(
      JSON.stringify(stripSensitiveFields(record), null, 2),
    );
  });

  it('text with no record is returned unchanged', () => {
    const text = 'Please apply for {"a": 1} today';
    expect(stripSensitiveText(text)).toBe(text);
    expect(stripSensitiveText('')).toBe('');
  });

  it('fails closed: a second record in the same message is stripped too', () => {
    const out = stripSensitiveText(
      `${opening(PARTICIPANTS[0])}\n\n${opening(PARTICIPANTS[1])}`,
    );
    expect(out).not.toContain(PARTICIPANTS[0].participant.calworks_id);
    expect(out).not.toContain(PARTICIPANTS[1].participant.calworks_id);
    expect(out.split(WITHHELD)).toHaveLength(3);
  });
});

describe('a withheld value is never a candidate to type', () => {
  it('flattenRecord skips WITHHELD leaves', () => {
    const leaves = flattenRecord(stripSensitiveFields(record));
    expect(Object.values(leaves)).not.toContain(WITHHELD);
    expect(leaves['participant.calworks_id']).toBeUndefined();
    expect(leaves.record_id).toBe(record.record_id);
  });
});
