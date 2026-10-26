import { describe, expect, it } from 'vitest';
import { PARTICIPANTS, buildApplicationPrompt } from '@/lib/data/participants';
import {
  SENSITIVE_RECORD_PATHS,
  hasSensitiveValueInProse,
  mustRefuseRelayText,
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

describe('linear scanning', () => {
  const timed = (fn: () => unknown) => {
    const t = performance.now();
    fn();
    return performance.now() - t;
  };

  it('200 KB of "{" completes quickly', () => {
    const text = '{'.repeat(200_000);
    expect(timed(() => stripSensitiveText(text))).toBeLessThan(100);
    expect(timed(() => extractParticipantFromText(text))).toBeLessThan(100);
  });

  it('200 KB of \'{"a":\' completes quickly', () => {
    const text = '{"a":'.repeat(40_000);
    expect(timed(() => stripSensitiveText(text))).toBeLessThan(100);
    expect(timed(() => hasSensitiveValueInProse(text))).toBeLessThan(500);
  });

  it('200 KB of "{}" and of nested braces completes quickly', () => {
    expect(timed(() => stripSensitiveText('{}'.repeat(100_000)))).toBeLessThan(
      300,
    );
    expect(
      timed(() =>
        stripSensitiveText('{'.repeat(100_000) + '}'.repeat(100_000)),
      ),
    ).toBeLessThan(300);
  });

  it('finds a valid record with 1000 braces inside string values', () => {
    const rec = structuredClone(record);
    rec.participant.name.first = '{'.repeat(1000);
    rec.participant.name.last = '}'.repeat(1000);
    const text = `before ${JSON.stringify(rec, null, 2)} after`;
    expect(extractParticipantFromText(text)?.record_id).toBe(rec.record_id);
    expect(stripSensitiveText(text)).not.toContain(
      record.participant.calworks_id,
    );
  });

  it('finds a record nested inside another JSON object', () => {
    const text = JSON.stringify({ wrapper: record });
    expect(extractParticipantFromText(text)?.record_id).toBe(record.record_id);
    expect(stripSensitiveText(text)).not.toContain(
      record.participant.calworks_id,
    );
  });

  it('a stray quote before a record does not hide it', () => {
    const text = `he said " then ${JSON.stringify(record)}`;
    expect(extractParticipantFromText(text)?.record_id).toBe(record.record_id);
  });
});

describe('records of an unrecognized shape', () => {
  it.each([
    [
      'name as a string',
      { record_id: 'r1', participant: { name: 'Ann Lee', ssn: '123-45-6789' } },
    ],
    [
      'missing record_id',
      { participant: { name: { first: 'A' }, ssn: '123-45-6789' } },
    ],
    [
      'numeric record_id',
      { record_id: 5, participant: { name: {}, calworks_id: 'A100103' } },
    ],
    [
      'no participant at all',
      { person: { social_security_number: '123456789' } },
    ],
  ])('%s is stripped', (_n, obj) => {
    const out = stripSensitiveText(`Do this: ${JSON.stringify(obj)} thanks`);
    expect(out).not.toMatch(/123-45-6789|123456789|A100103/);
    expect(out).toContain(WITHHELD);
    expect(out.startsWith('Do this: ')).toBe(true);
    expect(out.endsWith(' thanks')).toBe(true);
  });

  it('an object with nothing sensitive keeps its exact text, even a big number', () => {
    const text = 'x {"n":12345678901234567890,  "a":1} y';
    expect(stripSensitiveText(text)).toBe(text);
  });
});

describe('hasSensitiveValueInProse', () => {
  it.each([
    'CalWORKs ID: 12345 SSN 123-45-6789',
    'my social security number is 123456789',
    'password = hunter22',
    'card number 4111111111111111 exp soon',
    '{"ssn": "123-45-6789",}',
    '{"record_id": "r1", // note\n "calworks_id": "A100103"}',
    '{"participant": {"tax_id": "98-7654321",},}',
  ])('flags %j', (text) => {
    expect(hasSensitiveValueInProse(stripSensitiveText(text))).toBe(true);
  });

  it.each([
    'the participant has no SSN on file',
    "ask for the social worker's name",
    'Please use the home address and check the second page',
    'Fill in the PIN field later, then click next',
    'the appointment is on 10/26/2026 at 9am',
  ])('does not flag %j', (text) => {
    expect(hasSensitiveValueInProse(stripSensitiveText(text))).toBe(false);
  });

  it.each(PARTICIPANTS.map((p) => [p.record_id, p] as const))(
    'does not flag the normal opening message for %s',
    (_id, p) => {
      expect(hasSensitiveValueInProse(stripSensitiveText(opening(p)))).toBe(
        false,
      );
    },
  );
});

describe('fail closed on unparsed blocks and padded values', () => {
  const refuses = (text: string) =>
    mustRefuseRelayText(stripSensitiveText(text));

  it.each([
    ['label + 60 spaces + value', `SSN:${' '.repeat(60)}123-45-6789`],
    ['label + 60 spaces + non-SSN id', `calworks id${' '.repeat(60)}A100103`],
    ['value before label', '123-45-6789 is the SSN'],
    ['card number alone', 'here it is 4111 1111 1111 1111'],
    ['zero-width padded label', 'S​S​N​: 1​23-45-6789'],
    [
      'zero-width padding between label and value',
      `SSN${'​ '.repeat(40)}A12345`,
    ],
    ['full-width digits', 'ssn １２３－４５－６７８９'],
    [
      'unparseable block with a sensitive key',
      '{"a": 1, "ssn": "x1y2z3", oops}',
    ],
    [
      'unclosed block with a sensitive key',
      '{"participant": {"tax_id": "AB-12',
    ],
  ])('refuses %s', (_n, text) => {
    expect(refuses(text)).toBe(true);
  });

  it('strips or refuses a record nested 6 levels deep holding ssn', () => {
    let obj: unknown = { ssn: '123456789', x: 1 };
    for (let i = 0; i < 6; i++) obj = { level: obj };
    const text = JSON.stringify(obj);
    const out = stripSensitiveText(text);
    expect(out).not.toContain('123456789');
    expect(out).toContain(WITHHELD);
    expect(refuses(text)).toBe(false);
  });

  it('refuses a 6-deep block that is broken JSON with ssn inside', () => {
    const text = `${'{"l":'.repeat(6)}{"ssn": "x1y2z3",}${'}'.repeat(6)}`;
    expect(refuses(text)).toBe(true);
  });

  it('refuses when the scan budget runs out, and stays fast', () => {
    for (const text of ['{'.repeat(200_000), '{"a":'.repeat(40_000)]) {
      const t = performance.now();
      expect(refuses(text)).toBe(true);
      expect(performance.now() - t).toBeLessThan(500);
    }
  });

  it('a 200 KB message of distinct words is bounded', () => {
    const text = Array.from({ length: 30_000 }, (_, i) => `w${i}x`).join(' ');
    const t = performance.now();
    mustRefuseRelayText(text);
    expect(performance.now() - t).toBeLessThan(1000);
  });

  it.each([
    'the participant has no SSN on file',
    "ask for the social worker's name",
    'call the office at (555) 123-4567 or 555-123-4567 tomorrow',
    'the invoice total is 1234567890 units',
    'order placed at 1699999999999 ms',
  ])('does not refuse %j', (text) => {
    expect(refuses(text)).toBe(false);
  });

  it.each(PARTICIPANTS.map((p) => [p.record_id, p] as const))(
    'never refuses the normal opening message for %s',
    (_id, p) => {
      expect(refuses(opening(p))).toBe(false);
      expect(refuses(`Also check page 2.\n${opening(p)}\nThanks.`)).toBe(false);
    },
  );
});
