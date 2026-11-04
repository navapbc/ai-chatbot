import { describe, expect, it } from 'vitest';
import { sanitizeProgress } from '@/lib/relay/progress';

const sec = (name = 'Income') => ({ name, status: 'filling', filled: 2 });

describe('sanitizeProgress', () => {
  it('returns the clean object and strips unknown keys', () => {
    const out = sanitizeProgress({
      sections: [{ ...sec(), total: 5, value: 'SSN 123' }],
      current: 'Income',
      actions: ['a'],
      extra: 1,
    });
    expect(out).toEqual({
      sections: [{ name: 'Income', status: 'filling', filled: 2, total: 5 }],
      current: 'Income',
      actions: ['a'],
    });
  });
  it('rejects 41 sections, accepts 40', () => {
    const mk = (n: number) => ({
      sections: Array.from({ length: n }, (_, i) => sec(`s${i}`)),
    });
    expect(sanitizeProgress(mk(41))).toBeNull();
    expect(sanitizeProgress(mk(40))).not.toBeNull();
  });
  it('rejects bad fields', () => {
    expect(
      sanitizeProgress({ sections: [sec()], current: 'x'.repeat(121) }),
    ).toBeNull();
    expect(
      sanitizeProgress({ sections: [{ ...sec(), filled: -1 }] }),
    ).toBeNull();
    expect(
      sanitizeProgress({ sections: [{ ...sec(), filled: 1.5 }] }),
    ).toBeNull();
    expect(
      sanitizeProgress({
        sections: [sec()],
        actions: ['1', '2', '3', '4', '5', '6'],
      }),
    ).toBeNull();
    expect(sanitizeProgress({ sections: [sec('')] })).toBeNull();
    expect(
      sanitizeProgress({ sections: [{ ...sec(), status: 'bogus' }] }),
    ).toBeNull();
    expect(sanitizeProgress(null)).toBeNull();
    expect(sanitizeProgress('x')).toBeNull();
  });

  describe('field labels', () => {
    it('keeps labels and states and strips a value key on a field', () => {
      const out = sanitizeProgress({
        sections: [
          {
            ...sec(),
            fields: [
              { label: 'First name', state: 'filled', value: 'Ana' },
              { label: 'Phone', state: 'waiting' },
            ],
          },
        ],
      });
      expect(out?.sections[0].fields).toEqual([
        { label: 'First name', state: 'filled' },
        { label: 'Phone', state: 'waiting' },
      ]);
    });
    it('rejects an unknown state, a blank or 81-character label, 31 fields', () => {
      const withFields = (fields: unknown) =>
        sanitizeProgress({ sections: [{ ...sec(), fields }] });
      expect(withFields([{ label: 'A', state: 'typed' }])).toBeNull();
      expect(withFields([{ label: '', state: 'filled' }])).toBeNull();
      expect(
        withFields([{ label: 'x'.repeat(81), state: 'filled' }]),
      ).toBeNull();
      const many = (n: number) =>
        Array.from({ length: n }, (_, i) => ({
          label: `f${i}`,
          state: 'waiting',
        }));
      expect(withFields(many(31))).toBeNull();
      expect(withFields(many(30))).not.toBeNull();
    });
  });
});
