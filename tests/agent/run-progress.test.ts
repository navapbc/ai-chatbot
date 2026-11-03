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
});
