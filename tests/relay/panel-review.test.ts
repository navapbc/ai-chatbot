import { describe, expect, it } from 'vitest';
import {
  buildReviewModel,
  reviewActions,
} from '../../extension/panel/lib/review-model';

const sec = (
  name: string,
  status: 'waiting' | 'filling' | 'done' | 'needs_you',
  filled: number,
  total?: number,
) => ({ name, status, filled, ...(total === undefined ? {} : { total }) });

describe('buildReviewModel', () => {
  it('lists needs_you sections', () => {
    const m = buildReviewModel({
      sections: [sec('Income', 'needs_you', 2, 5), sec('Name', 'done', 3, 3)],
    });
    expect(m.hasProgress).toBe(true);
    expect(m.left).toEqual([
      { name: 'Income', reason: 'needs_you', count: '2 of 5' },
    ]);
  });

  it('lists a section whose filled is below a known total', () => {
    const m = buildReviewModel({ sections: [sec('Address', 'done', 1, 4)] });
    expect(m.left).toEqual([
      { name: 'Address', reason: 'incomplete', count: '1 of 4' },
    ]);
  });

  it('omits a section filled to its total, or with no total', () => {
    const m = buildReviewModel({
      sections: [sec('A', 'done', 4, 4), sec('B', 'filling', 2)],
    });
    expect(m.left).toEqual([]);
    expect(m.hasProgress).toBe(true);
  });

  it('has no progress for null or no sections', () => {
    expect(buildReviewModel(null)).toEqual({ left: [], hasProgress: false });
    expect(buildReviewModel({ sections: [] })).toEqual({
      left: [],
      hasProgress: false,
    });
    expect(buildReviewModel('x' as never).hasProgress).toBe(false);
  });

  it('caps at 40 sections', () => {
    const sections = Array.from({ length: 60 }, (_, i) =>
      sec(`S${i}`, 'needs_you', 0, 1),
    );
    expect(buildReviewModel({ sections }).left).toHaveLength(40);
  });

  it('sanitizes hostile names and counts', () => {
    const m = buildReviewModel({
      sections: [
        {
          name: `  ${'x'.repeat(200)}`,
          status: 'needs_you',
          filled: Number.POSITIVE_INFINITY,
          total: 1e12,
        },
        { name: 42, status: 'bogus', filled: -3, total: 5 } as never,
        null as never,
      ],
    });
    expect(Array.from(m.left[0].name)).toHaveLength(80);
    expect(m.left[0].count).toBe('0 of 1000000');
    expect(m.left[1]).toEqual({
      name: 'Untitled section',
      reason: 'incomplete',
      count: '0 of 5',
    });
  });

  it('never throws on junk', () => {
    expect(() => buildReviewModel({ sections: 7 } as never)).not.toThrow();
    expect(buildReviewModel({ sections: 7 } as never).hasProgress).toBe(false);
  });
});

describe('reviewActions', () => {
  it('keeps the last 50 strings, cut to 120 code points', () => {
    const log = Array.from({ length: 80 }, (_, i) => `a${i}`);
    const out = reviewActions([...log, 5 as never, 'y'.repeat(300)]);
    expect(out).toHaveLength(50);
    expect(Array.from(out[49])).toHaveLength(120);
    expect(out[0]).toBe('a31');
  });
  it('is empty for junk', () => {
    expect(reviewActions(undefined as never)).toEqual([]);
  });
});
