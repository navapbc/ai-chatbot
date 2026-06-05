import { describe, expect, test } from 'vitest';
import { adaptGapSections, adaptReviewSections } from '@/lib/types/form-cards';

describe('form-cards pagination', () => {
  test('keeps a six-field gap list on a single page', () => {
    const fields = Array.from({ length: 6 }, (_, i) => ({ field: `f${i + 1}` }));
    const pages = adaptGapSections({ missingFields: fields });
    expect(pages).toHaveLength(1);
    expect(pages[0].fields).toEqual(fields);
  });

  test('keeps a six-field review list on a single page', () => {
    const fields = Array.from({ length: 6 }, (_, i) => ({
      field: `r${i + 1}`,
      source: 'database' as const,
    }));
    const pages = adaptReviewSections({ fields });
    expect(pages).toHaveLength(1);
    expect(pages[0].fields).toEqual(fields);
  });
});
