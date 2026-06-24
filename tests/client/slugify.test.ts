import { describe, expect, test } from 'vitest';
import { slugify } from '@/lib/slugify';

describe('slugify', () => {
  test('lowercases and hyphenates spaces', () => {
    expect(slugify('Hello World')).toBe('hello-world');
  });

  test('collapses runs of non-alphanumeric characters into one hyphen', () => {
    expect(slugify('foo   bar!!!baz')).toBe('foo-bar-baz');
  });

  test('trims leading and trailing separators', () => {
    expect(slugify('  --Edge Case--  ')).toBe('edge-case');
  });

  test('returns empty string for input with no alphanumerics', () => {
    expect(slugify('!!! ???')).toBe('');
  });
});
