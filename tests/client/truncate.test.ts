import { describe, expect, test } from 'vitest';
import { truncate } from '@/lib/utils';

describe('truncate', () => {
  test('returns the string unchanged when within the limit', () => {
    expect(truncate('hello', 5)).toBe('hello');
    expect(truncate('hi', 10)).toBe('hi');
  });

  test('truncates and appends an ellipsis when over the limit', () => {
    expect(truncate('hello world', 5)).toBe('hello…');
  });

  test('returns an empty string for a non-positive max', () => {
    expect(truncate('hello', 0)).toBe('');
    expect(truncate('hello', -3)).toBe('');
  });

  test('handles an empty input string', () => {
    expect(truncate('', 5)).toBe('');
  });
});
