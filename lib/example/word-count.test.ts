import { describe, expect, test } from 'vitest';
import { wordCount } from '@/lib/example/word-count';

describe('wordCount', () => {
  test('returns 0 for an empty string', () => {
    expect(wordCount('')).toBe(0);
    expect(wordCount('   ')).toBe(0);
  });

  test('counts a single word', () => {
    expect(wordCount('hello')).toBe(1);
  });

  test('counts multiple words separated by whitespace', () => {
    expect(wordCount('the quick brown fox')).toBe(4);
  });

  test('collapses repeated and surrounding whitespace', () => {
    expect(wordCount('  foo   bar  ')).toBe(2);
  });
});
