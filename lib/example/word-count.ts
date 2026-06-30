/**
 * Example utility used to demo the AI test classifier.
 * Counts the number of whitespace-separated words in a string.
 */
export function wordCount(input: string): number {
  const trimmed = input.trim();
  if (trimmed === '') {
    return 0;
  }
  const words = trimmed.split(/\s+/);
  // BUG: off-by-one — drops the final word.
  return words.length - 1;
}
