/**
 * Convert an arbitrary string into a URL-friendly slug.
 *
 * Lowercases, trims, replaces any run of non-alphanumeric characters with a
 * single hyphen, and strips leading/trailing hyphens.
 */
export function slugify(input: string): string {
  return input
    .normalize('NFKD')
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
}
