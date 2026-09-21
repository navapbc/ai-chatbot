// Shared by lib/ai/tools/resolve-field-value.ts (legacy route) and
// agent/tools/resolve_field_value.ts (Eve), so both offer Jev the same
// candidate set.

/**
 * Flatten the record to `path -> value` leaves.
 *
 * Every scalar is offered rather than a curated subset: the cookbook guidance
 * for this pattern is to over-produce candidates, because a value that is not
 * in the list cannot be chosen. Paths are kept as the label so near-duplicate
 * values ("2400" as income vs as a case number) stay distinguishable.
 */
export const flattenRecord = (
  value: unknown,
  prefix = '',
  out: Record<string, string> = {},
): Record<string, string> => {
  if (value === null || value === undefined) return out;
  if (typeof value !== 'object') {
    const s = String(value);
    if (s !== '') out[prefix] = s;
    return out;
  }
  for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
    flattenRecord(v, prefix ? `${prefix}.${k}` : k, out);
  }
  return out;
};
