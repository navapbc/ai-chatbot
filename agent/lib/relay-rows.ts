// What the Jev tools may send to TypeSafe on an extension run. The form's
// field labels and the stripped participant record may go; a free-text note
// may carry anything the agent wrote, and a value from a sensitive field (or
// one shaped like an SSN) is never sent.

import { isSensitiveField, SSN_SHAPED } from '@/lib/relay/sensitive-fields';

type GapRow = { field: string; note?: string };
type SummaryRow = { field: string; value?: string };

const sensitive = (field: string): boolean =>
  isSensitiveField({ label: field, name: field });

/** Gap rows for Jev on an extension run: no free-text note. */
export const gapRowsForRelay = <T extends GapRow>(rows: T[]): T[] =>
  rows.map(({ note: _note, ...rest }) => rest as unknown as T);

/** Summary rows for Jev on an extension run: sensitive values removed. */
export const summaryRowsForRelay = <T extends SummaryRow>(rows: T[]): T[] =>
  rows.map((row) => {
    if (row.value === undefined) return row;
    if (sensitive(row.field) || SSN_SHAPED.test(row.value)) {
      const { value: _value, ...rest } = row;
      return rest as unknown as T;
    }
    return row;
  });
