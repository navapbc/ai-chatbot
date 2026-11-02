// What the Jev tools may send to TypeSafe on an extension run. The form's
// field labels and the stripped participant record may go; a free-text note
// may carry anything the agent wrote, and a value from a sensitive field (or
// one shaped like an SSN) is never sent.

import type { Participant } from '@/lib/data/participants';
import {
  mustRefuseRelayText,
  stripSensitiveFields,
  stripSensitiveText,
} from '@/lib/data/sensitive-record';
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

// Nine or more digits with up to five non-alphanumeric characters between
// them (so a phone number counts too: such a turn is dropped).
const LONG_DIGITS = /\d(?:[^A-Za-z0-9]{0,5}\d){8,}/;

/** The participant record with its sensitive fields withheld, as the server
 * already sends it; stripped again here so Jev never depends on that. */
export const participantForRelay = (
  participant: Participant | null,
): Participant | null =>
  participant ? stripSensitiveFields(participant) : null;

/**
 * The caseworker's turns for Jev on an extension run. Each turn is stripped,
 * and a turn that still pairs a sensitive label with a value, or holds an
 * SSN-shaped or 9+ digit run, is dropped whole.
 */
export const messagesForRelay = (messages: string[]): string[] =>
  messages
    .map((m) => stripSensitiveText(m).normalize('NFKC'))
    .filter(
      (m) =>
        !mustRefuseRelayText(m) && !SSN_SHAPED.test(m) && !LONG_DIGITS.test(m),
    );
