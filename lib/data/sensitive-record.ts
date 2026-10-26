// Removes sensitive values from a participant record before it reaches Eve.
//
// The client builds the opening message (`buildApplicationPrompt`) with the
// whole record in it. For extension runs the Next route strips the sensitive
// values from that text first, so the model, the Eve session state, Jev and
// the traces never hold them. The agent sees WITHHELD and asks the caseworker
// to type the value into the page.
//
// Pure and dependency-light: no `server-only`, so Eve code can import it.

import { findParticipantSpan } from '@/lib/jev/participant';
import { isSensitiveField } from '@/lib/relay/sensitive-fields';

export const WITHHELD = '[withheld: ask the caseworker]';

/** Dotted record paths that are sensitive whatever their key is called. */
export const SENSITIVE_RECORD_PATHS: readonly string[] = [
  'participant.calworks_id',
];

const isPlainObject = (v: unknown): v is Record<string, unknown> =>
  typeof v === 'object' && v !== null && !Array.isArray(v);

const withhold = (value: unknown): unknown =>
  // null / boolean carry no secret and keep the record's shape readable.
  value === null || value === undefined || typeof value === 'boolean'
    ? value
    : WITHHELD;

const strip = (value: unknown, path: string): unknown => {
  if (Array.isArray(value)) return value.map((v) => strip(v, path));
  if (!isPlainObject(value)) return value;
  const out: Record<string, unknown> = {};
  for (const [key, v] of Object.entries(value)) {
    const here = path ? `${path}.${key}` : key;
    out[key] =
      SENSITIVE_RECORD_PATHS.includes(here) ||
      isSensitiveField({ name: key, label: key })
        ? withhold(v)
        : strip(v, here);
  }
  return out;
};

export function stripSensitiveFields<T>(record: T): T {
  return strip(record, '') as T;
}

/**
 * Replace every participant record in `text` with its stripped form, in the
 * format `buildApplicationPrompt` writes. Fail closed: a second record in the
 * same message is stripped too. Text with no record comes back unchanged.
 */
export function stripSensitiveText(text: string): string {
  let out = '';
  let rest = text;
  for (;;) {
    const span = findParticipantSpan(rest);
    if (!span) return out + rest;
    out +=
      rest.slice(0, span.start) +
      JSON.stringify(stripSensitiveFields(span.participant), null, 2);
    rest = rest.slice(span.end);
  }
}
