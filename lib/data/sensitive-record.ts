// Removes sensitive values from a participant record before it reaches Eve.
//
// The client builds the opening message (`buildApplicationPrompt`) with the
// whole record in it. For extension runs the Next route strips the sensitive
// values from that text first, so the model, the Eve session state, Jev and
// the traces never hold them. The agent sees WITHHELD and asks the caseworker
// to type the value into the page.
//
// Pure and dependency-light: no `server-only`, so Eve code can import it.

import { scanJsonObjects } from '@/lib/jev/participant';
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
      SENSITIVE_RECORD_PATHS.some(
        (p) => here === p || here.endsWith(`.${p}`),
      ) || isSensitiveField({ name: key, label: key })
        ? withhold(v)
        : strip(v, here);
  }
  return out;
};

export function stripSensitiveFields<T>(record: T): T {
  return strip(record, '') as T;
}

/**
 * Strip every JSON object in `text`, whatever its shape: a record whose name
 * is a string, a missing or numeric record_id, or no participant fields at
 * all. Blocks that do not parse as JSON are left alone (the prose guard below
 * is the backstop). A block with nothing to strip keeps its exact original
 * text, so numbers beyond 2^53 are not rewritten. Text with no object comes
 * back unchanged.
 *
 * A record that changed is re-serialised in the format `buildApplicationPrompt`
 * uses (2-space indent), so it still parses with `extractParticipantFromText`.
 * Fail closed: every object in the message is stripped, not just the first.
 */
export function stripSensitiveText(text: string): string {
  let out = '';
  let at = 0;
  for (const span of scanJsonObjects(text)) {
    const stripped = stripSensitiveFields(span.value);
    const next = JSON.stringify(stripped, null, 2);
    if (next === JSON.stringify(span.value, null, 2)) continue;
    out += text.slice(at, span.start) + next;
    at = span.end;
  }
  return out + text.slice(at);
}

/** Longest relay message the route accepts, in characters. */
export const MAX_RELAY_TEXT_CHARS = 256_000;

// How far after a sensitive label a value may sit and still count.
const VALUE_WINDOW_CHARS = 40;
const WORD = /[A-Za-z][A-Za-z0-9'\u2019]*/g;
// 4+ characters containing a digit: an SSN, card, account or ID number.
const isValueLike = (token: string): boolean =>
  token.length >= 4 && /\d/.test(token);

/**
 * True when `text` still has a sensitive label (one to four words that
 * `isSensitiveField` classifies as sensitive) followed within 40 characters by
 * a value-like token. Run on the text AFTER `stripSensitiveText`: what is left
 * is a value typed as prose or in JSON that did not parse. The caller refuses
 * the turn; it must not echo or log the text.
 */
export function hasSensitiveValueInProse(text: string): boolean {
  const words = [...text.matchAll(WORD)];
  for (let i = 0; i < words.length; i++) {
    for (let len = 1; len <= 4 && i + len <= words.length; len++) {
      const last = words[i + len - 1];
      const label = words
        .slice(i, i + len)
        .map((w) => w[0])
        .join(' ');
      if (!isSensitiveField({ label })) continue;
      const after = text.slice(
        (last.index ?? 0) + last[0].length,
        (last.index ?? 0) + last[0].length + VALUE_WINDOW_CHARS,
      );
      if (after.split(/[\s:=,"'{}[\]()]+/).some(isValueLike)) return true;
      break;
    }
  }
  return false;
}
