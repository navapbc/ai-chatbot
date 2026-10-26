// Removes sensitive values from a participant record before it reaches Eve.
//
// The client builds the opening message (`buildApplicationPrompt`) with the
// whole record in it. For extension runs the Next route strips the sensitive
// values from that text first, so the model, the Eve session state, Jev and
// the traces never hold them. The agent sees WITHHELD and asks the caseworker
// to type the value into the page.
//
// Pure and dependency-light: no `server-only`, so Eve code can import it.

import { scanJsonBlocks, scanJsonObjects } from '@/lib/jev/participant';
import { SSN_SHAPED, isSensitiveField } from '@/lib/relay/sensitive-fields';

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

// ---------------------------------------------------------------------------
// Refusal checks, run on the text AFTER stripSensitiveText. Whatever they find
// is a sensitive value the stripper could not remove; the route refuses the
// turn and never echoes or logs the text.

/** Classifier calls allowed per check; past this the text is refused. */
const CLASSIFY_BUDGET = 20_000;
const TOKEN_WINDOW = 8;
const WORD = /[A-Za-z][A-Za-z0-9'\u2019]*/g;
const ZERO_WIDTH = /\u200b|\u200c|\u200d|\u2060|\ufeff|\u00ad/g;
const SSN_SEPARATED = /(?<!\d)\d{3}[\s.\-\u2013]\d{2}[\s.\-\u2013]\d{4}(?!\d)/;
const CARD_RUN = /(?<!\d)\d(?:[ -]?\d){12,18}(?!\d)/g;

/** NFKC, zero-width characters removed, whitespace collapsed. */
const normalizeText = (text: string): string =>
  text.normalize('NFKC').replace(ZERO_WIDTH, '').replace(/\s+/g, ' ');

const luhnValid = (digits: string): boolean => {
  let sum = 0;
  for (let i = 0; i < digits.length; i++) {
    let d = Number(digits[digits.length - 1 - i]);
    if (i % 2 === 1) {
      d *= 2;
      if (d > 9) d -= 9;
    }
    sum += d;
  }
  return sum % 10 === 0;
};

/** An SSN-shaped token or a Luhn-valid 13-19 digit number, anywhere. */
const hasValueShape = (text: string): boolean => {
  if (SSN_SHAPED.test(text) || SSN_SEPARATED.test(text)) return true;
  for (const m of text.matchAll(CARD_RUN)) {
    const digits = m[0].replace(/\D/g, '');
    if (digits.length >= 13 && digits.length <= 19 && luhnValid(digits)) {
      return true;
    }
  }
  return false;
};

// 4+ characters containing a digit: an ID, account or card number.
const isValueLike = (token: string): boolean =>
  token.length >= 4 && /\d/.test(token);
const hasValueLike = (tokens: string[]): boolean =>
  tokens.some((t) => t.split(/[:=,"'{}[\]()]+/).some(isValueLike));

const makeClassifier = () => {
  const cache = new Map<string, boolean>();
  let budget = CLASSIFY_BUDGET;
  return {
    sensitive: (label: string): boolean => {
      const hit = cache.get(label);
      if (hit !== undefined) return hit;
      budget--;
      const result = isSensitiveField({ label });
      cache.set(label, result);
      return result;
    },
    exhausted: () => budget < 0,
  };
};

const startsWithEmpty = (rest: string): boolean =>
  rest.startsWith(WITHHELD) ||
  rest.startsWith(`"${WITHHELD}`) ||
  /^(?:null|true|false|""|'')/.test(rest);

/**
 * True when the text still has a sensitive value: a value-shaped token (SSN or
 * a Luhn-valid card number) anywhere, or a sensitive label with a value-like
 * token within 8 tokens after it (and, for a label that is not a `key:`, 8
 * tokens before it). A label whose value is already withheld is fine. The
 * text is normalised first (NFKC, zero-width characters removed, whitespace
 * collapsed) so padding cannot push a value out of the window.
 */
export function hasSensitiveValueInProse(raw: string): boolean {
  const text = normalizeText(raw);
  if (hasValueShape(text)) return true;
  const classifier = makeClassifier();
  const words = [...text.matchAll(WORD)];
  for (let i = 0; i < words.length; i++) {
    if (classifier.exhausted()) return true;
    for (let len = 1; len <= 4 && i + len <= words.length; len++) {
      const first = words[i];
      const last = words[i + len - 1];
      const label = words
        .slice(i, i + len)
        .map((w) => w[0])
        .join(' ');
      if (!classifier.sensitive(label)) continue;
      const labelEnd = (last.index ?? 0) + last[0].length;
      const rest = text.slice(labelEnd, labelEnd + 400);
      const afterLabel = rest.replace(/^[\s"':=]+/, '');
      if (!startsWithEmpty(afterLabel)) {
        const isKey = /^["']?\s*[:=]/.test(rest);
        const next = rest.split(' ').filter(Boolean).slice(0, TOKEN_WINDOW);
        if (hasValueLike(next)) return true;
        if (!isKey) {
          const from = Math.max(0, (first.index ?? 0) - 400);
          const prev = text
            .slice(from, first.index ?? 0)
            .split(' ')
            .filter(Boolean)
            .slice(-TOKEN_WINDOW);
          if (hasValueLike(prev)) return true;
        }
      }
      break;
    }
  }
  return false;
}

/** True when a `key:` in `block` is sensitive and its value is not withheld. */
const blockHasSensitiveKey = (
  block: string,
  classifier: ReturnType<typeof makeClassifier>,
): boolean => {
  for (
    let at = block.indexOf(':');
    at !== -1;
    at = block.indexOf(':', at + 1)
  ) {
    if (classifier.exhausted()) return true;
    const before = block
      .slice(Math.max(0, at - 60), at)
      .replace(/["'\s]+$/, '');
    const all = [...before.matchAll(WORD)].slice(-4);
    const lastWord = all[all.length - 1];
    if (
      !lastWord ||
      (lastWord.index ?? 0) + lastWord[0].length !== before.length
    ) {
      continue;
    }
    for (let n = 1; n <= all.length; n++) {
      const label = all
        .slice(all.length - n)
        .map((w) => w[0])
        .join(' ');
      if (!classifier.sensitive(label)) continue;
      const value = block.slice(at + 1).replace(/^\s+/, '');
      if (!startsWithEmpty(value)) return true;
      break;
    }
  }
  return false;
};

/**
 * True when the text has a `{...}` block (balanced or not) that the stripper
 * could not parse and that holds a sensitive key with a value, or when the
 * scan ran out of work budget. Both mean: refuse the turn.
 */
export function hasUnparsedSensitiveBlock(raw: string): boolean {
  const text = normalizeText(raw);
  const scan = scanJsonBlocks(text);
  if (scan.overBudget) return true;
  const classifier = makeClassifier();
  return scan.unparsed.some((r) =>
    blockHasSensitiveKey(text.slice(r.start, r.end), classifier),
  );
}

/** Whether a relay turn must be refused, given the already-stripped text. */
export const mustRefuseRelayText = (stripped: string): boolean =>
  hasUnparsedSensitiveBlock(stripped) || hasSensitiveValueInProse(stripped);
