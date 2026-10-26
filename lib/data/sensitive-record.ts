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

// ---------------------------------------------------------------------------
// Refusal checks, run on the text AFTER stripSensitiveText. Whatever they find
// is a sensitive value the stripper could not remove; the route refuses the
// turn and never echoes or logs the text.

/** Classifier calls allowed per check; past this the text is refused. */
const CLASSIFY_BUDGET = 20_000;
const TOKEN_WINDOW = 8;
const WORD = /[A-Za-z][A-Za-z0-9'\u2019]*/g;
const ZERO_WIDTH = /\u200b|\u200c|\u200d|\u2060|\ufeff|\u00ad/g;
// SSN grouping 3-2-4 with a separator between the groups. Guard-local: the
// shared SSN_SHAPED accepts missing separators and so also matches ZIP+4
// (5-4). A contiguous 9-digit number is not flagged on its own; it is caught
// when a sensitive label sits within the token window.
const SSN_GROUPED =
  /(?<!\d)\d{3}[^\p{L}\p{N}]{1,3}\d{2}[^\p{L}\p{N}]{1,3}\d{4}(?!\d)/u;
// A maximal run of digits joined by up to 3 separator characters each.
const DIGIT_RUN = /\d(?:[^\p{L}\p{N}]{0,3}\d)*/gu;
// Card-length windows tested per run, and per message; over either => refuse.
const MAX_WINDOWS_PER_RUN = 256;
const MAX_WINDOWS_PER_TEXT = 20_000;

// Every Unicode decimal digit (Arabic-Indic, full-width, mathematical...) to
// ASCII. Decimal digits sit in contiguous runs starting at a zero, so the
// value is the offset from the start of the run, modulo 10.
const toAsciiDigit = (c: string): string => {
  const cp = c.codePointAt(0) as number;
  let start = cp;
  for (let n = 0; n < 100; n++) {
    if (!/^\p{Nd}$/u.test(String.fromCodePoint(start - 1))) break;
    start--;
  }
  return String((cp - start) % 10);
};

/** NFKC, zero-width removed, non-ASCII digits mapped, whitespace collapsed. */
const normalizeText = (text: string): string =>
  text
    .normalize('NFKC')
    .replace(ZERO_WIDTH, '')
    .replace(/(?![0-9])\p{Nd}/gu, toAsciiDigit)
    .replace(/\s+/g, ' ');

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

/**
 * An SSN-grouped (3-2-4) token, or a Luhn-valid 13-19 digit number anywhere.
 * Every 13-19 digit window of a digit run is tested, so a card next to a
 * phone, a ZIP or a longer run is still found. A run with more windows than
 * the cap is itself refused.
 */
const hasValueShape = (text: string): boolean => {
  if (SSN_GROUPED.test(text)) return true;
  let total = 0;
  for (const m of text.matchAll(DIGIT_RUN)) {
    const digits = m[0].replace(/\D/g, '');
    if (digits.length < 13) continue;
    let perRun = 0;
    for (let len = 13; len <= 19; len++) {
      for (let at = 0; at + len <= digits.length; at++) {
        if (++perRun > MAX_WINDOWS_PER_RUN || ++total > MAX_WINDOWS_PER_TEXT) {
          return true;
        }
        if (luhnValid(digits.slice(at, at + len))) return true;
      }
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

// The withheld placeholder is swapped for this mark before label scanning, so
// its own words ("ask the caseworker") never form part of a label window.
const EMPTY_MARK = '\u0001';
const EMPTY_VALUE = new RegExp(
  `^(?:"?${EMPTY_MARK}"?|"?${WITHHELD.replace(/[\\^$.*+?()[\]{}|]/g, '\\$&')}"?|null|true|false|""|'')(?=$|[\\s,}\\]])`,
);

/** Length of an empty/withheld value token at the start of `rest`, else 0. */
const emptyLength = (rest: string): number =>
  EMPTY_VALUE.exec(rest)?.[0].length ?? 0;

/**
 * True when the text still has a sensitive value: a value-shaped token (SSN or
 * a Luhn-valid card number) anywhere, or a sensitive label with a value-like
 * token within 8 tokens after it (and, for a label that is not a `key:`, 8
 * tokens before it). A label whose value is already withheld is fine. The
 * text is normalised first (NFKC, zero-width characters removed, whitespace
 * collapsed) so padding cannot push a value out of the window.
 */
export function hasSensitiveValueInProse(raw: string): boolean {
  const text = normalizeText(raw).split(WITHHELD).join(EMPTY_MARK);
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
      // An empty or withheld value is skipped, but the tokens after it are
      // still scanned ('ssn: null 123-45-6789' must not pass).
      const afterLabel = rest.replace(/^[\s"':=]+/, '');
      const skip = emptyLength(afterLabel);
      const next = afterLabel
        .slice(skip)
        .split(' ')
        .filter(Boolean)
        .slice(0, TOKEN_WINDOW);
      if (hasValueLike(next)) return true;
      const isKey = /^["']?\s*[:=]/.test(rest);
      if (!isKey) {
        const from = Math.max(0, (first.index ?? 0) - 400);
        const prev = text
          .slice(from, first.index ?? 0)
          .split(' ')
          .filter(Boolean)
          .slice(-TOKEN_WINDOW);
        if (hasValueLike(prev)) return true;
      }
      // The shortest sensitive window is the label; longer ones add words.
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
      const skip = emptyLength(value);
      if (skip === 0) return true;
      const next = value.slice(skip).split(' ').filter(Boolean);
      if (hasValueLike(next.slice(0, TOKEN_WINDOW))) return true;
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
  return scan.unparsed.some((r) => {
    const decoded = decodeEscapes(text.slice(r.start, r.end));
    // An escape that does not decode is hiding something: fail closed.
    return (
      decoded === null ||
      blockHasSensitiveKey(normalizeText(decoded), classifier) ||
      hasSensitiveValueInProse(decoded)
    );
  });
}

const SIMPLE_ESCAPES: Record<string, string> = {
  '"': '"',
  '\\': '\\',
  '/': '/',
  b: '\b',
  f: '\f',
  n: '\n',
  r: '\r',
  t: '\t',
};

/** Decode JSON string escapes; null when one is malformed. */
const decodeEscapes = (block: string): string | null => {
  let bad = false;
  const out = block.replace(
    /\\(u[0-9a-fA-F]{4}|[\s\S])/g,
    (_m, esc: string) => {
      if (esc.length === 5)
        return String.fromCharCode(Number.parseInt(esc.slice(1), 16));
      const simple = SIMPLE_ESCAPES[esc];
      if (simple === undefined) bad = true;
      return simple ?? '';
    },
  );
  return bad ? null : out;
};

const MAX_VALUE_NODES = 50_000;

/** True when a decoded string value or key of a parsed block has a value shape. */
const parsedBlocksHaveValueShape = (raw: string): boolean => {
  const scan = scanJsonBlocks(normalizeText(raw));
  const todo: unknown[] = scan.blocks.map((b) => b.value);
  for (let n = 0; todo.length > 0 && n < MAX_VALUE_NODES; n++) {
    const v = todo.pop();
    if (typeof v === 'string') {
      if (hasSensitiveValueInProse(v)) return true;
    } else if (typeof v === 'number') {
      if (hasSensitiveValueInProse(String(v))) return true;
    } else if (v && typeof v === 'object') {
      for (const [k, c] of Object.entries(v)) {
        if (hasSensitiveValueInProse(k)) return true;
        todo.push(c);
      }
    }
  }
  return todo.length > 0;
};

/** Whether a relay turn must be refused, given the already-stripped text. */
export const mustRefuseRelayText = (stripped: string): boolean =>
  hasUnparsedSensitiveBlock(stripped) ||
  hasSensitiveValueInProse(stripped) ||
  parsedBlocksHaveValueShape(stripped);
