// Deterministic sensitive-field classifier. Pure and dependency-free: it runs
// inside the extension service worker bundle, so no Node APIs and no imports.
//
// Classification fails toward sensitive: any one signal (input type,
// autocomplete token, or a label/name/placeholder/description match) is enough.

export const MASK = '[masked]';

export interface FieldDescriptor {
  role?: string;
  inputType?: string;
  autocomplete?: string;
  name?: string;
  label?: string;
  placeholder?: string;
  description?: string;
}

// Documentation only; the patterns below are the source of truth.
export const SENSITIVE_CATEGORIES: readonly string[] = [
  'SSN and ITIN',
  'Tax ID',
  'Bank account and routing numbers',
  'Credit and debit card numbers and security codes',
  'Passwords and passcodes',
  'PINs and security answers',
  'One-time codes',
  'Driver license and state ID numbers',
  'Passport numbers',
  'Immigration (A-) numbers',
  'Benefit case and ID numbers',
];

const SENSITIVE_AUTOCOMPLETE = new Set([
  'cc-number',
  'cc-csc',
  'cc-exp',
  'cc-exp-month',
  'cc-exp-year',
  'current-password',
  'new-password',
  'one-time-code',
]);

// Run against text normalized to lower case with punctuation collapsed to
// single spaces ('#' kept as its own token). A pattern must match whole
// words: no letter may touch either end, so "tin" does not match "tinted" and
// "ssn4" still matches "ssn". `PIN` alone is sensitive (so "pin code" is too);
// postal labels use zip or postal and are not.
const NUM = '(?:number|num|no|#)';
const PATTERNS: readonly string[] = [
  'ssn',
  'ss #',
  'ss (?:no|num|number)',
  'soc(?:ial)? sec(?:urity)?',
  'itin',
  'tin',
  'ein',
  'fein',
  'employer identification',
  'federal (?:tax )?id',
  'tax(?:payer)? id(?:entification)?',
  'routing',
  'aba',
  'iban',
  `acct ?${NUM}`,
  'acct num(?:ber)?',
  `account ${NUM}`,
  'bank (?:account|acct)',
  `(?:credit |debit )?card ${NUM}`,
  '(?:credit|debit) card(?! on file)',
  'cc num(?:ber)?',
  'cvv2?',
  'cvc2?',
  'csc',
  'card security',
  'security code',
  'card exp(?:iry|iration)?',
  'password',
  'pass ?word',
  'passwd',
  'pwd',
  'pass ?phrase',
  '(?:user|login|account) pass',
  'passcode',
  'pin',
  'security (?:answer|question)',
  'secret (?:answer|question)',
  'maiden name',
  'otp',
  'one time (?:code|password|passcode)',
  '(?:verification|authentication|auth|2fa|mfa) code',
  'drivers? licen[sc]e',
  `licen[sc]e ${NUM}`,
  `dl ?${NUM}`,
  'dl num(?:ber)?',
  `state id(?:entification)?(?: ${NUM})?`,
  'dmv id',
  'passport',
  'alien (?:registration|number)',
  'uscis',
  'a number(?! of)',
  'a #',
  'anumber',
  'medicare',
  `(?:client|beneficiary|member|participant|recipient) (?:id|${NUM})`,
  `(?:cal ?works|case|benefits?|ebt|medicaid|medi cal) (?:id|${NUM})`,
];

// A card expiry label is only sensitive next to another card signal; a lease
// "Expiration date" is an ordinary field.
const EXPIRY = '(?:exp(?:iration|iry|ires)?(?: date)?|mm yy)';
const CARD_SIGNAL = '(?:card|cc|credit|debit|cvv2?|cvc2?|csc)';
const wrap = (p: string): string => `(?<![a-z])${p}(?![a-z])`;
const EXPIRY_TEXT = new RegExp(wrap(EXPIRY));
const CARD_SIGNAL_TEXT = new RegExp(wrap(CARD_SIGNAL));

// Value-shaped placeholders: an SSN-like example or its masked form (x, #, *,
// underscore or bullet). A date (__/__/____) or phone ((___) ___-____) mask
// has a different shape and does not match.
const SSN_SHAPED =
  /(?<!\d)\d{3}-?\d{2}-?\d{4}(?!\d)|(?<![x#*_•●\d])[x#*_•●]{3}[- ]?[x#*_•●]{2}[- ]?[x#*_•●]{4}(?![x#*_•●\d])/i;

const SENSITIVE_TEXT = new RegExp(PATTERNS.map(wrap).join('|'));

const normalize = (text: string): string =>
  text
    .replace(/\b(?:[a-z]\.){2,}/gi, (m) => m.replace(/\./g, ''))
    .replace(/['’]/g, '')
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .toLowerCase()
    .replace(/#/g, ' # ')
    .replace(/[^a-z0-9#]+/g, ' ')
    .trim();

// Separator-free keys (`calworksid`, `pincode`) defeat the word-boundary
// patterns above. Tested against the WHOLE normalized text with separators
// removed, anchored at both ends, so `pinned` and `membership` still pass.
// Bare `cin` and `dl` are left out on purpose: too many false positives.
const COMPACT_SENSITIVE = /^(?:calworksid|calworksnumber|cwid|pincode|ebtcard|ebt|idnumber|dlnumber|drivinglicen[sc]e|drivinglicen[sc]enumber|driverlicen[sc]e|driverslicen[sc]e|ssnnumber|taxid|accountnumber|routingnumber|cardnumber)$/;
const compactIsSensitive = (text: string): boolean =>
  COMPACT_SENSITIVE.test(text.replace(/[^a-z0-9]/g, ''));

const autocompleteIsSensitive = (autocomplete: string | undefined): boolean =>
  typeof autocomplete === 'string' &&
  autocomplete
    .toLowerCase()
    .split(/\s+/)
    .some(
      (token) =>
        SENSITIVE_AUTOCOMPLETE.has(token) ||
        token.includes('ssn') ||
        token.includes('social'),
    );

export function isSensitiveField(field: FieldDescriptor): boolean {
  if (field.inputType?.trim().toLowerCase() === 'password') return true;
  if (autocompleteIsSensitive(field.autocomplete)) return true;
  if (
    typeof field.placeholder === 'string' &&
    SSN_SHAPED.test(field.placeholder)
  ) {
    return true;
  }
  const texts = [
    field.label,
    field.name,
    field.placeholder,
    field.description,
  ].map((t) => (typeof t === 'string' ? normalize(t) : ''));
  if (texts.some((t) => SENSITIVE_TEXT.test(t) || compactIsSensitive(t))) {
    return true;
  }
  const cardish =
    /(?:^|\s)cc-/i.test(field.autocomplete ?? '') ||
    texts.some((t) => CARD_SIGNAL_TEXT.test(t));
  return cardish && texts.some((t) => EXPIRY_TEXT.test(t));
}
