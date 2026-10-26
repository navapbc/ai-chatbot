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
  'social sec(?:urity)?',
  'itin',
  'tin',
  'ein',
  'tax(?:payer)? id(?:entification)?',
  'routing',
  'aba',
  'iban',
  `acct ${NUM}`,
  `account ${NUM}`,
  `bank account`,
  `(?:credit |debit )?card ${NUM}`,
  'cc num(?:ber)?',
  'cvv2?',
  'cvc2?',
  'csc',
  'card security',
  'security code',
  'card exp(?:iry|iration)?',
  'password',
  'passcode',
  'pin',
  'security (?:answer|question)',
  'secret (?:answer|question)',
  'otp',
  'one time (?:code|password|passcode)',
  '(?:verification|authentication|auth|2fa|mfa) code',
  'drivers? licen[sc]e',
  `dl ${NUM}`,
  `state id(?: ${NUM})?`,
  'passport',
  'alien (?:registration|number)',
  'uscis',
  'a number(?! of)',
  'a #',
  'anumber',
  `(?:cal ?works|case|benefits?|ebt|medicaid|medi cal) (?:id|${NUM})`,
];

const SENSITIVE_TEXT = new RegExp(
  PATTERNS.map((p) => `(?<![a-z])${p}(?![a-z])`).join('|'),
);

const normalize = (text: string): string =>
  text
    .replace(/['’]/g, '')
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .toLowerCase()
    .replace(/#/g, ' # ')
    .replace(/[^a-z0-9#]+/g, ' ')
    .trim();

const textIsSensitive = (text: string | undefined): boolean =>
  typeof text === 'string' && SENSITIVE_TEXT.test(normalize(text));

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
  if (field.inputType?.toLowerCase() === 'password') return true;
  if (autocompleteIsSensitive(field.autocomplete)) return true;
  return (
    textIsSensitive(field.label) ||
    textIsSensitive(field.name) ||
    textIsSensitive(field.placeholder) ||
    textIsSensitive(field.description)
  );
}
