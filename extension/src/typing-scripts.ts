// Page-side functions the bridge runs in its own isolated world (as CDP
// `expression` or `functionDeclaration` source) to read keyboard focus and
// to type a sensitive value for the caseworker (bridge.ts typeSensitive), to
// find the required fields still empty (showUnfilledField), and to read a
// confirmation number after a submission (readConfirmation). Pure strings,
// no chrome.*, so the e2e suite can run the exact source in a real page. They
// answer fixed words, counts or elements, never page text, except
// CONFIRMATION_READ's one labelled token.

import { SSN_SHAPED } from '../../lib/relay/sensitive-fields';

/** Elements that show another document. */
export const FRAME_OWNER_TAGS = /^(IFRAME|FRAME|OBJECT|EMBED|FENCEDFRAME)$/;

/**
 * The element keyboard focus is on, as a function of a document: followed
 * through open shadow roots and into reachable same-origin frames, stopping
 * at a frame whose document is not reachable (its owner is returned).
 */
export const DEEP_ACTIVE = `(doc) => {
    let el = doc.activeElement;
    for (let i = 0; i < 64 && el; i++) {
      while (el.shadowRoot && el.shadowRoot.activeElement) {
        el = el.shadowRoot.activeElement;
      }
      if (!${FRAME_OWNER_TAGS}.test(el.tagName)) return el;
      let inner = null;
      try { inner = el.contentDocument; } catch {}
      if (!inner || !inner.activeElement) return el;
      el = inner.activeElement;
    }
    return el;
  }`;

// Input types whose value is the text typed into them (an input without a
// type, or with an unknown one, reports `text`).
export const TEXT_ENTRY_TYPES = [
  'text',
  'password',
  'tel',
  'email',
  'number',
  'search',
  'url',
];

/**
 * Run on the focused element (page script cannot change what these getters
 * answer in an isolated world): 'entry' for an enabled, editable text input
 * or textarea, 'frame' for a frame owner focus stopped at (a frame the world
 * cannot see into), 'other' for anything else (contenteditable included).
 * Not by `instanceof`: an element of a same-origin frame has that frame's
 * prototypes.
 */
export const FIELD_KIND = `function () {
  const el = this;
  if (!el || el.nodeType !== 1 || !el.isConnected) return 'other';
  if (${FRAME_OWNER_TAGS}.test(el.tagName)) return 'frame';
  const tag = el.localName;
  if (tag === 'input') {
    if (!${JSON.stringify(TEXT_ENTRY_TYPES)}.includes(el.type)) return 'other';
  } else if (tag !== 'textarea') {
    return 'other';
  }
  if (el.getAttribute('aria-disabled') === 'true') return 'other';
  if (el.getAttribute('aria-readonly') === 'true') return 'other';
  return el.matches(':enabled:read-write') ? 'entry' : 'other';
}`;

/**
 * Types `text` into the element it runs on (the field that was checked),
 * replacing what it holds, in one script task: no other script runs between
 * the focus check and the insertion. Answers:
 * - 'moved': nothing typed, the element is no longer the focused one, or no
 *   longer connected, enabled and editable;
 * - 'typed': it now holds a different value than before, or already held
 *   exactly `text` (nothing to do);
 * - 'unchanged': its value did not change (a page that refused the input,
 *   or a page `beforeinput` handler that moved focus or the selection so
 *   the text went elsewhere).
 * The existing content is selected first, then replaced as typing does
 * (execCommand insertText: beforeinput and input events, undo), or, where
 * the browser will not do that, by the element's own value setter (found
 * here, so page overrides are bypassed, as frameworks expect) followed by
 * input and change events.
 */
export const TYPE_INTO = `function (text) {
  const el = this;
  if ((${DEEP_ACTIVE})(document) !== el) return 'moved';
  if (!el.isConnected || !el.matches(':enabled:read-write')) return 'moved';
  const before = el.value;
  if (before === text) return 'typed';
  try { el.select(); } catch {}
  let inserted = false;
  try {
    inserted = el.ownerDocument.execCommand('insertText', false, text);
  } catch {}
  if (!inserted) {
    let proto = Object.getPrototypeOf(el);
    let setter;
    while (proto && !setter) {
      setter = Object.getOwnPropertyDescriptor(proto, 'value')?.set;
      proto = Object.getPrototypeOf(proto);
    }
    if (!setter) return 'unchanged';
    setter.call(el, text);
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }
  return el.value !== before ? 'typed' : 'unchanged';
}`;

// Input types that hold no answer the caseworker gives (buttons, hidden).
export const NOT_ANSWER_TYPES = [
  'hidden',
  'submit',
  'button',
  'image',
  'reset',
];

/**
 * Required fields that are still empty, as a function of a document, for
 * showUnfilledField (bridge.ts): walked in DOM order through open shadow
 * roots (never into frames: a cross-origin frame is not reachable, and a
 * same-origin one is not scanned in v1). A field counts when it is visible
 * (has a box, not visibility:hidden), enabled (not :disabled, not
 * aria-disabled, not inert) and editable, and either fails
 * `validity.valueMissing` or has aria-required="true" with nothing chosen
 * (an empty text control or select, an unchecked checkbox, a radio group
 * with none checked). A radio group counts once, as its first counted radio.
 * ARIA widgets (role textbox, searchbox, combobox, checkbox, switch,
 * radiogroup, listbox) with aria-required="true" count when empty.
 *
 * Answers `{ els }`: the candidate elements. Values
 * are read here only to decide emptiness; nothing derived from a value, a
 * label, a name or page text is returned. The caller keeps the object by
 * reference and projects only the count (UNFILLED_SUMMARY).
 */
export const SCAN_UNFILLED = `(doc) => {
  const skip = ${JSON.stringify(NOT_ANSWER_TYPES)};
  const maxNodes = 20000;
  const els = [];
  const radioGroups = new Set();
  let visited = 0;
  const ariaRequired = (el) => el.getAttribute('aria-required') === 'true';
  const usable = (el) => {
    if (el.matches(':disabled')) return false;
    if (el.getAttribute('aria-disabled') === 'true') return false;
    if (el.closest('[inert]')) return false;
    if (el.getClientRects().length === 0) return false;
    const view = el.ownerDocument.defaultView;
    const style = view ? view.getComputedStyle(el) : null;
    return !style || (style.visibility !== 'hidden' && style.visibility !== 'collapse');
  };
  const radiosOf = (el) => {
    if (!el.name) return [el];
    const root = el.getRootNode();
    return Array.from(root.querySelectorAll('input[type="radio"]')).filter(
      (r) => r.name === el.name && r.form === el.form,
    );
  };
  const add = (el) => {
    els.push(el);
  };
  const consider = (el) => {
    const tag = el.localName;
    if (tag === 'input') {
      const type = el.type;
      if (skip.includes(type)) return;
      if (type === 'radio') {
        const group = radiosOf(el);
        const key = group[0] || el;
        if (radioGroups.has(key)) return;
        const missing =
          el.validity.valueMissing ||
          (group.some(ariaRequired) && !group.some((r) => r.checked));
        if (!missing || !usable(el)) return;
        radioGroups.add(key);
        add(el);
        return;
      }
      if (type === 'checkbox') {
        const missing = el.validity.valueMissing || (ariaRequired(el) && !el.checked);
        if (missing && usable(el)) add(el);
        return;
      }
      if (el.readOnly) return;
      const missing = el.validity.valueMissing || (ariaRequired(el) && el.value === '');
      if (missing && usable(el)) add(el);
      return;
    }
    if (tag === 'textarea') {
      if (el.readOnly) return;
      const missing = el.validity.valueMissing || (ariaRequired(el) && el.value === '');
      if (missing && usable(el)) add(el);
      return;
    }
    if (tag === 'select') {
      const missing = el.validity.valueMissing || (ariaRequired(el) && el.value === '');
      if (missing && usable(el)) add(el);
      return;
    }
    if (!ariaRequired(el)) return;
    if (el.getAttribute('aria-readonly') === 'true') return;
    const role = (el.getAttribute('role') || '').trim().toLowerCase();
    let empty;
    if (role === 'textbox' || role === 'searchbox' || role === 'combobox') {
      if (el.querySelector('input, select, textarea')) return;
      empty = (el.textContent || '').trim() === '';
    } else if (role === 'checkbox' || role === 'switch') {
      empty = el.getAttribute('aria-checked') !== 'true';
    } else if (role === 'radiogroup') {
      empty =
        !el.querySelector('[role="radio"][aria-checked="true"]') &&
        !el.querySelector('input[type="radio"]:checked');
    } else if (role === 'listbox') {
      empty = !el.querySelector('[aria-selected="true"]');
    } else {
      return;
    }
    if (empty && usable(el)) add(el);
  };
  const walk = (root) => {
    for (const el of root.querySelectorAll('*')) {
      visited += 1;
      if (visited > maxNodes) return;
      try {
        consider(el);
      } catch {}
      if (el.shadowRoot) walk(el.shadowRoot);
    }
  };
  walk(doc);
  return { els };
}`;

/** Run on SCAN_UNFILLED's answer: how many candidates, nothing else. */
export const UNFILLED_SUMMARY = `function () {
  return { count: this.els.length };
}`;

/** Run on SCAN_UNFILLED's answer: the candidate at `index`, by reference. */
export const UNFILLED_AT = `function (index) {
  return this.els[index];
}`;

/**
 * The highlight's fallback, run on the field: scroll it into view and give
 * it keyboard focus (the browser's focus ring marks it). Answers nothing.
 */
export const FOCUS_FIELD = `function () {
  try { this.scrollIntoView({ block: 'center', inline: 'nearest' }); } catch {}
  try { this.focus({ preventScroll: true }); } catch {}
}`;

/**
 * The labels a confirmation number may follow (P4c4-f), canonical (lower
 * case, single spaces), and the kind each one reports: only the
 * `confirmation` labels say the form was submitted (benefit portals show an
 * application number on every step). `case number` is not one: it is on the
 * sensitive-field list.
 */
export const CONFIRMATION_LABEL_KINDS: Readonly<
  Record<string, 'confirmation' | 'other'>
> = {
  'confirmation number': 'confirmation',
  'confirmation #': 'confirmation',
  'confirmation code': 'confirmation',
  'reference number': 'other',
  'application number': 'other',
  'tracking number': 'other',
};

/** How many characters of the page's visible text CONFIRMATION_READ reads. */
export const CONFIRMATION_TEXT_CAP = 20_000;
/** How far around a match CONFIRMATION_READ looks for an SSN echo. */
export const CONFIRMATION_NEAR_CHARS = 200;
/** A confirmation token: 4 to 32 of these characters, with a digit. */
export const CONFIRMATION_TOKEN = /^[A-Za-z0-9][A-Za-z0-9\-_/.]*$/;
export const CONFIRMATION_TOKEN_MIN = 4;
export const CONFIRMATION_TOKEN_MAX = 32;

// A label, as a label: a whole word (never "confirmation numbers are
// emailed"), then up to 8 of ':', '#' and whitespace, then the whole run of
// token characters (never cut short: a run over 32 is dropped, not trimmed).
// Group 1 is the label, group 2 the run.
const LABELLED_TOKEN =
  /(?<![A-Za-z0-9])(confirmation[ \t]*#|(?:confirmation[ \t]+(?:number|code)|(?:reference|application|tracking)[ \t]+number)(?![A-Za-z0-9]))[\s:#]{0,8}([A-Za-z0-9][A-Za-z0-9\-_/.]*)/gi;
// Sentence punctuation after a token ("... is AB-1234.").
const TRAILING_PUNCTUATION = /[._/-]+$/;
const HAS_DIGIT = /\d/;
// SSN_SHAPED covers dashes and none; this covers spaces and dots.
const SPACED_SSN = /(?<!\d)\d{3}[ .]\d{2}[ .]\d{4}(?!\d)/;
const WHITESPACE_RUN = /\s+/g;

/**
 * The confirmation number on the page, as a function of a document, for
 * readConfirmation (bridge.ts). Reads the body's visible text (innerText:
 * hidden elements are left out), the first CONFIRMATION_TEXT_CAP characters,
 * and matches only a label from CONFIRMATION_LABEL_KINDS followed by a token
 * of CONFIRMATION_TOKEN_MIN..MAX token characters with a digit (trailing
 * sentence punctuation trimmed). A `confirmation` label wins over an earlier
 * other one. Answers `{ token, label, nearSensitive }` (the label canonical,
 * from the fixed list, never as the page wrote it; `nearSensitive` true when
 * the text within CONFIRMATION_NEAR_CHARS of the match holds an SSN-shaped
 * value) or null. Never any other page text, never throws, never logs.
 */
export const CONFIRMATION_READ = `(doc) => {
  try {
    const body = doc && doc.body;
    if (!body) return null;
    const text = String(body.innerText || '')
      .slice(0, ${CONFIRMATION_TEXT_CAP})
      .normalize('NFKC');
    const kinds = ${JSON.stringify(CONFIRMATION_LABEL_KINDS)};
    const pattern = ${LABELLED_TOKEN};
    const ssn = [${SSN_SHAPED}, ${SPACED_SSN}];
    let other = null;
    let match;
    for (let i = 0; i < 100 && (match = pattern.exec(text)) !== null; i++) {
      const token = match[2].replace(${TRAILING_PUNCTUATION}, '');
      if (
        token.length < ${CONFIRMATION_TOKEN_MIN} ||
        token.length > ${CONFIRMATION_TOKEN_MAX} ||
        !${HAS_DIGIT}.test(token)
      ) {
        continue;
      }
      const raw = match[1].toLowerCase().replace(${WHITESPACE_RUN}, ' ');
      const label = raw.includes('#') ? 'confirmation #' : raw;
      if (!Object.prototype.hasOwnProperty.call(kinds, label)) continue;
      const end = match.index + match[0].length;
      const near = text.slice(
        Math.max(0, match.index - ${CONFIRMATION_NEAR_CHARS}),
        end + ${CONFIRMATION_NEAR_CHARS},
      );
      const found = {
        token: token,
        label: label,
        nearSensitive: ssn.some((r) => r.test(near)),
      };
      if (kinds[label] === 'confirmation') return found;
      if (other === null) other = found;
    }
    return other;
  } catch {
    return null;
  }
}`;
