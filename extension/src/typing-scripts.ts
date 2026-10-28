// Page-side functions the bridge runs in its own isolated world (as CDP
// `expression` or `functionDeclaration` source) to read keyboard focus and
// to type a sensitive value for the caseworker (bridge.ts typeSensitive).
// Pure strings, no chrome.*, so the e2e suite can run the exact source in a
// real page. They answer fixed words or an element, never page text.

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
