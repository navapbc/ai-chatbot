// Response redactor for extension (relay) runs. Pure and dependency-free
// beyond the classifier: it is bundled into the extension service worker, so
// no chrome.*, no Node APIs, and it never logs.
//
// Layers, applied to every CDP result and event before it leaves the worker:
// 1. Field masking (accessibility trees). Every text-bearing field of a node
//    classified sensitive, and of its whole subtree (StaticText, InlineTextBox
//    and option children repeat the value), becomes MASK; an empty field stays
//    empty so the agent can tell filled from empty. A filled field-like node
//    that cannot be classified (describe() failed, timed out or is missing)
//    is treated as sensitive. Nodes labelled or described by a masked node
//    are masked too.
// 2. Field masking (DOM nodes). The value attribute and text of every INPUT,
//    TEXTAREA, SELECT, OPTION and editable element is masked unconditionally;
//    a sensitive one keeps only an allowlist of attributes.
// 3. Known-value scrubbing. Every string anywhere in a result or event has
//    each registered value, in formatted, encoded and normalized forms,
//    replaced with MASK. AX text nodes that held a value, or that hold one
//    split across siblings, are masked whole. Protocol identifiers and URL
//    hosts are left alone so routing and node references keep working.
//
// Fail closed: any error in a redaction path returns a safe value
// (`{ nodes: [] }` for accessibility trees, `null` otherwise, and `null` for
// an event, meaning drop it), never the input. The input is never mutated.

import {
  type FieldDescriptor,
  isSensitiveField,
  MASK,
} from './sensitive-fields';

export interface Redactor {
  /**
   * Registers a sensitive value typed during the run, in memory only.
   * Returns `true` when the value is registered, already known, or shorter
   * than 3 characters (too short to scrub without destroying ordinary text;
   * field masking still applies). A 3-character value is scrubbed only as a
   * whole token. Returns `false`, keeping every earlier value, when the store
   * already holds `maxKnownValues` values, the value is longer than 512
   * characters, or it is not a string: the caller must then refuse to type
   * it (fail closed).
   */
  addKnownValue(value: string): boolean;
  /**
   * Read-only: true when `text` contains a registered value in any form the
   * scrubber recognizes. Used to mask pixels (screenshots) where the text
   * itself cannot be rewritten.
   */
  hasKnownValue(text: string): boolean;
  clear(): void;
  redactResult(
    method: string,
    result: unknown,
    describe?: (backendNodeId: number) => Promise<FieldDescriptor | undefined>,
  ): Promise<unknown>;
  /** Returns `null` when the event cannot be redacted; drop it. */
  redactEvent(method: string, params: unknown): unknown;
}

export interface RedactorOptions {
  maxKnownValues?: number;
  /** Per describe() call; a slower call counts as a failure. */
  describeTimeoutMs?: number;
  /**
   * For all describe() calls of one tree; calls not finished by then count
   * as failures (the field is masked).
   */
  describeDeadlineMs?: number;
  /**
   * An extra sensitive-field verdict for a form field the deterministic
   * classifier calls ordinary (see `askExtra` for which nodes). `true` masks
   * the field exactly like a deterministic sensitive one; `false`, a throw, a
   * rejection, a non-boolean answer or no answer within `classifyTimeoutMs`
   * (or the tree deadline) keeps the deterministic verdict. It can only add
   * masking. It receives a fresh descriptor of the field's own naming
   * strings, never a value, URL parameter or other page text; the caller
   * sanitizes it further before it leaves the worker.
   */
  classifyExtra?: (descriptor: FieldDescriptor) => Promise<boolean>;
  /** Per classifyExtra() call; default 2000. */
  classifyTimeoutMs?: number;
}

const MIN_KNOWN_LENGTH = 3;
const MAX_KNOWN_VALUE_LENGTH = 512;
const DEFAULT_MAX_KNOWN_VALUES = 32;
const DEFAULT_DESCRIBE_TIMEOUT_MS = 1500;
const DEFAULT_DESCRIBE_DEADLINE_MS = 5000;
const DESCRIBE_CONCURRENCY = 16;
const DEFAULT_CLASSIFY_TIMEOUT_MS = 2000;
const CLASSIFY_CONCURRENCY = 8;
const CLASSIFY_CACHE_SIZE = 256;
// FieldDescriptor keys in a fixed order: the descriptor handed to
// classifyExtra and its cache key are built from these and nothing else.
const DESCRIPTOR_KEYS: readonly (keyof FieldDescriptor)[] = [
  'role',
  'inputType',
  'autocomplete',
  'name',
  'label',
  'placeholder',
  'description',
];
const MAX_DEPTH = 256;
const SEPARATORS = /[\s\-./]+/;
const JOINERS = ['', ' ', '-', '.', '/'];
const INVISIBLE = /(?:\u00ad|\u200b|\u200c|\u200d|\u2060|\ufeff)/g;
const NON_ASCII = /[\u0080-\uffff]/;
const BULLETS_ONLY = /^[•●◦■*∗]+$/;
const MARKUP_TAG = /<[^>]*>/g;
const MARKUP_COMMENT = /<!--[\s\S]*?-->/g;
const ANGLE_BRACKETS = /[<>]/g;
const WHITESPACE_RUN = /\s+/g;
const HAS_WHITESPACE_RUN = /\s\s/;
const HAS_MARKUP = /[<>&]/;
const HTML_ENTITY =
  /&(?:(lt|gt|amp|quot|apos|nbsp)|#(\d{1,7})|#x([0-9a-f]{1,6}));/gi;
const NAMED_ENTITIES: Record<string, string> = {
  lt: '<',
  gt: '>',
  amp: '&',
  quot: '"',
  apos: "'",
  nbsp: ' ',
};
// Network header maps: their keys are page or server data, not fields.
const HEADER_KEYS = new Set(['headers', 'requestHeaders', 'responseHeaders']);
// Between the digits of a known number: whitespace, dashes, dots, slashes,
// soft hyphen, zero-width characters, and their URL, HTML and JSON escapes.
const DIGIT_SEPARATOR =
  '(?:[\\s\\u00ad\\u200b-\\u200d\\u2060\\ufeff\\u2010-\\u2015\\u2212./_+\\-]|%[0-9A-Fa-f]{2}|&#?[0-9A-Za-z]{1,8};|\\\\u[0-9A-Fa-f]{4})';
const DIGITS_AND_SEPARATORS = /^[\d\s‐-―−./_+-]+$/;

const AX_FULL_TREE = 'Accessibility.getFullAXTree';
const AX_PARTIAL_METHODS = new Set([
  'Accessibility.getPartialAXTree',
  'Accessibility.queryAXTree',
  'Accessibility.getChildAXNodes',
]);
const AX_SAFE_METHODS = new Set([
  'Accessibility.enable',
  'Accessibility.disable',
]);
// Results whose content cannot be tied to a classifiable node.
const DROPPED_RESULTS = new Set(['DOM.getOuterHTML']);
// Events that carry page content without the owning node.
const DROPPED_EVENTS = new Set([
  'Accessibility.nodesUpdated',
  'DOM.attributeModified',
  'DOM.characterDataModified',
  'DOM.setChildNodes',
  'DOM.childNodeInserted',
  'DOM.shadowRootPushed',
  'DOM.pseudoElementAdded',
]);
// Base64 image payloads: scrubbing them would corrupt the image and finds
// nothing. Screenshot masking is handled separately.
const BINARY_DATA_METHODS = new Set([
  'Page.captureScreenshot',
  'Page.screencastFrame',
]);
// Browser-generated identifiers. Exempt from scrubbing only in a protocol
// position and only in a browser id format, so routing and node references
// survive a known value (Tasks 4 and 10 route on the raw message anyway).
const ID_KEYS = new Set([
  'nodeId',
  'parentId',
  'childIds',
  'backendDOMNodeId',
  'backendNodeId',
  'frameId',
  'parentFrameId',
  'loaderId',
  'objectId',
  'sessionId',
  'targetId',
  'openerId',
  'browserContextId',
  'requestId',
  'executionContextId',
  'uniqueId',
  'scriptId',
]);
// 32-hex (targets, sessions, frames, loaders) or dotted integers (node,
// object, request and context ids).
const ID_FORMAT = /^(?:[0-9A-Fa-f]{32}|-?\d+(?:\.-?\d+)*)$/;
const HEX_ID = /^[0-9A-Fa-f]{32}$/;
// Protocol enums. Exempt only when made of letters; anything else is scrubbed.
const ENUM_KEYS = new Set([
  'role',
  'chromeRole',
  'type',
  'subtype',
  'nodeType',
]);
const ENUM_FORMAT = /^[A-Za-z]+$/;

const isProtocolId = (key: string, v: unknown): boolean =>
  typeof v === 'string' &&
  ((ID_KEYS.has(key) && ID_FORMAT.test(v)) || (key === 'id' && HEX_ID.test(v)));
// AXValue types whose `value` is a role enum.
const ROLE_AX_TYPES = new Set(['role', 'internalRole']);

// Runtime.RemoteObject types; such an object's `value` is page data.
const REMOTE_OBJECT_TYPES = new Set([
  'object',
  'function',
  'undefined',
  'string',
  'number',
  'boolean',
  'symbol',
  'bigint',
]);

// A RemoteObject carries a string `type`; an object without one (a property
// descriptor, say) is not one.
const isRemoteObject = (x: Obj): boolean =>
  typeof x.type === 'string' && REMOTE_OBJECT_TYPES.has(x.type);

// Accessibility roles that hold a user-entered value.
const INPUT_ROLES = new Set([
  'textbox',
  'searchbox',
  'combobox',
  'spinbutton',
  'listbox',
  'slider',
]);
// Roles whose name is the document title; a value there must not blank the
// whole page.
const DOCUMENT_ROLES = new Set(['RootWebArea', 'WebArea']);
// AXValue types that carry no page text.
const NON_TEXT_AX_TYPES = new Set([
  'boolean',
  'booleanOrUndefined',
  'tristate',
  'token',
  'role',
  'internalRole',
  'integer',
  'number',
]);
const RELATION_PROPERTIES = new Set(['labelledby', 'describedby']);

const VALUE_ELEMENTS = new Set(['INPUT', 'TEXTAREA', 'SELECT', 'OPTION']);
const BUTTON_INPUT_TYPES = new Set(['submit', 'button', 'reset']);
const VALUE_ATTRIBUTES = new Set([
  'value',
  'label',
  'aria-valuetext',
  'aria-valuenow',
]);
// The only attributes a sensitive DOM node keeps; every other page-authored
// value (aria-label, aria-description, title, role, data-*, ...) is masked,
// since a page can put the value there. `name`, `id` and `class` are
// identifiers the agent may need (known values in them are still scrubbed);
// `type` and `autocomplete` are kept only as well-formed tokens.
const SENSITIVE_KEEP_ATTRIBUTES: ReadonlyMap<string, (v: string) => boolean> =
  new Map([
    ['name', () => true],
    ['id', () => true],
    ['class', () => true],
    ['type', (v: string) => /^[A-Za-z-]{0,32}$/.test(v)],
    [
      'autocomplete',
      // Each token starts with a letter (section-*, shipping, cc-number), so
      // a number-shaped value cannot pass as a token.
      (v: string) =>
        v.length <= 64 &&
        v
          .trim()
          .split(/\s+/)
          .every((token) => token === '' || /^[a-z][a-z0-9-]*$/i.test(token)),
    ],
  ]);
const DOM_CHILD_KEYS = [
  'children',
  'shadowRoots',
  'pseudoElements',
  'contentDocument',
  'templateContent',
  'importedDocument',
];

type Obj = Record<string, unknown>;
type Describe = (backendNodeId: number) => Promise<FieldDescriptor | undefined>;
// A DOM node copy, its unscrubbed source, and the descriptor classifyExtra
// is asked about.
interface DomField {
  node: Obj;
  src: Obj;
  extra: FieldDescriptor;
}

const isObj = (x: unknown): x is Obj =>
  typeof x === 'object' && x !== null && !Array.isArray(x);

const escapeRegExp = (s: string): string =>
  s.replace(/[.*+?^${}()|[\]\\/-]/g, '\\$&');

const canonical = (s: string): string =>
  s.normalize('NFKC').replace(INVISIBLE, '');

const group = (digits: string, sizes: number[]): string[] => {
  const parts: string[] = [];
  let at = 0;
  for (const size of sizes) {
    parts.push(digits.slice(at, at + size));
    at += size;
  }
  return parts;
};

const decodeEntities = (s: string): string =>
  s.replace(
    HTML_ENTITY,
    (
      match,
      name: string | undefined,
      dec: string | undefined,
      hex?: string,
    ) => {
      if (name) return NAMED_ENTITIES[name.toLowerCase()] ?? match;
      const code = Number.parseInt(dec ?? hex ?? '', dec ? 10 : 16);
      try {
        return String.fromCodePoint(code);
      } catch {
        return match;
      }
    },
  );

// Forms of a string in which a known value may hide without being
// replaceable in place: whitespace collapsed, and markup (entities decoded,
// comments, tags or stray angle brackets removed). A match in any of them
// masks the whole string.
const wholeMaskForms = (s: string): string[] => {
  const forms: string[] = [];
  const collapse = (t: string) => t.replace(WHITESPACE_RUN, ' ');
  if (HAS_WHITESPACE_RUN.test(s)) forms.push(collapse(s));
  if (HAS_MARKUP.test(s)) {
    const text = decodeEntities(s).replace(MARKUP_COMMENT, '');
    forms.push(collapse(text.replace(MARKUP_TAG, '')));
    forms.push(collapse(text.replace(ANGLE_BRACKETS, '')));
  }
  return forms;
};

const numberForm = (n: number): string =>
  Number.isInteger(n) && Math.abs(n) >= 1e21 ? BigInt(n).toString() : String(n);

// Sets a property without the `__proto__` setter turning data into a
// prototype.
const setOwn = (o: Obj, key: string, value: unknown): void => {
  if (key === '__proto__') {
    Object.defineProperty(o, key, {
      value,
      enumerable: true,
      writable: true,
      configurable: true,
    });
  } else {
    o[key] = value;
  }
};

const htmlEscape = (s: string): string =>
  s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');

const encodings = (form: string): string[] => {
  const out = [
    form,
    form.replace(/ /g, '+'),
    JSON.stringify(form).slice(1, -1),
    htmlEscape(form),
  ];
  try {
    const uri = encodeURIComponent(form);
    out.push(
      uri,
      uri
        .replace(/%20/g, '+')
        .replace(
          /[!'()*]/g,
          (c) => `%${c.charCodeAt(0).toString(16).toUpperCase()}`,
        ),
    );
  } catch {
    // A lone surrogate cannot be URL-encoded; the other forms remain.
  }
  return out;
};

export function valueVariants(value: string): string[] {
  const out = new Set<string>();
  const trimmed = value.trim();
  const normalized = canonical(trimmed);
  const bases = [value, trimmed, normalized];

  const parts = normalized.split(SEPARATORS).filter(Boolean);
  if (parts.length > 1) {
    for (const joiner of JOINERS) bases.push(parts.join(joiner));
  }
  // Grouped forms only for a value made of digits and separators: pulling
  // the digits out of mixed text would scrub years and page numbers.
  const compact = parts.join('');
  if (/^\d+$/.test(compact)) {
    const sizes =
      compact.length === 9
        ? [3, 2, 4]
        : compact.length === 16
          ? [4, 4, 4, 4]
          : null;
    if (sizes) {
      const grouped = group(compact, sizes);
      for (const joiner of JOINERS) bases.push(grouped.join(joiner));
    }
  }

  for (const base of bases) {
    for (const form of [
      base,
      base.toLowerCase(),
      base.toUpperCase(),
      base.replace(/ß/g, 'ẞ'),
      base.replace(/ẞ/g, 'ß'),
      base.replace(/[ßẞ]/g, 'ss'),
    ]) {
      for (const encoded of encodings(form)) {
        if (encoded.length >= MIN_KNOWN_LENGTH) out.add(encoded);
      }
    }
  }
  return [...out];
}

interface Alternative {
  source: string;
  weight: number;
}

const wholeToken = (source: string): string =>
  `(?<![0-9A-Za-z])${source}(?![0-9A-Za-z])`;

// Regex alternatives for one known value. A number made of digits and
// separators becomes one pattern that tolerates any separator, width or
// escape between its digits; anything else becomes its literal variants.
const alternativesFor = (value: string): Alternative[] => {
  const normalized = canonical(value.trim());
  const digits = normalized.replace(/\D/g, '');
  if (DIGITS_AND_SEPARATORS.test(normalized) && digits.length >= 3) {
    const body = digits.split('').join(`${DIGIT_SEPARATOR}{0,6}`);
    const source =
      digits.length === 3
        ? wholeToken(body)
        : `(?:(?<![0-9])|(?<=%[0-9A-Fa-f]{2})|(?<=\\\\u[0-9A-Fa-f]{4}))${body}(?![0-9])`;
    return [{ source, weight: 1_000_000 + digits.length }];
  }
  const seen = new Set<string>();
  const out: Alternative[] = [];
  for (const variant of valueVariants(value)) {
    // The regex is case-insensitive; ASCII case variants are redundant.
    const key = variant.replace(/[A-Z]/g, (c) => c.toLowerCase());
    if (seen.has(key)) continue;
    seen.add(key);
    const escaped = escapeRegExp(variant);
    out.push({
      source: variant.length < 4 ? wholeToken(escaped) : escaped,
      weight: variant.length,
    });
  }
  return out;
};

const attributeMap = (attributes: unknown[]): Map<string, string> => {
  const map = new Map<string, string>();
  for (let i = 0; i + 1 < attributes.length; i += 2) {
    const name = attributes[i];
    const value = attributes[i + 1];
    if (typeof name === 'string' && typeof value === 'string') {
      map.set(name.toLowerCase(), value);
    }
  }
  return map;
};

/** Builds a FieldDescriptor from a DOM.describeNode `attributes` array. */
export function descriptorFromAttributes(
  attributes: unknown[],
): FieldDescriptor {
  const a = attributeMap(attributes);
  const d: FieldDescriptor = {};
  const set = (key: keyof FieldDescriptor, value: string | undefined) => {
    if (value !== undefined) d[key] = value;
  };
  set('inputType', a.get('type'));
  set('autocomplete', a.get('autocomplete'));
  const joined = (...values: (string | undefined)[]) => {
    const present = values.filter((v): v is string => v !== undefined);
    return present.length === 0 ? undefined : present.join(' ');
  };
  // `name` and `id` both carry naming signals (name=field_17 id=ssn).
  set('name', joined(a.get('name'), a.get('id')));
  set('placeholder', a.get('placeholder') ?? a.get('aria-placeholder'));
  set('label', a.get('aria-label'));
  set('description', joined(a.get('aria-description'), a.get('title')));
  return d;
}

// ------------------------------------------------------------ value shapes

// A run of 9 or more digits, single separators (space, dot, dashes) allowed
// between them: SSNs in any common format, account and card numbers. It
// subsumes the SSN shape and a Luhn check (every 13 to 19 digit card number
// is such a run), so a non-Luhn 16-digit number is masked too.
const DIGIT_RUN = /\d(?:[ .‐-―−-]?\d){8,}/g;
const PLAIN_DECIMAL = /^(\d+)\.\d+$/;

/**
 * True when `text` holds a value-shaped number (see DIGIT_RUN). A plain
 * decimal such as a coordinate (412.3333435058594) counts only when its
 * integer part alone has 9 or more digits.
 */
export function hasSensitiveShape(text: string): boolean {
  const normalized = canonical(text);
  for (const match of normalized.match(DIGIT_RUN) ?? []) {
    const decimal = PLAIN_DECIMAL.exec(match);
    if (!decimal || decimal[1].length >= 9) return true;
  }
  return false;
}

const decodeParam = (s: string): string => {
  try {
    return decodeURIComponent(s.replace(/\+/g, ' '));
  } catch {
    return s;
  }
};

// Masks the value of each `name=value` pair whose name is sensitive or
// whose decoded value is value-shaped; a bare value-shaped item is masked
// whole. `&` and `;` both separate parameters; empty segments are kept.
const maskParams = (params: string): string =>
  params
    .split(/([&;])/)
    .map((part, i) => {
      if (i % 2 === 1 || part === '') return part;
      const eq = part.indexOf('=');
      if (eq < 0) return hasSensitiveShape(decodeParam(part)) ? MASK : part;
      const name = decodeParam(part.slice(0, eq));
      const value = part.slice(eq + 1);
      if (value === '') return part;
      const sensitive =
        isSensitiveField({ name, label: name }) ||
        hasSensitiveShape(decodeParam(value));
      return sensitive ? `${part.slice(0, eq)}=${MASK}` : part;
    })
    .join('');

// A path segment: value-shaped (after decoding) is masked; `;name=value`
// path parameters go through maskParams.
const maskPathSegment = (segment: string): string => {
  if (segment === '') return segment;
  const semi = segment.indexOf(';');
  const head = semi < 0 ? segment : segment.slice(0, semi);
  const masked = hasSensitiveShape(decodeParam(head)) ? MASK : head;
  return semi < 0 ? masked : `${masked};${maskParams(segment.slice(semi + 1))}`;
};

// Masks one URL for the model's view (routing uses the raw message): any
// userinfo, value-shaped path segments, and sensitive query, `;` and
// fragment parameters. Scheme and host are kept.
const maskUrl = (url: string): string => {
  const schemeEnd = url.indexOf('://') + 3;
  const hashAt = url.indexOf('#');
  const end = hashAt < 0 ? url.length : hashAt;
  const queryAt = url.slice(0, end).indexOf('?');
  const pathEnd = queryAt < 0 ? end : queryAt;
  const rest = url.slice(schemeEnd, pathEnd);
  const slash = rest.indexOf('/');
  let authority = slash < 0 ? rest : rest.slice(0, slash);
  const path = slash < 0 ? '' : rest.slice(slash);
  const at = authority.lastIndexOf('@');
  if (at >= 0) authority = `${MASK}@${authority.slice(at + 1)}`;
  let out =
    url.slice(0, schemeEnd) +
    authority +
    path.split('/').map(maskPathSegment).join('/');
  if (queryAt >= 0) out += `?${maskParams(url.slice(queryAt + 1, end))}`;
  if (hashAt >= 0) {
    const fragment = url.slice(hashAt + 1);
    out += `#${
      fragment.includes('=')
        ? maskParams(fragment)
        : hasSensitiveShape(decodeParam(fragment))
          ? MASK
          : fragment
    }`;
  }
  return out;
};

const URL_IN_TEXT = /[a-z][a-z0-9+.-]*:\/\/[^\s"'<>`]+/gi;

// Every absolute URL in a string, whole or embedded in text.
const maskUrls = (s: string): string =>
  s.includes('://') ? s.replace(URL_IN_TEXT, maskUrl) : s;

// ------------------------------------------------------------ request bodies
// A request body (`postData`) is never parsed per content type. It is
// masked WHOLE when its raw text, in any of a few decoded views, names a
// sensitive field anywhere (form parameters, JSON keys, XML tags and
// attributes, multipart names quoted or not, free text) or holds a
// value-shaped number; otherwise it is returned unchanged. No caller reaches
// this in the live path today (filterEvent strips postData from Network
// events), so the rule is deliberately blunt.

const bodyViews = (body: string): string[] => {
  const views = [body, decodeParam(body), decodeEntities(body)];
  try {
    // Normalizes JSON escapes such as password.
    views.push(JSON.stringify(JSON.parse(body)));
  } catch {
    // Not JSON.
  }
  return views;
};

const bodyIsSensitive = (body: string): boolean =>
  maskParams(body) !== body ||
  bodyViews(body).some(
    (view) =>
      hasSensitiveShape(view) || isSensitiveField({ label: view, name: view }),
  );

const maskBody = (body: string): string =>
  bodyIsSensitive(body) ? MASK : body;

// Script results whose page data gets the value-shape guard.
const SCRIPT_RESULT_METHODS = new Set([
  'Runtime.evaluate',
  'Runtime.callFunctionOn',
  'Runtime.getProperties',
  'Runtime.awaitPromise',
]);

// Masks value-shaped strings, and 9+ digit integers in by-value page data.
// Handles (objectId and other protocol ids) and type tags are left alone
// outside page data; booleans, null, small numbers and decimals are never
// touched. Keys of page data that are value-shaped are renamed to MASK
// (numbered on collision). Works on a private copy.
const maskShapes = (x: unknown, pageData: boolean): unknown => {
  if (typeof x === 'string') return hasSensitiveShape(x) ? MASK : x;
  if (typeof x === 'number') {
    return pageData && Number.isInteger(x) && Math.abs(x) >= 1e8 ? MASK : x;
  }
  if (Array.isArray(x)) {
    for (let i = 0; i < x.length; i++) x[i] = maskShapes(x[i], pageData);
    return x;
  }
  if (!isObj(x)) return x;
  const remoteObject = isRemoteObject(x);
  const out: Obj = {};
  for (const key of Object.keys(x)) {
    const v = x[key];
    let name = key;
    if (pageData && hasSensitiveShape(key)) {
      name = MASK;
      for (let n = 2; Object.hasOwn(out, name); n++) name = `${MASK}#${n}`;
    }
    const keep =
      !pageData &&
      (key === 'type' || key === 'subtype' || isProtocolId(key, v));
    setOwn(
      out,
      name,
      keep ? v : maskShapes(v, pageData || (remoteObject && key === 'value')),
    );
  }
  return out;
};

// Page text carried by Page and Target payloads (dialogs, frame and window
// names, document titles): value-shaped text is masked; ids, URLs (masked
// separately) and enums are left alone.
const PAGE_TEXT_KEYS = new Set([
  'message',
  'defaultPrompt',
  'name',
  'windowName',
  'title',
  'errorText',
]);
const maskPageText = (x: unknown): unknown => {
  if (Array.isArray(x)) {
    for (let i = 0; i < x.length; i++) x[i] = maskPageText(x[i]);
    return x;
  }
  if (!isObj(x)) return x;
  for (const key of Object.keys(x)) {
    const v = x[key];
    if (typeof v === 'string') {
      if (PAGE_TEXT_KEYS.has(key) && hasSensitiveShape(v)) x[key] = MASK;
    } else if (typeof v === 'object' && v !== null) {
      maskPageText(v);
    }
  }
  return x;
};

// Which shape guard a method's result or event params get. The AX tree is
// left to field masking and known-value scrubbing (its text is the page the
// agent reads); Input and empty results carry nothing.
const shapeGuard = (method: string, x: unknown): unknown => {
  if (
    method.startsWith('Runtime.') ||
    method.startsWith('Network.') ||
    method.startsWith('DOM.')
  ) {
    return maskShapes(x, false);
  }
  if (method.startsWith('Page.') || method.startsWith('Target.')) {
    return maskPageText(x);
  }
  return x;
};

// ---------------------------------------------------------------- DOM nodes

// Masks value-bearing attributes and text of `root` and everything below it.
// A sensitive root keeps only allowlisted attributes. `root` is a private
// copy; `rawRoot` is its unscrubbed source, walked in parallel so attribute
// names are read from the page, not from the scrubbed copy.
const maskDomSubtree = (
  root: Obj,
  rawRoot: unknown,
  sensitive: boolean,
  keepRootValue: boolean,
): void => {
  const stack: [unknown, unknown][] = [[root, rawRoot]];
  while (stack.length > 0) {
    const [node, raw] = stack.pop() as [unknown, unknown];
    if (Array.isArray(node)) {
      const rawItems = Array.isArray(raw) ? raw : [];
      for (let i = 0; i < node.length; i++) stack.push([node[i], rawItems[i]]);
      continue;
    }
    if (!isObj(node)) continue;
    const src = isObj(raw) ? raw : node;
    if (typeof node.nodeValue === 'string' && node.nodeValue !== '') {
      node.nodeValue = MASK;
    }
    if (Array.isArray(node.attributes)) {
      const attributes = node.attributes;
      const names = Array.isArray(src.attributes) ? src.attributes : attributes;
      for (let i = 0; i + 1 < attributes.length; i += 2) {
        const name = names[i];
        if (typeof name !== 'string' || attributes[i + 1] === '') continue;
        const lower = name.toLowerCase();
        if (node === root && keepRootValue && lower === 'value') continue;
        const value = attributes[i + 1];
        const keep = SENSITIVE_KEEP_ATTRIBUTES.get(lower);
        const mask = sensitive
          ? !keep || typeof value !== 'string' || !keep(value)
          : VALUE_ATTRIBUTES.has(lower);
        if (mask) attributes[i + 1] = MASK;
      }
    }
    for (const key of DOM_CHILD_KEYS) {
      if (node[key] !== undefined) stack.push([node[key], src[key]]);
    }
  }
};

const isDomFieldSensitive = (attributes: unknown[]): boolean =>
  isSensitiveField(descriptorFromAttributes(attributes));

// Classifies from the raw source node (tag, type, attributes), then masks the
// copy: a known value such as 'password' or 'input' must not erase the
// signal from the scrubbed copy. Returns the descriptor of a field
// classifyExtra may still flag (see below), else undefined.
const maskDomNode = (node: Obj, src: Obj): FieldDescriptor | undefined => {
  if (typeof src.nodeName !== 'string' || typeof src.nodeType !== 'number') {
    return undefined;
  }
  const attributes = Array.isArray(src.attributes) ? src.attributes : [];
  const map = attributeMap(attributes);
  const tag = src.nodeName.toUpperCase();
  const holdsValue =
    VALUE_ELEMENTS.has(tag) ||
    map.has('contenteditable') ||
    INPUT_ROLES.has(map.get('role') ?? '');
  if (!holdsValue) return undefined;
  // A text button's value attribute is its visible label.
  const keepRootValue =
    tag === 'INPUT' &&
    BUTTON_INPUT_TYPES.has((map.get('type') ?? '').toLowerCase());
  const descriptor = descriptorFromAttributes(attributes);
  const sensitive = isSensitiveField(descriptor);
  maskDomSubtree(node, src, sensitive, keepRootValue && !sensitive);
  // classifyExtra is asked about the same value-holding elements, minus an
  // OPTION (its SELECT is asked and covers it) and a button-type INPUT (its
  // value is its label, not entered data).
  if (sensitive || tag === 'OPTION' || keepRootValue) return undefined;
  return descriptor;
};

// ------------------------------------------------------- accessibility tree

const axString = (v: unknown): string | undefined =>
  isObj(v) && typeof v.value === 'string' ? v.value : undefined;

const axProperties = (node: Obj): Obj[] =>
  Array.isArray(node.properties) ? node.properties.filter(isObj) : [];

const valueText = (node: Obj): string | undefined =>
  axString(axProperties(node).find((p) => p.name === 'valuetext')?.value);

const isFieldLike = (node: Obj): boolean => {
  const role = axString(node.role);
  if (role !== undefined && INPUT_ROLES.has(role)) return true;
  return node.value !== undefined || valueText(node) !== undefined;
};

const isFilled = (node: Obj): boolean => {
  const v = isObj(node.value) ? node.value.value : undefined;
  if (typeof v === 'number') return true;
  if (typeof v === 'string' && v !== '') return true;
  const vt = valueText(node);
  return vt !== undefined && vt !== '';
};

const looksMasked = (node: Obj): boolean => {
  const v = axString(node.value);
  const vt = valueText(node);
  return (
    (v !== undefined && BULLETS_ONLY.test(v)) ||
    (vt !== undefined && BULLETS_ONLY.test(vt))
  );
};

// The AX `autocomplete` property is aria-autocomplete (list/inline/both),
// not the HTML hint, so it is not read here; HTML autocomplete only comes
// from describe().
const axDescriptor = (node: Obj): FieldDescriptor => {
  const d: FieldDescriptor = {
    role: axString(node.role),
    label: axString(node.name),
    description: axString(node.description),
  };
  const sources =
    isObj(node.name) && Array.isArray(node.name.sources)
      ? node.name.sources.filter(isObj)
      : [];
  for (const source of sources) {
    if (source.type !== 'placeholder') continue;
    const placeholder =
      axString(source.attributeValue) ?? axString(source.value);
    if (placeholder) {
      d.placeholder = placeholder;
      break;
    }
  }
  return d;
};

type Verdict = 'sensitive' | 'ordinary' | 'ask';

const classifyAXNode = (node: Obj): Verdict => {
  if (!isFieldLike(node)) return 'ordinary';
  if (isSensitiveField(axDescriptor(node)) || looksMasked(node)) {
    return 'sensitive';
  }
  return isFilled(node) ? 'ask' : 'ordinary';
};

// Masks every non-empty string held under a `value` or `text` key, deeply.
const maskText = (x: unknown): void => {
  if (Array.isArray(x)) {
    for (const item of x) maskText(item);
    return;
  }
  if (!isObj(x)) return;
  for (const key of Object.keys(x)) {
    const v = x[key];
    if ((key === 'value' || key === 'text') && typeof v === 'string') {
      if (v !== '') x[key] = MASK;
    } else if (typeof v === 'object') {
      maskText(v);
    }
  }
};

const maskAXNode = (node: Obj): void => {
  maskText(node.name);
  maskText(node.description);
  if (isObj(node.value)) {
    if (typeof node.value.value === 'number') {
      // Keep the AXValue well typed: a string mask needs a string type.
      node.value.type = 'string';
      node.value.value = MASK;
    }
    maskText(node.value);
  }
  for (const property of axProperties(node)) {
    const value = property.value;
    if (isObj(value) && NON_TEXT_AX_TYPES.has(String(value.type))) continue;
    maskText(value);
  }
};

const axNodesOf = (tree: unknown): Obj[] => {
  if (!isObj(tree) || !Array.isArray(tree.nodes)) {
    throw new Error('not an accessibility tree');
  }
  const nodes = tree.nodes;
  if (!nodes.every(isObj)) throw new Error('malformed accessibility node');
  return nodes as Obj[];
};

const nodeTexts = (node: unknown): string =>
  isObj(node)
    ? [axString(node.name), axString(node.value), axString(node.description)]
        .map((t) => t ?? '')
        .join('\u0000')
    : '';

// Collects backendDOMNodeIds referenced through relatedNodes, deeply.
const referencedIds = (x: unknown, out: number[]): void => {
  if (Array.isArray(x)) {
    for (const item of x) referencedIds(item, out);
    return;
  }
  if (!isObj(x)) return;
  if (Array.isArray(x.relatedNodes)) {
    for (const related of x.relatedNodes) {
      if (isObj(related) && typeof related.backendDOMNodeId === 'number') {
        out.push(related.backendDOMNodeId);
      }
    }
  }
  for (const v of Object.values(x)) {
    if (typeof v === 'object') referencedIds(v, out);
  }
};

// An AX node copy and its unscrubbed source. Every decision (role, label,
// relation names, text) reads `src`; every mask writes `node`.
interface AXPair {
  node: Obj;
  src: Obj;
}

const axPairs = (nodes: Obj[], raw: unknown): AXPair[] => {
  if (!Array.isArray(raw) || raw.length !== nodes.length) {
    throw new Error('accessibility source mismatch');
  }
  return nodes.map((node, i) => {
    const src = raw[i];
    if (!isObj(src)) throw new Error('malformed accessibility node');
    return { node, src };
  });
};

// Applies field masking to a private copy of an AX node list.
// `sensitive`: copies classified sensitive. `pattern`: the known-value regex.
const maskAXTree = (
  pairs: AXPair[],
  sensitive: ReadonlySet<Obj>,
  pattern: RegExp | null,
): void => {
  const byId = new Map<string, Obj>();
  const srcOf = new Map<Obj, Obj>();
  for (const { node, src } of pairs) {
    byId.set(String(src.nodeId), node);
    srcOf.set(node, src);
  }
  const childrenOf = (node: Obj): Obj[] => {
    const src = srcOf.get(node) ?? node;
    if (!Array.isArray(src.childIds)) return [];
    const out: Obj[] = [];
    for (const id of src.childIds) {
      const child = byId.get(String(id));
      if (child) out.push(child);
    }
    return out;
  };

  const masked = new Set<Obj>();
  const maskSubtree = (root: Obj): void => {
    const stack = [root];
    while (stack.length > 0) {
      const node = stack.pop() as Obj;
      if (masked.has(node)) continue;
      masked.add(node);
      maskAXNode(node);
      for (const child of childrenOf(node)) {
        if (!masked.has(child)) stack.push(child);
      }
    }
  };

  // Which nodes had a known value scrubbed out of their own text.
  const scrubbed = pattern
    ? pairs.filter(({ node, src }) => nodeTexts(node) !== nodeTexts(src))
    : [];

  for (const node of sensitive) maskSubtree(node);

  if (pattern !== null) {
    // A node that held a known value: its children carry fragments of it
    // (InlineTextBox runs split mid-value).
    for (const { node, src } of scrubbed) {
      if (DOCUMENT_ROLES.has(axString(src.role) ?? '')) continue;
      for (const child of childrenOf(node)) maskSubtree(child);
    }
    maskSplitValues(pairs, childrenOf, pattern, maskSubtree);
  }

  // A node labelled or described by a masked node echoes its text.
  const maskedIds = new Set<number>();
  for (const node of masked) {
    const id = srcOf.get(node)?.backendDOMNodeId;
    if (typeof id === 'number') maskedIds.add(id);
  }
  if (maskedIds.size === 0) return;
  for (const { node, src } of pairs) {
    if (masked.has(node)) continue;
    const rawProps = Array.isArray(src.properties) ? src.properties : [];
    const copyProps = Array.isArray(node.properties) ? node.properties : [];
    const relations: unknown[] = [];
    const rawRelations: unknown[] = [];
    rawProps.forEach((p, i) => {
      if (isObj(p) && RELATION_PROPERTIES.has(String(p.name))) {
        rawRelations.push(p);
        const copied = copyProps[i];
        if (isObj(copied)) relations.push(copied.value);
      }
    });
    const ids: number[] = [];
    referencedIds([src.name, src.description, rawRelations], ids);
    if (!ids.some((id) => maskedIds.has(id))) continue;
    maskText(node.name);
    maskText(node.description);
    for (const relation of relations) maskText(relation);
  }
};

// Finds known values split across StaticText nodes: the raw StaticText texts
// are joined in document order and every node a match spans is masked whole.
// A match inside a single node is already scrubbed in place.
const maskSplitValues = (
  pairs: AXPair[],
  childrenOf: (node: Obj) => Obj[],
  pattern: RegExp,
  maskSubtree: (node: Obj) => void,
): void => {
  const rawText = new Map<Obj, string>();
  for (const { node, src } of pairs) {
    if (axString(src.role) !== 'StaticText') continue;
    const t = axString(src.name);
    if (t) rawText.set(node, t);
  }
  if (rawText.size < 2) return;

  const nodes = pairs.map((p) => p.node);
  const isChild = new Set<Obj>();
  for (const node of nodes) for (const c of childrenOf(node)) isChild.add(c);
  const order: Obj[] = [];
  const seen = new Set<Obj>();
  const walk = (root: Obj) => {
    const stack = [root];
    while (stack.length > 0) {
      const node = stack.pop() as Obj;
      if (seen.has(node)) continue;
      seen.add(node);
      order.push(node);
      const children = childrenOf(node);
      for (let i = children.length - 1; i >= 0; i--) stack.push(children[i]);
    }
  };
  for (const node of nodes) if (!isChild.has(node)) walk(node);
  for (const node of nodes) if (!seen.has(node)) walk(node);

  const segments: { node: Obj; start: number; end: number }[] = [];
  let joined = '';
  for (const node of order) {
    const t = rawText.get(node);
    if (t === undefined) continue;
    segments.push({
      node,
      start: joined.length,
      end: joined.length + t.length,
    });
    joined += t;
  }
  joined.replace(pattern, (match: string, ...args: unknown[]) => {
    const offset = args[args.length - 2] as number;
    const end = offset + match.length;
    const spanned = segments.filter((s) => s.start < end && s.end > offset);
    if (spanned.length > 1) for (const s of spanned) maskSubtree(s.node);
    return match;
  });
};

// ------------------------------------------------------------------ redactor

export function createRedactor(opts?: RedactorOptions): Redactor {
  const maxKnownValues = opts?.maxKnownValues ?? DEFAULT_MAX_KNOWN_VALUES;
  const describeTimeoutMs =
    opts?.describeTimeoutMs ?? DEFAULT_DESCRIBE_TIMEOUT_MS;
  const describeDeadlineMs =
    opts?.describeDeadlineMs ?? DEFAULT_DESCRIBE_DEADLINE_MS;
  const classifyExtra = opts?.classifyExtra;
  const classifyTimeoutMs =
    opts?.classifyTimeoutMs ?? DEFAULT_CLASSIFY_TIMEOUT_MS;
  const known = new Map<string, Alternative[]>();
  let pattern: RegExp | null = null;
  // Set only while copyOf() runs synchronously for one DOM result: the
  // ordinary fields classifyExtra may still flag.
  let domFields: DomField[] | null = null;

  const rebuild = () => {
    const all = new Map<string, number>();
    for (const list of known.values()) {
      for (const a of list) all.set(a.source, a.weight);
    }
    const sorted = [...all].sort((a, b) => b[1] - a[1]).map(([s]) => s);
    // Only ever used with String.prototype.replace, which resets lastIndex.
    pattern = sorted.length === 0 ? null : new RegExp(sorted.join('|'), 'gi');
  };

  const hasKnownValue = (s: string): boolean =>
    pattern !== null && s.replace(pattern, MASK) !== s;

  // Scrubs the whole string (URLs included: a value in a host is masked too).
  // Known values first, then URL parameters (with or without known values).
  const scrub = (s: string): string => {
    const out = scrubKnown(s);
    return out === MASK ? out : maskUrls(out);
  };

  const scrubKnown = (s: string): string => {
    if (pattern === null) return s;
    let out = s.replace(pattern, MASK);
    if (NON_ASCII.test(out)) {
      // Full-width digits, compatibility forms and zero-width characters.
      const normalized = canonical(out);
      if (normalized !== out) {
        const again = normalized.replace(pattern, MASK);
        if (again !== normalized) out = again;
      }
    }
    // A value split by markup or long whitespace (<td>123</td> <td>45</td>)
    // cannot be scrubbed in place, so the string is masked whole.
    for (const form of wholeMaskForms(out)) {
      if (hasKnownValue(form) || hasKnownValue(canonical(form))) return MASK;
    }
    return out;
  };

  // A number in page data that spells a known value becomes MASK.
  const scrubNumber = (n: number): unknown =>
    Number.isFinite(n) && hasKnownValue(numberForm(n)) ? MASK : n;

  // Deep copy; strings are scrubbed except at structural positions, and DOM
  // nodes are masked. Throws on a cycle or excessive depth.
  const copy = (
    x: unknown,
    ancestors: Set<object>,
    depth: number,
    scrubbing: boolean,
    skipData: boolean,
    trusted: boolean,
    pageKeys = false,
  ): unknown => {
    if (typeof x === 'string') return scrubbing ? scrub(x) : x;
    if (typeof x === 'number') {
      return scrubbing && !trusted && pattern !== null ? scrubNumber(x) : x;
    }
    if (typeof x !== 'object' || x === null) return x;
    if (depth > MAX_DEPTH || ancestors.has(x)) {
      throw new Error('unredactable structure');
    }
    ancestors.add(x);
    let out: unknown;
    if (Array.isArray(x)) {
      out = x.map((item) =>
        copy(item, ancestors, depth + 1, scrubbing, false, trusted),
      );
    } else {
      const source = x as Obj;
      // An AX property `{ name, value: AXValue }`: the name is an enum.
      const axProperty =
        trusted && isObj(source.value) && typeof source.value.type === 'string';
      // A role AXValue `{ type: 'role', value: 'textbox' }`.
      const roleValue = trusted && ROLE_AX_TYPES.has(String(source.type));
      // A RemoteObject's by-value payload is page data: its keys are not
      // protocol fields, so nothing below it is exempt.
      const remoteObject = isRemoteObject(source);
      // Keys of page data and header maps are data; protocol field names in
      // trusted positions are never renamed.
      const scrubKeys = scrubbing && pattern !== null && (!trusted || pageKeys);
      const o: Obj = {};
      for (const key of Object.keys(source)) {
        const v = source[key];
        let name = key;
        if (scrubKeys) {
          const scrubbed = scrub(key);
          name = scrubbed;
          // Two keys can scrub to the same name: number the later ones.
          for (let n = 2; Object.hasOwn(o, name); n++)
            name = `${scrubbed}#${n}`;
        }
        const enumLike =
          typeof v === 'string' &&
          ENUM_FORMAT.test(v) &&
          (ENUM_KEYS.has(key) ||
            (axProperty && key === 'name') ||
            (roleValue && key === 'value'));
        const raw =
          (trusted && (enumLike || isProtocolId(key, v))) ||
          (skipData && key === 'data' && typeof v === 'string');
        const pageData = remoteObject && key === 'value';
        let copied =
          trusted && ID_KEYS.has(key) && Array.isArray(v)
            ? copyIdList(v, ancestors, depth + 1, scrubbing)
            : copy(
                v,
                ancestors,
                depth + 1,
                scrubbing && !raw,
                false,
                trusted && !pageData,
                trusted && HEADER_KEYS.has(key),
              );
        if (key === 'postData' && typeof copied === 'string' && scrubbing) {
          copied = maskBody(copied);
        }
        setOwn(o, name, copied);
      }
      const extra = maskDomNode(o, source);
      if (extra && domFields) domFields.push({ node: o, src: source, extra });
      out = o;
    }
    ancestors.delete(x);
    return out;
  };

  // childIds and similar: each well-formed id is kept, anything else
  // scrubbed.
  const copyIdList = (
    list: unknown[],
    ancestors: Set<object>,
    depth: number,
    scrubbing: boolean,
  ): unknown[] =>
    list.map((item) =>
      typeof item === 'string' && ID_FORMAT.test(item)
        ? item
        : copy(item, ancestors, depth, scrubbing, false, false),
    );

  const copyOf = (method: string, x: unknown): unknown =>
    copy(x, new Set(), 0, true, BINARY_DATA_METHODS.has(method), true);

  // Resolves to the descriptor, or undefined on rejection, throw, timeout or
  // a non-object answer.
  const describeSafely = (
    describe: Describe,
    id: number,
    timeoutMs: number,
  ): Promise<FieldDescriptor | undefined> =>
    new Promise((resolve) => {
      const timer = setTimeout(() => resolve(undefined), timeoutMs);
      Promise.resolve()
        .then(() => describe(id))
        .then(
          (d) => (isObj(d) ? d : undefined),
          () => undefined,
        )
        .then((d) => {
          clearTimeout(timer);
          resolve(d);
        });
    });

  // At most DESCRIBE_CONCURRENCY calls in flight, one per distinct id, all
  // within describeDeadlineMs; an id not answered by then maps to undefined.
  const describeAll = async (
    describe: Describe,
    ids: number[],
  ): Promise<Map<number, FieldDescriptor | undefined>> => {
    const results = new Map<number, FieldDescriptor | undefined>();
    const deadline = Date.now() + describeDeadlineMs;
    let next = 0;
    const worker = async () => {
      while (next < ids.length) {
        const remaining = deadline - Date.now();
        if (remaining <= 0) return;
        const id = ids[next++];
        results.set(
          id,
          await describeSafely(
            describe,
            id,
            Math.min(describeTimeoutMs, remaining),
          ),
        );
      }
    };
    const workers: Promise<void>[] = [];
    for (let i = 0; i < Math.min(DESCRIBE_CONCURRENCY, ids.length); i++) {
      workers.push(worker());
    }
    await Promise.all(workers);
    return results;
  };

  // -------------------------------------------------------- classifyExtra
  // Verdicts (booleans only) keyed by the sanitized descriptor, least
  // recently used first; at most CLASSIFY_CACHE_SIZE. Emptied by clear().
  const verdicts = new Map<string, boolean>();
  // One call per key at a time, shared by concurrent trees.
  const pendingVerdicts = new Map<string, Promise<boolean | undefined>>();
  // Bumped by clear(): an answer to a call made before it is not cached.
  let generation = 0;
  // A slot is held from the call until the hook settles, so at most
  // CLASSIFY_CONCURRENCY calls are ever in flight, across trees.
  let slotsInUse = 0;
  const slotWaiters: (() => void)[] = [];

  const acquireSlot = (deadline: number): Promise<boolean> => {
    if (slotsInUse < CLASSIFY_CONCURRENCY) {
      slotsInUse++;
      return Promise.resolve(true);
    }
    return new Promise((resolve) => {
      const grant = () => {
        clearTimeout(timer);
        resolve(true);
      };
      const timer = setTimeout(
        () => {
          const at = slotWaiters.indexOf(grant);
          if (at >= 0) slotWaiters.splice(at, 1);
          resolve(false);
        },
        Math.max(0, deadline - Date.now()),
      );
      slotWaiters.push(grant);
    });
  };

  // Hands the slot to the next waiter, or frees it.
  const releaseSlot = () => {
    const next = slotWaiters.shift();
    if (next) next();
    else slotsInUse--;
  };

  const remember = (key: string, verdict: boolean) => {
    verdicts.delete(key);
    verdicts.set(key, verdict);
    while (verdicts.size > CLASSIFY_CACHE_SIZE) {
      const oldest = verdicts.keys().next().value;
      if (oldest === undefined) break;
      verdicts.delete(oldest);
    }
  };

  // Resolves to `p`'s answer, or undefined once `ms` have passed.
  const within = <T>(p: Promise<T>, ms: number): Promise<T | undefined> =>
    new Promise((resolve) => {
      const timer = setTimeout(() => resolve(undefined), Math.max(0, ms));
      p.then((v) => {
        clearTimeout(timer);
        resolve(v);
      });
    });

  // The descriptor classifyExtra sees: only the FieldDescriptor string keys,
  // in a fixed order, each scrubbed of known values and URL parameters and
  // masked whole when value-shaped. Never a value. Undefined when nothing
  // names the field.
  const extraDescriptor = (d: FieldDescriptor): FieldDescriptor | undefined => {
    const out: FieldDescriptor = {};
    for (const key of DESCRIPTOR_KEYS) {
      const v = d[key];
      if (typeof v !== 'string' || v.trim() === '') continue;
      const s = scrub(v);
      out[key] = hasSensitiveShape(s) ? MASK : s;
    }
    return Object.keys(out).length === 0 ? undefined : out;
  };

  // One verdict: cached, or from one bounded call. Undefined (keep the
  // deterministic verdict) on a throw, rejection, non-boolean answer,
  // per-call timeout, no free slot by `deadline`, or `deadline` itself.
  const askExtra = (
    hook: (d: FieldDescriptor) => Promise<boolean>,
    d: FieldDescriptor,
    deadline: number,
  ): Promise<boolean | undefined> => {
    const descriptor = extraDescriptor(d);
    if (!descriptor) return Promise.resolve(undefined);
    const key = JSON.stringify(descriptor);
    const cached = verdicts.get(key);
    if (cached !== undefined) {
      remember(key, cached);
      return Promise.resolve(cached);
    }
    let pending = pendingVerdicts.get(key);
    if (!pending) {
      // No new call once the tree's time is up.
      if (Date.now() >= deadline) return Promise.resolve(undefined);
      const asked = generation;
      const run = async (): Promise<boolean | undefined> => {
        if (!(await acquireSlot(deadline))) return undefined;
        const call = Promise.resolve()
          // A fresh copy: the hook cannot reach the redactor's state.
          .then(() => hook({ ...descriptor }))
          .then(
            (v) => (typeof v === 'boolean' ? v : undefined),
            () => undefined,
          );
        call.then((v) => {
          releaseSlot();
          // Only an answer is cached, never a failure; a late one still
          // counts for later snapshots, never for a reply already sent.
          if (v !== undefined && asked === generation) remember(key, v);
        });
        return within(
          call,
          Math.min(classifyTimeoutMs, deadline - Date.now()),
        );
      };
      const started = run();
      pending = started;
      pendingVerdicts.set(key, started);
      started.then(() => {
        if (pendingVerdicts.get(key) === started) pendingVerdicts.delete(key);
      });
    }
    return within(pending, deadline - Date.now());
  };

  // Every verdict, each bounded by `deadline`; resolves once all are in.
  const askAllExtra = (
    descriptors: FieldDescriptor[],
    deadline: number,
  ): Promise<(boolean | undefined)[]> => {
    const hook = classifyExtra;
    if (!hook || descriptors.length === 0) return Promise.resolve([]);
    return Promise.all(descriptors.map((d) => askExtra(hook, d, deadline)));
  };

  // Classifies each node from its raw source; returns the sensitive copies.
  //
  // classifyExtra is asked only about a node the deterministic rule leaves
  // ordinary that is a form field by the same test the rule uses
  // (`isFieldLike`: an INPUT_ROLES role, or a value or valuetext), with the
  // AX descriptor (merged with describe()'s when the field was filled and
  // vouched for). A `true` adds the node to the sensitive set, so it is
  // masked by the same code as any other. A filled field nothing could vouch
  // for is already sensitive and is not asked about.
  const classify = async (
    pairs: AXPair[],
    describe: Describe | undefined,
  ): Promise<Set<Obj>> => {
    // The whole pass ends by then, classifyExtra included.
    const deadline = Date.now() + describeDeadlineMs;
    const sensitive = new Set<Obj>();
    const asking: { pair: AXPair; id: number }[] = [];
    const ordinaryFields: { pair: AXPair; extra: FieldDescriptor }[] = [];
    for (const pair of pairs) {
      const verdict = classifyAXNode(pair.src);
      if (verdict === 'ordinary') {
        if (classifyExtra && isFieldLike(pair.src)) {
          ordinaryFields.push({ pair, extra: axDescriptor(pair.src) });
        }
        continue;
      }
      const id = pair.src.backendDOMNodeId;
      if (verdict === 'sensitive' || !describe || typeof id !== 'number') {
        // Fail closed: a filled field nothing can vouch for is sensitive.
        sensitive.add(pair.node);
        continue;
      }
      asking.push({ pair, id });
    }
    if (describe && asking.length > 0) {
      // One describe() per backendDOMNodeId within this call; never across
      // calls.
      const ids = [...new Set(asking.map((a) => a.id))];
      const answers = await describeAll(describe, ids);
      for (const { pair, id } of asking) {
        const fromDom = answers.get(id);
        if (
          !fromDom ||
          isSensitiveField(fromDom) ||
          isSensitiveField({ ...axDescriptor(pair.src), ...fromDom })
        ) {
          sensitive.add(pair.node);
        } else if (classifyExtra) {
          ordinaryFields.push({
            pair,
            extra: { ...axDescriptor(pair.src), ...fromDom },
          });
        }
      }
    }
    if (ordinaryFields.length > 0) {
      // Every verdict is in (or given up) before anything is masked.
      const extra = await askAllExtra(
        ordinaryFields.map((f) => f.extra),
        deadline,
      );
      ordinaryFields.forEach((f, i) => {
        if (extra[i] === true) sensitive.add(f.pair.node);
      });
    }
    return sensitive;
  };

  // DOM counterpart: a value-holding element left ordinary (see maskDomNode)
  // that classifyExtra flags is re-masked as sensitive, before the shape
  // guard rebuilds the result.
  const flagDomFields = async (fields: DomField[]): Promise<void> => {
    const deadline = Date.now() + describeDeadlineMs;
    const extra = await askAllExtra(
      fields.map((f) => f.extra),
      deadline,
    );
    fields.forEach((f, i) => {
      if (extra[i] === true) maskDomSubtree(f.node, f.src, true, false);
    });
  };

  const redactFullTree = async (
    result: unknown,
    describe: Describe | undefined,
  ): Promise<unknown> => {
    const tree = copyOf(AX_FULL_TREE, result);
    const pairs = axPairs(axNodesOf(tree), (result as Obj).nodes);
    const sensitive = await classify(pairs, describe);
    maskAXTree(pairs, sensitive, pattern);
    return tree;
  };

  return {
    addKnownValue(value) {
      if (typeof value !== 'string') return false;
      if (value.length > MAX_KNOWN_VALUE_LENGTH) return false;
      if (value.trim().length < MIN_KNOWN_LENGTH) return true;
      if (known.has(value)) return true;
      if (known.size >= maxKnownValues) return false;
      known.set(value, alternativesFor(value));
      rebuild();
      return true;
    },

    hasKnownValue(text) {
      return typeof text === 'string' && scrubKnown(text) !== text;
    },

    clear() {
      known.clear();
      rebuild();
      verdicts.clear();
      pendingVerdicts.clear();
      generation++;
    },

    async redactResult(method, result, describe) {
      const ax = method === AX_FULL_TREE || AX_PARTIAL_METHODS.has(method);
      try {
        if (method === AX_FULL_TREE) {
          return await redactFullTree(result, describe);
        }
        // A partial tree lacks the parents and siblings classification needs.
        if (AX_PARTIAL_METHODS.has(method)) return { nodes: [] };
        if (
          DROPPED_RESULTS.has(method) ||
          (method.startsWith('Accessibility.') && !AX_SAFE_METHODS.has(method))
        ) {
          return null;
        }
        // Collected synchronously, before any await, so concurrent calls
        // never share the list.
        domFields = classifyExtra ? [] : null;
        let copied: unknown;
        let fields: DomField[];
        try {
          copied = copyOf(method, result);
          fields = domFields ?? [];
        } finally {
          domFields = null;
        }
        if (fields.length > 0) await flagDomFields(fields);
        const out = shapeGuard(method, copied);
        if (method === 'DOM.getAttributes') {
          if (
            !isObj(out) ||
            !Array.isArray(out.attributes) ||
            !isObj(result) ||
            !Array.isArray(result.attributes)
          ) {
            return null;
          }
          maskDomSubtree(
            out,
            result,
            isDomFieldSensitive(result.attributes),
            false,
          );
        }
        return out;
      } catch {
        return ax ? { nodes: [] } : null;
      }
    },

    redactEvent(method, params) {
      try {
        if (DROPPED_EVENTS.has(method)) return null;
        const out = shapeGuard(method, copyOf(method, params));
        if (method === 'Accessibility.loadComplete') {
          if (!isObj(out) || !isObj(out.root) || !isObj(params)) return null;
          const pairs = axPairs([out.root], [params.root]);
          const sensitive = new Set(
            pairs
              .filter((p) => classifyAXNode(p.src) !== 'ordinary')
              .map((p) => p.node),
          );
          maskAXTree(pairs, sensitive, pattern);
        } else if (method.startsWith('Accessibility.')) {
          return null;
        }
        return out;
      } catch {
        return null;
      }
    },
  };
}
