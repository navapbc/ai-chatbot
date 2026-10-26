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
}

const MIN_KNOWN_LENGTH = 3;
const MAX_KNOWN_VALUE_LENGTH = 512;
const DEFAULT_MAX_KNOWN_VALUES = 32;
const DEFAULT_DESCRIBE_TIMEOUT_MS = 1500;
const MAX_DEPTH = 256;
const SEPARATORS = /[\s\-./]+/;
const JOINERS = ['', ' ', '-', '.', '/'];
const INVISIBLE = /(?:\u00ad|\u200b|\u200c|\u200d|\u2060|\ufeff)/g;
const NON_ASCII = /[\u0080-\uffff]/;
const BULLETS_ONLY = /^[•●◦■*∗]+$/;
// scheme://host with no userinfo; a URL with userinfo is scrubbed whole.
const URL_ORIGIN = /^[a-z][a-z0-9+.-]*:\/\/(?![^/?#]*@)[^/?#]*/i;
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
// Browser-generated identifiers and enums. Scrubbing them would break
// routing and node references (Tasks 4 and 10 route on the raw message).
const STRUCTURAL_KEYS = new Set([
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
  'role',
  'chromeRole',
  'type',
  'subtype',
  'nodeName',
  'localName',
  'nodeType',
]);

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
const BUTTON_INPUT_TYPES = new Set(['submit', 'button', 'reset', 'image']);
const VALUE_ATTRIBUTES = new Set([
  'value',
  'label',
  'aria-valuetext',
  'aria-valuenow',
]);
// The only attributes a sensitive DOM node keeps in clear.
const SENSITIVE_KEEP_ATTRIBUTES = new Set([
  'type',
  'name',
  'id',
  'autocomplete',
  'class',
  'role',
  'aria-label',
  'aria-labelledby',
  'aria-describedby',
  'required',
  'disabled',
  'readonly',
  'aria-required',
  'aria-invalid',
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
  set('name', a.get('name') ?? a.get('id'));
  set('placeholder', a.get('placeholder') ?? a.get('aria-placeholder'));
  set('label', a.get('aria-label'));
  set('description', a.get('aria-description'));
  return d;
}

// ---------------------------------------------------------------- DOM nodes

// Masks value-bearing attributes and text of `root` and everything below it.
// A sensitive root keeps only allowlisted attributes. `root` is a private copy.
const maskDomSubtree = (
  root: Obj,
  sensitive: boolean,
  keepRootValue: boolean,
): void => {
  const stack: unknown[] = [root];
  while (stack.length > 0) {
    const node = stack.pop();
    if (Array.isArray(node)) {
      for (const item of node) stack.push(item);
      continue;
    }
    if (!isObj(node)) continue;
    if (typeof node.nodeValue === 'string' && node.nodeValue !== '') {
      node.nodeValue = MASK;
    }
    if (Array.isArray(node.attributes)) {
      const attributes = node.attributes;
      for (let i = 0; i + 1 < attributes.length; i += 2) {
        const name = attributes[i];
        const value = attributes[i + 1];
        if (typeof name !== 'string' || value === '') continue;
        const lower = name.toLowerCase();
        if (node === root && keepRootValue && lower === 'value') continue;
        const mask = sensitive
          ? !SENSITIVE_KEEP_ATTRIBUTES.has(lower)
          : VALUE_ATTRIBUTES.has(lower);
        if (mask) attributes[i + 1] = MASK;
      }
    }
    for (const key of DOM_CHILD_KEYS) {
      if (node[key] !== undefined) stack.push(node[key]);
    }
  }
};

const maskDomNode = (node: Obj): void => {
  if (typeof node.nodeName !== 'string' || typeof node.nodeType !== 'number') {
    return;
  }
  const attributes = Array.isArray(node.attributes) ? node.attributes : [];
  const map = attributeMap(attributes);
  const tag = node.nodeName.toUpperCase();
  const holdsValue =
    VALUE_ELEMENTS.has(tag) ||
    map.has('contenteditable') ||
    INPUT_ROLES.has(map.get('role') ?? '');
  if (!holdsValue) return;
  // A button's value attribute is its visible label.
  const keepRootValue =
    tag === 'INPUT' &&
    BUTTON_INPUT_TYPES.has((map.get('type') ?? '').toLowerCase());
  const sensitive = isSensitiveField(descriptorFromAttributes(attributes));
  maskDomSubtree(node, sensitive, keepRootValue && !sensitive);
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

// Applies field masking to a private copy of an AX node list.
// `sensitive`: nodes classified sensitive. `raw`: the unscrubbed input nodes,
// index-aligned with `nodes`. `pattern`: the known-value regex, if any.
const maskAXTree = (
  nodes: Obj[],
  sensitive: ReadonlySet<Obj>,
  raw: unknown[],
  pattern: RegExp | null,
): void => {
  const byId = new Map<string, Obj>();
  for (const node of nodes) byId.set(String(node.nodeId), node);
  const childrenOf = (node: Obj): Obj[] => {
    if (!Array.isArray(node.childIds)) return [];
    const out: Obj[] = [];
    for (const id of node.childIds) {
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
  const scrubbed: Obj[] = [];
  if (pattern !== null) {
    nodes.forEach((node, i) => {
      if (nodeTexts(node) !== nodeTexts(raw[i])) scrubbed.push(node);
    });
  }

  for (const node of sensitive) maskSubtree(node);

  if (pattern !== null) {
    // A node that held a known value: its children carry fragments of it
    // (InlineTextBox runs split mid-value).
    for (const node of scrubbed) {
      if (DOCUMENT_ROLES.has(axString(node.role) ?? '')) continue;
      for (const child of childrenOf(node)) maskSubtree(child);
    }
    maskSplitValues(nodes, raw, childrenOf, pattern, maskSubtree);
  }

  // A node labelled or described by a masked node echoes its text.
  const maskedIds = new Set<number>();
  for (const node of masked) {
    if (typeof node.backendDOMNodeId === 'number') {
      maskedIds.add(node.backendDOMNodeId);
    }
  }
  if (maskedIds.size === 0) return;
  for (const node of nodes) {
    if (masked.has(node)) continue;
    const relations = axProperties(node).filter((p) =>
      RELATION_PROPERTIES.has(String(p.name)),
    );
    const ids: number[] = [];
    referencedIds([node.name, node.description, relations], ids);
    if (!ids.some((id) => maskedIds.has(id))) continue;
    maskText(node.name);
    maskText(node.description);
    for (const relation of relations) maskText(relation.value);
  }
};

// Finds known values split across StaticText nodes: the StaticText texts are
// joined in document order and every node a match spans is masked whole. A
// match inside a single node is already scrubbed in place.
const maskSplitValues = (
  nodes: Obj[],
  raw: unknown[],
  childrenOf: (node: Obj) => Obj[],
  pattern: RegExp,
  maskSubtree: (node: Obj) => void,
): void => {
  const rawText = new Map<Obj, string>();
  nodes.forEach((node, i) => {
    const r = raw[i];
    if (axString(node.role) === 'StaticText' && isObj(r)) {
      const t = axString(r.name);
      if (t) rawText.set(node, t);
    }
  });
  if (rawText.size < 2) return;

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
  const known = new Map<string, Alternative[]>();
  let pattern: RegExp | null = null;

  const rebuild = () => {
    const all = new Map<string, number>();
    for (const list of known.values()) {
      for (const a of list) all.set(a.source, a.weight);
    }
    const sorted = [...all].sort((a, b) => b[1] - a[1]).map(([s]) => s);
    // Only ever used with String.prototype.replace, which resets lastIndex.
    pattern = sorted.length === 0 ? null : new RegExp(sorted.join('|'), 'gi');
  };

  const scrubText = (s: string): string => {
    if (pattern === null) return s;
    const replaced = s.replace(pattern, MASK);
    if (!NON_ASCII.test(replaced)) return replaced;
    // Full-width digits, compatibility forms and zero-width characters.
    const normalized = canonical(replaced);
    if (normalized === replaced) return replaced;
    const again = normalized.replace(pattern, MASK);
    return again === normalized ? replaced : again;
  };

  const scrub = (s: string): string => {
    if (pattern === null) return s;
    const origin = URL_ORIGIN.exec(s);
    if (origin === null) return scrubText(s);
    // A known value in a host is not scrubbed (residual risk).
    return origin[0] + scrubText(s.slice(origin[0].length));
  };

  // Deep copy; strings are scrubbed except at structural positions, and DOM
  // nodes are masked. Throws on a cycle or excessive depth.
  const copy = (
    x: unknown,
    ancestors: Set<object>,
    depth: number,
    scrubbing: boolean,
    skipData: boolean,
    trusted: boolean,
  ): unknown => {
    if (typeof x === 'string') return scrubbing ? scrub(x) : x;
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
      // A RemoteObject's by-value payload is page data: its keys are not
      // protocol fields, so nothing below it is exempt.
      const remoteObject = REMOTE_OBJECT_TYPES.has(String(source.type));
      const o: Obj = {};
      for (const key of Object.keys(source)) {
        const v = source[key];
        const raw =
          (trusted && STRUCTURAL_KEYS.has(key)) ||
          (skipData && key === 'data' && typeof v === 'string') ||
          (axProperty && key === 'name' && typeof v === 'string');
        const pageData = remoteObject && key === 'value';
        o[key] = copy(
          v,
          ancestors,
          depth + 1,
          scrubbing && !raw,
          false,
          trusted && !pageData,
        );
      }
      maskDomNode(o);
      out = o;
    }
    ancestors.delete(x);
    return out;
  };

  const copyOf = (method: string, x: unknown): unknown =>
    copy(x, new Set(), 0, true, BINARY_DATA_METHODS.has(method), true);

  // Resolves to the descriptor, or undefined on rejection, throw, timeout or
  // a non-object answer.
  const describeSafely = (
    describe: Describe,
    id: number,
  ): Promise<FieldDescriptor | undefined> =>
    new Promise((resolve) => {
      const timer = setTimeout(() => resolve(undefined), describeTimeoutMs);
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

  const classify = async (
    nodes: Obj[],
    describe: Describe | undefined,
  ): Promise<Set<Obj>> => {
    const sensitive = new Set<Obj>();
    // One describe() per backendDOMNodeId within this call; never across
    // calls.
    const lookups = new Map<number, Promise<FieldDescriptor | undefined>>();
    const pending: Promise<void>[] = [];
    for (const node of nodes) {
      const verdict = classifyAXNode(node);
      if (verdict === 'ordinary') continue;
      const id = node.backendDOMNodeId;
      if (verdict === 'sensitive' || !describe || typeof id !== 'number') {
        // Fail closed: a filled field nothing can vouch for is sensitive.
        sensitive.add(node);
        continue;
      }
      let lookup = lookups.get(id);
      if (!lookup) {
        lookup = describeSafely(describe, id);
        lookups.set(id, lookup);
      }
      const fromAX = axDescriptor(node);
      pending.push(
        lookup.then((fromDom) => {
          if (
            !fromDom ||
            isSensitiveField(fromDom) ||
            isSensitiveField({ ...fromAX, ...fromDom })
          ) {
            sensitive.add(node);
          }
        }),
      );
    }
    await Promise.all(pending);
    return sensitive;
  };

  const redactFullTree = async (
    result: unknown,
    describe: Describe | undefined,
  ): Promise<unknown> => {
    const tree = copyOf(AX_FULL_TREE, result);
    const nodes = axNodesOf(tree);
    const raw = (result as { nodes: unknown[] }).nodes;
    const sensitive = await classify(nodes, describe);
    maskAXTree(nodes, sensitive, raw, pattern);
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

    clear() {
      known.clear();
      rebuild();
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
        const out = copyOf(method, result);
        if (method === 'DOM.getAttributes') {
          if (!isObj(out) || !Array.isArray(out.attributes)) return null;
          const sensitive = isSensitiveField(
            descriptorFromAttributes(out.attributes),
          );
          maskDomSubtree({ attributes: out.attributes }, sensitive, false);
        }
        return out;
      } catch {
        return ax ? { nodes: [] } : null;
      }
    },

    redactEvent(method, params) {
      try {
        if (DROPPED_EVENTS.has(method)) return null;
        const out = copyOf(method, params);
        if (method === 'Accessibility.loadComplete') {
          if (!isObj(out) || !isObj(out.root) || !isObj(params)) return null;
          const nodes = [out.root];
          const sensitive = new Set(
            nodes.filter((n) => classifyAXNode(n) !== 'ordinary'),
          );
          maskAXTree(nodes, sensitive, [(params as Obj).root], pattern);
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
