// Response redactor for extension (relay) runs. Pure and dependency-free
// beyond the classifier: it is bundled into the extension service worker, so
// no chrome.*, no Node APIs, and it never logs.
//
// Two layers, applied to every CDP result and event before it leaves the
// worker:
// 1. Field masking. In accessibility trees and DOM nodes, every text-bearing
//    field of a node classified sensitive (and of its descendants, which repeat
//    the value as StaticText, InlineTextBox or option names) becomes MASK; an
//    empty field stays empty so the agent can tell filled from empty.
// 2. Known-value scrubbing. Every string anywhere in a result or event has
//    each registered value, in its formatted variants, replaced with MASK.
//
// Fail closed: any error in a redaction path returns a safe empty value
// (`{ nodes: [] }` for accessibility trees, `null` otherwise), never the input.
// The input is never mutated.

import {
  type FieldDescriptor,
  isSensitiveField,
  MASK,
} from './sensitive-fields';

export interface Redactor {
  /**
   * Registers a sensitive value typed during the run, in memory only.
   * Returns `true` when the value is registered, already known, or shorter
   * than 4 characters (too short to scrub without destroying ordinary text;
   * field masking still applies). Returns `false`, keeping every earlier
   * value, when the store already holds `maxKnownValues` values or the value
   * is not a string: the caller must then refuse to type it (fail closed).
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

const MIN_KNOWN_LENGTH = 4;
const DEFAULT_MAX_KNOWN_VALUES = 32;
const MAX_DEPTH = 256;
const SEPARATORS = /[\s\-./]+/;
const JOINERS = ['', ' ', '-', '.', '/'];

const AX_TREE_METHODS = new Set([
  'Accessibility.getFullAXTree',
  'Accessibility.getPartialAXTree',
  'Accessibility.queryAXTree',
  'Accessibility.getChildAXNodes',
]);
const AX_TREE_EVENTS = new Set(['Accessibility.nodesUpdated']);
// Base64 image payloads: scrubbing them would corrupt the image and finds
// nothing. Screenshot masking is handled separately.
const BINARY_DATA_METHODS = new Set([
  'Page.captureScreenshot',
  'Page.screencastFrame',
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
const FIELD_ELEMENTS = new Set(['INPUT', 'TEXTAREA', 'SELECT']);
const VALUE_ATTRIBUTES = new Set([
  'value',
  'label',
  'aria-valuetext',
  'aria-valuenow',
]);

type Obj = Record<string, unknown>;

const isObj = (x: unknown): x is Obj =>
  typeof x === 'object' && x !== null && !Array.isArray(x);

const escapeRegExp = (s: string): string =>
  s.replace(/[.*+?^${}()|[\]\\/-]/g, '\\$&');

const group = (digits: string, sizes: number[]): string[] => {
  const parts: string[] = [];
  let at = 0;
  for (const size of sizes) {
    parts.push(digits.slice(at, at + size));
    at += size;
  }
  return parts;
};

export function valueVariants(value: string): string[] {
  const out = new Set<string>();
  const trimmed = value.trim();
  const bases = [value, trimmed];

  const parts = trimmed.split(SEPARATORS).filter(Boolean);
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

  const add = (v: string) => {
    if (v.length >= MIN_KNOWN_LENGTH) out.add(v);
  };
  for (const base of bases) {
    for (const form of [base, base.toLowerCase(), base.toUpperCase()]) {
      add(form);
      try {
        add(encodeURIComponent(form));
      } catch {
        // A lone surrogate cannot be URL-encoded; the plain form remains.
      }
      add(form.replace(/ /g, '+'));
    }
  }
  return [...out];
}

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

const axString = (v: unknown): string | undefined =>
  isObj(v) && typeof v.value === 'string' ? v.value : undefined;

const axProperties = (node: Obj): Obj[] =>
  Array.isArray(node.properties) ? node.properties.filter(isObj) : [];

const isFieldLike = (node: Obj): boolean => {
  const role = axString(node.role);
  if (role !== undefined && INPUT_ROLES.has(role)) return true;
  if (node.value !== undefined) return true;
  return axProperties(node).some((p) => p.name === 'valuetext');
};

const axDescriptor = (node: Obj): FieldDescriptor => {
  const d: FieldDescriptor = {
    role: axString(node.role),
    label: axString(node.name),
    description: axString(node.description),
  };
  const autocomplete = axProperties(node).find(
    (p) => p.name === 'autocomplete',
  );
  d.autocomplete = autocomplete ? axString(autocomplete.value) : undefined;
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
    const v = node.value.value;
    if ((typeof v === 'string' && v !== '') || typeof v === 'number') {
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

// Masks the sensitive nodes (classified from AX data, plus `extra`) and all
// their descendants. `nodes` must already be a private copy.
const maskAXTree = (nodes: Obj[], extra: ReadonlySet<Obj>): void => {
  const byId = new Map<string, Obj>();
  for (const node of nodes) byId.set(String(node.nodeId), node);
  const stack: Obj[] = [...extra];
  for (const node of nodes) {
    if (isFieldLike(node) && isSensitiveField(axDescriptor(node))) {
      stack.push(node);
    }
  }
  const visited = new Set<Obj>();
  while (stack.length > 0) {
    const node = stack.pop() as Obj;
    if (visited.has(node)) continue;
    visited.add(node);
    maskAXNode(node);
    if (!Array.isArray(node.childIds)) continue;
    for (const childId of node.childIds) {
      const child = byId.get(String(childId));
      if (child && !visited.has(child)) stack.push(child);
    }
  }
};

const DOM_CHILD_KEYS = [
  'children',
  'shadowRoots',
  'pseudoElements',
  'contentDocument',
  'templateContent',
  'importedDocument',
];

const isDomNode = (x: Obj): boolean =>
  typeof x.nodeName === 'string' && typeof x.nodeType === 'number';

const isDomField = (node: Obj, attributes: Map<string, string>): boolean =>
  FIELD_ELEMENTS.has(String(node.nodeName).toUpperCase()) ||
  attributes.has('contenteditable') ||
  INPUT_ROLES.has(attributes.get('role') ?? '');

// Masks the value-bearing attributes and text of a sensitive DOM node and of
// every node below it. `node` must already be a private copy.
const maskDomSubtree = (root: Obj): void => {
  const stack: unknown[] = [root];
  while (stack.length > 0) {
    const node = stack.pop();
    if (Array.isArray(node)) {
      stack.push(...node);
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
        if (
          typeof name === 'string' &&
          VALUE_ATTRIBUTES.has(name.toLowerCase()) &&
          attributes[i + 1] !== ''
        ) {
          attributes[i + 1] = MASK;
        }
      }
    }
    for (const key of DOM_CHILD_KEYS) {
      if (node[key] !== undefined) stack.push(node[key]);
    }
  }
};

const maskIfSensitiveDomNode = (node: Obj): void => {
  if (!isDomNode(node) || !Array.isArray(node.attributes)) return;
  const attributes = attributeMap(node.attributes);
  if (!isDomField(node, attributes)) return;
  if (isSensitiveField(descriptorFromAttributes(node.attributes))) {
    maskDomSubtree(node);
  }
};

export function createRedactor(opts?: { maxKnownValues?: number }): Redactor {
  const maxKnownValues = opts?.maxKnownValues ?? DEFAULT_MAX_KNOWN_VALUES;
  const known = new Map<string, string[]>();
  let pattern: RegExp | null = null;

  const rebuild = () => {
    const variants = new Set<string>();
    for (const list of known.values()) for (const v of list) variants.add(v);
    const sorted = [...variants].sort((a, b) => b.length - a.length);
    // Only ever used with String.prototype.replace, which resets lastIndex.
    pattern =
      sorted.length === 0
        ? null
        : new RegExp(sorted.map(escapeRegExp).join('|'), 'gi');
  };

  const scrub = (s: string): string =>
    pattern === null ? s : s.replace(pattern, MASK);

  // Deep copy with every string scrubbed and every sensitive DOM node masked.
  // Throws on a cycle or excessive depth, which callers turn into a safe
  // value.
  const copy = (
    x: unknown,
    ancestors: Set<object>,
    depth: number,
    skipDataKey: boolean,
  ): unknown => {
    if (typeof x === 'string') return scrub(x);
    if (typeof x !== 'object' || x === null) return x;
    if (depth > MAX_DEPTH || ancestors.has(x)) {
      throw new Error('unredactable structure');
    }
    ancestors.add(x);
    let out: unknown;
    if (Array.isArray(x)) {
      out = x.map((item) => copy(item, ancestors, depth + 1, false));
    } else {
      const o: Obj = {};
      for (const key of Object.keys(x)) {
        const v = (x as Obj)[key];
        o[key] =
          skipDataKey && key === 'data' && typeof v === 'string'
            ? v
            : copy(v, ancestors, depth + 1, false);
      }
      maskIfSensitiveDomNode(o);
      out = o;
    }
    ancestors.delete(x);
    return out;
  };

  const copyOf = (method: string, x: unknown): unknown =>
    copy(x, new Set(), 0, BINARY_DATA_METHODS.has(method));

  const describeSafely = (
    describe: (backendNodeId: number) => Promise<FieldDescriptor | undefined>,
    id: number,
  ): Promise<FieldDescriptor | undefined> =>
    Promise.resolve()
      .then(() => describe(id))
      .then((d) => (isObj(d) ? d : undefined))
      .catch(() => undefined);

  const redactAXResult = async (
    method: string,
    result: unknown,
    describe?: (backendNodeId: number) => Promise<FieldDescriptor | undefined>,
  ): Promise<unknown> => {
    const tree = copyOf(method, result);
    const nodes = axNodesOf(tree);
    const extra = new Set<Obj>();
    if (describe) {
      const lookups: Promise<void>[] = [];
      for (const node of nodes) {
        if (!isFieldLike(node) || typeof node.backendDOMNodeId !== 'number') {
          continue;
        }
        const fromAX = axDescriptor(node);
        if (isSensitiveField(fromAX)) continue;
        lookups.push(
          describeSafely(describe, node.backendDOMNodeId).then((fromDom) => {
            // describe() failing or unknown: AX-only classification stands.
            if (!fromDom) return;
            if (
              isSensitiveField(fromDom) ||
              isSensitiveField({ ...fromAX, ...fromDom })
            ) {
              extra.add(node);
            }
          }),
        );
      }
      await Promise.all(lookups);
    }
    maskAXTree(nodes, extra);
    return tree;
  };

  return {
    addKnownValue(value) {
      if (typeof value !== 'string') return false;
      if (value.trim().length < MIN_KNOWN_LENGTH) return true;
      if (known.has(value)) return true;
      if (known.size >= maxKnownValues) return false;
      known.set(value, valueVariants(value));
      rebuild();
      return true;
    },

    clear() {
      known.clear();
      rebuild();
    },

    async redactResult(method, result, describe) {
      const ax = AX_TREE_METHODS.has(method);
      try {
        if (ax) return await redactAXResult(method, result, describe);
        return copyOf(method, result);
      } catch {
        return ax ? { nodes: [] } : null;
      }
    },

    redactEvent(method, params) {
      try {
        const out = copyOf(method, params);
        if (AX_TREE_EVENTS.has(method)) maskAXTree(axNodesOf(out), new Set());
        return out;
      } catch {
        return null;
      }
    },
  };
}
