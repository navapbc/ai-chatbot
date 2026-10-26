import { type FieldDescriptor, isSensitiveField } from './sensitive-fields';

// Screenshot masking. Pure and dependency-free (no chrome.*, no Node APIs):
// bundled into the extension service worker, which supplies `send`.
//
// A screenshot is pixels, so the redactor cannot mask it after the fact.
// Instead every sensitive field on the page is hidden with an inline style
// for the duration of the capture, then the page is put back exactly.
//
// Classification happens in the worker, not the page: the injected function
// (PAGE_SOURCE) only collects each field's attributes, label text and current
// value, and the worker runs the real `isSensitiveField` over them (plus the
// accessible name from the AX tree, the value shape and the run's known
// values). The classifier's patterns are private to sensitive-fields.ts, so a
// copy serialized into the page would drift from it. Known values never go
// into the page.
//
// The injected function never runs in the page's own JavaScript world: each
// frame gets an isolated world (Page.createIsolatedWorld), which shares the
// DOM but has pristine builtins, so page script cannot patch what masks it.

export const SCREENSHOT_REFUSED =
  'The screenshot was refused because sensitive fields could not be masked.';

/** A refused screenshot. The message is fixed: it never carries page data. */
export class ScreenshotMaskError extends Error {
  constructor() {
    super(SCREENSHOT_REFUSED);
    this.name = 'ScreenshotMaskError';
  }
}

export interface MaskSession {
  apply(): Promise<void>;
  restore(): Promise<void>;
}

const RESTORE_FAILED = 'screenshot mask: a session could not be restored';

/** Restores every session, each even if another throws; reports failures. */
async function restoreAll(sessions: readonly MaskSession[]): Promise<void> {
  const results = await Promise.allSettled(
    sessions.map((s) => Promise.resolve().then(() => s.restore())),
  );
  // Logged without the error: a CDP error message can quote page text.
  if (results.some((r) => r.status === 'rejected'))
    console.warn(RESTORE_FAILED);
}

/**
 * Applies every session's mask in order, runs `capture`, and always restores.
 * If an apply rejects, the sessions applied so far (the failing one included:
 * it may have masked part of its page) are restored and the capture is
 * refused with ScreenshotMaskError, never run. A capture error is rethrown
 * as is; a restore failure is reported and never replaces it. A capture
 * that succeeded is returned even if a restore then fails: the caller
 * verifies the masks inside `capture` (see createMaskSession's `verify`).
 */
export async function withScreenshotMask<T>(
  sessions: readonly MaskSession[],
  capture: () => Promise<T>,
): Promise<T> {
  const applied: MaskSession[] = [];
  try {
    for (const session of sessions) {
      applied.push(session);
      await session.apply();
    }
  } catch {
    await restoreAll(applied);
    throw new ScreenshotMaskError();
  }
  try {
    return await capture();
  } finally {
    await restoreAll(applied);
  }
}

/**
 * Runs tasks one at a time, in call order. Two overlapping screenshots would
 * each record the other's mask as the page's own style, and the later
 * restore would leave the page masked; the bridge keeps one queue per tab.
 * A failed task does not stop the ones queued behind it.
 */
export function createScreenshotQueue() {
  let tail: Promise<unknown> = Promise.resolve();
  return <T>(task: () => Promise<T>): Promise<T> => {
    const result = tail.then(task);
    tail = result.then(
      () => {},
      () => {},
    );
    return result;
  };
}

// --- classification ------------------------------------------------------------

/** What PAGE_SOURCE reports for one field's metadata (label text in full). */
export interface FieldParts {
  type?: string;
  autocomplete?: string;
  name?: string;
  id?: string;
  ariaLabel?: string;
  /** Text of label[for], a wrapping label, and aria-labelledby targets. */
  labels?: string[];
  placeholder?: string;
  ariaPlaceholder?: string;
  ariaDescription?: string;
  title?: string;
  /** Text of aria-describedby targets. */
  describedBy?: string[];
  role?: string;
  /** The accessible name from the AX tree, added by the worker. */
  accessibleName?: string;
}

const str = (v: unknown): string | undefined =>
  typeof v === 'string' ? v : undefined;
const strs = (v: unknown): string[] =>
  Array.isArray(v) ? v.filter((x): x is string => typeof x === 'string') : [];
const joined = (...values: (string | undefined)[]) => {
  const present = values.filter(
    (v): v is string => v !== undefined && v !== '',
  );
  return present.length === 0 ? undefined : present.join(' ');
};

/**
 * The field's descriptor, built with the same joins as the redactor's
 * descriptorFromAttributes (name with id, aria-description with title), plus
 * the label text only the page can resolve.
 */
export function descriptorFromParts(parts: FieldParts): FieldDescriptor {
  const d: FieldDescriptor = {};
  const set = (key: keyof FieldDescriptor, value: string | undefined) => {
    if (value !== undefined) d[key] = value;
  };
  set('role', str(parts.role));
  set('inputType', str(parts.type));
  set('autocomplete', str(parts.autocomplete));
  set('name', joined(str(parts.name), str(parts.id)));
  set('placeholder', str(parts.placeholder) ?? str(parts.ariaPlaceholder));
  set(
    'label',
    joined(
      str(parts.ariaLabel),
      ...strs(parts.labels),
      str(parts.accessibleName),
    ),
  );
  set(
    'description',
    joined(
      str(parts.ariaDescription),
      str(parts.title),
      ...strs(parts.describedBy),
    ),
  );
  return d;
}

/**
 * True when the field is sensitive by `isSensitiveField`, judged on the
 * joined descriptor and on each label source alone (joining two labels can
 * only add signals, but a pattern's lookahead could see the next label).
 * Anything that is not an object fails toward sensitive.
 */
export function isSensitiveParts(parts: unknown): boolean {
  if (typeof parts !== 'object' || parts === null) return true;
  const p = parts as FieldParts;
  const descriptor = descriptorFromParts(p);
  if (isSensitiveField(descriptor)) return true;
  const labels = [str(p.ariaLabel), ...strs(p.labels), str(p.accessibleName)];
  return labels.some(
    (label) =>
      label !== undefined && isSensitiveField({ ...descriptor, label }),
  );
}

// --- the page side -------------------------------------------------------------

/**
 * How long the page waits before restoring itself if the worker never does
 * (the service worker died or the debugger detached mid-capture). Longer
 * than the bridge's whole mask-to-verify budget; if a slow capture outlives
 * it anyway, `verify` finds the masks gone and the screenshot is refused,
 * never sent unmasked. Passed to `mask()` so a test can shorten it.
 */
export const SELF_RESTORE_MS = 25_000;

/** Isolated world name; the bridge hides this world's contexts from the agent. */
export const MASK_WORLD = '__labs_screenshot_mask__';

/**
 * A JavaScript expression, evaluated in a frame's isolated world, whose value
 * is a records holder kept by the worker as a remote object. It enumerates
 * the frame's document and every open shadow root (frames are enumerated by
 * the worker from Page.getFrameTree, never through contentDocument).
 * Methods, called with Runtime.callFunctionOn on the holder:
 *   report()           -> { fields: [{ parts, value }], owners: [tag] }
 *   fieldList()        -> the field elements (for their backend node ids)
 *   owner(i)           -> the i-th visible frame owner element
 *   doc()              -> the document (for the closed-shadow-root scan)
 *   mask(indices, ms, ...textNodes) -> masks fields and the parents of
 *                         echoed text nodes, then watches for changes
 *   settle()           -> 'raf' after two animation frames, or 'timeout'
 *                         when none were rendered within 500ms
 *   verify()           -> { ok, values }: ok when nothing touched the masks,
 *                         values are the current values of unmasked fields
 *   restore()          -> puts every touched attribute back exactly
 * It is a string, not a stringified function, so no bundler transform can
 * change what the page runs. It logs nothing.
 */
export const PAGE_SOURCE = `(() => {
  'use strict';
  const MARK = 'data-labs-mask-' + Math.random().toString(36).slice(2, 10);
  const PROPS = [
    ['-webkit-text-fill-color', 'transparent'],
    ['color', 'transparent'],
    ['-webkit-text-security', 'disc'],
    ['-webkit-text-stroke-width', '0px'],
    ['text-shadow', 'none'],
    ['caret-color', 'transparent'],
    // Chromium does not transition -webkit-text-fill-color today; this
    // keeps a page's transition from fading the mask in if it ever does.
    ['transition-property', 'none'],
  ];
  const SKIP_TYPES = new Set(['hidden', 'checkbox', 'radio', 'button', 'submit',
    'reset', 'image', 'file', 'range', 'color']);
  const ROLES = '[role~="textbox"], [role~="searchbox"], [role~="combobox"], [role~="spinbutton"]';
  const FIELDS = 'input, textarea, select, [contenteditable], ' + ROLES;
  const OWNERS = 'iframe, frame, object, embed, fencedframe';
  const WATCHED = FIELDS + ', ' + OWNERS;
  const norm = (s) => s.replace(/\\s+/g, ' ').trim();
  const attr = (el, name) => {
    const v = el.getAttribute(name);
    return v === null ? undefined : norm(v);
  };
  const textOf = (el) => norm(el.textContent || '');
  const byIds = (el, name) => {
    const ids = (el.getAttribute(name) || '').split(/\\s+/).filter(Boolean);
    const root = el.getRootNode();
    const out = [];
    for (const id of ids) {
      const target = root.getElementById ? root.getElementById(id) : null;
      if (target) out.push(textOf(target));
    }
    return out;
  };
  const labelsOf = (el) => {
    const list = el.labels ? Array.from(el.labels) : [];
    const wrapping = el.parentElement ? el.parentElement.closest('label') : null;
    if (wrapping && !list.includes(wrapping)) list.push(wrapping);
    return list.map(textOf).concat(byIds(el, 'aria-labelledby')).filter(Boolean);
  };
  const parts = (el) => ({
    type: el.localName === 'input' ? attr(el, 'type') : undefined,
    autocomplete: attr(el, 'autocomplete'),
    name: attr(el, 'name'),
    id: attr(el, 'id'),
    ariaLabel: attr(el, 'aria-label'),
    labels: labelsOf(el),
    placeholder: attr(el, 'placeholder'),
    ariaPlaceholder: attr(el, 'aria-placeholder'),
    ariaDescription: attr(el, 'aria-description'),
    title: attr(el, 'title'),
    describedBy: byIds(el, 'aria-describedby'),
    role: attr(el, 'role'),
  });
  const valueOf = (el) => {
    if (el.localName === 'select') {
      return Array.from(el.selectedOptions, (o) => o.text + ' ' + o.value).join(' ');
    }
    if (el.localName === 'input' || el.localName === 'textarea') return String(el.value);
    return el.textContent || '';
  };
  const isField = (el) => {
    if (!el.matches(FIELDS)) return false;
    const type = (el.localName === 'input' && el.getAttribute('type')) || '';
    return !SKIP_TYPES.has(type.trim().toLowerCase());
  };
  // A frame owner with no content box paints none of its document.
  const visible = (el) => el.clientWidth > 0 && el.clientHeight > 0;

  const enumerate = () => {
    const roots = [];
    const fields = [];
    const owners = [];
    const walk = (root) => {
      roots.push(root);
      for (const el of root.querySelectorAll('*')) {
        if (isField(el)) fields.push(el);
        if (el.matches(OWNERS)) owners.push(el);
        if (el.shadowRoot) walk(el.shadowRoot);
      }
    };
    walk(document);
    return { roots, fields, owners };
  };
  const found = enumerate();
  const fieldSet = new Set(found.fields);
  const ownerSet = new Set(found.owners);
  const visibleOwners = found.owners.filter(visible);
  const rootElement = document.documentElement;

  const records = [];
  const recorded = new Set();
  const containers = [];
  const observers = [];
  const mutations = [];
  const sheets = [];
  let masked = new Set();
  let timer;

  const maskEl = (el) => {
    if (recorded.has(el)) return;
    recorded.add(el);
    const style = el.style;
    if (!style) return;
    const r = {
      el,
      before: el.getAttribute('style'),
      props: PROPS.map(([p]) => [p, style.getPropertyValue(p), style.getPropertyPriority(p)]),
      after: undefined,
      set: [],
    };
    records.push(r);
    el.setAttribute(MARK, '');
    for (const [p, v] of PROPS) style.setProperty(p, v, 'important');
    r.set = PROPS.map(([p]) => [p, style.getPropertyValue(p)]);
    r.after = el.getAttribute('style');
  };
  // Inline styles cannot reach these pseudo-elements, nor beat a page rule
  // on a descendant with higher specificity: a stylesheet keyed to MARK.
  const RULES = '[' + MARK + ']::selection, [' + MARK + '] *::selection {' +
    ' color: transparent !important; -webkit-text-fill-color: transparent !important;' +
    ' background: transparent !important; text-shadow: none !important; }' +
    ' [' + MARK + ']::placeholder { color: transparent !important;' +
    ' -webkit-text-fill-color: transparent !important; text-shadow: none !important; }' +
    ' [' + MARK + '], [' + MARK + '] * { -webkit-text-fill-color: transparent !important;' +
    ' color: transparent !important; text-shadow: none !important;' +
    ' -webkit-text-stroke-width: 0px !important; }';
  const CLEAR = /^(transparent|rgba\\(\\d+, \\d+, \\d+, 0\\))$/;
  const clear = (c) => CLEAR.test(c);
  const painted = (el, pseudo) => {
    const cs = getComputedStyle(el, pseudo);
    if (!clear(cs.webkitTextFillColor)) return true;
    if (cs.textShadow !== 'none' && cs.textShadow !== '') return true;
    if (pseudo === undefined && cs.webkitTextStrokeWidth !== '0px') return true;
    if (pseudo === '::selection' && !clear(cs.color)) return true;
    return false;
  };

  const restore = () => {
    for (const o of observers.splice(0)) o.disconnect();
    for (const r of records.splice(0).reverse()) {
      const el = r.el;
      el.removeAttribute(MARK);
      const now = el.getAttribute('style');
      if (r.after !== undefined && now === r.after) {
        if (r.before === null) el.removeAttribute('style');
        else el.setAttribute('style', r.before);
        continue;
      }
      // The page changed the style meanwhile: undo only our properties.
      for (const [p, v, priority] of r.props) {
        if (v) el.style.setProperty(p, v, priority);
        else el.style.removeProperty(p);
      }
      if (r.before === null && el.getAttribute('style') === '') {
        el.removeAttribute('style');
      }
    }
    recorded.clear();
    for (const [root, sheet] of sheets.splice(0)) {
      root.adoptedStyleSheets = root.adoptedStyleSheets.filter((s) => s !== sheet);
    }
    clearTimeout(timer);
    return true;
  };

  const touches = (node) =>
    node.nodeType === 1 &&
    (node.matches(WATCHED) || node.querySelector(WATCHED) !== null || recorded.has(node));

  return {
    report() {
      return {
        fields: found.fields.map((el) => ({ parts: parts(el), value: valueOf(el) })),
        owners: visibleOwners.map((el) => el.localName),
      };
    },
    fieldList() {
      return found.fields;
    },
    owner(i) {
      return visibleOwners[i];
    },
    doc() {
      return document;
    },
    mask(indices, ms, ...textNodes) {
      masked = new Set(indices);
      const targets = [];
      for (const i of indices) {
        const el = found.fields[i];
        if (el) targets.push(el);
      }
      for (const node of textNodes) {
        const el = node && node.nodeType === 1 ? node : node && node.parentElement;
        if (el) targets.push(el);
      }
      for (const el of targets) {
        maskEl(el);
        if (el.localName !== 'input') {
          containers.push(el);
          for (const d of el.querySelectorAll('*')) maskEl(d);
        }
      }
      const sheet = new CSSStyleSheet();
      sheet.replaceSync(RULES);
      for (const root of found.roots) {
        root.adoptedStyleSheets = root.adoptedStyleSheets.concat([sheet]);
        sheets.push([root, sheet]);
        const o = new MutationObserver((list) => {
          for (const m of list) mutations.push(m);
        });
        o.observe(root, {
          subtree: true,
          childList: true,
          attributes: true,
          attributeFilter: ['style', 'type', 'contenteditable', 'role', 'hidden', 'class'],
        });
        observers.push(o);
      }
      timer = setTimeout(restore, ms);
      return records.length;
    },
    settle() {
      return new Promise((resolve) => {
        let done = false;
        const end = (how) => {
          if (done) return;
          done = true;
          resolve(how);
        };
        requestAnimationFrame(() => requestAnimationFrame(() => end('raf')));
        setTimeout(() => end('timeout'), 500);
      });
    },
    verify() {
      const fail = { ok: false, values: [] };
      if (document.documentElement !== rootElement) return fail;
      const changes = mutations.splice(0);
      for (const o of observers) for (const m of o.takeRecords()) changes.push(m);
      for (const m of changes) {
        if (m.type === 'childList') {
          for (const n of m.addedNodes) if (touches(n)) return fail;
          for (const n of m.removedNodes) if (touches(n)) return fail;
        } else if (m.attributeName === 'style') {
          if (recorded.has(m.target)) return fail;
        } else if (m.attributeName === 'type' || m.attributeName === 'contenteditable' ||
            m.attributeName === 'role') {
          // The element may have become a field, or changed kind.
          if (fieldSet.has(m.target) || isField(m.target)) return fail;
        }
        // hidden and class changes cannot unmask: inline !important and the
        // MARK sheet win over class rules, and hidden fields are masked too.
      }
      const now = enumerate();
      for (const el of now.fields) if (!fieldSet.has(el)) return fail;
      for (const el of now.owners) if (!ownerSet.has(el)) return fail;
      for (const r of records) {
        if (!r.el.isConnected) return fail;
        for (const [p, v] of r.set) {
          if (r.el.style.getPropertyValue(p) !== v) return fail;
          if (r.el.style.getPropertyPriority(p) !== 'important') return fail;
        }
        if (painted(r.el)) return fail;
        if (painted(r.el, '::selection')) return fail;
        if ((r.el.localName === 'input' || r.el.localName === 'textarea') &&
            painted(r.el, '::placeholder')) return fail;
      }
      for (const el of containers) {
        for (const d of el.querySelectorAll('*')) {
          if (painted(d) || painted(d, '::selection')) return fail;
        }
      }
      const values = [];
      found.fields.forEach((el, i) => {
        if (!masked.has(i)) values.push(valueOf(el));
      });
      return { ok: true, values };
    },
    restore,
  };
})()`;

export type Send = (
  method: string,
  params?: Record<string, unknown>,
) => Promise<unknown>;

export interface MaskSessionOptions {
  /** Sends a CDP command to this session, never through the agent. */
  send: Send;
  /** CDP object group for this capture's remote objects; released on restore. */
  objectGroup: string;
  /**
   * True when a frame is masked by a CDP session of its own (an
   * out-of-process iframe's child session, whose target id is its frame id).
   */
  isCoveredFrame(frameId: string): boolean;
  /**
   * True when a field's current value must be masked whatever its label:
   * a value-shaped number or a registered known value. Values never leave
   * the worker.
   */
  isSensitiveValue(value: string): boolean;
  /** True when page text contains a registered known value. */
  hasKnownValue(text: string): boolean;
  /**
   * An out-of-process iframe's session: its renderer paints separately, so
   * `settle` must see a rendered frame, not only its timeout.
   */
  outOfProcess?: boolean;
  /** Called with each isolated world's execution context id. */
  onWorld?(executionContextId: number): void;
  /** See SELF_RESTORE_MS. */
  selfRestoreMs?: number;
}

export interface VerifiableMaskSession extends MaskSession {
  /**
   * After every session's apply, before the capture: waits until this
   * session's page has rendered a frame with the mask (an out-of-process
   * iframe paints into the tab's screenshot from its own renderer).
   * Rejects with ScreenshotMaskError on any failure.
   */
  settle(): Promise<void>;
  /**
   * After the capture: rejects with ScreenshotMaskError unless nothing
   * touched the masks (no field or frame added, removed or replaced, no
   * mask style changed or out-painted, no navigation) and no unmasked field
   * now holds a sensitive value.
   */
  verify(): Promise<void>;
}

const isObject = (v: unknown): v is Record<string, unknown> =>
  typeof v === 'object' && v !== null;

/**
 * The RemoteObject of an evaluate/callFunctionOn reply. A script that threw
 * still *resolves* with exceptionDetails, which must not pass for success.
 */
function remoteResult(reply: unknown): Record<string, unknown> {
  if (!isObject(reply) || reply.exceptionDetails !== undefined) {
    throw new Error('mask step failed');
  }
  if (!isObject(reply.result)) throw new Error('mask step failed');
  return reply.result;
}

const fail = (): never => {
  throw new Error('mask step failed');
};

/** Every frame id of a Page.getFrameTree reply, root first. */
export function frameIds(tree: unknown): string[] {
  const out: string[] = [];
  const walk = (node: unknown) => {
    if (!isObject(node) || !isObject(node.frame)) fail();
    const n = node as Record<string, unknown>;
    const id = (n.frame as Record<string, unknown>).id;
    if (typeof id !== 'string') fail();
    out.push(id as string);
    if (n.childFrames !== undefined) {
      if (!Array.isArray(n.childFrames)) fail();
      for (const child of n.childFrames as unknown[]) walk(child);
    }
  };
  if (!isObject(tree)) fail();
  walk((tree as Record<string, unknown>).frameTree);
  return out;
}

const FIELD_NODE = new Set(['INPUT', 'TEXTAREA', 'SELECT']);
const FIELD_ROLE = /(?:^|\s)(?:textbox|searchbox|combobox|spinbutton)(?:\s|$)/;

/**
 * True when a DOM.describeNode tree (depth -1, pierce) holds a closed shadow
 * root with a field in it: page script cannot see into it, so it cannot be
 * masked.
 */
export function hasClosedShadowField(node: unknown): boolean {
  const isFieldNode = (n: Record<string, unknown>): boolean => {
    if (typeof n.nodeName === 'string' && FIELD_NODE.has(n.nodeName)) {
      return true;
    }
    const attrs = Array.isArray(n.attributes) ? n.attributes : [];
    for (let i = 0; i + 1 < attrs.length; i += 2) {
      if (attrs[i] === 'contenteditable') return true;
      if (attrs[i] === 'role' && FIELD_ROLE.test(String(attrs[i + 1]))) {
        return true;
      }
    }
    return false;
  };
  const children = (n: Record<string, unknown>): unknown[] => [
    ...(Array.isArray(n.children) ? n.children : []),
    ...(Array.isArray(n.shadowRoots) ? n.shadowRoots : []),
    ...(isObject(n.contentDocument) ? [n.contentDocument] : []),
    ...(isObject(n.templateContent) ? [n.templateContent] : []),
  ];
  const containsField = (n: unknown): boolean =>
    isObject(n) && (isFieldNode(n) || children(n).some(containsField));
  const scan = (n: unknown): boolean => {
    if (!isObject(n)) return false;
    if (n.shadowRootType === 'closed' && containsField(n)) return true;
    return children(n).some(scan);
  };
  return scan(node);
}

interface World {
  holder: string;
  frameId: string;
  contextId: number;
}

/** One CDP session's mask (the tab, or an iframe's child session). */
export function createMaskSession(
  options: MaskSessionOptions,
): VerifiableMaskSession {
  const { send, objectGroup, isCoveredFrame, isSensitiveValue, hasKnownValue } =
    options;
  const selfRestoreMs = options.selfRestoreMs ?? SELF_RESTORE_MS;
  const worlds: World[] = [];
  // Set by restore. An apply still running then (the bridge gave up on it)
  // stops before its next step, so it cannot mask after the restore.
  let closed = false;
  const release = () => {
    send('Runtime.releaseObjectGroup', { objectGroup }).catch(() => {});
  };
  const open = () => {
    if (closed) {
      release();
      throw new Error('mask session closed');
    }
  };

  const call = async (
    objectId: string,
    body: string,
    args: Array<{ value: unknown } | { objectId: string }> = [],
    returnByValue = true,
    awaitPromise = false,
  ) =>
    remoteResult(
      await send('Runtime.callFunctionOn', {
        objectId,
        functionDeclaration: body,
        arguments: args,
        returnByValue,
        awaitPromise,
        silent: true,
        objectGroup,
      }),
    );
  const objectIdOf = (o: Record<string, unknown>) =>
    typeof o.objectId === 'string' ? o.objectId : fail();

  /** Masks one frame (in its isolated world); returns nothing on success. */
  const applyFrame = async (frameId: string, known: Set<string>) => {
    const world = await send('Page.createIsolatedWorld', {
      frameId,
      worldName: MASK_WORLD,
      grantUniveralAccess: false,
    });
    const contextId = isObject(world) ? world.executionContextId : undefined;
    if (typeof contextId !== 'number') fail();
    options.onWorld?.(contextId as number);
    open();
    const holder = objectIdOf(
      remoteResult(
        await send('Runtime.evaluate', {
          expression: PAGE_SOURCE,
          contextId,
          objectGroup,
          returnByValue: false,
          silent: true,
        }),
      ),
    );
    open();
    const w: World = { holder, frameId, contextId: contextId as number };
    worlds.push(w);

    const report = (await call(holder, 'function () { return this.report(); }'))
      .value;
    if (
      !isObject(report) ||
      !Array.isArray(report.fields) ||
      !Array.isArray(report.owners)
    ) {
      fail();
    }
    const fields = (report as { fields: unknown[] }).fields;
    const owners = (report as { owners: unknown[] }).owners;

    // Backstop: every visible frame owner's frame is masked by this session
    // (in the frame tree) or by a child session of its own.
    for (let i = 0; i < owners.length; i++) {
      open();
      const owner = await call(
        holder,
        'function (i) { return this.owner(i); }',
        [{ value: i }],
        false,
      );
      const described = await send('DOM.describeNode', {
        objectId: objectIdOf(owner),
      });
      const node = isObject(described) ? described.node : undefined;
      const ownerFrame = isObject(node) ? node.frameId : undefined;
      if (typeof ownerFrame === 'string') {
        if (!known.has(ownerFrame) && !isCoveredFrame(ownerFrame)) fail();
      } else if (owners[i] !== 'object' && owners[i] !== 'embed') {
        // An object or embed showing an image or plugin has no frame.
        fail();
      }
    }

    // Accessible names (labels the DOM walk cannot resolve: ::before
    // content, img alt, hidden text) and echoed known values.
    open();
    const ax = await send('Accessibility.getFullAXTree', { frameId });
    const axNodes = isObject(ax) && Array.isArray(ax.nodes) ? ax.nodes : fail();
    const names = new Map<number, string>();
    const echoes: number[] = [];
    for (const n of axNodes as unknown[]) {
      if (!isObject(n) || typeof n.backendDOMNodeId !== 'number') continue;
      const name =
        isObject(n.name) && typeof n.name.value === 'string'
          ? n.name.value
          : '';
      names.set(n.backendDOMNodeId, name);
      const role = isObject(n.role) ? n.role.value : undefined;
      if (role === 'StaticText' && name !== '' && hasKnownValue(name)) {
        echoes.push(n.backendDOMNodeId);
      }
    }

    open();
    const list = objectIdOf(
      await call(holder, 'function () { return this.fieldList(); }', [], false),
    );
    const props = await send('Runtime.getProperties', {
      objectId: list,
      ownProperties: true,
    });
    const entries =
      isObject(props) && Array.isArray(props.result) ? props.result : fail();
    const fieldObjects = new Map<number, string>();
    for (const p of entries as unknown[]) {
      if (!isObject(p) || !/^\d+$/.test(String(p.name))) continue;
      if (isObject(p.value) && typeof p.value.objectId === 'string') {
        fieldObjects.set(Number(p.name), p.value.objectId);
      }
    }
    const indices: number[] = [];
    for (let i = 0; i < fields.length; i++) {
      const field = fields[i];
      if (!isObject(field) || !isObject(field.parts)) fail();
      const f = field as { parts: FieldParts; value: unknown };
      const objectId = fieldObjects.get(i) ?? fail();
      open();
      const described = await send('DOM.describeNode', { objectId });
      const node = isObject(described) ? described.node : undefined;
      const backend = isObject(node) ? node.backendNodeId : undefined;
      const accessibleName =
        typeof backend === 'number' ? names.get(backend) : undefined;
      const sensitive =
        // No AX node: nothing to check the label against, so mask.
        accessibleName === undefined ||
        isSensitiveParts({ ...f.parts, accessibleName }) ||
        typeof f.value !== 'string' ||
        isSensitiveValue(f.value);
      if (sensitive) indices.push(i);
    }

    const textNodes: Array<{ objectId: string }> = [];
    for (const backendNodeId of echoes) {
      open();
      const resolved = await send('DOM.resolveNode', {
        backendNodeId,
        executionContextId: contextId,
        objectGroup,
      });
      const object = isObject(resolved) ? resolved.object : undefined;
      textNodes.push({ objectId: objectIdOf(isObject(object) ? object : {}) });
    }

    open();
    const masked = (
      await call(
        holder,
        'function (indices, ms, ...nodes) { return this.mask(indices, ms, ...nodes); }',
        [{ value: indices }, { value: selfRestoreMs }, ...textNodes],
      )
    ).value;
    if (typeof masked !== 'number') fail();
  };

  return {
    async apply() {
      const tree = await send('Page.getFrameTree');
      const ids = frameIds(tree);
      const known = new Set(ids);
      const [root] = ids;
      // A frame listed here but owned by a child session is masked there.
      const own = ids.filter((id) => id === root || !isCoveredFrame(id));
      for (const frameId of own) {
        open();
        await applyFrame(frameId, known);
      }
      // Page script cannot see into a closed shadow root: refuse when one
      // holds a field (scanned once; pierce reaches same-process frames).
      open();
      const doc = objectIdOf(
        await call(
          worlds[0]?.holder ?? fail(),
          'function () { return this.doc(); }',
          [],
          false,
        ),
      );
      const tree2 = await send('DOM.describeNode', {
        objectId: doc,
        depth: -1,
        pierce: true,
      });
      if (!isObject(tree2) || hasClosedShadowField(tree2.node)) fail();
    },

    async settle() {
      const root = worlds[0];
      if (root === undefined) throw new ScreenshotMaskError();
      const attempt = async () => {
        try {
          return (
            await call(
              root.holder,
              'function () { return this.settle(); }',
              [],
              true,
              true,
            )
          ).value;
        } catch {
          throw new ScreenshotMaskError();
        }
      };
      let how = await attempt();
      // An out-of-process frame that rendered nothing may not have painted
      // its mask into the tab's next frame: try once more, then refuse.
      if (options.outOfProcess && how === 'timeout') how = await attempt();
      if (how !== 'raf' && !(how === 'timeout' && !options.outOfProcess)) {
        throw new ScreenshotMaskError();
      }
    },

    async verify() {
      if (worlds.length === 0) throw new ScreenshotMaskError();
      for (const w of worlds) {
        let result: unknown;
        try {
          result = (
            await call(w.holder, 'function () { return this.verify(); }')
          ).value;
        } catch {
          throw new ScreenshotMaskError();
        }
        if (!isObject(result) || result.ok !== true) {
          throw new ScreenshotMaskError();
        }
        const values = Array.isArray(result.values) ? result.values : [];
        if (values.some((v) => typeof v !== 'string' || isSensitiveValue(v))) {
          throw new ScreenshotMaskError();
        }
      }
    },

    async restore() {
      closed = true;
      const all = worlds.splice(0);
      try {
        // After a navigation a holder is gone along with its document: this
        // rejects, and there is nothing left to restore there.
        const results = await Promise.allSettled(
          all.map((w) =>
            call(w.holder, 'function () { return this.restore(); }'),
          ),
        );
        if (results.some((r) => r.status === 'rejected')) {
          throw new Error('restore failed');
        }
      } finally {
        release();
      }
    },
  };
}
