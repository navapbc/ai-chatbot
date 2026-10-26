import { type FieldDescriptor, isSensitiveField } from './sensitive-fields';

// Screenshot masking. Pure and dependency-free (no chrome.*, no Node APIs):
// bundled into the extension service worker, which supplies `send`.
//
// A screenshot is pixels, so the redactor cannot mask it after the fact.
// Instead every sensitive field on the page is hidden with an inline style
// for the duration of the capture, then the page is put back exactly.
//
// Classification happens in the worker, not the page: the injected function
// (PAGE_SOURCE) only collects each field's attributes and label text, and
// the worker runs the real `isSensitiveField` over them. The classifier's
// patterns are private to sensitive-fields.ts, so a copy serialized into the
// page would drift from it; this way the two cannot disagree.

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
 * restore would leave the page masked; the bridge keeps one queue per run.
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

/** What PAGE_SOURCE reports for one field: raw page strings, never values. */
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
  set('label', joined(str(parts.ariaLabel), ...strs(parts.labels)));
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
  const labels = [str(p.ariaLabel), ...strs(p.labels)];
  return labels.some(
    (label) =>
      label !== undefined && isSensitiveField({ ...descriptor, label }),
  );
}

// --- the page side -------------------------------------------------------------

/**
 * How long the page waits before restoring itself if the worker never does
 * (the service worker died or the debugger detached mid-capture). Longer
 * than the bridge's whole capture budget, so a live capture never sees it.
 */
export const SELF_RESTORE_MS = 25_000;

/**
 * A JavaScript expression, sent with Runtime.evaluate, whose value is a
 * records holder kept by the worker as a remote object (never put on
 * `window`, so the page cannot find or replace it). It collects the
 * candidate fields of the document, of every same-origin frame (recursively)
 * and of every open shadow root, and the visible frames it could not inspect.
 * Methods, called with Runtime.callFunctionOn on the holder:
 *   report()      -> { fields: FieldParts[], opaque: number }
 *   frame(i)      -> the i-th uninspectable frame element
 *   mask(indices) -> number of fields masked
 *   verify()      -> true when every mask is still in place and no inspected
 *                    frame has changed document
 *   restore()     -> puts every touched style attribute back exactly
 * It is a string, not a stringified function, so no bundler transform can
 * change what the page runs. It logs nothing.
 */
export const PAGE_SOURCE = `(() => {
  'use strict';
  const PROPS = [
    ['-webkit-text-fill-color', 'transparent'],
    ['color', 'transparent'],
    ['-webkit-text-security', 'disc'],
    ['text-shadow', 'none'],
    ['caret-color', 'transparent'],
  ];
  const SKIP_TYPES = new Set(['hidden', 'checkbox', 'radio', 'button', 'submit',
    'reset', 'image', 'file', 'range', 'color']);
  const FIELDS = 'input, textarea, select, [contenteditable], [role~="textbox"],' +
    ' [role~="searchbox"], [role~="combobox"], [role~="spinbutton"]';
  const clip = (s) => s.replace(/\\s+/g, ' ').trim().slice(0, 500);
  const attr = (el, name) => {
    const v = el.getAttribute(name);
    return v === null ? undefined : clip(v);
  };
  const textOf = (el) => clip(el.textContent || '');
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

  const fields = [];
  const opaque = [];
  const frames = [];
  const docOf = (frame) => {
    try {
      return frame.contentDocument;
    } catch (e) {
      return null;
    }
  };
  // Each inspected document with its root element: a frame that navigated,
  // or a document rewritten by document.open(), shows content never masked.
  const docs = [];
  const walk = (root) => {
    if (root.nodeType === 9) docs.push([root, root.documentElement]);
    for (const el of root.querySelectorAll(FIELDS)) {
      const type = (el.localName === 'input' && el.getAttribute('type')) || '';
      if (!SKIP_TYPES.has(type.trim().toLowerCase())) fields.push(el);
    }
    for (const el of root.querySelectorAll('*')) {
      if (el.shadowRoot) walk(el.shadowRoot);
    }
    for (const frame of root.querySelectorAll('iframe, frame')) {
      const doc = docOf(frame);
      if (doc) {
        frames.push([frame, doc]);
        walk(doc);
        continue;
      }
      // A frame with no content box (display: none, or 0 by 0 inside its
      // border) paints none of its document into a screenshot.
      if (frame.clientWidth > 0 && frame.clientHeight > 0) opaque.push(frame);
    }
  };
  if (typeof document !== 'undefined') walk(document);

  const records = [];
  let timer;
  const restore = () => {
    for (const r of records.splice(0).reverse()) {
      const el = r.el;
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
    try {
      clearTimeout(timer);
    } catch (e) {}
    return true;
  };
  return {
    report() {
      return { fields: fields.map(parts), opaque: opaque.length };
    },
    frame(i) {
      return opaque[i];
    },
    mask(indices) {
      for (const i of indices) {
        const el = fields[i];
        if (!el || !el.isConnected) continue;
        const style = el.style;
        const r = {
          el,
          before: el.getAttribute('style'),
          props: PROPS.map(([p]) => [p, style.getPropertyValue(p), style.getPropertyPriority(p)]),
          after: undefined,
          set: [],
        };
        records.push(r);
        for (const [p, v] of PROPS) style.setProperty(p, v, 'important');
        r.set = PROPS.map(([p]) => [p, style.getPropertyValue(p)]);
        r.after = el.getAttribute('style');
      }
      try {
        timer = setTimeout(restore, ${SELF_RESTORE_MS});
      } catch (e) {}
      return records.length;
    },
    verify() {
      for (const [frame, doc] of frames) {
        if (frame.isConnected && docOf(frame) !== doc) return false;
      }
      for (const [doc, root] of docs) {
        if (doc.documentElement !== root) return false;
      }
      for (const r of records) {
        if (!r.el.isConnected) continue;
        for (const [p, v] of r.set) {
          if (r.el.style.getPropertyValue(p) !== v) return false;
          if (r.el.style.getPropertyPriority(p) !== 'important') return false;
        }
      }
      return true;
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
   * True when a frame the page could not inspect is masked by a session of
   * its own (an out-of-process iframe's child session, whose target id is
   * the frame id).
   */
  isCoveredFrame(frameId: string): boolean;
}

export interface VerifiableMaskSession extends MaskSession {
  /**
   * After the capture: rejects with ScreenshotMaskError unless every mask is
   * still in place in the same documents (a navigation, a re-render that
   * dropped the style, or a lost session means the image may be unmasked).
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

/** One CDP session's mask (the tab, or an iframe's child session). */
export function createMaskSession(
  options: MaskSessionOptions,
): VerifiableMaskSession {
  const { send, objectGroup, isCoveredFrame } = options;
  let holder: string | undefined;

  const call = async (
    objectId: string,
    body: string,
    args: unknown[] = [],
    returnByValue = true,
  ) =>
    remoteResult(
      await send('Runtime.callFunctionOn', {
        objectId,
        functionDeclaration: body,
        arguments: args.map((value) => ({ value })),
        returnByValue,
        silent: true,
        objectGroup,
      }),
    );

  return {
    async apply() {
      const evaluated = remoteResult(
        await send('Runtime.evaluate', {
          expression: PAGE_SOURCE,
          objectGroup,
          returnByValue: false,
          silent: true,
          awaitPromise: false,
        }),
      );
      if (typeof evaluated.objectId !== 'string') {
        throw new Error('mask step failed');
      }
      const objectId = evaluated.objectId;
      holder = objectId;

      const report = (
        await call(objectId, 'function () { return this.report(); }')
      ).value;
      if (
        !isObject(report) ||
        !Array.isArray(report.fields) ||
        typeof report.opaque !== 'number'
      ) {
        throw new Error('mask step failed');
      }
      // A frame the page cannot see into is masked only if a session of
      // its own covers it; otherwise nothing proves it is masked.
      for (let i = 0; i < report.opaque; i++) {
        const frame = await call(
          objectId,
          'function (i) { return this.frame(i); }',
          [i],
          false,
        );
        const described = await send('DOM.describeNode', {
          objectId: frame.objectId,
        });
        const node = isObject(described) ? described.node : undefined;
        const frameId = isObject(node) ? node.frameId : undefined;
        if (typeof frameId !== 'string' || !isCoveredFrame(frameId)) {
          throw new Error('frame not covered');
        }
      }
      const indices = report.fields.flatMap((parts, i) =>
        isSensitiveParts(parts) ? [i] : [],
      );
      const masked = (
        await call(
          objectId,
          'function (indices) { return this.mask(indices); }',
          [indices],
        )
      ).value;
      if (typeof masked !== 'number') throw new Error('mask step failed');
    },

    async verify() {
      if (holder === undefined) throw new ScreenshotMaskError();
      let intact: unknown;
      try {
        intact = (await call(holder, 'function () { return this.verify(); }'))
          .value;
      } catch {
        throw new ScreenshotMaskError();
      }
      if (intact !== true) throw new ScreenshotMaskError();
    },

    async restore() {
      const objectId = holder;
      holder = undefined;
      if (objectId === undefined) return;
      try {
        // After a navigation the holder is gone along with the masked
        // document: this rejects, and there is nothing left to restore.
        await call(objectId, 'function () { return this.restore(); }');
      } finally {
        send('Runtime.releaseObjectGroup', { objectGroup }).catch(() => {});
      }
    },
  };
}
