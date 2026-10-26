import { inScope, isJavascriptUrl, normalizeOrigin } from './policy';

// Pure module: bundled into the extension service worker, so no node: or
// server-only imports.
//
// Frame scope for the frames that share the tab's CDP session. A same-site
// but cross-origin iframe (another port or subdomain of the run's site)
// lives in the tab's renderer, so it has no child session of its own: the
// bridge tracks the tab's frames and execution contexts from raw events and
// judges every frame-, context- and object-addressed command against them.

export interface FrameInfo {
  url: string;
  parentId?: string;
}

/** What the bridge knows about the frames of one CDP session. */
export interface FrameScope {
  frames: Map<string, FrameInfo>;
  /** executionContextId -> frameId. */
  contexts: Map<number, string>;
  /** Execution context uniqueId -> executionContextId. */
  uniqueContexts: Map<string, number>;
  /**
   * executionContextId -> the origin of the document it was created for (as
   * Chrome reported it, or as discovery read it), or REFUSED_ORIGIN once
   * its frame committed a document of another origin. Isolated worlds and
   * inheriting documents have no entry and follow their frame.
   */
  contextOrigins: Map<number, string>;
  /**
   * Frames and contexts the agent was told of (their attach, commit or
   * creation was forwarded), so their end is forwarded too, wherever they
   * went since: the end events carry ids only.
   */
  announcedFrames: Set<string>;
  announcedContexts: Set<number>;
}

export interface ScopeOrigins {
  origin: string;
  allowedFrameOrigins: readonly string[];
}

export const createFrameScope = (): FrameScope => ({
  frames: new Map(),
  contexts: new Map(),
  uniqueContexts: new Map(),
  contextOrigins: new Map(),
  announcedFrames: new Set(),
  announcedContexts: new Set(),
});

const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

/**
 * True for a URL whose document takes its origin from the parent: about:blank
 * and about:srcdoc (fragment or query allowed), and the empty URL of a frame
 * that has not navigated yet (it holds its initial about:blank).
 */
function inheritsScope(url: string): boolean {
  if (url === '') return true;
  try {
    const parsed = new URL(url);
    return (
      parsed.protocol === 'about:' &&
      (parsed.pathname === 'blank' || parsed.pathname === 'srcdoc')
    );
  } catch {
    return false;
  }
}

/** Marks a context that can no longer be trusted to its frame's verdict. */
export const REFUSED_ORIGIN = 'refused';

/**
 * The origin a document's URL gives its contexts: undefined when the
 * document inherits it (about:blank, about:srcdoc), REFUSED_ORIGIN when it
 * has no https origin.
 */
export function documentOrigin(url: string): string | undefined {
  if (inheritsScope(url)) return undefined;
  return normalizeOrigin(url) ?? REFUSED_ORIGIN;
}

const urlAllowed = (url: string, o: ScopeOrigins) =>
  inScope(url, o.origin) ||
  o.allowedFrameOrigins.some((origin) => inScope(url, origin));

/**
 * True when the frame's document is in the run origin or an allowed frame
 * origin. about:blank and about:srcdoc frames take their parent's scope. An
 * unknown frame, an unknown parent and a parent cycle are out of scope.
 */
export function frameInScope(
  frames: ReadonlyMap<string, FrameInfo>,
  frameId: string,
  origin: string,
  allowedFrameOrigins: readonly string[],
): boolean {
  const seen = new Set<string>();
  let id: string | undefined = frameId;
  while (id !== undefined && !seen.has(id)) {
    seen.add(id);
    const frame = frames.get(id);
    if (!frame) return false;
    if (!inheritsScope(frame.url)) {
      return urlAllowed(frame.url, { origin, allowedFrameOrigins });
    }
    id = frame.parentId;
  }
  return false;
}

/**
 * The execution context id inside a V8 remote object id, which Chromium
 * formats as `<isolate>.<context>.<object>` (measured: a node resolved in a
 * frame carries that frame's main-world context id).
 */
export function contextIdFromObjectId(objectId: string): number | undefined {
  const match = /^-?\d+\.(\d+)\.-?\d+$/.exec(objectId);
  return match ? Number(match[1]) : undefined;
}

export interface Addresses {
  frameIds: unknown[];
  contextIds: unknown[];
  uniqueContextIds: unknown[];
  objectIds: unknown[];
  nodes: Array<{ backendNodeId?: unknown; nodeId?: unknown }>;
}

/**
 * Every handle in a command's params that names a frame, an execution
 * context, a remote object or a DOM node. Values are returned as found, so a
 * malformed one (a numeric frameId) reaches the caller and is refused.
 */
export function commandAddresses(params: unknown): Addresses {
  const out: Addresses = {
    frameIds: [],
    contextIds: [],
    uniqueContextIds: [],
    objectIds: [],
    nodes: [],
  };
  if (!isObject(params)) return out;
  if ('frameId' in params) out.frameIds.push(params.frameId);
  for (const key of ['executionContextId', 'contextId']) {
    if (key in params) out.contextIds.push(params[key]);
  }
  if ('uniqueContextId' in params) {
    out.uniqueContextIds.push(params.uniqueContextId);
  }
  if ('objectId' in params) out.objectIds.push(params.objectId);
  if (Array.isArray(params.arguments)) {
    for (const arg of params.arguments) {
      if (isObject(arg) && 'objectId' in arg) out.objectIds.push(arg.objectId);
    }
  }
  // Every handle present is collected and resolved: a command naming an
  // in-scope node and an out-of-scope one is refused.
  if ('backendNodeId' in params) {
    out.nodes.push({ backendNodeId: params.backendNodeId });
  }
  if ('nodeId' in params) out.nodes.push({ nodeId: params.nodeId });
  return out;
}

type Keep = (frameId: string) => boolean;

/**
 * Removes the AX nodes of frames `keep` refuses, every node below them, and
 * every reference to a removed node, so the tree stays consistent.
 */
export function pruneAXTree(result: unknown, keep: Keep): unknown {
  if (!isObject(result) || !Array.isArray(result.nodes)) return result;
  const nodes = result.nodes.filter(isObject);
  const dropped = new Set<unknown>();
  for (const node of nodes) {
    if (!('frameId' in node)) continue;
    if (typeof node.frameId !== 'string' || !keep(node.frameId)) {
      dropped.add(node.nodeId);
    }
  }
  // Descendants, by childIds and by parentId, until nothing changes.
  for (let changed = dropped.size > 0; changed; ) {
    changed = false;
    for (const node of nodes) {
      if (dropped.has(node.nodeId) && Array.isArray(node.childIds)) {
        for (const child of node.childIds) {
          if (!dropped.has(child)) {
            dropped.add(child);
            changed = true;
          }
        }
      } else if (!dropped.has(node.nodeId) && dropped.has(node.parentId)) {
        dropped.add(node.nodeId);
        changed = true;
      }
    }
  }
  if (dropped.size === 0) return result;
  return {
    ...result,
    nodes: nodes
      .filter((node) => !dropped.has(node.nodeId))
      .map((node) =>
        Array.isArray(node.childIds)
          ? {
              ...node,
              childIds: node.childIds.filter((id) => !dropped.has(id)),
            }
          : node,
      ),
  };
}

/** Each frame of a Page.getFrameTree result, parents first. */
export function framesOfTree(result: unknown): Array<[string, FrameInfo]> {
  const out: Array<[string, FrameInfo]> = [];
  const walk = (tree: unknown) => {
    if (!isObject(tree) || !isObject(tree.frame)) return;
    const { id, url, parentId } = tree.frame;
    if (typeof id !== 'string') return;
    out.push([
      id,
      {
        url: typeof url === 'string' ? url : '',
        parentId: typeof parentId === 'string' ? parentId : undefined,
      },
    ]);
    if (Array.isArray(tree.childFrames)) {
      for (const child of tree.childFrames) walk(child);
    }
  };
  if (isObject(result)) walk(result.frameTree);
  return out;
}

/**
 * A Page.getFrameTree result without the subtrees of frames `keep` refuses,
 * so their URLs and ids never reach the agent. Null when the root itself is
 * refused (or the result is malformed).
 */
export function pruneFrameTree(result: unknown, keep: Keep): unknown {
  const prune = (tree: unknown): Record<string, unknown> | null => {
    if (!isObject(tree) || !isObject(tree.frame)) return null;
    const id = tree.frame.id;
    if (typeof id !== 'string' || !keep(id)) return null;
    if (!Array.isArray(tree.childFrames)) return tree;
    const childFrames = tree.childFrames
      .map(prune)
      .filter((child) => child !== null);
    return { ...tree, childFrames };
  };
  if (!isObject(result)) return null;
  const frameTree = prune(result.frameTree);
  return frameTree ? { ...result, frameTree } : null;
}

const NODE_LIST_KEYS = ['children', 'shadowRoots', 'pseudoElements'];
const NODE_KEYS = ['contentDocument', 'templateContent', 'importedDocument'];

/**
 * A DOM.describeNode result in which every frame owner whose frame `keep`
 * refuses has lost its `frameId` and `contentDocument` (a pierced describe
 * would otherwise carry that frame's URL and attributes). Never mutates the
 * input.
 */
export function pruneDescribedNode(result: unknown, keep: Keep): unknown {
  const prune = (node: unknown): unknown => {
    if (!isObject(node)) return node;
    const { frameId, contentDocument, ...rest } = node;
    const refused =
      'frameId' in node && (typeof frameId !== 'string' || !keep(frameId));
    const out: Record<string, unknown> = refused ? rest : { ...node };
    for (const key of NODE_LIST_KEYS) {
      if (Array.isArray(out[key]))
        out[key] = (out[key] as unknown[]).map(prune);
    }
    for (const key of NODE_KEYS) {
      if (key in out) out[key] = prune(out[key]);
    }
    return out;
  };
  if (!isObject(result) || !('node' in result)) return result;
  return { ...result, node: prune(result.node) };
}

const PRE_COMMIT_EVENTS = new Set([
  'Page.frameRequestedNavigation',
  'Page.frameStartedNavigating',
  'Page.frameScheduledNavigation',
]);
const DOCUMENT_NETWORK_EVENTS = new Set([
  'Network.requestWillBeSent',
  'Network.responseReceived',
]);

/**
 * The URL a navigation event is heading to before it commits (a pre-commit
 * Page event, or a Document request or response), or undefined for any
 * other event. A malformed one gives ''.
 */
export function navigationDestination(
  method: string,
  params: unknown,
): string | undefined {
  if (!isObject(params)) return undefined;
  let url: unknown;
  if (PRE_COMMIT_EVENTS.has(method)) {
    url = params.url;
  } else if (DOCUMENT_NETWORK_EVENTS.has(method)) {
    if (params.type !== 'Document') return undefined;
    const carrier = isObject(params.request)
      ? params.request
      : isObject(params.response)
        ? params.response
        : undefined;
    url = carrier?.url;
  } else {
    return undefined;
  }
  return typeof url === 'string' ? url : '';
}

/**
 * A navigation destination that keeps the frame in scope: an allowed
 * origin, a URL that inherits the frame's scope, or javascript:, which runs
 * in the current document and commits nothing.
 */
export const destinationAllowed = (url: string, origins: ScopeOrigins) =>
  url !== '' &&
  (isJavascriptUrl(url) || inheritsScope(url) || urlAllowed(url, origins));

/**
 * Updates the scope from one raw event of its session and says whether the
 * event may be forwarded: false when it is about a frame (or a context of a
 * frame) out of scope, when a navigation is heading out of scope, or when it
 * names a context the scope never saw. Events that name no frame pass.
 * The caller must apply every event, forwarded or not, in order.
 * `alsoKeep` admits frames judged elsewhere (an out-of-process frame, swapped
 * out of this session, is judged by its own child session).
 */
export function scopeFrameEvent(
  scope: FrameScope,
  origins: ScopeOrigins,
  method: string,
  params: unknown,
  alsoKeep: (frameId: string) => boolean = () => false,
): boolean {
  const keep = (frameId: unknown) =>
    typeof frameId === 'string' &&
    (frameInScope(
      scope.frames,
      frameId,
      origins.origin,
      origins.allowedFrameOrigins,
    ) ||
      alsoKeep(frameId));
  const announceFrame = (frameId: string) => {
    const forward = keep(frameId);
    if (forward) scope.announcedFrames.add(frameId);
    return forward;
  };
  if (!isObject(params)) return true;

  switch (method) {
    case 'Page.frameAttached': {
      const { frameId, parentFrameId } = params;
      if (typeof frameId !== 'string') return false;
      if (!scope.frames.has(frameId)) {
        scope.frames.set(frameId, {
          url: '',
          parentId:
            typeof parentFrameId === 'string' ? parentFrameId : undefined,
        });
      }
      return announceFrame(frameId);
    }
    case 'Page.frameNavigated': {
      const frame = isObject(params.frame) ? params.frame : undefined;
      if (typeof frame?.id !== 'string') return false;
      const url = typeof frame.url === 'string' ? frame.url : '';
      scope.frames.set(frame.id, {
        url,
        parentId:
          typeof frame.parentId === 'string' ? frame.parentId : undefined,
      });
      // The frame's contexts that belong to a document of another origin
      // (the one it left) never pass again, whatever the frame does next.
      const committed = documentOrigin(url);
      if (committed !== undefined) {
        for (const [id, frameId] of scope.contexts) {
          if (frameId !== frame.id) continue;
          if (scope.contextOrigins.get(id) !== committed) {
            scope.contextOrigins.set(id, REFUSED_ORIGIN);
          }
        }
      }
      return announceFrame(frame.id);
    }
    case 'Page.navigatedWithinDocument': {
      const { frameId, url } = params;
      if (typeof frameId !== 'string') return false;
      const frame = scope.frames.get(frameId);
      if (frame) frame.url = typeof url === 'string' ? url : '';
      return keep(frameId);
    }
    case 'Page.frameDetached': {
      const { frameId } = params;
      if (typeof frameId !== 'string') return false;
      const told = scope.announcedFrames.has(frameId) || keep(frameId);
      scope.frames.delete(frameId);
      scope.announcedFrames.delete(frameId);
      return told;
    }
    case 'Runtime.executionContextCreated': {
      const context = isObject(params.context) ? params.context : undefined;
      const aux = isObject(context?.auxData) ? context.auxData : undefined;
      const frameId = aux?.frameId;
      if (typeof frameId !== 'string' || typeof context?.id !== 'number') {
        return false;
      }
      scope.contexts.set(context.id, frameId);
      if (typeof context.uniqueId === 'string') {
        scope.uniqueContexts.set(context.uniqueId, context.id);
      }
      // The context can be created before its frame's frameNavigated: its
      // own origin must be in scope too (isolated worlds report none).
      const origin = context.origin;
      if (origin !== '' && origin !== '://') {
        scope.contextOrigins.set(
          context.id,
          typeof origin === 'string'
            ? (normalizeOrigin(origin) ?? REFUSED_ORIGIN)
            : REFUSED_ORIGIN,
        );
      } else {
        scope.contextOrigins.delete(context.id);
      }
      const forward =
        contextInScope(scope, origins, context.id, alsoKeep) && keep(frameId);
      if (forward) scope.announcedContexts.add(context.id);
      return forward;
    }
    case 'Runtime.executionContextDestroyed': {
      const { executionContextId: id, executionContextUniqueId: uniqueId } =
        params;
      const frameId =
        typeof id === 'number' ? scope.contexts.get(id) : undefined;
      const told =
        typeof id === 'number' &&
        (scope.announcedContexts.has(id) ||
          (frameId !== undefined && keep(frameId)));
      if (typeof id === 'number') {
        scope.contexts.delete(id);
        scope.contextOrigins.delete(id);
        scope.announcedContexts.delete(id);
      }
      if (typeof uniqueId === 'string') scope.uniqueContexts.delete(uniqueId);
      return told;
    }
    case 'Runtime.executionContextsCleared':
      scope.contexts.clear();
      scope.uniqueContexts.clear();
      scope.contextOrigins.clear();
      scope.announcedContexts.clear();
      return true;
    default:
      break;
  }

  if (!('frameId' in params)) return true;
  if (!keep(params.frameId)) return false;
  const destination = navigationDestination(method, params);
  return destination === undefined || destinationAllowed(destination, origins);
}

/**
 * True when a context may be used: its frame is in scope and its own
 * recorded origin (if any) is allowed. Unknown contexts are refused.
 */
export function contextInScope(
  scope: FrameScope,
  origins: ScopeOrigins,
  contextId: number,
  alsoKeep: (frameId: string) => boolean = () => false,
): boolean {
  const frameId = scope.contexts.get(contextId);
  if (frameId === undefined) return false;
  const origin = scope.contextOrigins.get(contextId);
  if (origin !== undefined && !urlAllowed(origin, origins)) return false;
  return (
    frameInScope(
      scope.frames,
      frameId,
      origins.origin,
      origins.allowedFrameOrigins,
    ) || alsoKeep(frameId)
  );
}
