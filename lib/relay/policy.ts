import { ALLOWED_CDP_METHODS } from './allowed-cdp-methods';

// Pure module: bundled into the extension service worker, so no node: or
// server-only imports.

export type CommandCheck = { ok: true } | { ok: false; reason: string };

export function normalizeOrigin(url: string): string | null {
  try {
    const parsed = new URL(url);
    return parsed.protocol === 'https:' ? parsed.origin : null;
  } catch {
    return null;
  }
}

export function inScope(url: string, origin: string): boolean {
  const normalized = normalizeOrigin(url);
  return normalized !== null && normalized === origin;
}

export function isMethodAllowed(method: string): boolean {
  return ALLOWED_CDP_METHODS.has(method);
}

const URL_CHECKED_METHODS = new Set(['Page.navigate', 'Target.createTarget']);

export function checkCommand(
  cmd: { method: string; params?: unknown },
  ctx: { origin: string },
): CommandCheck {
  const { method, params } = cmd;
  const isCreateTarget = method === 'Target.createTarget';
  if (!isCreateTarget && !isMethodAllowed(method)) {
    return { ok: false, reason: `method not allowed: ${method}` };
  }
  if (!URL_CHECKED_METHODS.has(method)) {
    // CDP params are an object when present; anything else is malformed.
    if (
      params !== undefined &&
      (typeof params !== 'object' || params === null || Array.isArray(params))
    ) {
      return { ok: false, reason: `${method} params must be an object` };
    }
    return { ok: true };
  }

  const url =
    typeof params === 'object' && params !== null
      ? (params as { url?: unknown }).url
      : undefined;
  if (typeof url !== 'string') {
    return { ok: false, reason: `${method} requires a string url` };
  }
  if (!inScope(url, ctx.origin)) {
    return { ok: false, reason: `${method} url is out of scope` };
  }
  return { ok: true };
}

/**
 * Events the agent may receive, by exact name. Anything else is dropped, so a
 * new or content-bearing event (console output, exceptions, DOM mutations,
 * accessibility updates, dialog text, Log/Console entries) never reaches the
 * agent by default. Each entry was measured while driving every documented
 * agent-browser command (tests/relay/e2e/allowlist.e2e.test.ts), or carries a
 * reason in tests/relay/policy.test.ts. The redactor still scrubs what is sent.
 *
 * Dropped from the measured set, with the reason:
 * - Accessibility.loadComplete: carries the whole AX tree; agent-browser
 *   asks for it with Accessibility.getFullAXTree.
 * - Page.javascriptDialogOpening / Page.javascriptDialogClosed: carry the
 *   dialog message and typed prompt text; `dialog accept|dismiss` sends
 *   Page.handleJavaScriptDialog without waiting for either.
 * - Page.backForwardCacheNotUsed: diagnostic detail only.
 */
export const ALLOWED_EVENT_METHODS: ReadonlySet<string> = new Set([
  'DOM.documentUpdated',
  'Network.loadingFailed',
  'Network.loadingFinished',
  'Network.requestServedFromCache',
  'Network.requestWillBeSent',
  'Network.responseReceived',
  'Page.domContentEventFired',
  'Page.frameAttached',
  'Page.frameDetached',
  'Page.frameNavigated',
  'Page.frameRequestedNavigation',
  'Page.frameStartedLoading',
  'Page.frameStartedNavigating',
  'Page.frameStoppedLoading',
  'Page.loadEventFired',
  'Page.navigatedWithinDocument',
  'Runtime.executionContextCreated',
  'Runtime.executionContextDestroyed',
  'Runtime.executionContextsCleared',
  'Target.attachedToTarget',
  'Target.detachedFromTarget',
  'Target.targetInfoChanged',
]);

// Network events the agent may see. agent-browser needs them for load and
// idle waits; everything else in the Network domain is dropped.
const FORWARDED_NETWORK_EVENTS = new Set([
  'Network.requestWillBeSent',
  'Network.responseReceived',
  'Network.loadingFinished',
  'Network.loadingFailed',
  'Network.requestServedFromCache',
]);

// Any key naming headers, cookies or request bodies: headers, requestHeaders,
// headersText, requestHeadersText, associatedCookies, blockedCookies,
// cookiePartitionKey, postData, hasPostData, postDataEntries.
const SENSITIVE_KEY = /header|cookie|postdata/i;

function stripSensitive(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(stripSensitive);
  if (typeof value !== 'object' || value === null) return value;
  const out: Record<string, unknown> = {};
  for (const [key, child] of Object.entries(value)) {
    if (SENSITIVE_KEY.test(key)) continue;
    out[key] = stripSensitive(child);
  }
  return out;
}

/**
 * Applied by the extension to every CDP event before it leaves the browser.
 * Returns null to drop an event outside ALLOWED_EVENT_METHODS. Network events
 * lose all header, cookie and post-data fields; other allowed events pass
 * through unchanged.
 */
export function filterEvent(
  method: string,
  params: unknown,
): { method: string; params: unknown } | null {
  if (!ALLOWED_EVENT_METHODS.has(method)) return null;
  if (!method.startsWith('Network.')) return { method, params };
  if (!FORWARDED_NETWORK_EVENTS.has(method)) return null;
  return { method, params: stripSensitive(params) };
}

/**
 * True only for a URL a browser parses as `javascript:` (case-insensitive,
 * after the leading whitespace and control characters, and inner tabs and
 * newlines, that the URL parser strips). Such a navigation runs script in the
 * current document and never commits a new one. Anything else, including a
 * malformed string, is false, so callers fail closed.
 */
export function isJavascriptUrl(url: string): boolean {
  try {
    return new URL(url).protocol === 'javascript:';
  } catch {
    return false;
  }
}
