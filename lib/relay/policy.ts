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
  if (!URL_CHECKED_METHODS.has(method)) return { ok: true };

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
