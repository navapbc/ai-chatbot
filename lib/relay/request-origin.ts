// The extension-origin check shared by the relay route guard and the eve-chat
// relay turns. Pure apart from a one-time, value-free warning.
//
// RELAY_EXTENSION_ORIGIN is the extension's full origin:
// `chrome-extension://` plus its 32-character id (letters a-p). Any other shape
// (empty, trailing slash, uppercase, a bare id) is treated as unset. Unset
// means no check at all (local development and curl).
//
// Observed Chromium behavior: a GET from the extension carries no Origin
// header, a POST carries `Origin: chrome-extension://<id>`. So a present
// Origin must always match, and a missing one is refused only where the caller
// asks (`requireOrigin`, for state-changing requests).

const EXTENSION_ORIGIN = /^chrome-extension:\/\/[a-p]{32}$/;

let warned = false;

export const extensionOrigin = (
  env: Record<string, string | undefined> = process.env,
): string | null => {
  const value = env.RELAY_EXTENSION_ORIGIN;
  if (value === undefined) return null;
  if (EXTENSION_ORIGIN.test(value)) return value;
  if (!warned) {
    warned = true;
    console.warn('[relay-origin] ignoring malformed RELAY_EXTENSION_ORIGIN');
  }
  return null;
};

export const originRefusal = (
  request: Request,
  opts: { requireOrigin?: boolean } = {},
  env: Record<string, string | undefined> = process.env,
): 'mismatch' | 'missing' | null => {
  const expected = extensionOrigin(env);
  if (expected === null) return null;
  const origin = request.headers.get('origin');
  if (origin === null) return opts.requireOrigin ? 'missing' : null;
  return origin === expected ? null : 'mismatch';
};

/** Test hook: lets a test observe the once-per-process warning again. */
export const resetOriginWarningForTests = (): void => {
  warned = false;
};
