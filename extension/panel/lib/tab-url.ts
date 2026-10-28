// What the allow-site card says about the tab's actual URL. Only the origin
// and the path: the query and fragment (where identifiers and tokens usually
// live) and any credentials are never shown. The path itself may still hold
// an identifier, so the result is shown once and never logged or stored.

export const MAX_TAB_PATH = 80;
export const UNKNOWN_PAGE = 'an unknown page';

const ELLIPSIS = '…';

/**
 * `origin + pathname` for an http(s) URL, the path cut to MAX_TAB_PATH code
 * points with a marker when cut; UNKNOWN_PAGE for anything else. The origin
 * is the parser's: lowercase, credentials dropped, an IDN host in punycode
 * (so a look-alike host shows as what it is). Pure.
 */
export const describeTabUrl = (url: string | undefined): string => {
  if (typeof url !== 'string' || url === '') return UNKNOWN_PAGE;
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    return UNKNOWN_PAGE;
  }
  if (parsed.protocol !== 'https:' && parsed.protocol !== 'http:') {
    return UNKNOWN_PAGE;
  }
  const path = Array.from(parsed.pathname);
  const shown =
    path.length > MAX_TAB_PATH
      ? `${path.slice(0, MAX_TAB_PATH).join('')}${ELLIPSIS}`
      : parsed.pathname;
  return `${parsed.origin}${shown}`;
};
