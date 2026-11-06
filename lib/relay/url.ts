// Pure URL helpers shared by the token route, the Eve relay tool and tests.
// No Node-only or server-only imports: Eve's bundler must be able to take it.

export const LOOPBACK_HOSTS = new Set(['localhost', '127.0.0.1', '[::1]']);

/**
 * Normalizes a relay URL to its websocket form: https and wss become wss; http
 * and ws are accepted for loopback hosts only and become ws. Rejects
 * credentials, query and fragment. Returns `<ws|wss>://host[/path]` with
 * trailing slashes removed, or null for anything else.
 */
export function toRelayWsUrl(input: string): string | null {
  try {
    const url = new URL(input);
    if (url.username || url.password || url.search || url.hash) return null;
    const loopback = LOOPBACK_HOSTS.has(url.hostname);
    let scheme: 'ws' | 'wss';
    if (url.protocol === 'https:' || url.protocol === 'wss:') {
      scheme = 'wss';
    } else if (
      (url.protocol === 'http:' || url.protocol === 'ws:') &&
      loopback
    ) {
      scheme = 'ws';
    } else {
      return null;
    }
    const path = url.pathname.replace(/\/+$/, '');
    return `${scheme}://${url.host}${path}`;
  } catch {
    return null;
  }
}
