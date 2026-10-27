import { createServer as createHttpServer, type Server } from 'node:http';
import { createServer as createHttpsServer } from 'node:https';
import type { AddressInfo } from 'node:net';
import { makeLocalhostCert } from '../../helpers/harness';

// A stand-in for the web app in the cookie-auth spike: `/login` sets one
// session-like cookie per SameSite shape, and `/whoami` records which of them
// arrived. It sends no CORS headers, so a response the extension can read
// proves its host permission (not CORS) let the request through.

/** Cookie name -> the attributes `/login` sets it with. */
export const COOKIE_SHAPES = {
  sid_lax: 'SameSite=Lax',
  sid_strict: 'SameSite=Strict',
  sid_none: 'SameSite=None; Secure',
  sid_unset: '',
} as const;

export type CookieName = keyof typeof COOKIE_SHAPES;

/** What the server saw for one `/whoami?id=<id>` request. */
export interface SeenRequest {
  id: string;
  method: string;
  cookies: string[];
  secFetchSite: string | null;
  origin: string | null;
}

export interface CookieServer {
  /** https://localhost:<port> or http://127.0.0.1:<port>. */
  origin: string;
  seen: SeenRequest[];
  stop(): Promise<void>;
}

const cookieNames = (header: string | undefined) =>
  (header ?? '')
    .split(';')
    .map((part) => part.trim().split('=')[0])
    .filter((name) => name in COOKIE_SHAPES)
    .sort();

export async function startCookieServer(
  protocol: 'https' | 'http',
): Promise<CookieServer> {
  const seen: SeenRequest[] = [];
  const handler: Parameters<typeof createHttpServer>[1] = (req, res) => {
    const url = new URL(req.url ?? '/', 'http://cookie-server');
    if (url.pathname === '/login') {
      res
        .writeHead(200, {
          'set-cookie': Object.entries(COOKIE_SHAPES).map(([name, attrs]) =>
            [`${name}=v`, 'Path=/', 'HttpOnly', attrs]
              .filter(Boolean)
              .join('; '),
          ),
          'content-type': 'text/html; charset=utf-8',
        })
        .end('<!doctype html><html lang="en"><title>Signed in</title></html>');
      return;
    }
    if (url.pathname === '/whoami') {
      const record: SeenRequest = {
        id: url.searchParams.get('id') ?? '',
        method: req.method ?? '',
        cookies: cookieNames(req.headers.cookie),
        secFetchSite: (req.headers['sec-fetch-site'] as string) ?? null,
        origin: req.headers.origin ?? null,
      };
      seen.push(record);
      req.resume();
      req.on('end', () =>
        res
          .writeHead(200, { 'content-type': 'application/json' })
          .end(JSON.stringify({ cookies: record.cookies })),
      );
      return;
    }
    res.writeHead(404).end();
  };

  const server: Server =
    protocol === 'https'
      ? (createHttpsServer(makeLocalhostCert(), handler) as unknown as Server)
      : createHttpServer(handler);
  const port = await new Promise<number>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () =>
      resolve((server.address() as AddressInfo).port),
    );
  });
  return {
    origin:
      protocol === 'https'
        ? `https://localhost:${port}`
        : `http://127.0.0.1:${port}`,
    seen,
    async stop() {
      server.closeAllConnections();
      await new Promise<void>((resolve) => server.close(() => resolve()));
    },
  };
}
