import { createServer, type IncomingMessage, type Server } from 'node:http';
import type { Duplex } from 'node:stream';
import { type RawData, type WebSocket, WebSocketServer } from 'ws';
import { type RelayClaims, verifyRelayToken } from '../lib/relay/token';
import type { Bus } from './bus';
import { type PeerSocket, RunHub } from './hub';

export interface RelayServerOptions {
  secret: string;
  allowedHostSuffixes: string[];
  allowedHosts: string[];
  extensionId?: string;
  bus: Bus;
  log: (line: Record<string, unknown>) => void;
  /** Largest accepted message, in bytes. Defaults to 32 MiB. */
  maxPayloadBytes?: number;
  /** Logs `{event: 'method', method}` once per distinct forwarded CDP method. */
  recordMethods?: boolean;
}

const DEFAULT_MAX_PAYLOAD = 32 * 1024 * 1024;
const POLICY_CLOSE = 1008;
const EXTENSION_ORIGIN_PREFIX = 'chrome-extension://';
// Hub events that carry a method the agent sent on to the extension.
const FORWARDED_METHOD_EVENTS = new Set(['command', 'command-failed']);

const ignore = () => {};

const HOST_PATTERN = /^([a-z0-9.-]+|\[[0-9a-f:.]+\])(:\d+)?$/;

/** Strictly parsed hostname, or null when the Host header is malformed. */
const hostnameOf = (req: IncomingMessage): string | null => {
  const match = HOST_PATTERN.exec((req.headers.host ?? '').toLowerCase());
  if (!match) return null;
  const hostname = match[1];
  if (hostname.endsWith('.') || hostname.startsWith('.')) return null;
  return hostname;
};

// Request paths are attacker-controlled and may carry a token, so logs only
// ever see one of these fixed values.
const classifyPath = (path: string): string =>
  path === '/extension' || path === '/cdp' || path === '/healthz'
    ? path
    : 'other';

const rawToString = (data: RawData): string => {
  if (Array.isArray(data)) return Buffer.concat(data).toString('utf8');
  if (data instanceof ArrayBuffer) return Buffer.from(data).toString('utf8');
  return data.toString('utf8');
};

/** Adapts a `ws` socket to the hub's PeerSocket; binary frames are dropped. */
function toPeer(ws: WebSocket): PeerSocket {
  return {
    send: (data) => ws.send(data),
    close: (code, reason) => ws.close(code, reason),
    on(event: 'message' | 'close', cb: (data: string) => void) {
      if (event === 'close') {
        ws.on('close', () => (cb as () => void)());
        return;
      }
      ws.on('message', (data, isBinary) => {
        if (isBinary) return;
        cb(rawToString(data));
      });
    },
  };
}

export function createRelayServer(opts: RelayServerOptions): Server {
  const suffixes = opts.allowedHostSuffixes.map((s) =>
    s.toLowerCase().replace(/^\./, ''),
  );
  const hosts = new Set(opts.allowedHosts.map((h) => h.toLowerCase()));

  // Nothing here may be handed a URL, header or token value.
  const log = (line: Record<string, unknown>) => {
    try {
      opts.log(line);
    } catch {
      // A failing logger must not break the relay.
    }
  };
  const seenMethods = new Set<string>();
  const hubLog = (line: Record<string, unknown>) => {
    log(line);
    if (
      opts.recordMethods &&
      typeof line.event === 'string' &&
      FORWARDED_METHOD_EVENTS.has(line.event) &&
      typeof line.method === 'string' &&
      !seenMethods.has(line.method)
    ) {
      seenMethods.add(line.method);
      log({ event: 'method', method: line.method });
    }
  };

  const hub = new RunHub({ bus: opts.bus, log: hubLog });
  const wss = new WebSocketServer({
    noServer: true,
    maxPayload: opts.maxPayloadBytes ?? DEFAULT_MAX_PAYLOAD,
  });

  const hostOk = (req: IncomingMessage): boolean => {
    const host = hostnameOf(req);
    if (host === null) return false;
    return (
      hosts.has(host) ||
      suffixes.some((s) => host === s || host.endsWith(`.${s}`))
    );
  };

  const originOk = (path: string, origin: string | undefined): boolean => {
    if (path === '/cdp') return origin === undefined;
    if (origin === undefined) return false;
    return opts.extensionId
      ? origin === `${EXTENSION_ORIGIN_PREFIX}${opts.extensionId}`
      : origin.startsWith(EXTENSION_ORIGIN_PREFIX);
  };

  const server = createServer((req, res) => {
    if (!hostOk(req)) {
      res.statusCode = 421;
      res.end();
      return;
    }
    const path = (req.url ?? '').split('?')[0];
    if (req.method === 'GET' && path === '/healthz') {
      res.statusCode = 200;
      res.end('ok');
      return;
    }
    res.statusCode = 404;
    res.end();
  });

  const reject = (socket: Duplex, path: string, reason: string) => {
    log({ event: 'upgrade-rejected', path: classifyPath(path), reason });
    socket.destroy();
  };

  server.on('upgrade', (req, socket, head) => {
    socket.on('error', ignore);
    const started = Date.now();
    let path = '';
    let token = '';
    try {
      const url = new URL(req.url ?? '', 'http://relay.invalid');
      path = url.pathname;
      token = url.searchParams.get('token') ?? '';
    } catch {
      return reject(socket, '', 'bad-url');
    }
    if (!hostOk(req)) return reject(socket, path, 'host');
    if (path !== '/extension' && path !== '/cdp') {
      return reject(socket, path, 'path');
    }
    if (token === '') return reject(socket, path, 'no-token');
    const claims: RelayClaims | null = verifyRelayToken(token, opts.secret);
    if (!claims) return reject(socket, path, 'token');
    const wantRole = path === '/extension' ? 'extension' : 'agent';
    if (claims.role !== wantRole) return reject(socket, path, 'role');
    if (!originOk(path, req.headers.origin)) {
      return reject(socket, path, 'origin');
    }

    wss.handleUpgrade(req, socket, head, (ws) => {
      ws.on('error', ignore);
      const peer = toPeer(ws);
      log({
        event: 'accepted',
        runId: claims.runId,
        role: claims.role,
        durationMs: Date.now() - started,
      });
      // The token's sub binds the run to one user: the hub refuses a socket
      // whose sub differs from the one the run's first socket carried.
      if (claims.role === 'extension') {
        hub.attachExtension(claims.runId, claims.origin, claims.sub, peer);
        return;
      }
      const result = hub.attachAgent(
        claims.runId,
        claims.origin,
        claims.sub,
        peer,
      );
      if (result === 'busy') ws.close(POLICY_CLOSE, 'run busy');
      else if (result === 'origin-mismatch') {
        ws.close(POLICY_CLOSE, 'origin mismatch');
      } else if (result === 'sub-mismatch') {
        ws.close(POLICY_CLOSE, 'user mismatch');
      }
    });
  });

  server.on('close', () => wss.close());
  return server;
}
