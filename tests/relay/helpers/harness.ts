import { execFile, execFileSync } from 'node:child_process';
import { randomBytes } from 'node:crypto';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { createServer as createHttpsServer } from 'node:https';
import type { Server, ServerResponse } from 'node:http';
import type { AddressInfo, Socket } from 'node:net';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { type BrowserContext, chromium, type Worker } from 'playwright';
import { mintRelayToken } from '../../../lib/relay/token';
import { createLocalBus } from '../../../relay/bus';
import { createRelayServer } from '../../../relay/server';

// End-to-end harness: HTTPS fixtures, a relay on ws://127.0.0.1, Chromium
// with the bridge extension, and agent-browser driving it through the relay.

const ROOT = path.resolve(__dirname, '../../..');
const FIXTURES = path.join(ROOT, 'tests/relay/fixtures');
const EXTENSION_DIST = path.join(ROOT, 'extension/dist');
const AGENT_BROWSER = path.join(ROOT, 'node_modules/.bin/agent-browser');

export const EXTENSION_ID = readFileSync(
  path.join(ROOT, 'extension/EXTENSION_ID'),
  'utf8',
).trim();

/** Isolates this run's agent-browser daemons from any the user has running. */
const NAMESPACE = `relay-e2e-${process.pid}-${randomBytes(3).toString('hex')}`;

const listen = (server: Server, port = 0) =>
  new Promise<number>((resolve, reject) => {
    server.once('error', reject);
    server.listen(port, '127.0.0.1', () =>
      resolve((server.address() as AddressInfo).port),
    );
  });

const closeServer = (server: Server) =>
  new Promise<void>((resolve) => server.close(() => resolve()));

// --- fixtures ----------------------------------------------------------------

export interface Fixtures {
  /** https://localhost:<port>, the run origin. */
  origin: string;
  /** https://127.0.0.1:<port>, cross-site, so its iframe is out of process. */
  frameOrigin: string;
  formUrl: string;
  /** Value of the HttpOnly cookie set by `/set-cookie`. */
  cookieSentinel: string;
  stop(): Promise<void>;
}

/** Serves the fixtures over HTTPS from a certificate generated for this run. */
export async function serveFixtures(): Promise<Fixtures> {
  const dir = mkdtempSync(path.join(tmpdir(), 'relay-e2e-cert-'));
  let tls: { key: Buffer; cert: Buffer };
  try {
    execFileSync(
      'openssl',
      [
        'req',
        '-x509',
        '-newkey',
        'rsa:2048',
        '-nodes',
        '-subj',
        '/CN=localhost',
        '-days',
        '1',
        '-keyout',
        path.join(dir, 'key.pem'),
        '-out',
        path.join(dir, 'cert.pem'),
      ],
      { stdio: 'ignore' },
    );
    tls = {
      key: readFileSync(path.join(dir, 'key.pem')),
      cert: readFileSync(path.join(dir, 'cert.pem')),
    };
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }

  const cookieSentinel = `cookie-${randomBytes(12).toString('hex')}`;
  const form = readFileSync(path.join(FIXTURES, 'form.html'), 'utf8');
  const oopif = readFileSync(path.join(FIXTURES, 'oopif.html'), 'utf8');
  let frameOrigin = '';

  const sendHtml = (res: ServerResponse, body: string) =>
    res
      .writeHead(200, { 'content-type': 'text/html; charset=utf-8' })
      .end(body);

  const main = createHttpsServer(tls, (req, res) => {
    const url = new URL(req.url ?? '/', 'https://localhost');
    if (url.pathname === '/form.html') {
      sendHtml(res, form.replaceAll('{{FRAME_ORIGIN}}', frameOrigin));
    } else if (url.pathname === '/set-cookie') {
      res
        .writeHead(200, {
          'set-cookie': `bridge_sid=${cookieSentinel}; Path=/; Secure; HttpOnly; SameSite=Strict`,
          'content-type': 'application/json',
        })
        .end('{}');
    } else if (url.pathname === '/page2.html') {
      sendHtml(
        res,
        '<!doctype html><html lang="en"><title>Second Page</title><p>Second</p></html>',
      );
    } else if (url.pathname === '/submitted' && req.method === 'POST') {
      req.resume();
      req.on('end', () =>
        sendHtml(
          res,
          '<!doctype html><html lang="en"><title>Submitted</title><p>Done</p></html>',
        ),
      );
    } else if (url.pathname === '/api') {
      res.writeHead(200, { 'content-type': 'application/json' }).end('{}');
    } else {
      res.writeHead(404).end();
    }
  });
  const frame = createHttpsServer(tls, (req, res) => {
    const url = new URL(req.url ?? '/', 'https://127.0.0.1');
    sendHtml(
      res,
      url.pathname === '/oopif.html'
        ? oopif
        : '<!doctype html><html lang="en"><title>Away</title><p>Away</p></html>',
    );
  });

  const mainPort = await listen(main as unknown as Server);
  const framePort = await listen(frame as unknown as Server);
  frameOrigin = `https://127.0.0.1:${framePort}`;
  const origin = `https://localhost:${mainPort}`;
  return {
    origin,
    frameOrigin,
    formUrl: `${origin}/form.html`,
    cookieSentinel,
    async stop() {
      main.closeAllConnections();
      frame.closeAllConnections();
      await Promise.all([
        closeServer(main as unknown as Server),
        closeServer(frame as unknown as Server),
      ]);
    },
  };
}

// --- relay -------------------------------------------------------------------

export interface Relay {
  /** ws://127.0.0.1:<port>, the base URL handed to startRun. */
  url: string;
  port: number;
  secret: string;
  logs: Array<Record<string, unknown>>;
  /** Every message published on the relay's bus, in order. */
  frames: string[];
  stop(): Promise<void>;
}

/** Starts a relay; pass the previous port and secret to simulate a restart. */
export async function startRelay(
  opts: { port?: number; secret?: string } = {},
): Promise<Relay> {
  const secret = opts.secret ?? randomBytes(32).toString('hex');
  const logs: Array<Record<string, unknown>> = [];
  const bus = createLocalBus();
  // Records every bus message, so tests can assert on what crossed the relay.
  const frames: string[] = [];
  const publish = bus.publish.bind(bus);
  bus.publish = (channel, message) => {
    frames.push(message);
    return publish(channel, message);
  };
  const server = createRelayServer({
    secret,
    allowedHostSuffixes: [],
    allowedHosts: ['127.0.0.1', 'localhost'],
    extensionId: EXTENSION_ID,
    bus,
    log: (line) => logs.push(line),
  });
  // server.close() leaves upgraded WebSockets open; track them to cut them.
  const sockets = new Set<Socket>();
  server.on('upgrade', (_req, socket: Socket) => {
    sockets.add(socket);
    socket.once('close', () => sockets.delete(socket));
  });
  const port = await listen(server, opts.port);
  return {
    url: `ws://127.0.0.1:${port}`,
    port,
    secret,
    logs,
    frames,
    async stop() {
      for (const socket of sockets) socket.destroy();
      await closeServer(server);
      await bus.close();
    },
  };
}

export function mintRunTokens(secret: string, origin: string) {
  const runId = `run-${randomBytes(6).toString('hex')}`;
  return {
    runId,
    extension: mintRelayToken({ runId, role: 'extension', origin }, secret),
    agent: mintRelayToken({ runId, role: 'agent', origin }, secret),
  };
}

export const cdpUrl = (relay: Relay, agentToken: string) =>
  `${relay.url}/cdp?token=${agentToken}`;

// --- browser -----------------------------------------------------------------

export interface ExtensionBrowser {
  context: BrowserContext;
  sw: Worker;
  stop(): Promise<void>;
}

/** Builds the extension and launches Chromium with it loaded. */
export async function launchExtensionBrowser(): Promise<ExtensionBrowser> {
  execFileSync('pnpm', ['extension:build'], { cwd: ROOT, stdio: 'ignore' });
  const profile = mkdtempSync(path.join(tmpdir(), 'relay-e2e-profile-'));
  const context = await chromium.launchPersistentContext(profile, {
    channel: 'chromium',
    headless: true,
    args: [
      `--disable-extensions-except=${EXTENSION_DIST}`,
      `--load-extension=${EXTENSION_DIST}`,
      '--ignore-certificate-errors',
    ],
  });
  const sw =
    context.serviceWorkers()[0] ??
    (await context.waitForEvent('serviceworker'));
  return {
    context,
    sw,
    async stop() {
      await context.close();
      rmSync(profile, { recursive: true, force: true });
    },
  };
}

export type BridgeResponse = { ok: true } | { ok: false; error: string };

/** Sends a bridge message to the service worker's onMessage handler. */
export const bridgeMessage = (sw: Worker, message: Record<string, unknown>) =>
  sw.evaluate(
    (msg) =>
      (
        globalThis as unknown as {
          bridgeHandleMessage: (m: unknown) => Promise<unknown>;
        }
      ).bridgeHandleMessage(msg),
    message,
  ) as Promise<BridgeResponse>;

// --- agent-browser -----------------------------------------------------------

export interface AgentResult {
  success: boolean;
  data?: unknown;
  error?: string | null;
}

// `--json` prints one object, or an array for `batch`; logs may precede it.
const parseOutput = (text: string): AgentResult => {
  try {
    return JSON.parse(text);
  } catch {
    return JSON.parse(text.split('\n').at(-1) ?? '');
  }
};

/**
 * Runs one agent-browser command against the relay and returns its parsed
 * `--json` output. `session` names the agent-browser daemon to use.
 */
export function agentBrowser(
  args: string[],
  cdpUrl: string,
  session = 'e2e',
): Promise<AgentResult> {
  return new Promise((resolve, reject) => {
    execFile(
      AGENT_BROWSER,
      ['--cdp', cdpUrl, '--session', session, '--json', ...args],
      {
        env: { ...process.env, AGENT_BROWSER_NAMESPACE: NAMESPACE },
        timeout: 60_000,
      },
      (error, stdout) => {
        const text = stdout.toString().trim();
        try {
          resolve(parseOutput(text));
        } catch {
          // error.message repeats the command line, which carries the token.
          const code = error && 'code' in error ? String(error.code) : 'none';
          reject(
            new Error(`agent-browser ${args[0]} gave no JSON (exit ${code})`),
          );
        }
      },
    );
  });
}

/** Shuts down every agent-browser daemon this test run started. */
export function closeAgentSessions(): Promise<void> {
  return new Promise((resolve) => {
    execFile(
      AGENT_BROWSER,
      ['close', '--all'],
      {
        env: { ...process.env, AGENT_BROWSER_NAMESPACE: NAMESPACE },
        timeout: 20_000,
      },
      () => resolve(),
    );
  });
}
