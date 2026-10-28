import { execFile, execFileSync } from 'node:child_process';
import { randomBytes } from 'node:crypto';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
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

/**
 * The bridge's origin hold in the e2e build (the shipped bridge holds 60 s):
 * short, so a run that leaves its origin still ends within seconds, and long
 * enough for a test to see the hold and answer it. Baked in at build time
 * through EXTENSION_TEST_ORIGIN_HOLD_MS, which only a test-hooks build takes.
 * Every launchExtensionBrowser build uses this same value (the e2e files can
 * build extension/dist concurrently).
 */
export const E2E_ORIGIN_HOLD_MS = 2_500;

export const EXTENSION_ID = readFileSync(
  path.join(ROOT, 'extension/EXTENSION_ID'),
  'utf8',
).trim();

/** Isolates this run's agent-browser daemons from any the user has running. */
const NAMESPACE = `relay-e2e-${process.pid}-${randomBytes(3).toString('hex')}`;

/**
 * The namespace of this run's agent-browser daemons. A test that drives the
 * CLI through the tool (which reads AGENT_BROWSER_NAMESPACE) sets that
 * variable to this, so `closeAgentSessions` stops those daemons too.
 */
export const AGENT_NAMESPACE = NAMESPACE;

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
  /**
   * https://localhost:<port>, same site as `origin` but another origin, so
   * its iframe shares the tab's renderer (and CDP session). The same server
   * also answers as https://127.0.0.1:<port>, same site as `frameOrigin`.
   */
  sameSiteOrigin: string;
  formUrl: string;
  /** Value of the HttpOnly cookie set by `/set-cookie`. */
  cookieSentinel: string;
  stop(): Promise<void>;
}

/** A self-signed certificate for localhost, generated for this run. */
export function makeLocalhostCert(): { key: Buffer; cert: Buffer } {
  const dir = mkdtempSync(path.join(tmpdir(), 'relay-e2e-cert-'));
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
    return {
      key: readFileSync(path.join(dir, 'key.pem')),
      cert: readFileSync(path.join(dir, 'cert.pem')),
    };
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

/**
 * Confirmation pages the `/submitted` POST answers with, by `?v=`, for the
 * worker's confirmation read (bridge readConfirmation).
 */
export const CONFIRMATION_PAGES: Record<string, string> = {
  // A hidden number first: innerText leaves it out.
  alnum:
    '<p style="display:none">Confirmation code: HID-999</p>' +
    '<p>Thank you, Jane Doe.</p><p>Confirmation number: AB-1234</p>',
  other: '<p>Application number: 20240017</p>',
  nine: '<p>Confirmation number: 123456789</p>',
  ten: '<p>Confirmation number: 1234567890</p>',
  ssn: '<p>Applicant SSN: 123-45-6789</p><p>Confirmation number: AB-1234</p>',
  sentence:
    '<p>Confirmation numbers are emailed within 2 days (ref 12345).</p>',
  late:
    '<p id="late">Loading...</p><script>setTimeout(() => {' +
    "document.getElementById('late').textContent = " +
    "'Confirmation number: LATE-77'; }, 1500);</script>",
};

/** Serves the fixtures over HTTPS from a certificate generated for this run. */
export async function serveFixtures(): Promise<Fixtures> {
  const tls = makeLocalhostCert();
  const cookieSentinel = `cookie-${randomBytes(12).toString('hex')}`;
  const form = readFileSync(path.join(FIXTURES, 'form.html'), 'utf8');
  const oopif = readFileSync(path.join(FIXTURES, 'oopif.html'), 'utf8');
  const sensitive = readFileSync(path.join(FIXTURES, 'sensitive.html'), 'utf8');
  const oopifSensitive = readFileSync(
    path.join(FIXTURES, 'oopif-sensitive.html'),
    'utf8',
  );
  const fixture = (name: string) =>
    readFileSync(path.join(FIXTURES, name), 'utf8');
  let frameOrigin = '';
  let sameSiteOrigin = '';

  const sendHtml = (res: ServerResponse, body: string) =>
    res
      .writeHead(200, { 'content-type': 'text/html; charset=utf-8' })
      .end(body);

  const main = createHttpsServer(tls, (req, res) => {
    const url = new URL(req.url ?? '/', 'https://localhost');
    if (url.pathname === '/form.html') {
      sendHtml(res, form.replaceAll('{{FRAME_ORIGIN}}', frameOrigin));
    } else if (url.pathname === '/sensitive.html') {
      sendHtml(res, sensitive.replaceAll('{{FRAME_ORIGIN}}', frameOrigin));
    } else if (url.pathname === '/get-submit.html') {
      sendHtml(res, fixture('get-submit.html'));
    } else if (url.pathname === '/samesite.html') {
      sendHtml(
        res,
        fixture('samesite.html').replaceAll(
          '{{SAME_SITE_ORIGIN}}',
          sameSiteOrigin,
        ),
      );
    } else if (url.pathname === '/inner.html') {
      sendHtml(res, fixture('inner.html'));
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
      // `?v=` picks a confirmation page (readConfirmation); none: "Done".
      const body = CONFIRMATION_PAGES[url.searchParams.get('v') ?? ''];
      req.resume();
      req.on('end', () =>
        sendHtml(
          res,
          `<!doctype html><html lang="en"><title>Submitted</title>${body ?? '<p>Done</p>'}</html>`,
        ),
      );
    } else if (url.pathname === '/submit-redirect' && req.method === 'POST') {
      // Post/Redirect/Get: 303 to the page `to` names (this origin's
      // page2.html by default, or the frame origin's away page).
      const away = url.searchParams.get('to') === 'away';
      req.resume();
      req.on('end', () =>
        res
          .writeHead(303, {
            location: away ? `${frameOrigin}/away.html` : '/page2.html',
          })
          .end(),
      );
    } else if (url.pathname === '/no-content' && req.method === 'POST') {
      // A form POST the server answers with nothing: no navigation commits.
      req.resume();
      req.on('end', () => res.writeHead(204).end());
    } else if (url.pathname === '/api') {
      res.writeHead(200, { 'content-type': 'application/json' }).end('{}');
    } else {
      res.writeHead(404).end();
    }
  });
  const frame = createHttpsServer(tls, (req, res) => {
    const url = new URL(req.url ?? '/', 'https://127.0.0.1');
    if (url.pathname === '/oopif.html') {
      sendHtml(res, oopif);
    } else if (url.pathname === '/oopif-sensitive.html') {
      sendHtml(res, oopifSensitive);
    } else if (url.pathname === '/oopif-nested.html') {
      // Nests a frame of the same site (127.0.0.1) but another origin, so it
      // shares this out-of-process frame's session.
      const nested = sameSiteOrigin.replace('localhost', '127.0.0.1');
      sendHtml(
        res,
        `<!doctype html><html lang="en"><title>Nesting frame</title><iframe id="nested" title="nested" src="${nested}/samesite-frame.html"></iframe></html>`,
      );
    } else {
      sendHtml(
        res,
        '<!doctype html><html lang="en"><title>Away</title><p>Away</p></html>',
      );
    }
  });

  const sameSite = createHttpsServer(tls, (req, res) => {
    const url = new URL(req.url ?? '/', 'https://localhost');
    sendHtml(
      res,
      url.pathname === '/samesite-frame.html'
        ? fixture('samesite-frame.html')
        : '<!doctype html><html lang="en"><title>Elsewhere</title></html>',
    );
  });

  const mainPort = await listen(main as unknown as Server);
  const framePort = await listen(frame as unknown as Server);
  const sameSitePort = await listen(sameSite as unknown as Server);
  frameOrigin = `https://127.0.0.1:${framePort}`;
  sameSiteOrigin = `https://localhost:${sameSitePort}`;
  const origin = `https://localhost:${mainPort}`;
  return {
    origin,
    frameOrigin,
    sameSiteOrigin,
    formUrl: `${origin}/form.html`,
    cookieSentinel,
    async stop() {
      main.closeAllConnections();
      frame.closeAllConnections();
      sameSite.closeAllConnections();
      await Promise.all([
        closeServer(main as unknown as Server),
        closeServer(frame as unknown as Server),
        closeServer(sameSite as unknown as Server),
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

/** Both tokens for a fresh run; they share `sub`, as the hub requires. */
export function mintRunTokens(
  secret: string,
  origin: string,
  sub = 'e2e-user',
) {
  const runId = `run-${randomBytes(6).toString('hex')}`;
  return {
    runId,
    extension: mintRelayToken(
      { runId, role: 'extension', origin, sub },
      secret,
    ),
    agent: mintRelayToken({ runId, role: 'agent', origin, sub }, secret),
  };
}

/** A relay running elsewhere (e.g. a container), known only by URL and secret. */
export type ExternalRelay = Pick<Relay, 'url' | 'secret'>;

/** Points the harness at an already-running relay; http(s) becomes ws(s). */
export const externalRelay = (url: string, secret: string): ExternalRelay => ({
  url: url.replace(/^http/, 'ws').replace(/\/+$/, ''),
  secret,
});

export const cdpUrl = (relay: Pick<Relay, 'url'>, agentToken: string) =>
  `${relay.url}/cdp?token=${agentToken}`;

// --- browser -----------------------------------------------------------------

export interface ExtensionBrowser {
  context: BrowserContext;
  sw: Worker;
  stop(): Promise<void>;
}

/**
 * Builds the extension (with the bridge's test hooks, which `bridgeMessage`
 * needs) and launches Chromium with it loaded. Pass `extensionDir` to load an
 * extension the caller already built (no build runs; build it with
 * EXTENSION_TEST_HOOKS=1 if the test calls `bridgeMessage`).
 */
export async function launchExtensionBrowser(
  opts: { extensionDir?: string; headless?: boolean } = {},
): Promise<ExtensionBrowser> {
  const extensionDir = opts.extensionDir ?? EXTENSION_DIST;
  if (!opts.extensionDir) {
    const { EXTENSION_RELEASE: _release, ...env } = process.env;
    execFileSync('pnpm', ['extension:build'], {
      cwd: ROOT,
      env: {
        ...env,
        EXTENSION_TEST_HOOKS: '1',
        EXTENSION_TEST_ORIGIN_HOLD_MS: String(E2E_ORIGIN_HOLD_MS),
      },
      stdio: 'ignore',
    });
    // An empty extension page, only in this test build: bridgeMessageFromPage
    // reaches a worker Playwright has no handle for (after a restart).
    writeFileSync(
      path.join(extensionDir, TEST_PAGE),
      '<!doctype html><html lang="en"><title>test</title></html>',
    );
  }
  const profile = mkdtempSync(path.join(tmpdir(), 'relay-e2e-profile-'));
  const context = await chromium.launchPersistentContext(profile, {
    channel: 'chromium',
    headless: opts.headless ?? true,
    args: [
      `--disable-extensions-except=${extensionDir}`,
      `--load-extension=${extensionDir}`,
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

const TEST_PAGE = 'test-page.html';

/**
 * Sends a bridge message through chrome.runtime.sendMessage from an extension
 * page, which also wakes the worker. For a worker restarted under the test,
 * which Playwright reports no handle for. Needs launchExtensionBrowser's own
 * build (it writes the page).
 */
export async function bridgeMessageFromPage(
  context: BrowserContext,
  message: Record<string, unknown>,
): Promise<BridgeResponse> {
  const page = await context.newPage();
  try {
    await page.goto(`chrome-extension://${EXTENSION_ID}/${TEST_PAGE}`);
    return (await page.evaluate(
      (msg) => chrome.runtime.sendMessage(msg),
      message,
    )) as BridgeResponse;
  } finally {
    await page.close();
  }
}

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
