import { execFileSync } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { mkdtempSync, rmSync } from 'node:fs';
import {
  createServer,
  type IncomingMessage,
  type Server,
  type ServerResponse,
} from 'node:http';
import type { AddressInfo } from 'node:net';
import { tmpdir } from 'node:os';
import path from 'node:path';
import type { Page } from 'playwright';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { mintRelayToken } from '../../../lib/relay/token';
import {
  agentBrowser,
  bridgeMessage,
  cdpUrl,
  closeAgentSessions,
  EXTENSION_ID,
  type ExtensionBrowser,
  type Fixtures,
  launchExtensionBrowser,
  type Relay,
  serveFixtures,
  startRelay,
} from '../helpers/harness';

// Jev classification in the worker (plan 4b, Task 8), end to end: the
// extension is built with APP_ORIGIN pointing at a fake app, which answers
// POST /api/relay/classify-field. A field the deterministic classifier calls
// ordinary ("Reference code") is one the fake app calls sensitive, so the
// snapshot the agent receives masks it. With the app failing (500) the
// snapshot still works and the field stays visible (deterministic verdict),
// and once the app recovers a later snapshot in the SAME run masks it, so the
// failure was not cached.
//
// Auth is the real extension's: the caseworker signs in with a first-party
// navigation to the fake app (a SameSite=Lax session cookie), and the fake
// route answers 401 without that cookie and 403 without the extension's
// Origin header.
//
// Run with `pnpm test:relay e2e/jev-classification.e2e.test.ts` (it builds
// its own extension copy into a temp dir).

const ROOT = path.resolve(__dirname, '../../..');
const MASK = '[masked]';
const LABEL = 'Reference code';
// Neither SSN-shaped nor a 9+ digit run, so only Jev can mask it.
const VALUE = 'QX-4417-ref';
const SESSION_COOKIE = 'app_session';
const EXTENSION_ORIGIN = `chrome-extension://${EXTENSION_ID}`;

interface FakeApp {
  origin: string;
  mode: 'ok' | 'fail';
  /** Raw bodies of every classify-field request that passed auth. */
  bodies: string[];
  /** Requests refused for a missing cookie or Origin. */
  refused: number;
  stop(): Promise<void>;
}

const readBody = (req: IncomingMessage) =>
  new Promise<string>((resolve) => {
    const chunks: Buffer[] = [];
    req.on('data', (c: Buffer) => chunks.push(c));
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
  });

async function startFakeApp(): Promise<FakeApp> {
  const app: FakeApp = {
    origin: '',
    mode: 'ok',
    bodies: [],
    refused: 0,
    stop: async () => {},
  };
  const send = (res: ServerResponse, status: number, body?: unknown) =>
    res
      .writeHead(status, {
        'content-type': 'application/json',
        'cache-control': 'no-store',
      })
      .end(body === undefined ? undefined : JSON.stringify(body));
  const server: Server = createServer(async (req, res) => {
    const url = new URL(req.url ?? '/', 'http://fake-app');
    if (url.pathname === '/login') {
      res
        .writeHead(200, {
          'set-cookie': `${SESSION_COOKIE}=signed-in; Path=/; HttpOnly; SameSite=Lax`,
          'content-type': 'text/html; charset=utf-8',
        })
        .end('<!doctype html><html lang="en"><title>Signed in</title></html>');
      return;
    }
    if (url.pathname === '/api/relay/classify-field' && req.method === 'POST') {
      const body = await readBody(req);
      const signedIn = (req.headers.cookie ?? '')
        .split(';')
        .some((c) => c.trim() === `${SESSION_COOKIE}=signed-in`);
      if (!signedIn) {
        app.refused++;
        send(res, 401);
        return;
      }
      if (req.headers.origin !== EXTENSION_ORIGIN) {
        app.refused++;
        send(res, 403);
        return;
      }
      app.bodies.push(body);
      if (app.mode === 'fail') {
        send(res, 500);
        return;
      }
      const { fields } = JSON.parse(body) as {
        fields: Array<Record<string, unknown>>;
      };
      send(res, 200, {
        sensitive: fields.map((f) =>
          String(f.label ?? '')
            .toLowerCase()
            .includes('reference code'),
        ),
      });
      return;
    }
    res.writeHead(404).end();
  });
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => resolve());
  });
  app.origin = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
  app.stop = async () => {
    server.closeAllConnections();
    await new Promise<void>((resolve) => server.close(() => resolve()));
  };
  return app;
}

const cleanups: Array<() => Promise<void> | void> = [];

/** Builds the extension into a temp dir with `appOrigin` as APP_ORIGIN. */
function buildExtension(appOrigin: string): string {
  const out = mkdtempSync(path.join(tmpdir(), 'jev-extension-'));
  cleanups.push(() => rmSync(out, { recursive: true, force: true }));
  const {
    APP_ORIGIN: _origin,
    EXTENSION_RELEASE: _release,
    ...env
  } = process.env;
  execFileSync(
    path.join(ROOT, 'node_modules/.bin/tsx'),
    ['scripts/build-extension.ts', '--out', out],
    { cwd: ROOT, env: { ...env, APP_ORIGIN: appOrigin }, stdio: 'ignore' },
  );
  return out;
}

let app: FakeApp;
let fixtures: Fixtures;
let relay: Relay;
let browser: ExtensionBrowser;
let page: Page;
let tabId: number;
let pageUrl: string;

async function until(check: () => boolean | Promise<boolean>, ms = 10_000) {
  const deadline = Date.now() + ms;
  while (!(await check())) {
    if (Date.now() > deadline) throw new Error('condition not met in time');
    await new Promise((r) => setTimeout(r, 50));
  }
}

/** Starts a run on the tab; its id is a UUID, as a real (chat id) run's is. */
async function startRun(): Promise<string> {
  const runId = randomUUID();
  const claims = { runId, origin: fixtures.origin, sub: 'e2e-user' };
  const extension = mintRelayToken(
    { ...claims, role: 'extension' },
    relay.secret,
  );
  const agent = mintRelayToken({ ...claims, role: 'agent' }, relay.secret);
  const registered = relay.logs.length;
  expect(
    await bridgeMessage(browser.sw, {
      type: 'startRun',
      relayUrl: relay.url,
      token: extension,
      tabId,
      allowedFrameOrigins: [],
    }),
  ).toEqual({ ok: true });
  await until(() =>
    relay.logs
      .slice(registered)
      .some((l) => l.event === 'registered' && l.runId === runId),
  );
  return cdpUrl(relay, agent);
}

/** The snapshot as one string; asserts it succeeded. */
async function snapshot(cdp: string, session: string): Promise<string> {
  const result = await agentBrowser(['snapshot', '-i'], cdp, session);
  expect(result.success).toBe(true);
  return JSON.stringify(result.data);
}

beforeAll(async () => {
  app = await startFakeApp();
  cleanups.push(() => app.stop());
  fixtures = await serveFixtures();
  cleanups.push(() => fixtures.stop());
  relay = await startRelay();
  cleanups.push(() => relay.stop());
  browser = await launchExtensionBrowser({
    extensionDir: buildExtension(app.origin),
  });
  cleanups.push(() => browser.stop());

  // Sign in the way a caseworker would: a first-party navigation.
  const appTab = await browser.context.newPage();
  await appTab.goto(`${app.origin}/login`);
  await appTab.close();

  pageUrl = `${fixtures.origin}/page2.html`;
  page = await browser.context.newPage();
  await page.goto(pageUrl);
  await page.evaluate(
    ({ label, value }) => {
      const box = document.createElement('div');
      box.innerHTML = `<label for="refcode">${label}</label> <input id="refcode" value="${value}">`;
      document.body.append(box);
    },
    { label: LABEL, value: VALUE },
  );
  tabId = (await browser.sw.evaluate(
    async (u) => (await chrome.tabs.query({ url: u }))[0]?.id,
    pageUrl,
  )) as number;
}, 90_000);

afterAll(async () => {
  try {
    await closeAgentSessions();
    if (browser) {
      await Promise.race([
        bridgeMessage(browser.sw, { type: 'stopRun' }).catch(() => {}),
        new Promise((resolve) => setTimeout(resolve, 3_000)),
      ]);
    }
  } finally {
    for (const cleanup of cleanups.reverse()) await cleanup();
  }
}, 60_000);

/** Every classify body so far: no value key, no typed value, label present. */
const expectBodiesValueFree = () => {
  expect(app.bodies.length).toBeGreaterThan(0);
  for (const body of app.bodies) {
    expect(body.includes(VALUE)).toBe(false);
    const { fields } = JSON.parse(body) as {
      fields: Array<Record<string, unknown>>;
    };
    expect(fields.length).toBeGreaterThan(0);
    for (const field of fields)
      expect(Object.keys(field)).not.toContain('value');
  }
  expect(app.bodies.some((b) => b.includes(LABEL))).toBe(true);
};

describe('worker Jev classification e2e', { timeout: 90_000 }, () => {
  it('the extension was built with the app origin', async () => {
    const hostPermissions = await browser.sw.evaluate(
      () => chrome.runtime.getManifest().host_permissions ?? [],
    );
    expect(hostPermissions).toEqual([`${app.origin}/*`]);
  });

  it('masks a field the app calls sensitive', async () => {
    app.mode = 'ok';
    const before = app.bodies.length;
    const cdp = await startRun();
    const snap = await snapshot(cdp, 'jev-ok');
    await closeAgentSessions();
    expect(app.bodies.length).toBeGreaterThan(before);
    expect(snap.includes(VALUE)).toBe(false);
    expect(snap.includes(MASK)).toBe(true);
    // Nothing else up the relay carried the value either.
    expect(relay.frames.some((f) => f.includes(VALUE))).toBe(false);
    expectBodiesValueFree();
    expect(app.refused).toBe(0);
    expect(await bridgeMessage(browser.sw, { type: 'stopRun' })).toEqual({
      ok: true,
    });
  });

  it('with the app failing the field stays visible, and a later snapshot after recovery masks it', async () => {
    app.mode = 'fail';
    const before = app.bodies.length;
    const cdp = await startRun();
    const failing = await snapshot(cdp, 'jev-fail');
    await closeAgentSessions();
    // Positive control: the hook did ask, and the app failed.
    expect(app.bodies.length).toBeGreaterThan(before);
    expect(failing.includes(VALUE)).toBe(true);

    app.mode = 'ok';
    const recovered = await snapshot(cdp, 'jev-recovered');
    await closeAgentSessions();
    expect(recovered.includes(VALUE)).toBe(false);
    expect(recovered.includes(MASK)).toBe(true);
    expectBodiesValueFree();
    expect(app.refused).toBe(0);
    expect(await bridgeMessage(browser.sw, { type: 'stopRun' })).toEqual({
      ok: true,
    });
  });
});
