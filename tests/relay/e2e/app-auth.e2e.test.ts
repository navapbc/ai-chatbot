import { execFileSync } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import type { Page, Worker } from 'playwright';
import { afterAll, describe, expect, it } from 'vitest';
import {
  EXTENSION_ID,
  type ExtensionBrowser,
  launchExtensionBrowser,
} from '../helpers/harness';
import {
  COOKIE_SHAPES,
  type CookieServer,
  startCookieServer,
} from './fixtures/cookie-server';

// Spike (plan 4b, Task 1): does the web app's session cookie reach the app
// when the extension calls it with fetch(url, { credentials: 'include' })?
//
// A local app sets one cookie per SameSite shape through a real first-party
// navigation, then the extension fetches `/whoami` from its service worker and
// from an extension page, with GET and POST. The server records which cookies
// arrived and the request's Sec-Fetch-Site. The table below asserts what
// Chromium did, so a future Chromium change fails here with the moved cell.
//
// Observed (Chromium 145.0.0.0, macOS, headless and headed identical; the
// worker and the extension page identical; GET and POST identical):
// - With `host_permissions` for the app origin, every cookie arrives:
//   SameSite=Lax, SameSite=Strict, SameSite=None; Secure and no SameSite.
//   Chromium sends the request with Sec-Fetch-Site: none, i.e. it does not
//   treat it as cross-site, and the response is readable without CORS
//   headers. GET carries no Origin header; POST carries
//   Origin: chrome-extension://<id>. Same over http://127.0.0.1.
// - Without host_permissions (control), the request is cross-site: only the
//   SameSite=None cookie is sent and the response is unreadable (CORS).
// So a Lax session cookie (NextAuth's default) is enough; P4b-a needs no
// SameSite=None change, provided the build sets APP_ORIGIN.
//
// Run with `pnpm test:relay e2e/app-auth.e2e.test.ts` (it builds its own
// extension copies into temp dirs). RELAY_E2E_HEADED=1 runs it headed.

const ROOT = path.resolve(__dirname, '../../..');
const HEADED = process.env.RELAY_E2E_HEADED === '1';

type Context = 'worker' | 'page';
type Method = 'GET' | 'POST';
interface Observation {
  /** Whether the extension could read the response. */
  readable: boolean;
  /** Cookies the server saw on the request, or null if it never arrived. */
  cookies: string[] | null;
  secFetchSite: string | null;
  origin: string | null;
}
type Table = Record<`${Context} ${Method}`, Observation>;

const cleanups: Array<() => Promise<void> | void> = [];
afterAll(async () => {
  for (const cleanup of cleanups.reverse()) await cleanup();
});

/** Builds the extension into a temp dir, with `appOrigin` as APP_ORIGIN. */
function buildExtension(appOrigin: string | null): string {
  const out = mkdtempSync(path.join(tmpdir(), 'app-auth-extension-'));
  cleanups.push(() => rmSync(out, { recursive: true, force: true }));
  // Drop any APP_ORIGIN / EXTENSION_RELEASE from the caller's shell.
  const {
    APP_ORIGIN: _origin,
    EXTENSION_RELEASE: _release,
    ...env
  } = process.env;
  if (appOrigin) env.APP_ORIGIN = appOrigin;
  execFileSync(
    path.join(ROOT, 'node_modules/.bin/tsx'),
    ['scripts/build-extension.ts', '--out', out],
    { cwd: ROOT, env, stdio: 'ignore' },
  );
  // A throwaway extension page, only in this temp build: the extension has no
  // pages yet, and the future side panel is one.
  writeFileSync(
    path.join(out, 'spike.html'),
    '<!doctype html><html lang="en"><title>spike</title></html>',
  );
  return out;
}

/** Runs in the worker or the page: one credentialed fetch to `/whoami`. */
const credentialedFetch = async ({
  url,
  method,
}: {
  url: string;
  method: string;
}) => {
  try {
    const res = await fetch(url, {
      method,
      credentials: 'include',
      ...(method === 'POST' ? { body: 'x' } : {}),
    });
    return { readable: true, status: res.status };
  } catch {
    return { readable: false, status: 0 };
  }
};

async function observe(
  server: CookieServer,
  sw: Worker,
  page: Page,
  label: string,
): Promise<Table> {
  const table = {} as Table;
  for (const context of ['worker', 'page'] as const) {
    for (const method of ['GET', 'POST'] as const) {
      const id = `${label}-${context}-${method}`;
      const arg = { url: `${server.origin}/whoami?id=${id}`, method };
      const result =
        context === 'worker'
          ? await sw.evaluate(credentialedFetch, arg)
          : await page.evaluate(credentialedFetch, arg);
      const seen = server.seen.find((s) => s.id === id && s.method === method);
      table[`${context} ${method}`] = {
        readable: result.readable,
        cookies: seen ? seen.cookies : null,
        secFetchSite: seen ? seen.secFetchSite : null,
        origin: seen ? seen.origin : null,
      };
    }
  }
  return table;
}

/** Starts a server and a browser with the built extension, signs in, measures. */
async function runScenario(
  protocol: 'https' | 'http',
  grantOrigin: boolean,
): Promise<{ table: Table; setCookies: string[]; userAgent: string }> {
  const server = await startCookieServer(protocol);
  cleanups.push(() => server.stop());
  const dir = buildExtension(grantOrigin ? server.origin : null);
  const browser: ExtensionBrowser = await launchExtensionBrowser({
    extensionDir: dir,
    headless: !HEADED,
  });
  cleanups.push(() => browser.stop());

  // The pinned manifest key must survive the build, or the id (and with it
  // RELAY_EXTENSION_ORIGIN) would change.
  expect(await browser.sw.evaluate(() => chrome.runtime.id)).toBe(EXTENSION_ID);
  const hostPermissions = await browser.sw.evaluate(
    () => chrome.runtime.getManifest().host_permissions ?? [],
  );
  expect(hostPermissions).toEqual(grantOrigin ? [`${server.origin}/*`] : []);

  // Sign in the way a caseworker would: a first-party navigation in a tab.
  const appTab = await browser.context.newPage();
  await appTab.goto(`${server.origin}/login`);
  // Not cookies(url): Playwright filters Secure cookies out for an http URL.
  const host = new URL(server.origin).hostname;
  const setCookies = (await browser.context.cookies())
    .filter((c) => c.domain === host && c.name in COOKIE_SHAPES)
    .map((c) => c.name)
    .sort();

  const [extensionPage] = await Promise.all([
    browser.context.waitForEvent('page'),
    browser.sw.evaluate(() =>
      chrome.tabs.create({ url: chrome.runtime.getURL('spike.html') }),
    ),
  ]);
  await extensionPage.waitForLoadState();
  expect(extensionPage.url()).toBe(
    `chrome-extension://${EXTENSION_ID}/spike.html`,
  );

  const userAgent = await browser.sw.evaluate(() => navigator.userAgent);
  const table = await observe(
    server,
    browser.sw,
    extensionPage,
    `${protocol}-${grantOrigin ? 'granted' : 'control'}`,
  );
  return { table, setCookies, userAgent };
}

const ALL_COOKIES = Object.keys(COOKIE_SHAPES).sort();
const EXTENSION_ORIGIN = `chrome-extension://${EXTENSION_ID}`;

/** With host permission: first-party-like, every cookie sent, readable. */
const GRANTED: Table = {
  'worker GET': {
    readable: true,
    cookies: ALL_COOKIES,
    secFetchSite: 'none',
    origin: null,
  },
  'worker POST': {
    readable: true,
    cookies: ALL_COOKIES,
    secFetchSite: 'none',
    origin: EXTENSION_ORIGIN,
  },
  'page GET': {
    readable: true,
    cookies: ALL_COOKIES,
    secFetchSite: 'none',
    origin: null,
  },
  'page POST': {
    readable: true,
    cookies: ALL_COOKIES,
    secFetchSite: 'none',
    origin: EXTENSION_ORIGIN,
  },
};

/** Without host permission: cross-site, only SameSite=None, unreadable. */
const crossSite: Observation = {
  readable: false,
  cookies: ['sid_none'],
  secFetchSite: 'cross-site',
  origin: EXTENSION_ORIGIN,
};
const CONTROL: Table = {
  'worker GET': crossSite,
  'worker POST': crossSite,
  'page GET': crossSite,
  'page POST': crossSite,
};

describe('app cookie auth from the extension (spike)', () => {
  it.each([
    ['https', true, GRANTED],
    ['https', false, CONTROL],
    ['http', true, GRANTED],
  ] as const)(
    '%s app, host permission granted: %s',
    async (protocol, grant, expected) => {
      const { table, setCookies, userAgent } = await runScenario(
        protocol,
        grant,
      );
      // Every shape must have been stored, or the table measures nothing.
      expect(setCookies).toEqual(ALL_COOKIES);
      expect(
        table,
        `Chromium's cookie behavior for extension fetches changed (${userAgent}); update the Observed note and plan ruling P4b-a`,
      ).toEqual(expected);
    },
    // Each case builds the extension (worker and side panel) and launches
    // Chromium: past vitest's 5 s default when the whole e2e suite runs.
    30_000,
  );
});
