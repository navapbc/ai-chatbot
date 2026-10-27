import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { createServer, type Server } from 'node:http';
import type { AddressInfo } from 'node:net';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { mintRelayToken } from '../../../lib/relay/token';
import { extractTokens } from '../../../scripts/extension-build-config';
import {
  EXTENSION_ID,
  type ExtensionBrowser,
  type Fixtures,
  launchExtensionBrowser,
  type Relay,
  serveFixtures,
  startRelay,
} from '../helpers/harness';

// The panel shell against a fake app: GET /api/relay/config answers 401 first
// (signed out), then 200. The signed-out screen shows with the primary color
// applied, and Try again leads to the Home placeholder. Opened as a tab: a
// real side panel cannot be driven headless. Run with
// `pnpm test:relay e2e/panel-shell.e2e.test.ts`; RELAY_E2E_HEADED=1 runs it
// headed.
//
// Starting a run: the fake app's POST /api/eve-chat answers one SSE chunk and
// keeps the stream open; POST /api/relay/token mints a real extension token
// for a real relay (ws://127.0.0.1) and the fixture origin, so the worker's
// startRun succeeds against a real tab and the panel reaches Running.

const ROOT = path.resolve(__dirname, '../../..');
const HEADED = process.env.RELAY_E2E_HEADED === '1';

let out: string;
let browser: ExtensionBrowser;
let app: Server;
let configCalls = 0;
let relay: Relay;
let fixtures: Fixtures;
/** Every start-flow request the fake app saw, in order: "METHOD path". */
const appCalls: string[] = [];
const eveChatBodies: unknown[] = [];
const eveChatOrigins: (string | undefined)[] = [];
/** The status the next POST /api/eve-chat answers with. */
let eveChatStatus = 200;

const readBody = (req: import('node:http').IncomingMessage) =>
  new Promise<string>((resolve) => {
    let text = '';
    req.on('data', (chunk) => {
      text += chunk;
    });
    req.on('end', () => resolve(text));
  });

beforeAll(async () => {
  relay = await startRelay();
  fixtures = await serveFixtures();
  app = createServer(async (req, res) => {
    if (req.method === 'POST' && req.url === '/api/eve-chat') {
      appCalls.push('POST /api/eve-chat');
      eveChatOrigins.push(req.headers.origin);
      eveChatBodies.push(JSON.parse(await readBody(req)));
      if (eveChatStatus !== 200) {
        res.writeHead(eveChatStatus, { 'content-type': 'application/json' });
        res.end('{"code":"rate_limit:chat","message":"limit"}');
        return;
      }
      // One chunk, then the turn "keeps going": the stream stays open.
      res.writeHead(200, { 'content-type': 'text/event-stream' });
      res.write('data: {"type":"start"}\n\n');
      return;
    }
    if (req.method === 'POST' && req.url === '/api/relay/token') {
      appCalls.push('POST /api/relay/token');
      const { runId } = JSON.parse(await readBody(req)) as { runId: string };
      const now = Math.floor(Date.now() / 1000);
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(
        JSON.stringify({
          extensionToken: mintRelayToken(
            {
              runId,
              role: 'extension',
              origin: fixtures.origin,
              sub: 'e2e-user',
            },
            relay.secret,
            { now },
          ),
          relayUrl: relay.url,
          expiresAt: now + 1800,
        }),
      );
      return;
    }
    if (
      req.method === 'POST' &&
      req.url?.startsWith('/api/relay/runs/') &&
      req.url.endsWith('/stop')
    ) {
      appCalls.push(`POST ${req.url}`);
      res.writeHead(404).end();
      return;
    }
    if (req.url === '/api/relay/config') {
      configCalls += 1;
      if (configCalls === 1) {
        res.writeHead(401, { 'content-type': 'application/json' });
        res.end('{"error":"unauthorized"}');
        return;
      }
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(
        JSON.stringify({
          sites: [
            {
              id: 'example',
              name: 'Example Benefits',
              url: 'https://example.gov/apply',
            },
          ],
          models: [{ id: 'claude-opus-5' }, { id: 'claude-haiku-4-5' }],
          defaultModel: 'claude-opus-5',
        }),
      );
      return;
    }
    if (req.url?.startsWith('/api/participants?q=')) {
      const q = new URL(req.url, 'http://x').searchParams.get('q');
      const body = (name: string) =>
        JSON.stringify({
          participants: [{ recordId: 'rec-1', name, county: 'Kern' }],
        });
      const send = (name: string) => {
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(body(name));
      };
      // A slow answer to the earlier query must not overwrite the later one.
      if (q === 'al') setTimeout(() => send('Stale Al'), 1200);
      else send('Alice <b>x</b>');
      return;
    }
    res.writeHead(404).end();
  });
  await new Promise<void>((resolve) => app.listen(0, '127.0.0.1', resolve));
  const origin = `http://127.0.0.1:${(app.address() as AddressInfo).port}`;

  out = mkdtempSync(path.join(tmpdir(), 'panel-shell-extension-'));
  const {
    APP_ORIGIN: _origin,
    EXTENSION_RELEASE: _release,
    EXTENSION_TEST_HOOKS: _hooks,
    ...env
  } = process.env;
  execFileSync(
    path.join(ROOT, 'node_modules/.bin/tsx'),
    ['scripts/build-extension.ts', '--out', out],
    { cwd: ROOT, env: { ...env, APP_ORIGIN: origin }, stdio: 'ignore' },
  );
  browser = await launchExtensionBrowser({
    extensionDir: out,
    headless: !HEADED,
  });
}, 60_000);

afterAll(async () => {
  await browser?.stop();
  app?.closeAllConnections();
  app?.close();
  await relay?.stop();
  await fixtures?.stop();
  if (out) rmSync(out, { recursive: true, force: true });
});

describe('the panel shell', () => {
  it('shows signed out, then Home after Try again', async () => {
    const page = await browser.context.newPage();
    const errors: string[] = [];
    page.on('console', (msg) => {
      // The fake app's deliberate 401 is logged by Chromium itself.
      if (msg.text().includes('status of 401')) return;
      if (msg.type() === 'error' || msg.type() === 'warning') {
        errors.push(`${msg.type()}: ${msg.text()}`);
      }
    });
    page.on('pageerror', (error) => errors.push(`pageerror: ${error.message}`));

    await page.goto(`chrome-extension://${EXTENSION_ID}/panel.html`);
    await page.getByText('You are signed out').waitFor();
    await page.getByRole('button', { name: 'Open the app' }).waitFor();
    const retry = page.getByRole('button', { name: 'Try again' });

    const primary = extractTokens(
      readFileSync(path.join(ROOT, 'app/globals.css'), 'utf8'),
    ).match(/--primary:\s*([^;]+);/)?.[1];
    expect(primary).toBeTruthy();
    // Let the browser convert the token, so no hand-rounded rgb is pinned.
    const expected = await page.evaluate((hsl) => {
      const probe = document.createElement('div');
      document.body.append(probe);
      probe.style.backgroundColor = `hsl(${hsl})`;
      const color = getComputedStyle(probe).backgroundColor;
      probe.remove();
      return color;
    }, primary as string);
    const actual = await page
      .getByRole('button', { name: 'Open the app' })
      .evaluate((el) => getComputedStyle(el).backgroundColor);
    expect(expected).not.toBe('rgba(0, 0, 0, 0)');
    expect(actual).toBe(expected);

    await page.evaluate(() =>
      chrome.storage.local.set({
        recentRuns: [
          {
            runId: '00000000-0000-4000-8000-000000000001',
            siteName: 'Seeded Site',
            startedAt: Date.now() - 5 * 60_000,
          },
          {
            runId: '00000000-0000-4000-8000-000000000002',
            siteName: 'Planted Site',
            startedAt: Date.now() - 60_000,
            participant: 'Jane Doe',
          },
        ],
      }),
    );
    await retry.click();
    await page.getByRole('heading', { name: 'Home' }).waitFor();
    await page.getByRole('button', { name: 'Example Benefits' }).waitFor();
    await page.getByRole('button', { name: 'Use this tab' }).waitFor();
    await page.getByText('Seeded Site, 5 minutes ago').waitFor();
    expect(await page.getByText('Planted Site').count()).toBe(0);
    expect(configCalls).toBe(2);

    // No console or CSP error beyond that 401.
    expect(errors).toEqual([]);
    await page.close();
  }, 30_000);

  it('searches participants, ignores a stale answer, and chooses one', async () => {
    const page = await browser.context.newPage();
    // The panel is a tab here, so "Use this tab" would see the panel itself;
    // stub the tab query to return an https page instead.
    await page.addInitScript(() => {
      chrome.tabs.query = (async () => [
        { id: 7, url: 'https://example.gov/apply' },
      ]) as unknown as typeof chrome.tabs.query;
    });
    await page.goto(`chrome-extension://${EXTENSION_ID}/panel.html`);
    await page.getByRole('button', { name: 'Use this tab' }).click();
    await page.getByText('Site: example.gov').waitFor();

    const box = page.getByRole('searchbox', { name: 'Search participants' });
    await box.fill('al');
    await page.waitForTimeout(400);
    await box.fill('alice');
    await page.getByText('Alice <b>x</b>').waitFor();
    await page.waitForTimeout(1500);
    expect(await page.getByText('Stale Al').count()).toBe(0);
    expect(await page.locator('main b').count()).toBe(0);
    await page.getByText('1 result').waitFor();

    await page.getByRole('button', { name: /Alice/ }).click();
    await page.getByRole('heading', { name: 'Ready' }).waitFor();
    await page.getByText('Alice <b>x</b>').waitFor();
    await page.getByText('https://example.gov').waitFor();
    await page.getByText('The agent works only on this site.').waitFor();
    await page.getByText('It asks you for anything that is missing.').waitFor();
    await page
      .getByText('It stops before Submit — you submit the form yourself.')
      .waitFor();
    const model = page.getByRole('combobox', { name: 'Model' });
    expect(await model.inputValue()).toBe('claude-opus-5');
    await model.selectOption('claude-haiku-4-5');
    expect(await model.inputValue()).toBe('claude-haiku-4-5');

    // Back returns to the search screen.
    await page.getByRole('button', { name: 'Back' }).click();
    await page
      .getByRole('searchbox', { name: 'Search participants' })
      .waitFor();
    await page
      .getByRole('searchbox', { name: 'Search participants' })
      .fill('alice');
    await page.getByRole('button', { name: /Alice/ }).click();
    await page.getByRole('heading', { name: 'Ready' }).waitFor();

    // The stubbed tab 7 does not exist: the starter refuses before any
    // request, and Ready shows why.
    const before = appCalls.length;
    await page.getByRole('button', { name: 'Start' }).click();
    await page.getByText('The tab was closed').waitFor();
    await page.getByRole('heading', { name: 'Ready' }).waitFor();
    expect(appCalls.length).toBe(before);
    await page.close();
  }, 30_000);

  const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;

  type Status = {
    ok: boolean;
    active: boolean;
    runId?: string;
    attached?: boolean;
  };

  /** A panel on Ready for a real tab at the fixture origin. */
  const readyOnRealTab = async () => {
    const site = await browser.context.newPage();
    await site.goto(fixtures.formUrl);
    const page = await browser.context.newPage();
    // The panel is a tab here: "Use this tab" must see the site's real tab
    // (the worker checks its real URL), not the panel itself.
    await page.addInitScript((origin: string) => {
      const query = chrome.tabs.query.bind(chrome.tabs);
      chrome.tabs.query = (async () =>
        (await query({})).filter((t) =>
          t.url?.startsWith(`${origin}/`),
        )) as unknown as typeof chrome.tabs.query;
    }, fixtures.origin);
    await page.goto(`chrome-extension://${EXTENSION_ID}/panel.html`);
    await page.getByRole('button', { name: 'Use this tab' }).click();
    await page
      .getByRole('searchbox', { name: 'Search participants' })
      .fill('alice');
    await page.getByRole('button', { name: /Alice/ }).click();
    await page.getByRole('heading', { name: 'Ready' }).waitFor();
    return { site, page };
  };

  const runStatus = (page: import('playwright').Page) =>
    page.evaluate(
      () =>
        chrome.runtime.sendMessage({ type: 'getRunStatus' }) as Promise<Status>,
    );

  const stopWorkerRun = (page: import('playwright').Page) =>
    page.evaluate(() => chrome.runtime.sendMessage({ type: 'stopRun' }));

  it('starts a run: eve-chat, then the token, then the worker; shows Running', async () => {
    const { site, page } = await readyOnRealTab();
    appCalls.length = 0;
    eveChatBodies.length = 0;
    eveChatOrigins.length = 0;
    eveChatStatus = 200;

    await page.getByRole('button', { name: 'Start' }).click();
    await page.getByRole('heading', { name: 'Running' }).waitFor();

    expect(appCalls).toEqual(['POST /api/eve-chat', 'POST /api/relay/token']);
    expect(eveChatOrigins).toEqual([`chrome-extension://${EXTENSION_ID}`]);
    const body = eveChatBodies[0] as { id: string; relay: unknown };
    expect(body.id).toMatch(UUID);
    expect(body).toEqual({
      id: body.id,
      relay: { origin: fixtures.origin, participant: { recordId: 'rec-1' } },
    });

    // The worker holds the run and registered it with the real relay (the
    // debugger attaches only once an agent asks, so `attached` stays false).
    const status = await runStatus(page);
    expect({ active: status.active, runId: status.runId }).toEqual({
      active: true,
      runId: body.id,
    });
    await expect
      .poll(
        () =>
          relay.logs.some(
            (l) => l.event === 'registered' && l.runId === body.id,
          ),
        { timeout: 10_000 },
      )
      .toBe(true);
    const recents = await page.evaluate(
      async () => (await chrome.storage.local.get('recentRuns')).recentRuns,
    );
    expect(recents).toEqual(
      expect.arrayContaining([
        {
          runId: body.id,
          siteName: 'localhost',
          startedAt: expect.any(Number),
        },
      ]),
    );

    await stopWorkerRun(page);
    await page.close();
    await site.close();
  }, 30_000);

  it('shows the run-limit message on 429, then Start works again', async () => {
    const { site, page } = await readyOnRealTab();
    appCalls.length = 0;
    eveChatBodies.length = 0;
    eveChatStatus = 429;

    await page.getByRole('button', { name: 'Start' }).click();
    await page.getByText('Too many runs right now').waitFor();
    await page.getByRole('heading', { name: 'Ready' }).waitFor();
    const refusedId = (eveChatBodies[0] as { id: string }).id;
    // A run may exist once eve-chat was asked, so the server stop is called.
    await expect.poll(() => appCalls.length).toBe(2);
    expect(appCalls).toEqual([
      'POST /api/eve-chat',
      `POST /api/relay/runs/${refusedId}/stop`,
    ]);
    expect((await runStatus(page)).active).toBe(false);

    eveChatStatus = 200;
    const start = page.getByRole('button', { name: 'Start' });
    expect(await start.isDisabled()).toBe(false);
    await start.click();
    await page.getByRole('heading', { name: 'Running' }).waitFor();
    expect(appCalls.slice(2)).toEqual([
      'POST /api/eve-chat',
      'POST /api/relay/token',
    ]);
    expect((eveChatBodies[1] as { id: string }).id).not.toBe(refusedId);

    await stopWorkerRun(page);
    await page.close();
    await site.close();
  }, 30_000);

  it('saves details for 8 hours, reuses them, expires them and clears them', async () => {
    const seen: string[] = [];
    const watch = (page: import('playwright').Page) => {
      page.on('console', (msg) => seen.push(msg.text()));
      page.on('framenavigated', (frame) => seen.push(frame.url()));
    };
    const stubTab = (page: import('playwright').Page) =>
      page.addInitScript(() => {
        chrome.tabs.query = (async () => [
          { id: 7, url: 'https://example.gov/apply' },
        ]) as unknown as typeof chrome.tabs.query;
      });
    const toDetails = async (page: import('playwright').Page) => {
      await page.goto(`chrome-extension://${EXTENSION_ID}/panel.html`);
      await page.getByRole('button', { name: 'Use this tab' }).click();
      await page.getByRole('button', { name: 'Enter details instead' }).click();
      await page.getByRole('heading', { name: 'Enter details' }).waitFor();
    };

    const page = await browser.context.newPage();
    watch(page);
    await stubTab(page);
    await toDetails(page);
    expect(
      await page.getByRole('button', { name: 'Reuse saved details' }).count(),
    ).toBe(0);
    expect(
      await page.getByRole('button', { name: 'Continue' }).isDisabled(),
    ).toBe(true);
    await page.getByLabel('First name').fill('Zelda');
    await page.getByLabel('City').fill('Fresno');
    await page
      .getByRole('checkbox', { name: 'Save these for 8 hours' })
      .click();
    await page.getByRole('button', { name: 'Continue' }).click();
    await page.getByRole('heading', { name: 'Ready' }).waitFor();
    await page.close();

    const again = await browser.context.newPage();
    watch(again);
    await stubTab(again);
    await toDetails(again);
    await again.getByRole('button', { name: 'Reuse saved details' }).click();
    await expect
      .poll(() => again.getByLabel('First name').inputValue())
      .toBe('Zelda');
    await expect
      .poll(() => again.getByLabel('City').inputValue())
      .toBe('Fresno');
    await again.close();

    const late = await browser.context.newPage();
    watch(late);
    await stubTab(late);
    await late.addInitScript(() => {
      const real = Date.now;
      Date.now = () => real() + 9 * 3_600_000;
    });
    await toDetails(late);
    expect(
      await late.getByRole('button', { name: 'Reuse saved details' }).count(),
    ).toBe(0);
    expect(
      await late.evaluate(
        async () =>
          (await chrome.storage.local.get('savedDetails')).savedDetails,
      ),
    ).toBeUndefined();
    await late.close();

    const clear = await browser.context.newPage();
    watch(clear);
    await stubTab(clear);
    await toDetails(clear);
    await clear.getByLabel('City').fill('Reno');
    await clear
      .getByRole('checkbox', { name: 'Save these for 8 hours' })
      .click();
    await clear.getByRole('button', { name: 'Continue' }).click();
    await clear.getByRole('heading', { name: 'Ready' }).waitFor();
    await clear.close();
    const last = await browser.context.newPage();
    watch(last);
    await stubTab(last);
    await toDetails(last);
    await last.getByRole('button', { name: 'Clear saved details' }).click();
    await last.getByText('Saved details cleared.').waitFor();
    expect(
      await last.getByRole('button', { name: 'Reuse saved details' }).count(),
    ).toBe(0);
    expect(
      await last.evaluate(
        async () =>
          (await chrome.storage.local.get('savedDetails')).savedDetails,
      ),
    ).toBeUndefined();
    await last.close();

    for (const line of seen) {
      expect(line).not.toMatch(/Zelda|Fresno|Reno/);
    }
  }, 60_000);
});
