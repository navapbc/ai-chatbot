import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, statSync } from 'node:fs';
import { createServer, type Server } from 'node:http';
import type { AddressInfo } from 'node:net';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { afterAll, beforeAll, beforeEach, describe, expect, it } from 'vitest';
import { mintRelayToken } from '../../../lib/relay/token';
import { extractTokens } from '../../../scripts/extension-build-config';
import {
  cdpUrl,
  closeAgentSessions,
  EXTENSION_ID,
  type ExtensionBrowser,
  agentBrowser,
  type Fixtures,
  idOfNode,
  launchExtensionBrowser,
  recordOverlay,
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
//
// Typing a sensitive value (plan 4c-3): the panel is driven as a tab, as
// above, but the worker, relay and page are real and the agent is
// agent-browser through the relay. Fill it in goes to the real worker; the
// secret must appear nowhere the fake app, the console, storage, the panel or
// any relay frame can show it, including after a same-run token refresh.
//
// Review and submit (plan 4c-4): the same way, panel as a tab. The agent
// is agent-browser through the relay, the page is real and its own submit
// button posts to the fixture's confirmation pages; the three scenarios are
// the unfilled-field highlight with a real submit, an echoed SSN, and a Next
// step answered No.
//
// Reliability (plan 4d): one scenario drives the same way (panel as a tab,
// real worker, real relay, real page, fake app) through a panel closed during
// a hung Starting (the next open stops its run once), typing after a token
// refresh with the agent silent, an allowed extra origin surviving a refresh,
// and a later open that must not stop the run the worker holds.

const ROOT = path.resolve(__dirname, '../../..');
const HEADED = process.env.RELAY_E2E_HEADED === '1';
/**
 * The bridge's origin hold in this file's build (the shipped bridge holds
 * 60 s): long enough for the panel's 2 s worker poll to show the card and a
 * test to press Allow, short enough that a run nobody allows ends in seconds.
 * Only a test-hooks build takes it (the hooks change nothing the panel sees).
 */
const PANEL_HOLD_MS = 8_000;

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
/** Runs eve-chat created: the fake server rows the panel reads and stops. */
const runs = new Map<
  string,
  { ended: boolean; phase?: string; pending?: unknown[]; progress?: unknown }
>();
/** Answer turns (`inputResponses` bodies) POSTed to /api/eve-chat. */
const answerBodies: unknown[] = [];
/** Every request the fake app saw: method, URL, headers and body text. */
const appRequests: string[] = [];
const answerOrigins: (string | undefined)[] = [];
/**
 * When set, POST /api/eve-chat creates the run row and then never answers
 * (a first turn that hangs while the panel is on Starting).
 */
let eveChatHang = false;
/** When set, the fake app answers POST /stop with a 500 (row still ends). */
let stopFails = false;
/** The refusal reason the next answer turn gets as a 400, or null for 200. */
let answerRefusal: string | null = null;
/** GET /api/relay/runs/:id polls, kept apart from `appCalls`. */
const pollCalls: string[] = [];

/** Canned progress the fake app reports for every run. */
const CANNED_PROGRESS = {
  sections: [
    { name: 'Applicant', status: 'done', filled: 4, total: 4 },
    { name: '<b>Household</b>', status: 'filling', filled: 2, total: 6 },
    { name: 'Income', status: 'waiting', filled: 0 },
  ],
  current: '<b>Household</b>',
  actions: ['Typed the household size', 'Chose the county'],
};

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
    const raw = await readBody(req);
    appRequests.push(JSON.stringify([req.method, req.url, req.headers, raw]));
    if (req.method === 'POST' && req.url === '/api/eve-chat') {
      const parsed = JSON.parse(raw) as { inputResponses?: unknown };
      if (parsed.inputResponses !== undefined) {
        // An answer turn: the run row is left as it is.
        answerBodies.push(parsed);
        answerOrigins.push(req.headers.origin);
        if (answerRefusal) {
          res.writeHead(400, { 'content-type': 'application/json' });
          res.end(
            JSON.stringify({
              code: 'bad_request:api',
              message: "The request couldn't be processed.",
              cause: answerRefusal,
            }),
          );
          return;
        }
        res.writeHead(200, { 'content-type': 'text/event-stream' });
        res.write('data: {"type":"start"}\n\n');
        return;
      }
      appCalls.push('POST /api/eve-chat');
      eveChatOrigins.push(req.headers.origin);
      eveChatBodies.push(parsed);
      if (eveChatHang) {
        runs.set((eveChatBodies.at(-1) as { id: string }).id, { ended: false });
        return;
      }
      if (eveChatStatus !== 200) {
        res.writeHead(eveChatStatus, { 'content-type': 'application/json' });
        res.end('{"code":"rate_limit:chat","message":"limit"}');
        return;
      }
      // One chunk, then the turn "keeps going": the stream stays open.
      runs.set((eveChatBodies.at(-1) as { id: string }).id, { ended: false });
      res.writeHead(200, { 'content-type': 'text/event-stream' });
      res.write('data: {"type":"start"}\n\n');
      return;
    }
    if (req.method === 'POST' && req.url === '/api/relay/token') {
      appCalls.push('POST /api/relay/token');
      const { runId } = JSON.parse(raw) as { runId: string };
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
      const row = runs.get(req.url.split('/')[4] ?? '');
      if (!row) {
        res.writeHead(404).end();
        return;
      }
      const alreadyEnded = row.ended;
      row.ended = true;
      if (stopFails) {
        res.writeHead(500).end();
        return;
      }
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ ended: true, alreadyEnded }));
      return;
    }
    if (req.method === 'GET' && req.url?.startsWith('/api/relay/runs/')) {
      pollCalls.push(`GET ${req.url}`);
      const runId = req.url.split('/')[4] ?? '';
      const row = runs.get(runId);
      if (!row) {
        res.writeHead(404).end();
        return;
      }
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(
        JSON.stringify({
          runId,
          origin: fixtures.origin,
          ended: row.ended,
          endedAt: row.ended ? new Date().toISOString() : null,
          progress: row.progress ?? CANNED_PROGRESS,
          pending: row.ended ? [] : (row.pending ?? []),
          phase: row.ended ? 'ended' : (row.phase ?? 'working'),
        }),
      );
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
    if (req.method === 'POST' && req.url === '/api/participants') {
      const q = (JSON.parse(raw || '{}') as { q?: string }).q ?? null;
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
    {
      cwd: ROOT,
      env: {
        ...env,
        APP_ORIGIN: origin,
        EXTENSION_TEST_HOOKS: '1',
        EXTENSION_TEST_ORIGIN_HOLD_MS: String(PANEL_HOLD_MS),
      },
      stdio: 'ignore',
    },
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

// Every panel open sweeps unended recents older than 90 s (orphan-sweep.ts).
// Runs earlier tests left unended would be stopped by a later test's panel,
// adding stop calls to its exact `appCalls` and, for a run the fake app has
// no row for, a 404 console error. Each test starts with no recents.
beforeEach(async () => {
  const sw = browser.context.serviceWorkers()[0] ?? browser.sw;
  await sw.evaluate(() => chrome.storage.local.remove('recentRuns'));
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

    // The seeded recent has no end reason and the worker holds nothing, so
    // the open-time sweep stops it: give the fake app its row (else the stop
    // is a 404, which Chromium logs as a console error).
    const seededId = '00000000-0000-4000-8000-000000000001';
    runs.set(seededId, { ended: false });
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
    await expect
      .poll(async () =>
        page.evaluate(
          async () =>
            (
              (await chrome.storage.local.get('recentRuns')).recentRuns as {
                endedReason?: string;
              }[]
            )?.[0]?.endedReason,
        ),
      )
      .toBe('swept');
    expect(appCalls).toContain(`POST /api/relay/runs/${seededId}/stop`);
    expect(runs.get(seededId)?.ended).toBe(true);

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
    // The term rode a POST body: no request URL the app saw contains it.
    const searches = appRequests
      .map((r) => JSON.parse(r) as [string, string, unknown, string])
      .filter(([, url]) => url.startsWith('/api/participants'));
    expect(searches.length).toBeGreaterThan(0);
    for (const [method, url, , body] of searches) {
      expect(method).toBe('POST');
      expect(url).toBe('/api/participants');
      expect(url).not.toMatch(/al/i);
    }
    expect(searches.some(([, , , body]) => body.includes('alice'))).toBe(true);

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
    // Counts the panel's worker stop messages (the worker has no log of them).
    await page.addInitScript(() => {
      const w = window as unknown as { __workerStops: number };
      w.__workerStops = 0;
      const send = chrome.runtime.sendMessage.bind(chrome.runtime);
      chrome.runtime.sendMessage = ((...args: unknown[]) => {
        if ((args[0] as { type?: string })?.type === 'stopRun') {
          w.__workerStops += 1;
        }
        return (send as (...a: unknown[]) => unknown)(...args);
      }) as typeof chrome.runtime.sendMessage;
    });
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

    try {
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
    } finally {
      await stopWorkerRun(page);
      await page.close();
      await site.close();
    }
  }, 30_000);

  it('shows the run-limit message on 429, then Start works again', async () => {
    const { site, page } = await readyOnRealTab();
    appCalls.length = 0;
    eveChatBodies.length = 0;
    eveChatStatus = 429;

    try {
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
    } finally {
      eveChatStatus = 200;
      await stopWorkerRun(page);
      await page.close();
      await site.close();
    }
  }, 30_000);

  /** Starts a run from a fresh panel and waits until the relay has it. */
  const startOnRealTab = async () => {
    const { site, page } = await readyOnRealTab();
    appCalls.length = 0;
    eveChatBodies.length = 0;
    eveChatStatus = 200;
    await page.getByRole('button', { name: 'Start' }).click();
    await page.getByRole('heading', { name: 'Running' }).waitFor();
    const runId = (eveChatBodies[0] as { id: string }).id;
    await expect
      .poll(
        () =>
          relay.logs.some((l) => l.event === 'registered' && l.runId === runId),
        { timeout: 10_000 },
      )
      .toBe(true);
    return { site, page, runId };
  };

  const stopCalls = (runId: string) =>
    appCalls.filter((c) => c === `POST /api/relay/runs/${runId}/stop`).length;

  /** Posts a form in the site tab to the `v` confirmation page. */
  const submitIn = async (site: import('playwright').Page, v: string) => {
    await site.evaluate((page) => {
      const form = document.createElement('form');
      form.method = 'post';
      form.action = `/submitted?v=${page}`;
      document.body.append(form);
      form.requestSubmit();
    }, v);
    await site.waitForURL(`${fixtures.origin}/submitted?v=${v}`);
  };
  const armed = (page: import('playwright').Page) =>
    expect
      .poll(
        async () =>
          ((await runStatus(page)) as Status & { armed?: boolean }).armed,
        { timeout: 10_000 },
      )
      .toBe(true);
  const workerStops = (page: import('playwright').Page) =>
    page.evaluate(
      () => (window as unknown as { __workerStops: number }).__workerStops,
    );

  it('Running shows the site, a timer and Stop; Stop ends the run on the server and the worker', async () => {
    const { site, page, runId } = await startOnRealTab();
    try {
      await page.getByText('Site: localhost').waitFor();
      // The participant chosen on Ready is not shown while the run is live.
      expect(await page.getByText('Alice', { exact: false }).count()).toBe(0);
      const elapsed = page.getByText(/^Elapsed: \d+:\d\d$/);
      const first = await elapsed.textContent();
      await expect
        .poll(() => elapsed.textContent(), { timeout: 5_000 })
        .not.toBe(first);
      // Canned progress: bar, collapsed done section, open active section.
      await page.getByText('6 fields so far').waitFor();
      await page.getByText('Section 2 of 3').waitFor();
      const sectionList = page.getByRole('list', { name: 'Form sections' });
      await sectionList.getByText('Applicant').waitFor();
      await sectionList.getByText('4 of 4').waitFor();
      await sectionList.getByText('Typed the household size').waitFor();
      await sectionList.getByText('Chose the county').waitFor();
      // Markup in a section name renders as text, not as an element.
      await sectionList.getByText('<b>Household</b>').waitFor();
      expect(await sectionList.locator('b').count()).toBe(0);
      expect(await sectionList.locator('[aria-current="step"]').count()).toBe(
        1,
      );
      // No way off a live run except Stop.
      expect(
        await page.getByRole('button', { name: 'Back to Home' }).count(),
      ).toBe(0);
      // The panel polls the server's run state.
      await expect
        .poll(() => pollCalls.includes(`GET /api/relay/runs/${runId}`), {
          timeout: 10_000,
        })
        .toBe(true);

      await page.getByRole('button', { name: 'Stop' }).click();
      await page.getByRole('heading', { name: 'Run ended' }).waitFor();
      await page.getByText('You stopped this run.').waitFor();
      expect(stopCalls(runId)).toBe(1);
      expect(runs.get(runId)?.ended).toBe(true);
      expect((await runStatus(page)).active).toBe(false);
      const recents = await page.evaluate(
        async () => (await chrome.storage.local.get('recentRuns')).recentRuns,
      );
      expect(recents).toEqual(
        expect.arrayContaining([
          expect.objectContaining({ runId, endedReason: 'user' }),
        ]),
      );

      await page.getByRole('button', { name: 'Back to Home' }).click();
      await page.getByRole('heading', { name: 'Home' }).waitFor();
    } finally {
      await stopWorkerRun(page);
      await page.close();
      await site.close();
    }
  }, 30_000);

  it('Review: shows only while the phase is review; I submitted it stops both sides and confirms; Start another and Back to Home', async () => {
    const { site, page, runId } = await startOnRealTab();
    const row = runs.get(runId);
    if (!row) throw new Error('no run row');
    try {
      await page.getByText('6 fields so far').waitFor();
      const heading = page.getByRole('heading', {
        name: 'Ready for your review',
      });
      // Working: no review panel.
      expect(await heading.count()).toBe(0);
      row.progress = {
        sections: [
          { name: 'Applicant', status: 'done', filled: 4, total: 4 },
          { name: 'Income', status: 'needs_you', filled: 1, total: 3 },
        ],
        actions: ['Typed the income'],
      };
      row.phase = 'review';
      await heading.waitFor({ timeout: 8_000 });
      await page
        .getByText('The agent has stopped. It has not submitted the form.')
        .waitFor();
      const left = page.getByRole('region', { name: 'Ready for your review' });
      await left.getByText('Income', { exact: true }).waitFor();
      await left.getByText('1 of 3').waitFor();
      expect(await left.getByText('Applicant', { exact: true }).count()).toBe(
        0,
      );
      // The action log is collapsed by default and shows the agent's steps.
      const details = left.locator('details');
      expect(
        await details.evaluate((d) => (d as HTMLDetailsElement).open),
      ).toBe(false);
      await details.locator('summary').click();
      await details.getByText('Typed the income').waitFor();
      await page
        .getByText('Check the form on the page, then submit it there yourself.')
        .waitFor();
      // Show me the unfilled field: one worker call per press, the answer is
      // fixed copy in a status line. The test page has no required field.
      await page.evaluate(() => {
        const w = window as unknown as { __showCalls: number };
        w.__showCalls = 0;
        const original = chrome.runtime.sendMessage.bind(chrome.runtime);
        // biome-ignore lint/suspicious/noExplicitAny: test spy
        (chrome.runtime as any).sendMessage = (m: any, ...rest: any[]) => {
          if (m?.type === 'showUnfilledField') w.__showCalls += 1;
          // biome-ignore lint/suspicious/noExplicitAny: test spy
          return (original as any)(m, ...rest);
        };
      });
      const show = left.getByRole('button', {
        name: 'Show me the unfilled field',
      });
      // The unfilled-field line (the submission line is named).
      const status = left.locator('[role="status"]:not([aria-label])');
      expect(await status.textContent()).toBe('');
      await show.dblclick();
      await expect
        .poll(async () => (await status.textContent()) ?? '', {
          timeout: 10_000,
        })
        .toMatch(
          /^(No empty required fields found\. Check the page before you submit\.|Showing the next field|The extension|The run is paused|Could not check)/,
        );
      expect(
        await page.evaluate(
          () => (window as unknown as { __showCalls: number }).__showCalls,
        ),
      ).toBe(1);
      expect(await show.isDisabled()).toBe(false);

      // Back to working: the panel goes away.
      row.phase = 'working';
      await expect.poll(() => heading.count(), { timeout: 8_000 }).toBe(0);
      row.phase = 'review';
      await heading.waitFor({ timeout: 8_000 });
      // The status line did not survive leaving review.
      expect(
        await page.getByRole('status').filter({ hasText: /\S/ }).count(),
      ).toBe(0);

      await page.getByRole('button', { name: 'I submitted it' }).click();
      await page.getByRole('heading', { name: 'Submitted' }).waitFor();
      await page
        .getByText('You confirmed the form was submitted on localhost.')
        .waitFor();
      expect(stopCalls(runId)).toBe(1);
      expect(
        await page.evaluate(
          () => (window as unknown as { __workerStops: number }).__workerStops,
        ),
      ).toBe(1);
      expect(row.ended).toBe(true);
      expect((await runStatus(page)).active).toBe(false);
      // The stop worked: no stop notice. Focus is on the Confirmed heading.
      expect(await page.getByText(/could not reach/).count()).toBe(0);
      expect(
        await page.evaluate(() => document.activeElement?.textContent),
      ).toBe('Submitted');
      const recents = await page.evaluate(
        async () => (await chrome.storage.local.get('recentRuns')).recentRuns,
      );
      expect(recents).toEqual(
        expect.arrayContaining([
          expect.objectContaining({ runId, endedReason: 'submitted' }),
        ]),
      );

      // Another form for the same person on the same site.
      await page
        .getByRole('button', { name: 'Start another form for this person' })
        .click();
      await page.getByRole('heading', { name: 'Ready' }).waitFor();
      await page.getByText('Alice', { exact: false }).first().waitFor();
      await page.getByText('localhost', { exact: false }).first().waitFor();
      await page.getByRole('button', { name: 'Start' }).waitFor();
    } finally {
      await stopWorkerRun(page);
      await page.close();
      await site.close();
    }
  }, 60_000);

  it('Confirmed: a failing server stop still confirms, with the stop notice; Back to Home clears the person and the site', async () => {
    const { site, page, runId } = await startOnRealTab();
    stopFails = true;
    const row = runs.get(runId);
    if (!row) throw new Error('no run row');
    try {
      row.phase = 'review';
      await page
        .getByRole('button', { name: 'I submitted it' })
        .click({ timeout: 10_000 });
      await page.getByRole('heading', { name: 'Submitted' }).waitFor();
      await page
        .getByText(
          'We could not reach the app or the extension to confirm the stop. The agent may still be working: check the page and use Stop if it is.',
        )
        .waitFor();
      expect(await page.getByRole('button', { name: 'Stop' }).count()).toBe(0);
      await page.getByRole('button', { name: 'Back to Home' }).click();
      await page.getByRole('heading', { name: 'Home' }).waitFor();
      expect(await page.getByText('Alice', { exact: false }).count()).toBe(0);
      expect(
        await page.getByRole('button', { name: 'Use this tab' }).count(),
      ).toBe(1);
    } finally {
      stopFails = false;
      await stopWorkerRun(page);
      await page.close();
      await site.close();
    }
  }, 40_000);

  it('Review: a form POST showing a confirmation number confirms with it; another label asks, No keeps the run, Yes confirms', async () => {
    // A confirmation number: Confirmed at once, the number as text.
    {
      const { site, page, runId } = await startOnRealTab();
      const row = runs.get(runId);
      if (!row) throw new Error('no run row');
      try {
        row.phase = 'review';
        await page
          .getByRole('heading', { name: 'Ready for your review' })
          .waitFor({ timeout: 8_000 });
        await armed(page);
        await submitIn(site, 'alnum');
        // Asked at once, the number filled in when the worker read it; the
        // run is not stopped until the caseworker says so.
        const prompt = page.getByRole('region', {
          name: 'Did you submit the form?',
        });
        await prompt.waitFor({ timeout: 10_000 });
        await prompt.getByText('AB-1234', { exact: true }).waitFor();
        await prompt
          .getByText('Confirmation number', { exact: true })
          .waitFor();
        expect(stopCalls(runId)).toBe(0);
        expect(await workerStops(page)).toBe(0);
        expect((await runStatus(page)).active).toBe(true);
        await prompt.getByRole('button', { name: 'Yes, I did' }).click();
        await page.getByRole('heading', { name: 'Submitted' }).waitFor();
        await page
          .getByText('You confirmed the form was submitted on localhost.')
          .waitFor();
        await page.getByText('Confirmation number', { exact: true }).waitFor();
        await page.getByText('AB-1234', { exact: true }).waitFor();
        expect(stopCalls(runId)).toBe(1);
        expect(await workerStops(page)).toBe(1);
        expect(row.ended).toBe(true);
        // The number went nowhere but the screen.
        expect(JSON.stringify(appRequests)).not.toContain('AB-1234');
        const stored = await page.evaluate(async () =>
          JSON.stringify([
            await chrome.storage.local.get(null),
            await chrome.storage.session.get(null),
          ]),
        );
        expect(stored).not.toContain('AB-1234');
        expect(stored).toContain(runId);
        await page
          .getByRole('button', { name: 'Start another form for this person' })
          .click();
        await page.getByRole('heading', { name: 'Ready' }).waitFor();
        expect(await page.getByText('AB-1234').count()).toBe(0);
      } finally {
        await stopWorkerRun(page);
        await page.close();
        await site.close();
      }
    }

    // Another label: asked; No keeps the run going and the next submission
    // is seen again; Yes confirms with the number shown on the page.
    {
      const { site, page, runId } = await startOnRealTab();
      const row = runs.get(runId);
      if (!row) throw new Error('no run row');
      try {
        row.phase = 'review';
        const review = page.getByRole('region', {
          name: 'Ready for your review',
        });
        await review.waitFor({ timeout: 8_000 });
        await armed(page);
        await submitIn(site, 'other');
        const prompt = page.getByRole('region', {
          name: 'Did you submit the form?',
        });
        await prompt.waitFor({ timeout: 10_000 });
        await prompt.getByText('20240017').waitFor();
        await prompt
          .getByText('Number shown on the page', { exact: true })
          .waitFor();
        await prompt.getByRole('button', { name: 'No, not yet' }).click();
        await expect.poll(() => prompt.count()).toBe(0);
        // Still running, still in review.
        await review.waitFor();
        expect(stopCalls(runId)).toBe(0);
        expect((await runStatus(page)).active).toBe(true);
        await site.goto(fixtures.formUrl);
        await submitIn(site, 'other');
        // The prompt opens at once; the number fills in when the worker's
        // read gives it (Yes carries what is shown at that moment).
        await prompt.waitFor({ timeout: 10_000 });
        await prompt.getByText('20240017', { exact: true }).waitFor();
        await prompt.getByRole('button', { name: 'Yes, I did' }).click();
        await page.getByRole('heading', { name: 'Submitted' }).waitFor();
        await page
          .getByText('You confirmed the form was submitted on localhost.')
          .waitFor();
        await page.getByText('Number shown on the page').waitFor();
        await page.getByText('20240017', { exact: true }).waitFor();
        expect(
          await page.getByText('Confirmation number', { exact: true }).count(),
        ).toBe(0);
        expect(stopCalls(runId)).toBe(1);
        expect(await workerStops(page)).toBe(1);
        expect(JSON.stringify(appRequests)).not.toContain('20240017');
      } finally {
        await stopWorkerRun(page);
        await page.close();
        await site.close();
      }
    }
  }, 90_000);

  // --- review and submit, integrated (plan 4c-4 Task 6) -----------------------
  // Driven as the rest of this file is: the panel is a tab, the worker, the
  // relay and the page are real, the agent is agent-browser through the
  // relay, and only the fake app's answers are canned.

  /** Required fields of every kind, at the top of the page. */
  const REQUIRED_FORM =
    '<form id="app-form" method="post" action="/submitted?v=alnum">' +
    '<label for="e-name">Nickname</label><input id="e-name" required>' +
    '<label for="e-county">County</label><select id="e-county" required>' +
    '<option value="">--</option><option>Alameda</option></select>' +
    '<fieldset><legend>Housing</legend>' +
    '<label><input type="radio" id="e-r1" name="e-r" required> Own</label>' +
    '<label><input type="radio" id="e-r2" name="e-r"> Rent</label></fieldset>' +
    '<label><input type="checkbox" id="e-chk" required> I confirm</label>' +
    '<label for="e-ssn">Social Security Number</label>' +
    '<input id="e-ssn" name="field_31" required>' +
    '<label for="e-phone">Phone</label>' +
    '<input id="e-phone" aria-required="true">' +
    '<button type="submit" id="e-go">Submit</button></form>';

  const SSN_ECHO = '123-45-6789';

  /** The panel, the console and the fake app: where a value must not be. */
  const nowhereBut = async (
    page: import('playwright').Page,
    site: import('playwright').Page,
    values: string[],
    seen: string[],
    appSeen: number,
  ) => {
    const stored = await page.evaluate(async () =>
      JSON.stringify([
        await chrome.storage.local.get(null),
        await chrome.storage.session.get(null),
      ]),
    );
    const panelText = await page.evaluate(
      () => `${document.body.innerHTML}${document.body.innerText}`,
    );
    const toApp = JSON.stringify(appRequests.slice(appSeen));
    for (const v of values) {
      expect(panelText).not.toContain(v);
      expect(stored).not.toContain(v);
      expect(toApp).not.toContain(v);
      expect(await page.title()).not.toContain(v);
      expect(page.url()).not.toContain(v);
      expect(await site.title()).not.toContain(v);
      expect(site.url()).not.toContain(v);
      for (const line of seen) expect(line).not.toContain(v);
    }
  };

  it('Review end to end: Show me the unfilled field counts and highlights, the agent and the screenshot see no highlight, a real submit confirms', async () => {
    const { site, page, runId } = await startOnRealTab();
    const row = runs.get(runId);
    if (!row) throw new Error('no run row');
    const agentToken = mintRelayToken(
      { runId, role: 'agent', origin: fixtures.origin, sub: 'e2e-user' },
      relay.secret,
      { now: Math.floor(Date.now() / 1000) },
    );
    const cdp = cdpUrl(relay, agentToken);
    const frameStart = relay.frames.length;
    const dir = mkdtempSync(path.join(tmpdir(), 'review-shot-'));
    const recorder = await recordOverlay(browser.sw);
    try {
      await site.evaluate((html) => {
        window.scrollTo(0, 0);
        document.body.insertAdjacentHTML('afterbegin', html);
      }, REQUIRED_FORM);
      row.phase = 'review';
      const review = page.getByRole('region', {
        name: 'Ready for your review',
      });
      await review.waitFor({ timeout: 8_000 });
      await armed(page);
      const show = review.getByRole('button', {
        name: 'Show me the unfilled field',
      });
      const status = review.locator('[role="status"]:not([aria-label])');
      const statusIs = (text: string) =>
        expect
          .poll(async () => (await status.textContent()) ?? '', {
            timeout: 10_000,
          })
          .toContain(text);
      const activeId = () =>
        site.evaluate(() => document.activeElement?.id ?? '');

      // The worker highlights the first empty required field (an Overlay
      // highlight: no DOM footprint, no focus change) and the status line
      // counts all six.
      const tabId = ((await runStatus(page)) as Status & { tabId: number })
        .tabId;
      const highlighted = async () =>
        idOfNode(
          browser.sw,
          tabId,
          (
            (await recorder.calls())
              .filter((c) => c.method === 'Overlay.highlightNode' && c.ok)
              .at(-1) as { backendNodeId: number }
          ).backendNodeId,
        );
      const highlights = async () =>
        (await recorder.calls()).filter(
          (c) => c.method === 'Overlay.highlightNode' && c.ok,
        ).length;
      const focusBefore = await activeId();
      const formBefore = await site.evaluate(
        () => document.getElementById('app-form')?.outerHTML,
      );
      await show.click();
      await statusIs(
        'Showing the next field that is still empty. 6 left. Look for the field in your page that is highlighted or focused.',
      );
      await expect.poll(highlights).toBe(1);
      expect(await highlighted()).toBe('e-name');
      expect(await activeId()).toBe(focusBefore);
      expect(
        await site.evaluate(
          () => document.getElementById('app-form')?.outerHTML,
        ),
      ).toBe(formBefore);

      // The agent attaches and reads the page: it sees nothing of the
      // highlight, and the screenshot still works with the sensitive field
      // masked and left as it was.
      await site.fill('#e-name', 'Ash');
      await site.selectOption('#e-county', 'Alameda');
      await site.fill('#e-ssn', '900-11-2222');
      expect(
        (await agentBrowser(['snapshot', '-i'], cdp, 'review')).success,
      ).toBe(true);
      const shape = await site.evaluate(() => ({
        style: document.getElementById('e-ssn')?.getAttribute('style'),
        html: document.getElementById('app-form')?.outerHTML,
        active: document.activeElement?.id ?? '',
      }));
      await show.click();
      await statusIs('3 left');
      await expect.poll(highlights).toBe(2);
      expect(await highlighted()).toBe('e-r1');
      expect(
        await site.evaluate(() => ({
          style: document.getElementById('e-ssn')?.getAttribute('style'),
          html: document.getElementById('app-form')?.outerHTML,
          active: document.activeElement?.id ?? '',
        })),
      ).toEqual(shape);
      const seenByAgent = [
        await agentBrowser(['snapshot'], cdp, 'review'),
        await agentBrowser(['snapshot', '-i'], cdp, 'review'),
      ];
      expect(seenByAgent.map((r) => r.success)).toEqual([true, true]);
      expect(JSON.stringify(seenByAgent)).not.toContain('900-11-2222');
      expect(JSON.stringify(seenByAgent).toLowerCase()).not.toContain(
        'highlight',
      );
      const file = path.join(dir, 'shot.png');
      expect(
        (await agentBrowser(['screenshot', file], cdp, 'review')).success,
      ).toBe(true);
      expect(statSync(file).size).toBeGreaterThan(1000);
      const after = await site.evaluate(() => ({
        style: document.getElementById('e-ssn')?.getAttribute('style'),
        masks: document.querySelectorAll('[data-labs-mask]').length,
        sheets: document.adoptedStyleSheets.length,
      }));
      expect(after).toEqual({ style: shape.style, masks: 0, sheets: 0 });
      // No relay frame carried the sensitive value or an Overlay message.
      expect(relay.frames.length).toBeGreaterThan(frameStart);
      for (const frame of relay.frames.slice(frameStart)) {
        expect(frame).not.toContain('900-11-2222');
        expect(frame).not.toContain('Overlay.');
      }

      // The count follows the page: the sensitive field is next, flagged.
      await site.fill('#e-ssn', '');
      await site.check('#e-r2');
      await site.check('#e-chk');
      await show.click();
      await statusIs('2 left');
      await statusIs('It is a sensitive field');
      await expect.poll(highlights).toBe(3);
      expect(await highlighted()).toBe('e-ssn');
      await site.fill('#e-ssn', '900-11-2222');
      await site.fill('#e-phone', '5550100');
      await show.click();
      await statusIs(
        'No empty required fields found. Check the page before you submit.',
      );

      // The caseworker presses the page's own submit button.
      await site.click('#e-go');
      const prompt = page.getByRole('region', {
        name: 'Did you submit the form?',
      });
      await prompt.waitFor({ timeout: 10_000 });
      await prompt.getByText('AB-1234', { exact: true }).waitFor();
      expect(stopCalls(runId)).toBe(0);
      await prompt.getByRole('button', { name: 'Yes, I did' }).click();
      await page.getByRole('heading', { name: 'Submitted' }).waitFor();
      await page.getByText('Confirmation number', { exact: true }).waitFor();
      await page.getByText('AB-1234', { exact: true }).waitFor();
      expect(stopCalls(runId)).toBe(1);
      expect(await workerStops(page)).toBe(1);
      expect(row.ended).toBe(true);
      expect((await runStatus(page)).active).toBe(false);
      expect(JSON.stringify(appRequests)).not.toContain('AB-1234');
      expect(JSON.stringify(appRequests)).not.toContain('900-11-2222');
    } finally {
      await recorder.stop();
      rmSync(dir, { recursive: true, force: true });
      await closeAgentSessions();
      await stopWorkerRun(page);
      await page.close();
      await site.close();
    }
  }, 120_000);

  it('Review end to end: a page echoing an SSN beside a labelled number shows no number and the SSN reaches nobody; a ten-digit number the same', async () => {
    const { site, page, runId } = await startOnRealTab();
    const row = runs.get(runId);
    if (!row) throw new Error('no run row');
    const seen: string[] = [];
    const onConsole = (msg: import('playwright').ConsoleMessage) =>
      seen.push(msg.text());
    browser.context.on('console', onConsole);
    const appSeen = appRequests.length;
    const frameStart = relay.frames.length;
    const values = [SSN_ECHO, '123456789', '1234567890'];
    const readOver = () =>
      expect
        .poll(
          async () =>
            (await page
              .getByRole('status', { name: 'Submission check' })
              .textContent()) ?? 'x',
          { timeout: 20_000 },
        )
        .toBe('');
    const noNumber = async () => {
      for (const label of ['Confirmation number', 'Number shown on the page']) {
        expect(await page.getByText(label, { exact: true }).count()).toBe(0);
      }
    };
    try {
      row.phase = 'review';
      const review = page.getByRole('region', {
        name: 'Ready for your review',
      });
      await review.waitFor({ timeout: 8_000 });
      await armed(page);
      await submitIn(site, 'ssnecho');
      const prompt = page.getByRole('region', {
        name: 'Did you submit the form?',
      });
      // Asked at once; the looking line clears when the worker's read is
      // over (12 s), and only then is "no number" an answer.
      await prompt.waitFor({ timeout: 10_000 });
      await readOver();
      // No number is offered, and the run is still going.
      await noNumber();
      expect(stopCalls(runId)).toBe(0);
      expect((await runStatus(page)).active).toBe(true);
      await nowhereBut(page, site, values, seen, appSeen);

      // No keeps the run; a ten-digit number is not offered either.
      await prompt.getByRole('button', { name: 'No, not yet' }).click();
      await expect.poll(() => prompt.count()).toBe(0);
      await review.waitFor();
      await site.goto(fixtures.formUrl);
      await armed(page);
      await submitIn(site, 'ten');
      await prompt.waitFor({ timeout: 10_000 });
      await readOver();
      await noNumber();
      await nowhereBut(page, site, values, seen, appSeen);
      await prompt.getByRole('button', { name: 'Yes, I did' }).click();
      await page.getByRole('heading', { name: 'Submitted' }).waitFor();
      await noNumber();
      expect(stopCalls(runId)).toBe(1);
      expect(await workerStops(page)).toBe(1);
      await nowhereBut(page, site, values, seen, appSeen);
      for (const frame of relay.frames.slice(frameStart)) {
        for (const v of values) expect(frame).not.toContain(v);
      }
    } finally {
      browser.context.off('console', onConsole);
      await stopWorkerRun(page);
      await page.close();
      await site.close();
    }
  }, 120_000);

  it('Review end to end: a Next step asks, No keeps the run alive, and the real submission after it is detected', async () => {
    const { site, page, runId } = await startOnRealTab();
    const row = runs.get(runId);
    if (!row) throw new Error('no run row');
    try {
      row.phase = 'review';
      const review = page.getByRole('region', {
        name: 'Ready for your review',
      });
      await review.waitFor({ timeout: 8_000 });
      await armed(page);
      await site.evaluate(() => {
        document.body.insertAdjacentHTML(
          'afterbegin',
          '<form id="step1" method="post" action="/submitted?v=step">' +
            '<button type="submit" id="next-step">Next</button></form>',
        );
      });
      await site.click('#next-step');
      const prompt = page.getByRole('region', {
        name: 'Did you submit the form?',
      });
      await prompt.waitFor({ timeout: 10_000 });
      await prompt.getByText('20240017').waitFor();
      await prompt.getByRole('button', { name: 'No, not yet' }).click();
      await expect.poll(() => prompt.count()).toBe(0);
      await review.waitFor();
      expect(stopCalls(runId)).toBe(0);
      expect((await runStatus(page)).active).toBe(true);
      expect(row.ended).toBe(false);
      // The re-rendered form's own button: seen again, with its number.
      await armed(page);
      await site.click('#again-go');
      await prompt.waitFor({ timeout: 10_000 });
      await prompt.getByText('AB-1234', { exact: true }).waitFor();
      expect(stopCalls(runId)).toBe(0);
      await prompt.getByRole('button', { name: 'Yes, I did' }).click();
      await page.getByRole('heading', { name: 'Submitted' }).waitFor();
      await page.getByText('AB-1234', { exact: true }).waitFor();
      expect(stopCalls(runId)).toBe(1);
      expect(await workerStops(page)).toBe(1);
      expect(row.ended).toBe(true);
    } finally {
      await stopWorkerRun(page);
      await page.close();
      await site.close();
    }
  }, 90_000);

  it('a run the worker ended by itself is stopped on the server once', async () => {
    const { site, page, runId } = await startOnRealTab();
    try {
      // The worker ends the run without the panel (as on leaving the origin).
      // The panel waits out 7 s of "no run" (WORKER_GONE_AFTER_MS) first.
      await stopWorkerRun(page);
      await page
        .getByText('The run ended in the browser.', { exact: false })
        .waitFor({ timeout: 20_000 });
      await expect.poll(() => stopCalls(runId), { timeout: 5_000 }).toBe(1);
      expect(runs.get(runId)?.ended).toBe(true);
      expect(await page.getByRole('button', { name: 'Stop' }).count()).toBe(0);
      // Leaving the ended run (polling stopped with it; the unit tests pin
      // the single stop across later polls).
      await page.getByRole('button', { name: 'Back to Home' }).click();
      await page.getByRole('heading', { name: 'Home' }).waitFor();
      expect(stopCalls(runId)).toBe(1);
    } finally {
      await stopWorkerRun(page);
      await page.close();
      await site.close();
    }
  }, 40_000);

  it('rebuilds Running when the panel reopens mid-run, without a second eve-chat or a token refresh', async () => {
    const { site, page, runId } = await startOnRealTab();
    await page.close();
    const again = await browser.context.newPage();
    try {
      await again.goto(`chrome-extension://${EXTENSION_ID}/panel.html`);
      await again.getByRole('heading', { name: 'Running' }).waitFor();
      await again.getByText('Site: localhost').waitFor();
      expect(appCalls.filter((c) => c === 'POST /api/eve-chat')).toHaveLength(
        1,
      );
      // The worker reports the token's real expiry, so reopening mid-run does
      // not refresh the token (a same-run-id startRun) and shows no notice.
      await again.waitForTimeout(5_000);
      expect(appCalls.filter((c) => c === 'POST /api/relay/token').length).toBe(
        1,
      );
      expect(await again.getByText('This run may end soon.').count()).toBe(0);
      const status = await runStatus(again);
      expect({ active: status.active, runId: status.runId }).toEqual({
        active: true,
        runId,
      });
      expect(stopCalls(runId)).toBe(0);

      await again.getByRole('button', { name: 'Stop' }).click();
      await again.getByText('You stopped this run.').waitFor();
      expect(stopCalls(runId)).toBe(1);
    } finally {
      await stopWorkerRun(again);
      await again.close();
      await site.close();
    }
  }, 40_000);

  it('Needs You: answers a text and a sensitive question in one batch, only while the phase is needs_you', async () => {
    const ANSWER = 'Kern-County-Answer-42';
    const seen: string[] = [];
    const PENDING = [
      {
        requestId: 'q-text',
        prompt: 'County of residence <b>x</b>',
        display: 'text',
        options: [],
        allowFreeform: true,
      },
      {
        requestId: 'q-ssn',
        prompt: 'Social Security number',
        display: 'select',
        options: [
          { id: 'provided', label: 'I entered it myself' },
          { id: 'skip', label: 'Skip' },
        ],
        allowFreeform: false,
      },
    ];
    const { site, page, runId } = await startOnRealTab();
    page.on('console', (msg) => seen.push(msg.text()));
    answerBodies.length = 0;
    answerOrigins.length = 0;
    answerRefusal = null;
    const row = runs.get(runId);
    if (!row) throw new Error('no run row');
    const card = page.getByRole('region', { name: 'The agent needs you' });
    const pollsSince = () =>
      pollCalls.filter((c) => c === `GET /api/relay/runs/${runId}`).length;
    try {
      // Pending questions while the agent works or after it stopped: no card.
      for (const phase of ['working', 'review']) {
        row.phase = phase;
        row.pending = PENDING;
        const before = pollsSince();
        await expect
          .poll(pollsSince, { timeout: 10_000 })
          .toBeGreaterThan(before + 1);
        expect(await card.count()).toBe(0);
      }

      row.phase = 'needs_you';
      await card.waitFor({ timeout: 10_000 });
      // The prompt is text, not markup.
      await card.getByText('County of residence <b>x</b>').waitFor();
      expect(await card.locator('b').count()).toBe(0);
      await card.getByText('Social Security number').waitFor();
      await card
        .getByText(
          'This one is sensitive. Type it in the page yourself first, then choose "I typed it in the page" and press Send. Or type it here and press Fill it in.',
        )
        .waitFor();
      // The sensitive question gets a masked input, not a textarea: one only.
      expect(await card.locator('textarea').count()).toBe(1);

      const send = card.getByRole('button', { name: 'Send' });
      expect(await send.isDisabled()).toBe(true);
      const box = card.getByLabel('County of residence <b>x</b>');
      expect(await box.getAttribute('autocomplete')).toBe('off');
      expect(await box.getAttribute('maxlength')).toBe('2000');
      // The fixed no-numbers line sits under the input and describes it.
      const warning = card.getByText(
        'Do not type Social Security, bank, card or ID numbers here. Type those in the page.',
      );
      const warningId = await warning.getAttribute('id');
      expect(warningId).toBeTruthy();
      expect(await box.getAttribute('aria-describedby')).toBe(warningId);
      await box.fill(ANSWER);
      expect(await send.isDisabled()).toBe(true);

      // The same pending set re-reported by later polls keeps what was typed.
      let mark = pollsSince();
      await expect
        .poll(pollsSince, { timeout: 10_000 })
        .toBeGreaterThan(mark + 1);
      expect(await box.inputValue()).toBe(ANSWER);
      // A changed id set resets the card.
      row.pending = [{ ...PENDING[0], requestId: 'q-text-2' }, PENDING[1]];
      await expect.poll(() => box.inputValue(), { timeout: 10_000 }).toBe('');
      row.pending = PENDING;
      mark = pollsSince();
      await expect
        .poll(pollsSince, { timeout: 10_000 })
        .toBeGreaterThan(mark + 1);
      await box.fill(ANSWER);
      expect(await send.isDisabled()).toBe(true);
      await card
        .getByRole('button', { name: 'I typed it in the page' })
        .click();
      expect(await send.isDisabled()).toBe(false);

      // The server refuses the text: fixed copy, the text is cleared.
      answerRefusal = 'sensitive_value_in_message';
      await send.click();
      await card
        .getByRole('alert')
        .getByText("We can't send that here")
        .waitFor();
      await card
        .getByRole('alert')
        .getByText('Type that value in the page yourself, then press Skip.')
        .waitFor();
      // Focus moved to the alert, not lost with the disabled Send.
      expect(
        await page.evaluate(() => document.activeElement?.getAttribute('role')),
      ).toBe('alert');
      expect(await box.inputValue()).toBe('');
      expect(await send.isDisabled()).toBe(true);
      expect(answerBodies).toHaveLength(1);

      // Answer again: one POST with exactly the batch.
      answerRefusal = null;
      await box.fill(ANSWER);
      expect(await send.isDisabled()).toBe(false);
      await send.click();
      await card.getByText('Sent. Waiting for the agent...').waitFor();
      expect(
        await page.evaluate(() => document.activeElement?.getAttribute('role')),
      ).toBe('status');
      expect(answerBodies).toHaveLength(2);
      expect(answerBodies[1]).toEqual({
        id: runId,
        inputResponses: [
          { requestId: 'q-text', text: ANSWER },
          { requestId: 'q-ssn', optionId: 'provided' },
        ],
      });
      expect(answerOrigins).toEqual([
        `chrome-extension://${EXTENSION_ID}`,
        `chrome-extension://${EXTENSION_ID}`,
      ]);
      // Still the same pending set: the card stays sent and disabled.
      const before = pollsSince();
      await expect
        .poll(pollsSince, { timeout: 10_000 })
        .toBeGreaterThan(before + 1);
      await card.getByText('Sent. Waiting for the agent...').waitFor();
      expect(await send.isDisabled()).toBe(true);
      expect(await box.isDisabled()).toBe(true);
      expect(answerBodies).toHaveLength(2);

      // The agent took the answers: the card goes away.
      row.phase = 'working';
      row.pending = [];
      await card.waitFor({ state: 'detached', timeout: 10_000 });

      // A lone sensitive question (plus a malformed entry the card ignores):
      // its button only selects; Send sends.
      row.pending = [null, PENDING[1]];
      row.phase = 'needs_you';
      await card.waitFor({ timeout: 10_000 });
      expect(await card.locator('textarea').count()).toBe(0);
      await card.getByRole('button', { name: 'Skip' }).click();
      expect(
        await card
          .getByRole('button', { name: 'Skip' })
          .getAttribute('aria-pressed'),
      ).toBe('true');
      expect(answerBodies).toHaveLength(2);
      await send.click();
      await card.getByText('Sent. Waiting for the agent...').waitFor();
      expect(answerBodies).toHaveLength(3);
      expect(answerBodies[2]).toEqual({
        id: runId,
        inputResponses: [{ requestId: 'q-ssn', optionId: 'skip' }],
      });
      row.phase = 'working';
      row.pending = [];
      await card.waitFor({ state: 'detached', timeout: 10_000 });

      // Fill it in: the masked input, fixed refusal copy, then success. The
      // value reaches only the worker message, never an answer POST.
      const SECRET = 'Zq9-Tangerine-Value-731';
      await page.evaluate(() => {
        const w = window as unknown as {
          __fillCalls: { runId: string; value: string }[];
          __fillReply: { ok: boolean; error?: string };
        };
        w.__fillCalls = [];
        w.__fillReply = { ok: false, error: 'no focused field' };
        const send = chrome.runtime.sendMessage.bind(chrome.runtime);
        chrome.runtime.sendMessage = ((...args: unknown[]) => {
          const m = args[0] as { type?: string; runId: string; value: string };
          if (m?.type === 'typeSensitive') {
            w.__fillCalls.push({ runId: m.runId, value: m.value });
            // The panel's client passes a callback; answer it as Chrome does.
            if (typeof args[1] === 'function') {
              (args[1] as (r: unknown) => void)(w.__fillReply);
              return undefined;
            }
            return Promise.resolve(w.__fillReply);
          }
          return (send as (...a: unknown[]) => unknown)(...args);
        }) as typeof chrome.runtime.sendMessage;
      });
      row.pending = [PENDING[1]];
      row.phase = 'needs_you';
      await card.waitFor({ timeout: 10_000 });
      const secret = card.getByLabel('Value to fill in');
      const fillIt = card.getByRole('button', { name: 'Fill it in' });
      expect(await secret.getAttribute('type')).toBe('password');
      expect(await secret.getAttribute('autocomplete')).toBe('off');
      expect(await secret.getAttribute('name')).toBeNull();
      expect(await secret.getAttribute('id')).toBeNull();
      expect(await fillIt.isDisabled()).toBe(true);
      await card
        .getByText('Click the field in the page, then press Fill it in.')
        .waitFor();
      await secret.fill(`  ${SECRET}\n`);
      const eye = card.getByRole('button', { name: 'Show what I typed' });
      expect(await eye.getAttribute('aria-pressed')).toBe('false');
      await eye.click();
      expect(await eye.getAttribute('aria-pressed')).toBe('true');
      expect(await secret.getAttribute('type')).toBe('text');
      await eye.click();
      expect(await secret.getAttribute('type')).toBe('password');
      await fillIt.click();
      await card
        .getByRole('alert')
        .filter({
          hasText: 'Click the field in the page first, then press Fill it in.',
        })
        .waitFor();
      // The value is kept for a retry and sent trimmed.
      expect(await secret.inputValue()).toContain(SECRET);
      expect(await send.isDisabled()).toBe(true);
      await page.evaluate(() => {
        (window as unknown as { __fillReply: { ok: boolean } }).__fillReply = {
          ok: true,
        };
      });
      await fillIt.click();
      await card.getByText('Filled in.').waitFor();
      expect(await card.getByLabel('Value to fill in').count()).toBe(0);
      expect(
        await page.evaluate(
          () =>
            (
              window as unknown as {
                __fillCalls: { runId: string; value: string }[];
              }
            ).__fillCalls,
        ),
      ).toEqual([
        { runId, value: SECRET },
        { runId, value: SECRET },
      ]);
      // The value is nowhere in the panel's DOM text after success.
      expect(await page.evaluate(() => document.body.innerHTML)).not.toContain(
        'Tangerine',
      );
      expect(await send.isDisabled()).toBe(false);
      await send.click();
      await card.getByText('Sent. Waiting for the agent...').waitFor();
      expect(answerBodies).toHaveLength(4);
      expect(answerBodies[3]).toEqual({
        id: runId,
        inputResponses: [{ requestId: 'q-ssn', optionId: 'provided' }],
      });
      expect(JSON.stringify(answerBodies)).not.toContain('Tangerine');
      row.phase = 'working';
      row.pending = [];
      await card.waitFor({ state: 'detached', timeout: 10_000 });

      // A free-text question with a flagged prompt gets the typing card (a
      // masked input, no text box) and is answered with the exact text.
      const freeSsn = [
        {
          requestId: 'q-free-ssn',
          prompt: 'What is your Social Security number?',
          display: 'text',
          options: [],
          allowFreeform: true,
        },
      ];
      row.pending = freeSsn;
      row.phase = 'needs_you';
      await card
        .getByRole('button', { name: 'Fill it in' })
        .waitFor({ timeout: 10_000 });
      expect(await card.getByLabel('Value to fill in').count()).toBe(1);
      expect(await card.locator('textarea').count()).toBe(0);
      expect(await card.getByText('cannot be answered here').count()).toBe(0);
      expect(await send.isDisabled()).toBe(true);
      await card
        .getByRole('button', { name: 'I typed it in the page' })
        .click();
      await send.click();
      await card.getByText('Sent. Waiting for the agent...').waitFor();
      expect(answerBodies).toHaveLength(5);
      expect(answerBodies[4]).toEqual({
        id: runId,
        inputResponses: [{ requestId: 'q-free-ssn', text: 'provided' }],
      });
      row.phase = 'working';
      row.pending = [];
      await card.waitFor({ state: 'detached', timeout: 10_000 });

      // Skip on the same kind of question is the text `skip`.
      row.pending = freeSsn;
      row.phase = 'needs_you';
      await card
        .getByRole('button', { name: 'Skip' })
        .click({ timeout: 10_000 });
      await send.click();
      await card.getByText('Sent. Waiting for the agent...').waitFor();
      expect(answerBodies).toHaveLength(6);
      expect(answerBodies[5]).toEqual({
        id: runId,
        inputResponses: [{ requestId: 'q-free-ssn', text: 'skip' }],
      });
      row.phase = 'working';
      row.pending = [];
      await card.waitFor({ state: 'detached', timeout: 10_000 });

      // A yes/no whose prompt only mentions a sensitive topic stays a choice.
      row.pending = [
        {
          requestId: 'q-benefits',
          prompt:
            'Does anyone in the household receive Social Security benefits?',
          display: 'select',
          options: [
            { id: 'yes', label: 'Yes' },
            { id: 'no', label: 'No' },
          ],
          allowFreeform: false,
        },
      ];
      row.phase = 'needs_you';
      await card.getByRole('radio', { name: 'Yes' }).click({ timeout: 10_000 });
      await send.click();
      await card.getByText('Sent. Waiting for the agent...').waitFor();
      expect(answerBodies).toHaveLength(7);
      expect(answerBodies[6]).toEqual({
        id: runId,
        inputResponses: [{ requestId: 'q-benefits', optionId: 'yes' }],
      });
      row.phase = 'working';
      row.pending = [];
      await card.waitFor({ state: 'detached', timeout: 10_000 });

      // The typed answer and the filled-in value reached no console line,
      // the title or storage.
      for (const secret of [ANSWER, SECRET]) {
        expect(await page.title()).not.toContain(secret);
        expect(page.url()).not.toContain(secret);
        const stored = await page.evaluate(async () =>
          JSON.stringify([
            await chrome.storage.local.get(null),
            await chrome.storage.session.get(null),
          ]),
        );
        expect(stored).not.toContain(secret);
        for (const line of seen) expect(line).not.toContain(secret);
      }
    } finally {
      answerRefusal = null;
      await stopWorkerRun(page);
      await page.close();
      await site.close();
    }
  }, 90_000);

  it('types a sensitive value end to end with the real worker, through a token refresh', async () => {
    // Driven as the rest of this file is: the panel is a tab, the worker, the
    // relay and the page are real, and the agent is agent-browser through the
    // relay. Only the fake app's answers are canned.
    const SECRET = 'Qx4-Mango-Value-905';
    const PENDING = [
      {
        requestId: 'q-ssn',
        prompt: 'Social Security number',
        display: 'select',
        options: [
          { id: 'provided', label: 'I entered it myself' },
          { id: 'skip', label: 'Skip' },
        ],
        allowFreeform: false,
      },
    ];
    const { site, page, runId } = await startOnRealTab();
    const seen: string[] = [];
    // The panel's, the page's and (where Playwright reports them) the
    // worker's console lines.
    const onConsole = (msg: import('playwright').ConsoleMessage) =>
      seen.push(msg.text());
    browser.context.on('console', onConsole);
    answerBodies.length = 0;
    answerOrigins.length = 0;
    answerRefusal = null;
    appCalls.length = 0;
    const appSeen = appRequests.length;
    const row = runs.get(runId);
    if (!row) throw new Error('no run row');
    const now = Math.floor(Date.now() / 1000);
    const agentToken = mintRelayToken(
      { runId, role: 'agent', origin: fixtures.origin, sub: 'e2e-user' },
      relay.secret,
      { now },
    );
    const cdp = cdpUrl(relay, agentToken);
    const frameStart = relay.frames.length;
    try {
      await site.evaluate(() => {
        document.body.insertAdjacentHTML(
          'beforeend',
          '<div id="typing">' +
            '<label for="e2e-ssn">Social Security Number</label>' +
            '<input id="e2e-ssn" name="field_17">' +
            '<label for="e2e-mid">Middle name</label>' +
            '<input id="e2e-mid" name="field_18"></div>',
        );
      });
      // The agent reads the page: this attaches the debugger to the tab.
      expect(
        (await agentBrowser(['snapshot', '-i'], cdp, 'typing')).success,
      ).toBe(true);

      row.pending = PENDING;
      row.phase = 'needs_you';
      const card = page.getByRole('region', { name: 'The agent needs you' });
      await card.waitFor({ timeout: 10_000 });
      const fillIt = card.getByRole('button', { name: 'Fill it in' });
      const send = card.getByRole('button', { name: 'Send' });
      await card.getByLabel('Value to fill in').fill(SECRET);

      // An ordinary field is focused: refused with the fixed copy, nothing
      // typed, and the value is kept for a retry.
      await site.focus('#e2e-mid');
      await fillIt.click();
      await card
        .getByRole('alert')
        .getByText(
          'The extension does not treat that field as sensitive, so it will not type there. Type it in the page yourself.',
        )
        .waitFor();
      expect(await site.inputValue('#e2e-mid')).toBe('');
      expect(await site.inputValue('#e2e-ssn')).toBe('');
      expect(await card.getByLabel('Value to fill in').inputValue()).toBe(
        SECRET,
      );

      // The caseworker clicks the masked field, then Fill it in.
      await site.focus('#e2e-ssn');
      await fillIt.click();
      await card.getByText('Filled in.').waitFor();
      expect(await site.inputValue('#e2e-ssn')).toBe(SECRET);
      expect(await site.inputValue('#e2e-mid')).toBe('');
      expect(await page.evaluate(() => document.body.innerHTML)).not.toContain(
        SECRET,
      );
      await send.click();
      await card.getByText('Sent. Waiting for the agent...').waitFor();
      expect(answerBodies).toEqual([
        {
          id: runId,
          inputResponses: [{ requestId: 'q-ssn', optionId: 'provided' }],
        },
      ]);
      row.phase = 'working';
      row.pending = [];
      await card.waitFor({ state: 'detached', timeout: 10_000 });

      // A token refresh in the middle: a same-run startRun on the run's tab,
      // then the panel's registrationComplete, as the run session sends them.
      const tabId = ((await runStatus(page)) as Status & { tabId: number })
        .tabId;
      const refreshed = mintRelayToken(
        {
          runId,
          role: 'extension',
          origin: fixtures.origin,
          sub: 'e2e-user',
        },
        relay.secret,
        { now: Math.floor(Date.now() / 1000) },
      );
      const registrations = relay.logs.filter(
        (l) => l.event === 'registered' && l.runId === runId,
      ).length;
      expect(
        await page.evaluate(
          ([relayUrl, token, tab, id]) =>
            chrome.runtime
              .sendMessage({ type: 'startRun', relayUrl, token, tabId: tab })
              .then(() =>
                chrome.runtime.sendMessage({
                  type: 'registrationComplete',
                  runId: id,
                  count: 0,
                }),
              ),
          [relay.url, refreshed, tabId, runId] as const,
        ),
      ).toEqual({ ok: true });
      await expect
        .poll(
          () =>
            relay.logs.filter(
              (l) => l.event === 'registered' && l.runId === runId,
            ).length,
          { timeout: 10_000 },
        )
        .toBeGreaterThan(registrations);
      expect((await runStatus(page)).runId).toBe(runId);

      // The page now echoes the value in ordinary text. The agent, on a
      // fresh connection after the refresh (the old one closed with the
      // restarted run), still never sees it.
      await site.evaluate((v) => {
        const echo = document.createElement('p');
        echo.textContent = `You entered ${v}`;
        document.getElementById('typing')?.append(echo);
        document.title = `Review ${v}`;
      }, SECRET);
      await closeAgentSessions();
      const agentSaw: unknown[] = [];
      for (const args of [
        ['snapshot'],
        ['snapshot', '-i'],
        ['get', 'title'],
        ['get', 'url'],
        ['screenshot'],
      ]) {
        agentSaw.push(await agentBrowser(args, cdp, 'after-refresh'));
      }
      // The reads work; the screenshot is refused while the registered
      // value is in the page text, and says nothing of it.
      expect(
        (agentSaw as { success: boolean }[]).map((r) => r.success),
      ).toEqual([true, true, true, true, false]);
      expect(JSON.stringify(agentSaw.at(-1))).toContain(
        'sensitive fields could not be masked',
      );
      expect(JSON.stringify(agentSaw)).not.toContain(SECRET);
      expect(JSON.stringify(agentSaw)).not.toContain('Mango');
      // The value is still in the page field (the caseworker's own view).
      expect(await site.inputValue('#e2e-ssn')).toBe(SECRET);

      // Nothing the agent's relay carried held the value.
      expect(relay.frames.length).toBeGreaterThan(frameStart);
      for (const frame of relay.frames.slice(frameStart)) {
        expect(frame).not.toContain(SECRET);
      }
      // The fake app saw no value in any request body, URL or header.
      const toApp = JSON.stringify([
        appRequests.slice(appSeen),
        answerBodies,
        answerOrigins,
      ]);
      expect(toApp).not.toContain(SECRET);
      expect(toApp).not.toContain('Mango');
      // Nor did a console line, the panel's title or URL, the panel's DOM,
      // or storage.
      expect(await page.title()).not.toContain(SECRET);
      expect(page.url()).not.toContain(SECRET);
      expect(await page.evaluate(() => document.body.innerHTML)).not.toContain(
        SECRET,
      );
      const stored = await page.evaluate(async () =>
        JSON.stringify([
          await chrome.storage.local.get(null),
          await chrome.storage.session.get(null),
        ]),
      );
      expect(stored).not.toContain(SECRET);
      for (const line of seen) expect(line).not.toContain(SECRET);
    } finally {
      browser.context.off('console', onConsole);
      await closeAgentSessions();
      await stopWorkerRun(page);
      await page.close();
      await site.close();
    }
  }, 90_000);

  type HoldStatus = Status & { pendingOrigin?: string };

  /** Sends the run's tab to the other fixture origin: the worker holds. */
  const leaveForFrameOrigin = async (
    site: import('playwright').Page,
    page: import('playwright').Page,
  ) => {
    // A query and a fragment the card must never show.
    const away = `${fixtures.frameOrigin}/oopif.html?case=Zelda-Quill#frag-77`;
    await site.evaluate((u) => {
      location.href = u;
    }, away);
    await expect
      .poll(async () => ((await runStatus(page)) as HoldStatus).pendingOrigin, {
        timeout: 5_000,
      })
      .toBe(fixtures.frameOrigin);
  };

  const stoppedFor = (runId: string, reason: string) =>
    relay.logs.some(
      (l) =>
        l.event === 'run-stopped' &&
        (l as { runId?: string }).runId === runId &&
        (l as { reason?: string }).reason === reason,
    );

  it('Allow this site: names the origin and the tab URL; Allow lets the run go on', async () => {
    const { site, page, runId } = await startOnRealTab();
    const seen: string[] = [];
    page.on('console', (msg) => seen.push(msg.text()));
    try {
      // The progress comes from the first server poll (3 s), after the first
      // worker poll (2 s): the panel has seen the run clear, so it sees the
      // hold begin and can show a countdown.
      await page.getByText('6 fields so far').waitFor();
      await leaveForFrameOrigin(site, page);
      const heading = page.getByRole('heading', { name: 'Allow this site?' });
      await heading.waitFor({ timeout: 5_000 });
      // Focus moves to the card once, when it appears.
      expect(await page.evaluate(() => document.activeElement?.id ?? '')).toBe(
        'allow-site-title',
      );
      await page
        .getByText(
          `The page moved to ${fixtures.frameOrigin}. The agent is paused until you decide.`,
        )
        .waitFor();
      // The tab's actual URL: origin plus path, never the query or fragment.
      await page
        .getByText(`The tab is now on: ${fixtures.frameOrigin}/oopif.html`, {
          exact: true,
        })
        .waitFor();
      // The panel saw the hold begin (it polled the clear run first).
      // The build's shortened hold (the panel gets the same define).
      const left = await page
        .getByText(/^About \d+ seconds? left$/)
        .textContent();
      expect(Number(left?.match(/\d+/)?.[0])).toBeLessThanOrEqual(
        PANEL_HOLD_MS / 1000,
      );
      const shown = await page.locator('body').innerText();
      expect(shown).not.toMatch(/Zelda|frag-77/);
      // Nothing allowed yet: the worker still holds.
      expect(((await runStatus(page)) as HoldStatus).pendingOrigin).toBe(
        fixtures.frameOrigin,
      );

      // A hold takes precedence: a review phase shows no review panel.
      const heldRun = runs.get(runId);
      if (heldRun) heldRun.phase = 'review';
      // Wait for the panel's next server poll to have read the review phase.
      const polled = pollCalls.length;
      await expect
        .poll(() => pollCalls.length, { timeout: 5_000 })
        .toBeGreaterThan(polled);
      expect(await heading.count()).toBe(1);
      expect(
        await page
          .getByRole('heading', { name: 'Ready for your review' })
          .count(),
      ).toBe(0);
      if (heldRun) heldRun.phase = 'working';

      await page
        .getByRole('button', { name: 'Allow this site for this run' })
        .click();
      // One predicate: the same run, still active, and no longer held.
      await expect
        .poll(
          async () => {
            const status = (await runStatus(page)) as HoldStatus;
            return {
              active: status.active,
              runId: status.runId,
              held: status.pendingOrigin ?? null,
            };
          },
          { timeout: 5_000 },
        )
        .toEqual({ active: true, runId, held: null });
      await expect.poll(() => heading.count(), { timeout: 5_000 }).toBe(0);

      // Past the hold: the run goes on at the new origin.
      await new Promise((r) => setTimeout(r, PANEL_HOLD_MS + 1_000));
      const after = (await runStatus(page)) as HoldStatus;
      expect({ active: after.active, runId: after.runId }).toEqual({
        active: true,
        runId,
      });
      expect(after.pendingOrigin).toBeUndefined();
      expect(stoppedFor(runId, 'left-origin')).toBe(false);
      await page.getByRole('heading', { name: 'Running' }).waitFor();
      for (const line of seen) expect(line).not.toMatch(/Zelda|frag-77|oopif/);
    } finally {
      await stopWorkerRun(page);
      await page.close();
      await site.close();
    }
  }, 60_000);

  it('Allow this site: nobody answers, the run ends after the hold and the card goes', async () => {
    const { site, page, runId } = await startOnRealTab();
    try {
      await page.getByText('Site: localhost').waitFor();
      await leaveForFrameOrigin(site, page);
      const heading = page.getByRole('heading', { name: 'Allow this site?' });
      await heading.waitFor({ timeout: 5_000 });
      // The hold runs out; the panel waits out WORKER_GONE_AFTER_MS (7 s).
      await page
        .getByText('The run ended in the browser.', { exact: false })
        .waitFor({ timeout: PANEL_HOLD_MS + 20_000 });
      expect(stoppedFor(runId, 'left-origin')).toBe(true);
      expect(await heading.count()).toBe(0);
      expect(
        await page
          .getByRole('button', { name: 'Allow this site for this run' })
          .count(),
      ).toBe(0);
      expect((await runStatus(page)).active).toBe(false);
      await expect.poll(() => stopCalls(runId), { timeout: 5_000 }).toBe(1);
    } finally {
      await stopWorkerRun(page);
      await page.close();
      await site.close();
    }
  }, 60_000);

  it('runs one form end to end: progress, questions, an allowed move, review, submitted, another form', async () => {
    // Driven as panel.html in a tab (a real side panel cannot be driven
    // headless), against the real worker and relay; the fake app's run row
    // changes over time the way the server's would. The only stubs are the
    // tab query (the panel is a tab) and the unlisted-origin move, which is a
    // real navigation of the run's tab.
    const ANSWER = 'End-To-End-Answer-91';
    const PENDING = [
      {
        requestId: 'q-text',
        prompt: 'County of residence',
        display: 'text',
        options: [],
        allowFreeform: true,
      },
      {
        requestId: 'q-ssn',
        prompt: 'Social Security number',
        display: 'select',
        options: [
          { id: 'provided', label: 'I entered it myself' },
          { id: 'skip', label: 'Skip' },
        ],
        allowFreeform: false,
      },
    ];
    const { site, page, runId } = await startOnRealTab();
    const seen: string[] = [];
    page.on('console', (msg) => seen.push(msg.text()));
    page.on('pageerror', (error) => seen.push(error.message));
    answerBodies.length = 0;
    answerRefusal = null;
    const row = runs.get(runId);
    if (!row) throw new Error('no run row');
    const needsYou = page.getByRole('region', { name: 'The agent needs you' });
    const allow = page.getByRole('heading', { name: 'Allow this site?' });
    const review = page.getByRole('heading', { name: 'Ready for your review' });
    const none = async (...gone: import('playwright').Locator[]) => {
      for (const l of gone) expect(await l.count()).toBe(0);
    };
    try {
      // Working, with progress: sections and actions, nothing else.
      await page.getByText('6 fields so far').waitFor();
      await none(needsYou, allow, review);

      // Text and sensitive questions: the card, only on needs_you.
      row.pending = PENDING;
      row.phase = 'needs_you';
      await needsYou.waitFor({ timeout: 10_000 });
      await needsYou.getByText('Social Security number').waitFor();
      await none(allow, review);
      await needsYou.getByLabel('County of residence').fill(ANSWER);
      await needsYou
        .getByRole('button', { name: 'I typed it in the page' })
        .click();
      await needsYou.getByRole('button', { name: 'Send' }).click();
      await needsYou.getByText('Sent. Waiting for the agent...').waitFor();
      expect(answerBodies).toEqual([
        {
          id: runId,
          inputResponses: [
            { requestId: 'q-text', text: ANSWER },
            { requestId: 'q-ssn', optionId: 'provided' },
          ],
        },
      ]);

      // The agent took them: working again, no stale card.
      row.pending = [];
      row.phase = 'working';
      await needsYou.waitFor({ state: 'detached', timeout: 10_000 });
      await page.getByText('6 fields so far').waitFor();
      await none(allow, review);

      // The tab leaves for another origin: the allow card, then Allow.
      await leaveForFrameOrigin(site, page);
      await allow.waitFor({ timeout: 5_000 });
      await none(needsYou, review);
      await page
        .getByRole('button', { name: 'Allow this site for this run' })
        .click();
      await expect.poll(() => allow.count(), { timeout: 5_000 }).toBe(0);
      const status = (await runStatus(page)) as HoldStatus;
      expect({ active: status.active, runId: status.runId }).toEqual({
        active: true,
        runId,
      });

      // The turn idles: the review panel, and nothing stale beside it.
      row.phase = 'review';
      await review.waitFor({ timeout: 10_000 });
      await none(needsYou, allow);

      // Submitted: one stop on each side, Confirmed, then another form.
      await page.getByRole('button', { name: 'I submitted it' }).click();
      await page.getByRole('heading', { name: 'Submitted' }).waitFor();
      await none(needsYou, allow, review);
      expect(row.ended).toBe(true);
      expect(stopCalls(runId)).toBe(1);
      expect((await runStatus(page)).active).toBe(false);
      await page
        .getByRole('button', { name: 'Start another form for this person' })
        .click();
      await page.getByRole('heading', { name: 'Ready' }).waitFor();
      await page.getByRole('button', { name: 'Start' }).waitFor();
      await none(needsYou, allow, review);

      // The fake app saw each call once, in this order; the answer turn is
      // the one eve-chat POST after the first, which `answerBodies` counted.
      expect(appCalls).toEqual([
        'POST /api/eve-chat',
        'POST /api/relay/token',
        `POST /api/relay/runs/${runId}/stop`,
      ]);
      expect(answerBodies).toHaveLength(1);
      // No applicant text in the console, and nothing from the move's URL.
      for (const line of seen) {
        expect(line).not.toContain(ANSWER);
        expect(line).not.toMatch(/Zelda|frag-77|oopif/);
      }
    } finally {
      await stopWorkerRun(page);
      await page.close();
      await site.close();
    }
  }, 90_000);

  it('survives the reliability faults: an orphaned start is swept once, typing and an allowed origin survive a refresh, a held run is never swept', async () => {
    const SECRET = 'Rv7-Papaya-Value-318';
    const PENDING = [
      {
        requestId: 'q-ssn',
        prompt: 'Social Security number',
        display: 'select',
        options: [
          { id: 'provided', label: 'I entered it myself' },
          { id: 'skip', label: 'Skip' },
        ],
        allowFreeform: false,
      },
    ];
    const stopsSeen = (id: string) =>
      appRequests
        .map((r) => JSON.parse(r) as [string, string])
        .filter(
          ([method, url]) =>
            method === 'POST' && url === `/api/relay/runs/${id}/stop`,
        ).length;
    /** Moves every stored recent back in time; `extra` are added as given. */
    const ageRecents = (ms: number, extra: object[] = []) =>
      (browser.context.serviceWorkers()[0] ?? browser.sw).evaluate(
        async ([age, more]: [number, object[]]) => {
          const list =
            ((await chrome.storage.local.get('recentRuns')).recentRuns as {
              startedAt: number;
            }[]) ?? [];
          await chrome.storage.local.set({
            recentRuns: [
              ...list.map((r) => ({ ...r, startedAt: Date.now() - age })),
              ...more,
            ],
          });
        },
        [ms, extra] as [number, object[]],
      );
    const recentOf = (id: string) =>
      (browser.context.serviceWorkers()[0] ?? browser.sw).evaluate(
        async (runId: string) => {
          const found = (
            (await chrome.storage.local.get('recentRuns')).recentRuns as {
              runId: string;
              endedReason?: string;
            }[]
          ).find((r) => r.runId === runId);
          return found ? (found.endedReason ?? 'unended') : 'none';
        },
        id,
      );
    const refresh = async (page: import('playwright').Page, runId: string) => {
      const registrations = () =>
        relay.logs.filter((l) => l.event === 'registered' && l.runId === runId)
          .length;
      const before = registrations();
      const tabId = ((await runStatus(page)) as Status & { tabId: number })
        .tabId;
      const token = mintRelayToken(
        { runId, role: 'extension', origin: fixtures.origin, sub: 'e2e-user' },
        relay.secret,
        { now: Math.floor(Date.now() / 1000) },
      );
      const reply = await page.evaluate(
        ([relayUrl, t, tab, id]) =>
          chrome.runtime
            .sendMessage({ type: 'startRun', relayUrl, token: t, tabId: tab })
            .then((r: unknown) =>
              chrome.runtime
                .sendMessage({
                  type: 'registrationComplete',
                  runId: id,
                  count: 0,
                })
                .then(() => r),
            ),
        [relay.url, token, tabId, runId] as const,
      );
      expect(reply).toEqual({ ok: true });
      await expect
        .poll(registrations, { timeout: 10_000 })
        .toBeGreaterThan(before);
    };

    // (4a) A panel closed on a hung Starting leaves a recent with no end.
    const first = await readyOnRealTab();
    eveChatBodies.length = 0;
    eveChatHang = true;
    let orphanId: string;
    try {
      await first.page.getByRole('button', { name: 'Start' }).click();
      await expect.poll(() => eveChatBodies.length).toBe(1);
      orphanId = (eveChatBodies[0] as { id: string }).id;
      expect(runs.has(orphanId)).toBe(true);
      expect(await recentOf(orphanId)).toBe('unended');
      expect((await runStatus(first.page)).active).toBe(false);
    } finally {
      eveChatHang = false;
      await first.page.close();
      await first.site.close();
    }
    // Past the 90 s age gate (seeded, not waited out), with no stop yet.
    await ageRecents(5 * 60_000);
    expect(stopsSeen(orphanId)).toBe(0);

    // The next open sweeps it: one stop, marked swept. Then a real run starts.
    const { site, page, runId } = await startOnRealTab();
    try {
      // The sweep is fire and forget: wait for it rather than assume it ran.
      await expect.poll(() => stopsSeen(orphanId)).toBe(1);
      expect(runs.get(orphanId)?.ended).toBe(true);
      await expect.poll(() => recentOf(orphanId)).toBe('swept');
      expect(stopsSeen(runId)).toBe(0);

      const row = runs.get(runId);
      if (!row) throw new Error('no run row');
      answerBodies.length = 0;
      answerRefusal = null;

      // (3) After a refresh, with the agent silent, Fill it in types.
      await site.evaluate(() => {
        document.body.insertAdjacentHTML(
          'beforeend',
          '<label for="rv-ssn">Social Security Number</label>' +
            '<input id="rv-ssn" name="field_21">',
        );
      });
      await refresh(page, runId);
      row.pending = PENDING;
      row.phase = 'needs_you';
      const card = page.getByRole('region', { name: 'The agent needs you' });
      await card.waitFor({ timeout: 10_000 });
      await card.getByLabel('Value to fill in').fill(SECRET);
      await site.focus('#rv-ssn');
      await card.getByRole('button', { name: 'Fill it in' }).click();
      await card.getByText('Filled in.').waitFor();
      expect(await site.inputValue('#rv-ssn')).toBe(SECRET);
      await card.getByRole('button', { name: 'Send' }).click();
      await card.getByText('Sent. Waiting for the agent...').waitFor();
      expect(answerBodies).toEqual([
        {
          id: runId,
          inputResponses: [{ requestId: 'q-ssn', optionId: 'provided' }],
        },
      ]);
      row.phase = 'working';
      row.pending = [];
      await card.waitFor({ state: 'detached', timeout: 10_000 });
      expect(await page.evaluate(() => document.body.innerHTML)).not.toContain(
        SECRET,
      );
      expect(((await runStatus(page)) as Status).runId).toBe(runId);

      // (2) The tab moves to another origin; Allow; then a refresh keeps it.
      await page.getByText('6 fields so far').waitFor();
      await leaveForFrameOrigin(site, page);
      await page
        .getByRole('button', { name: 'Allow this site for this run' })
        .click();
      await expect
        .poll(
          async () => ((await runStatus(page)) as HoldStatus).pendingOrigin,
          { timeout: 5_000 },
        )
        .toBeUndefined();
      await refresh(page, runId);
      const status = (await runStatus(page)) as HoldStatus;
      expect({
        active: status.active,
        runId: status.runId,
        held: status.pendingOrigin ?? null,
      }).toEqual({ active: true, runId, held: null });
      const agentToken = mintRelayToken(
        { runId, role: 'agent', origin: fixtures.origin, sub: 'e2e-user' },
        relay.secret,
        { now: Math.floor(Date.now() / 1000) },
      );
      await closeAgentSessions();
      try {
        const cdp = cdpUrl(relay, agentToken);
        expect(
          (await agentBrowser(['snapshot', '-i'], cdp, 'reliability')).success,
        ).toBe(true);
        expect(
          (await agentBrowser(['get', 'title'], cdp, 'reliability')).success,
        ).toBe(true);
      } finally {
        await closeAgentSessions();
      }
      expect((await runStatus(page)).active).toBe(true);
      expect(stoppedFor(runId, 'left-origin')).toBe(false);

      // (4b) A later open: the run the worker holds is never stopped, and an
      // older orphan is. Both recents are past the age gate.
      const lateId = '00000000-0000-4000-8000-0000000000c1';
      runs.set(lateId, { ended: false });
      await ageRecents(3 * 60_000, [
        {
          runId: lateId,
          siteName: 'Orphan',
          startedAt: Date.now() - 10 * 60_000,
        },
      ]);
      const reopened = await browser.context.newPage();
      try {
        await reopened.goto(`chrome-extension://${EXTENSION_ID}/panel.html`);
        await reopened.getByRole('heading', { name: 'Running' }).waitFor();
        await expect
          .poll(() => recentOf(lateId), { timeout: 10_000 })
          .toBe('swept');
        expect(stopsSeen(lateId)).toBe(1);
        expect(stopsSeen(orphanId)).toBe(1);
        expect(stopsSeen(runId)).toBe(0);
        expect(runs.get(runId)?.ended).toBe(false);
        expect(await recentOf(runId)).toBe('unended');
        const held = await runStatus(reopened);
        expect({ active: held.active, runId: held.runId }).toEqual({
          active: true,
          runId,
        });
      } finally {
        await reopened.close();
      }
    } finally {
      await stopWorkerRun(page);
      await page.close();
      await site.close();
    }
  }, 120_000);

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
