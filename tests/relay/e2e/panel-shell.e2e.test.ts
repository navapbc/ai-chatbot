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
  { ended: boolean; phase?: string; pending?: unknown[] }
>();
/** Answer turns (`inputResponses` bodies) POSTed to /api/eve-chat. */
const answerBodies: unknown[] = [];
const answerOrigins: (string | undefined)[] = [];
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
    if (req.method === 'POST' && req.url === '/api/eve-chat') {
      const text = await readBody(req);
      const parsed = JSON.parse(text) as { inputResponses?: unknown };
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
      const row = runs.get(req.url.split('/')[4] ?? '');
      if (!row) {
        res.writeHead(404).end();
        return;
      }
      const alreadyEnded = row.ended;
      row.ended = true;
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
          progress: CANNED_PROGRESS,
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

  it('rebuilds Running when the panel reopens mid-run, without a second eve-chat', async () => {
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
      // The expiry is unknown after reopening, so the token is refreshed at
      // once (a same-run-id startRun) and the notice goes away.
      await expect
        .poll(
          () => appCalls.filter((c) => c === 'POST /api/relay/token').length,
          { timeout: 10_000 },
        )
        .toBe(2);
      await expect
        .poll(() => again.getByText('This run may end soon.').count(), {
          timeout: 10_000,
        })
        .toBe(0);
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
          'This one is sensitive. Type it in the page yourself, then press the button.',
        )
        .waitFor();
      // No text input for the sensitive question: only the one textarea.
      expect(await card.getByRole('textbox').count()).toBe(1);

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
      await card.getByRole('button', { name: 'I entered it myself' }).click();
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
      expect(await card.getByRole('textbox').count()).toBe(0);
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

      // A free-text question with a flagged prompt: a notice, no input, no
      // buttons, and Send never enables.
      row.pending = [
        {
          requestId: 'q-free-ssn',
          prompt: 'What is your Social Security number?',
          display: 'text',
          options: [],
          allowFreeform: true,
        },
      ];
      row.phase = 'needs_you';
      await card
        .getByText(
          'This question cannot be answered here. The agent is waiting; you can Stop the run.',
        )
        .waitFor({ timeout: 10_000 });
      expect(await card.getByRole('textbox').count()).toBe(0);
      expect(
        await card.getByRole('button', { name: 'I entered it myself' }).count(),
      ).toBe(0);
      expect(await send.isDisabled()).toBe(true);

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
      await card.getByRole('radio', { name: 'Yes' }).click({ timeout: 10_000 });
      await send.click();
      await card.getByText('Sent. Waiting for the agent...').waitFor();
      expect(answerBodies).toHaveLength(4);
      expect(answerBodies[3]).toEqual({
        id: runId,
        inputResponses: [{ requestId: 'q-benefits', optionId: 'yes' }],
      });
      row.phase = 'working';
      row.pending = [];
      await card.waitFor({ state: 'detached', timeout: 10_000 });

      // The typed answer reached no console line, the title or storage.
      expect(await page.title()).not.toContain(ANSWER);
      expect(page.url()).not.toContain(ANSWER);
      const stored = await page.evaluate(async () =>
        JSON.stringify([
          await chrome.storage.local.get(null),
          await chrome.storage.session.get(null),
        ]),
      );
      expect(stored).not.toContain(ANSWER);
      for (const line of seen) expect(line).not.toContain(ANSWER);
    } finally {
      answerRefusal = null;
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
      await page.getByText(/^Decide within \d+ seconds?$/).waitFor();
      const shown = await page.locator('body').innerText();
      expect(shown).not.toMatch(/Zelda|frag-77/);
      // Nothing allowed yet: the worker still holds.
      expect(((await runStatus(page)) as HoldStatus).pendingOrigin).toBe(
        fixtures.frameOrigin,
      );

      await page
        .getByRole('button', { name: 'Allow this site for this run' })
        .click();
      await expect
        .poll(async () => (await runStatus(page)) as HoldStatus, {
          timeout: 5_000,
        })
        .toMatchObject({ active: true, runId });
      expect(
        ((await runStatus(page)) as HoldStatus).pendingOrigin,
      ).toBeUndefined();
      // The P4c2-e note, which outlives the card.
      await page
        .getByText(
          /^Allowed\. This run can no longer be extended, so it ends at .+\.$/,
        )
        .waitFor();
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
      await page.getByText(/^Allowed\. This run/).waitFor();
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
