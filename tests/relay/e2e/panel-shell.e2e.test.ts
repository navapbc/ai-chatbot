import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { createServer, type Server } from 'node:http';
import type { AddressInfo } from 'node:net';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { extractTokens } from '../../../scripts/extension-build-config';
import {
  EXTENSION_ID,
  type ExtensionBrowser,
  launchExtensionBrowser,
} from '../helpers/harness';

// The panel shell against a fake app: GET /api/relay/config answers 401 first
// (signed out), then 200. The signed-out screen shows with the primary color
// applied, and Try again leads to the Home placeholder. Opened as a tab: a
// real side panel cannot be driven headless. Run with
// `pnpm test:relay e2e/panel-shell.e2e.test.ts`; RELAY_E2E_HEADED=1 runs it
// headed.

const ROOT = path.resolve(__dirname, '../../..');
const HEADED = process.env.RELAY_E2E_HEADED === '1';

let out: string;
let browser: ExtensionBrowser;
let app: Server;
let configCalls = 0;

beforeAll(async () => {
  app = createServer((req, res) => {
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
          models: [{ id: 'claude-opus-5' }],
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
  app?.close();
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
    await page.close();
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
