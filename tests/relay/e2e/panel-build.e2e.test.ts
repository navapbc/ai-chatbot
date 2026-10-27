import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { extractTokens } from '../../../scripts/extension-build-config';
import {
  EXTENSION_ID,
  type ExtensionBrowser,
  launchExtensionBrowser,
} from '../helpers/harness';

// The side panel page as Chromium loads it from the built extension: the
// placeholder renders, the app's primary color applies through the panel's
// Tailwind build and tokens, and the page raises no console error or CSP
// violation (MV3 forbids inline script). Opened as a tab: a real side panel
// cannot be driven headless. Run with
// `pnpm test:relay e2e/panel-build.e2e.test.ts`; RELAY_E2E_HEADED=1 runs it
// headed.

const ROOT = path.resolve(__dirname, '../../..');
const HEADED = process.env.RELAY_E2E_HEADED === '1';

let out: string;
let browser: ExtensionBrowser;

beforeAll(async () => {
  out = mkdtempSync(path.join(tmpdir(), 'panel-build-extension-'));
  const {
    APP_ORIGIN: _origin,
    EXTENSION_RELEASE: _release,
    EXTENSION_TEST_HOOKS: _hooks,
    ...env
  } = process.env;
  execFileSync(
    path.join(ROOT, 'node_modules/.bin/tsx'),
    ['scripts/build-extension.ts', '--out', out],
    { cwd: ROOT, env, stdio: 'ignore' },
  );
  browser = await launchExtensionBrowser({
    extensionDir: out,
    headless: !HEADED,
  });
}, 60_000);

afterAll(async () => {
  await browser?.stop();
  if (out) rmSync(out, { recursive: true, force: true });
});

describe('the side panel page', () => {
  it('renders the placeholder in the app primary color, without errors', async () => {
    const page = await browser.context.newPage();
    const errors: string[] = [];
    page.on('console', (msg) => {
      if (msg.type() === 'error' || msg.type() === 'warning') {
        errors.push(`${msg.type()}: ${msg.text()}`);
      }
    });
    page.on('pageerror', (error) => errors.push(`pageerror: ${error.message}`));

    await page.goto(`chrome-extension://${EXTENSION_ID}/panel.html`);
    const button = page.getByRole('button', { name: 'Start' });
    await button.waitFor();

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
    const actual = await button.evaluate(
      (el) => getComputedStyle(el).backgroundColor,
    );
    expect(expected).not.toBe('rgba(0, 0, 0, 0)');
    expect(actual).toBe(expected);

    // The system font stack applies (the app's next/font variables do not
    // exist here), not the browser's default serif.
    const font = await button.evaluate((el) => getComputedStyle(el).fontFamily);
    expect(font).toContain('system-ui');

    expect(errors).toEqual([]);
    await page.close();
  }, 30_000);
});
