import type { Page } from 'playwright';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import {
  type AgentResult,
  agentBrowser,
  bridgeMessage,
  cdpUrl,
  closeAgentSessions,
  type ExtensionBrowser,
  type ExternalRelay,
  externalRelay,
  type Fixtures,
  launchExtensionBrowser,
  mintRunTokens,
  serveFixtures,
} from '../helpers/harness';

// Smoke test against a relay running outside this process (the relay image).
// Skipped unless both variables are set:
//
//   docker run -d --rm --name relay-smoke -p 127.0.0.1:8080:8080 \
//     -e PORT=8080 -e RELAY_TOKEN_SECRET=<32+ chars> \
//     -e RELAY_ALLOWED_HOSTS=127.0.0.1,localhost \
//     -e RELAY_EXTENSION_ID="$(cat extension/EXTENSION_ID)" relay:local
//   RELAY_E2E_URL=http://127.0.0.1:8080 RELAY_E2E_SECRET=<same> \
//     pnpm test:relay container-smoke
//
// The relay's logs are not reachable from here, so nothing below waits on or
// asserts against them.

const URL_ENV = process.env.RELAY_E2E_URL;
const SECRET_ENV = process.env.RELAY_E2E_SECRET;

/** Retries an agent command until it succeeds (the extension may still be registering). */
async function untilOk(run: () => Promise<AgentResult>, ms = 15_000) {
  const deadline = Date.now() + ms;
  let last: AgentResult = await run();
  while (!last.success && Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 250));
    last = await run();
  }
  return last;
}

describe.skipIf(!URL_ENV || !SECRET_ENV)(
  'relay container smoke',
  { timeout: 60_000 },
  () => {
    let relay: ExternalRelay;
    let fixtures: Fixtures;
    let browser: ExtensionBrowser;
    let page: Page;
    let tabId: number;

    beforeAll(async () => {
      relay = externalRelay(URL_ENV as string, SECRET_ENV as string);
      fixtures = await serveFixtures();
      browser = await launchExtensionBrowser();
      page = await browser.context.newPage();
      await page.goto(fixtures.formUrl);
      tabId = (await browser.sw.evaluate(
        async (u) => (await chrome.tabs.query({ url: u }))[0]?.id,
        fixtures.formUrl,
      )) as number;
    }, 60_000);

    afterAll(async () => {
      await closeAgentSessions();
      await bridgeMessage(browser?.sw, { type: 'stopRun' }).catch(() => {});
      await browser?.stop();
      await fixtures?.stop();
    }, 60_000);

    it('drives a run through the external relay and stops it', async () => {
      const tokens = mintRunTokens(relay.secret, fixtures.origin);
      const started = await bridgeMessage(browser.sw, {
        type: 'startRun',
        relayUrl: relay.url,
        token: tokens.extension,
        tabId,
        allowedFrameOrigins: [fixtures.frameOrigin],
      });
      expect(started).toEqual({ ok: true });
      const cdp = cdpUrl(relay, tokens.agent);

      const snap = await untilOk(() =>
        agentBrowser(['snapshot', '-i'], cdp, 'smoke'),
      );
      expect(snap.success).toBe(true);
      const text = (snap.data as { snapshot: string }).snapshot;
      expect(text).toContain('textbox "First name"');

      const refs = (
        snap.data as { refs: Record<string, { role: string; name: string }> }
      ).refs;
      const first = Object.entries(refs).find(
        ([, r]) => r.role === 'textbox' && r.name.trim() === 'First name',
      );
      if (!first) throw new Error('no textbox "First name" in snapshot');
      const fill = await agentBrowser(
        ['fill', `@${first[0]}`, 'Ada'],
        cdp,
        'smoke',
      );
      expect(fill.success).toBe(true);
      expect(
        await page.$eval('#first', (e) => (e as HTMLInputElement).value),
      ).toBe('Ada');

      const stopping = Date.now();
      expect(await bridgeMessage(browser.sw, { type: 'stopRun' })).toEqual({
        ok: true,
      });
      const next = await agentBrowser(['get', 'title'], cdp, 'smoke');
      expect(next.success).toBe(false);
      expect(Date.now() - stopping).toBeLessThan(5_000);
    });
  },
);
