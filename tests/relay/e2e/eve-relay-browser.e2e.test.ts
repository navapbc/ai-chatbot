import { execFileSync } from 'node:child_process';
import type { ToolContext } from 'eve/tools';
import type { Page } from 'playwright';
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest';
import { runRelayCommand } from '@/lib/relay/eve-relay-browser';
import {
  bridgeMessage,
  type ExtensionBrowser,
  type Fixtures,
  launchExtensionBrowser,
  mintRunTokens,
  type Relay,
  serveFixtures,
  startRelay,
} from '../helpers/harness';

type Principal = {
  attributes: Record<string, string | readonly string[]>;
  authenticator: string;
  principalId: string;
  principalType: string;
  subject?: string;
};

const principal = (attributes: Principal['attributes']): Principal => ({
  attributes,
  authenticator: 'test',
  principalId: 'p',
  principalType: 'user',
  subject: 's',
});

const ctxWith = (
  initiator: Principal | null,
  current: Principal | null = null,
  id = 'wrun_relay_1',
) =>
  ({ session: { id, auth: { initiator, current } } }) as unknown as ToolContext;

describe('runRelayCommand end to end', { timeout: 60_000 }, () => {
  let fixtures: Fixtures;
  let relay: Relay;
  let browser: ExtensionBrowser;
  let page: Page;
  let ctx: ToolContext;
  let agentToken: string;
  let priorSecret: string | undefined;

  beforeAll(async () => {
    process.env.AGENT_BROWSER_NAMESPACE = `eve-relay-${process.pid}`;
    fixtures = await serveFixtures();
    relay = await startRelay();
    browser = await launchExtensionBrowser();
    page = await browser.context.newPage();
    await page.goto(fixtures.formUrl);
    const tabId = (await browser.sw.evaluate(
      async (u) => (await chrome.tabs.query({ url: u }))[0]?.id,
      fixtures.formUrl,
    )) as number;
    const tokens = mintRunTokens(relay.secret, fixtures.origin);
    const registered = relay.logs.length;
    const started = await bridgeMessage(browser.sw, {
      type: 'startRun',
      relayUrl: relay.url,
      token: tokens.extension,
      tabId,
      allowedFrameOrigins: [fixtures.frameOrigin],
    });
    expect(started).toEqual({ ok: true });
    const deadline = Date.now() + 10_000;
    while (
      !relay.logs
        .slice(registered)
        .some((l) => l.event === 'registered' && l.runId === tokens.runId)
    ) {
      if (Date.now() > deadline) throw new Error('run did not register');
      await new Promise((r) => setTimeout(r, 50));
    }
    // The session carries only the run reference; the tool fetches the token
    // from the app's agent-token route, which this stands in for.
    agentToken = tokens.agent;
    priorSecret = process.env.EVE_INTERNAL_SECRET;
    process.env.EVE_INTERNAL_SECRET = 'e2e-internal-secret';
    // Like the route, it mints only for the run's own Eve session.
    vi.stubGlobal('fetch', async (_url: string, init: RequestInit) => {
      const { runId, sessionId } = JSON.parse(String(init.body));
      if (runId !== tokens.runId || sessionId !== 'wrun_relay_e2e') {
        return new Response(null, { status: 404 });
      }
      return Response.json({ token: tokens.agent, expiresAt: 4_000_000_000 });
    });
    ctx = ctxWith(
      principal({ relayUrl: `${relay.url}/`, relayRunId: tokens.runId }),
      null,
      'wrun_relay_e2e',
    );
  }, 60_000);

  afterAll(async () => {
    vi.unstubAllGlobals();
    if (priorSecret === undefined)
      Reflect.deleteProperty(process.env, 'EVE_INTERNAL_SECRET');
    else process.env.EVE_INTERNAL_SECRET = priorSecret;
    await runRelayCommand(ctx, ['close']).catch(() => {});
    await bridgeMessage(browser.sw, { type: 'stopRun' }).catch(() => {});
    await browser?.stop();
    await relay?.stop();
    await fixtures?.stop();
  }, 60_000);

  it('snapshots and fills a fixture field through the relay', async () => {
    const snap = await runRelayCommand(ctx, ['snapshot', '-i']);
    expect(snap.success).toBe(true);
    expect(snap.liveViewUrl).toBeNull();
    const refs = (
      snap.data as { refs: Record<string, { role: string; name: string }> }
    ).refs;
    const hit = Object.entries(refs).find(
      ([, r]) => r.role === 'textbox' && r.name.trim() === 'First name',
    );
    expect(hit).toBeDefined();

    const fill = await runRelayCommand(ctx, ['fill', `@${hit?.[0]}`, 'Ada']);
    expect(fill.success).toBe(true);
    await expect(page.locator('#first').inputValue()).resolves.toBe('Ada');
  });

  it('keeps the relay token out of every process command line', () => {
    const token = agentToken;
    const processes = execFileSync('ps', ['-axww', '-o', 'command='], {
      encoding: 'utf8',
    });
    expect(processes).not.toContain(token);
  });
});
