import { execFileSync } from 'node:child_process';
import type { ToolContext } from 'eve/tools';
import type { Page } from 'playwright';
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest';
import {
  relayCredentialState,
  relayCredentialsFrom,
  runRelayCommand,
} from '@/lib/relay/eve-relay-browser';
import {
  bridgeMessage,
  type ExtensionBrowser,
  type Fixtures,
  launchExtensionBrowser,
  mintRunTokens,
  type Relay,
  serveFixtures,
  startRelay,
} from './helpers/harness';

const FAKE_TOKEN = 'FAKE-TOKEN-do-not-use-0123456789abcdef';

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

describe('relayCredentialsFrom', () => {
  it('returns null with no attributes', () => {
    expect(relayCredentialsFrom(ctxWith(null))).toBeNull();
    expect(relayCredentialsFrom(ctxWith(principal({})))).toBeNull();
  });

  it('returns the credentials when both attributes are present', () => {
    const ctx = ctxWith(
      principal({ relayUrl: 'wss://relay.example', relayToken: FAKE_TOKEN }),
    );
    expect(relayCredentialsFrom(ctx)).toEqual({
      url: 'wss://relay.example',
      token: FAKE_TOKEN,
    });
  });

  it('prefers the initiator over the current caller', () => {
    const ctx = ctxWith(
      principal({ relayUrl: 'wss://a.example', relayToken: 'initiator' }),
      principal({ relayUrl: 'wss://b.example', relayToken: 'current' }),
    );
    expect(relayCredentialsFrom(ctx)?.token).toBe('initiator');
  });

  it('falls back to the current caller when the initiator has none', () => {
    const ctx = ctxWith(
      principal({}),
      principal({ relayUrl: 'wss://b.example', relayToken: 'current' }),
    );
    expect(relayCredentialsFrom(ctx)?.token).toBe('current');
  });

  it('returns null when only one of the two is set', () => {
    expect(
      relayCredentialsFrom(ctxWith(principal({ relayUrl: 'wss://a.example' }))),
    ).toBeNull();
    expect(
      relayCredentialsFrom(ctxWith(principal({ relayToken: FAKE_TOKEN }))),
    ).toBeNull();
  });
});

describe('relayCredentialState', () => {
  it('distinguishes none, complete and partial', () => {
    expect(relayCredentialState(ctxWith(null)).kind).toBe('none');
    expect(
      relayCredentialState(
        ctxWith(principal({ relayUrl: 'wss://a.example', relayToken: 't' })),
      ).kind,
    ).toBe('complete');
    expect(
      relayCredentialState(ctxWith(principal({ relayToken: 't' }))).kind,
    ).toBe('partial');
    expect(
      relayCredentialState(ctxWith(principal({ relayUrl: 'wss://a.example' })))
        .kind,
    ).toBe('partial');
  });
});

describe('relay URL allowlist', () => {
  const run = (url: string) =>
    runRelayCommand(
      ctxWith(principal({ relayUrl: url, relayToken: FAKE_TOKEN })),
      ['get', 'url'],
    );

  it('refuses another origin when RELAY_PUBLIC_URL is set', async () => {
    vi.stubEnv('RELAY_PUBLIC_URL', 'wss://relay.example');
    const result = await run('wss://evil.example');
    expect(result).toMatchObject({
      success: false,
      error: 'The relay URL is not the configured relay.',
    });
    expect(result.error).not.toContain(FAKE_TOKEN);
    vi.unstubAllEnvs();
  });

  it('accepts the configured origin (fails later, at connect, not at policy)', async () => {
    vi.stubEnv('RELAY_PUBLIC_URL', 'wss://relay.invalid.example/path');
    const result = await run('wss://relay.invalid.example');
    expect(result.error).not.toBe('The relay URL is not the configured relay.');
    expect(result.error ?? '').not.toContain(FAKE_TOKEN);
    vi.unstubAllEnvs();
  });

  it('accepts only loopback ws: when it is unset', async () => {
    vi.stubEnv('RELAY_PUBLIC_URL', '');
    expect((await run('ws://relay.example')).error).toMatch(/must use wss/);
    expect((await run('ws://127.0.0.1:9')).error ?? '').not.toMatch(
      /must use wss/,
    );
    vi.unstubAllEnvs();
  });
});

describe('runRelayCommand failures', () => {
  const cases: Array<[string, string]> = [
    ['an insecure remote ws: url', 'ws://relay.example.com'],
    ['a non-websocket url', 'https://relay.example.com'],
    ['an unparseable url', 'not a url'],
  ];
  it.each(cases)('refuses %s without echoing the token', async (_n, url) => {
    const ctx = ctxWith(principal({ relayUrl: url, relayToken: FAKE_TOKEN }));
    const result = await runRelayCommand(ctx, ['snapshot']);
    expect(result.success).toBe(false);
    expect(result.liveViewUrl).toBeNull();
    expect(result.error).toBeTruthy();
    expect(result.error).not.toContain(FAKE_TOKEN);
  });

  it('fails cleanly when the context carries no credentials', async () => {
    const result = await runRelayCommand(ctxWith(null), ['snapshot']);
    expect(result.success).toBe(false);
    expect(result.error).toBeTruthy();
  });
});

describe('browser tool routing', () => {
  it('fails closed on partial credentials without touching the Kernel path', async () => {
    vi.resetModules();
    const runBrowserCommand = vi.fn();
    vi.doMock('@/lib/kernel/eve-browser', () => ({ runBrowserCommand }));
    const { default: tool } = await import('@/agent/tools/browser');
    const result = await (
      tool as unknown as {
        execute: (i: unknown, c: ToolContext) => Promise<unknown>;
      }
    ).execute(
      { command: ['snapshot'] },
      ctxWith(principal({ relayToken: FAKE_TOKEN })),
    );
    expect(result).toMatchObject({
      success: false,
      error: 'relay credentials incomplete',
    });
    expect(JSON.stringify(result)).not.toContain(FAKE_TOKEN);
    expect(runBrowserCommand).not.toHaveBeenCalled();
    vi.doUnmock('@/lib/kernel/eve-browser');
  });

  it('uses the Kernel path when no relay credentials are present', async () => {
    vi.resetModules();
    const runBrowserCommand = vi.fn().mockResolvedValue({
      success: true,
      data: 'ok',
      liveViewUrl: null,
    });
    vi.doMock('@/lib/kernel/eve-browser', () => ({ runBrowserCommand }));
    const { default: tool } = await import('@/agent/tools/browser');
    const ctx = ctxWith(principal({}));
    const result = await (
      tool as unknown as {
        execute: (i: unknown, c: ToolContext) => Promise<{ success: boolean }>;
      }
    ).execute({ command: ['snapshot'] }, ctx);
    expect(runBrowserCommand).toHaveBeenCalledWith(ctx, ['snapshot']);
    expect(result.success).toBe(true);
    vi.doUnmock('@/lib/kernel/eve-browser');
  });
});

describe('runRelayCommand end to end', { timeout: 60_000 }, () => {
  let fixtures: Fixtures;
  let relay: Relay;
  let browser: ExtensionBrowser;
  let page: Page;
  let ctx: ToolContext;

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
    ctx = ctxWith(
      principal({ relayUrl: `${relay.url}/`, relayToken: tokens.agent }),
      null,
      'wrun_relay_e2e',
    );
  }, 60_000);

  afterAll(async () => {
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
    const token = (
      ctx.session.auth.initiator?.attributes as Record<string, string>
    ).relayToken;
    const processes = execFileSync('ps', ['-axww', '-o', 'command='], {
      encoding: 'utf8',
    });
    expect(processes).not.toContain(token);
  });
});
