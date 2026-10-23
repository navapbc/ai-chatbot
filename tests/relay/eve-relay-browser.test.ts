import type { ToolContext } from 'eve/tools';
import { describe, expect, it, vi } from 'vitest';
import {
  relayCredentialState,
  relayCredentialsFrom,
  runRelayCommand,
} from '@/lib/relay/eve-relay-browser';

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
    const result = await run('wss://relay.invalid.example/path');
    expect(result.error).not.toBe('The relay URL is not the configured relay.');
    expect(result.error ?? '').not.toContain(FAKE_TOKEN);
    vi.unstubAllEnvs();
  });

  it('accepts the https spelling of the configured relay, as deployed', async () => {
    // Terraform sets RELAY_PUBLIC_URL to the https service uri; sessions carry
    // the wss form the token route derives from it.
    vi.stubEnv('RELAY_PUBLIC_URL', 'https://relay.example');
    const result = await run('wss://relay.example');
    expect(result.error ?? '').not.toBe(
      'The relay URL is not the configured relay.',
    );
    expect(result.error ?? '').not.toContain(FAKE_TOKEN);
    vi.unstubAllEnvs();
  });

  it('accepts a wss spelling of RELAY_PUBLIC_URL too', async () => {
    vi.stubEnv('RELAY_PUBLIC_URL', 'wss://relay.example');
    const result = await run('wss://relay.example');
    expect(result.error ?? '').not.toBe(
      'The relay URL is not the configured relay.',
    );
    vi.unstubAllEnvs();
  });

  it('refuses a different host or a different path', async () => {
    vi.stubEnv('RELAY_PUBLIC_URL', 'https://relay.example');
    for (const url of ['wss://evil.example', 'wss://relay.example/other']) {
      const result = await run(url);
      expect(result).toMatchObject({
        success: false,
        error: 'The relay URL is not the configured relay.',
      });
      expect(result.error).not.toContain(FAKE_TOKEN);
    }
    vi.unstubAllEnvs();
  });

  it('refuses when RELAY_PUBLIC_URL itself is unusable', async () => {
    vi.stubEnv('RELAY_PUBLIC_URL', 'http://relay.example');
    const result = await run('wss://relay.example');
    expect(result).toMatchObject({
      success: false,
      error: 'The relay is not configured correctly.',
    });
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

describe('check_submit_gate routing', () => {
  const load = async () => {
    vi.resetModules();
    const runBrowserCommand = vi.fn().mockResolvedValue({
      success: false,
      error: 'kernel path ran',
      liveViewUrl: null,
    });
    vi.doMock('@/lib/kernel/eve-browser', () => ({ runBrowserCommand }));
    const { default: tool } = await import('@/agent/tools/check_submit_gate');
    const execute = (c: ToolContext) =>
      (
        tool as unknown as {
          execute: (i: unknown, c: ToolContext) => Promise<unknown>;
        }
      ).execute({ forceEnable: true }, c);
    return { runBrowserCommand, execute };
  };

  it.each([
    ['complete', { relayUrl: 'wss://relay.example', relayToken: FAKE_TOKEN }],
    ['partial', { relayToken: FAKE_TOKEN }],
  ])('refuses on %s relay credentials without a browser', async (_n, attrs) => {
    const { runBrowserCommand, execute } = await load();
    const result = await execute(ctxWith(principal(attrs)));
    expect(result).toEqual({
      success: false,
      error: 'check_submit_gate is not available in extension runs',
      state: null,
      action: null,
    });
    expect(JSON.stringify(result)).not.toContain(FAKE_TOKEN);
    expect(runBrowserCommand).not.toHaveBeenCalled();
    vi.doUnmock('@/lib/kernel/eve-browser');
  });

  it('still runs the Kernel path without relay credentials', async () => {
    const { runBrowserCommand, execute } = await load();
    const ctx = ctxWith(principal({}));
    const result = await execute(ctx);
    expect(runBrowserCommand).toHaveBeenCalledTimes(1);
    expect(runBrowserCommand.mock.calls[0][0]).toBe(ctx);
    expect(result).toMatchObject({ success: false, error: 'kernel path ran' });
    vi.doUnmock('@/lib/kernel/eve-browser');
  });
});
