import type { ToolContext } from 'eve/tools';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

// Separate from eve-relay-browser.test.ts: that file drives the real
// agent-browser CLI, this one mocks the transport to inspect what it receives.
const runCommand = vi.fn();
vi.mock('@/lib/kernel/cli', () => ({ runCommand }));

const { _resetAgentTokenCache } = await import('@/lib/relay/agent-token');
const { runRelayCommand } = await import('@/lib/relay/eve-relay-browser');

const RUN = '0b6f7c4e-5d1a-4c55-9a53-3f7d2f0c9a11';
const TOKEN = 'FETCHED+token/with=specials&more';
const LEGACY = 'SENTINEL-LEGACY-TOKEN';
const SECRET = 'sekret-value';

const ctxWith = (attributes: Record<string, string>) =>
  ({
    session: {
      id: 'wrun_relay_run',
      auth: {
        initiator: {
          attributes,
          authenticator: 't',
          principalId: 'p',
          principalType: 'user',
          subject: 's',
        },
        current: null,
      },
    },
  }) as unknown as ToolContext;

const fetchMock = vi.fn();
const ctx = () =>
  ctxWith({
    relayUrl: 'ws://127.0.0.1:9',
    relayRunId: RUN,
    relayToken: LEGACY,
  });

beforeEach(() => {
  _resetAgentTokenCache();
  runCommand.mockReset();
  fetchMock.mockReset();
  vi.stubEnv('EVE_INTERNAL_SECRET', SECRET);
  vi.stubEnv('RELAY_PUBLIC_URL', '');
  fetchMock.mockImplementation(
    async () =>
      new Response(JSON.stringify({ token: TOKEN, expiresAt: 4_000_000_000 }), {
        status: 200,
      }),
  );
  vi.stubGlobal('fetch', fetchMock);
});
afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

describe('runRelayCommand transport', () => {
  it('gives the CLI the fetched token, never a legacy attribute token', async () => {
    runCommand.mockResolvedValue({ success: true, data: 'ok' });
    const result = await runRelayCommand(ctx(), ['snapshot']);
    expect(result).toMatchObject({ success: true, data: 'ok' });
    const opts = runCommand.mock.calls[0][1] as {
      cdpUrl: string;
      cdpViaEnv: boolean;
      redact: string[];
    };
    expect(opts.cdpUrl).toBe(
      `ws://127.0.0.1:9/cdp?token=${encodeURIComponent(TOKEN)}`,
    );
    expect(opts.cdpUrl).not.toContain(LEGACY);
    expect(opts.cdpViaEnv).toBe(true);
    expect(opts.redact).toEqual([TOKEN, opts.cdpUrl]);
    expect(JSON.stringify(runCommand.mock.calls)).not.toContain(LEGACY);
  });

  it('redacts the token and URL, raw and encoded, from a thrown error', async () => {
    runCommand.mockImplementation(
      async (_c: unknown, o: { cdpUrl: string }) => {
        throw new Error(
          `failed ${TOKEN} ${encodeURIComponent(TOKEN)} ${o.cdpUrl}`,
        );
      },
    );
    const result = await runRelayCommand(ctx(), ['snapshot']);
    expect(result.success).toBe(false);
    const text = JSON.stringify(result);
    expect(text).not.toContain(TOKEN);
    expect(text).not.toContain(encodeURIComponent(TOKEN));
    expect(text).not.toContain('/cdp?token=');
    expect(text).not.toContain(LEGACY);
  });

  it('fails closed without spawning when EVE_INTERNAL_SECRET is missing', async () => {
    vi.stubEnv('EVE_INTERNAL_SECRET', '');
    const result = await runRelayCommand(ctx(), ['snapshot']);
    expect(result).toEqual({
      success: false,
      error: 'The relay token service is unavailable.',
      liveViewUrl: null,
    });
    expect(runCommand).not.toHaveBeenCalled();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('does not fetch a token for a refused relay URL', async () => {
    vi.stubEnv('RELAY_PUBLIC_URL', 'wss://relay.example');
    const result = await runRelayCommand(
      ctxWith({ relayUrl: 'wss://evil.example', relayRunId: RUN }),
      ['snapshot'],
    );
    expect(result.error).toBe('The relay URL is not the configured relay.');
    expect(fetchMock).not.toHaveBeenCalled();
    expect(runCommand).not.toHaveBeenCalled();
  });

  it('returns a fixed failure when the token cannot be encoded', async () => {
    fetchMock.mockImplementation(
      async () =>
        new Response(
          JSON.stringify({ token: '\ud800lone', expiresAt: 4_000_000_000 }),
          { status: 200 },
        ),
    );
    const result = await runRelayCommand(ctx(), ['snapshot']);
    expect(result).toEqual({
      success: false,
      error: 'The relay token could not be used.',
      liveViewUrl: null,
    });
    expect(runCommand).not.toHaveBeenCalled();
  });
});
