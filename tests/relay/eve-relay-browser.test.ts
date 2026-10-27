import { execFileSync } from 'node:child_process';
import type { ToolContext } from 'eve/tools';
import {
  afterAll,
  afterEach,
  beforeAll,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from 'vitest';
import { _resetAgentTokenCache } from '@/lib/relay/agent-token';
import {
  relayCredentialState,
  relayRunRefFrom,
  runRelayCommand,
} from '@/lib/relay/eve-relay-browser';

const FAKE_TOKEN = 'FAKE-TOKEN-do-not-use-0123456789abcdef';
const RUN = '0b6f7c4e-5d1a-4c55-9a53-3f7d2f0c9a11';

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

describe('relayRunRefFrom', () => {
  it('returns null with no attributes', () => {
    expect(relayRunRefFrom(ctxWith(null))).toBeNull();
    expect(relayRunRefFrom(ctxWith(principal({})))).toBeNull();
  });

  it('returns the run reference when both attributes are present', () => {
    const ctx = ctxWith(
      principal({ relayUrl: 'wss://relay.example', relayRunId: RUN }),
    );
    expect(relayRunRefFrom(ctx)).toEqual({
      url: 'wss://relay.example',
      runId: RUN,
    });
  });

  it('prefers the initiator over the current caller', () => {
    const ctx = ctxWith(
      principal({ relayUrl: 'wss://a.example', relayRunId: 'initiator' }),
      principal({ relayUrl: 'wss://b.example', relayRunId: 'current' }),
    );
    expect(relayRunRefFrom(ctx)?.runId).toBe('initiator');
  });

  it('falls back to the current caller when the initiator has none', () => {
    const ctx = ctxWith(
      principal({}),
      principal({ relayUrl: 'wss://b.example', relayRunId: 'current' }),
    );
    expect(relayRunRefFrom(ctx)?.runId).toBe('current');
  });

  it('returns null when only one of the two is set', () => {
    expect(
      relayRunRefFrom(ctxWith(principal({ relayUrl: 'wss://a.example' }))),
    ).toBeNull();
    expect(relayRunRefFrom(ctxWith(principal({ relayRunId: RUN })))).toBeNull();
  });
});

describe('relayCredentialState', () => {
  it('distinguishes none, complete and partial', () => {
    expect(relayCredentialState(ctxWith(null)).kind).toBe('none');
    expect(
      relayCredentialState(
        ctxWith(principal({ relayUrl: 'wss://a.example', relayRunId: 't' })),
      ).kind,
    ).toBe('complete');
    expect(
      relayCredentialState(ctxWith(principal({ relayRunId: 't' }))).kind,
    ).toBe('partial');
    expect(
      relayCredentialState(ctxWith(principal({ relayUrl: 'wss://a.example' })))
        .kind,
    ).toBe('partial');
  });
});

// Tests below that reach the real CLI (against a relay that is not there) leave
// an agent-browser daemon behind. A private namespace scopes `close --all` to
// the daemons this file started.
const NAMESPACE = `relay-unit-${process.pid}`;
beforeAll(() => {
  process.env.AGENT_BROWSER_NAMESPACE = NAMESPACE;
});
afterAll(() => {
  try {
    execFileSync('agent-browser', ['close', '--all'], {
      env: process.env,
      stdio: 'ignore',
      timeout: 15_000,
    });
  } catch {
    // No daemon to close.
  }
  Reflect.deleteProperty(process.env, 'AGENT_BROWSER_NAMESPACE');
});

const fetchMock = vi.fn();
beforeEach(() => {
  _resetAgentTokenCache();
  vi.stubEnv(
    'EVE_INTERNAL_SECRET',
    'sekret-value-padded-to-be-over-thirty-two-chars',
  );
  fetchMock.mockReset();
  fetchMock.mockImplementation(
    async () =>
      new Response(
        JSON.stringify({ token: FAKE_TOKEN, expiresAt: 4_000_000_000 }),
        { status: 200 },
      ),
  );
  vi.stubGlobal('fetch', fetchMock);
});
afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

describe('relay URL allowlist', () => {
  const run = (url: string) =>
    runRelayCommand(ctxWith(principal({ relayUrl: url, relayRunId: RUN })), [
      'get',
      'url',
    ]);

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
    const ctx = ctxWith(principal({ relayUrl: url, relayRunId: RUN }));
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

describe('runRelayCommand with a fetched token', () => {
  it('fails without starting agent-browser when the token fetch fails', async () => {
    fetchMock.mockResolvedValue(new Response('no', { status: 500 }));
    const result = await runRelayCommand(
      ctxWith(principal({ relayUrl: 'ws://127.0.0.1:9', relayRunId: RUN })),
      ['snapshot'],
    );
    expect(result).toMatchObject({ success: false, liveViewUrl: null });
    expect(result.error).toBe('The relay token request was refused.');
    expect(JSON.stringify(result)).not.toContain(FAKE_TOKEN);
    expect(JSON.stringify(result)).not.toContain(
      'sekret-value-padded-to-be-over-thirty-two-chars',
    );
    vi.unstubAllEnvs();
  });

  it('ignores a legacy relayToken attribute and the retired header value', async () => {
    const SENTINEL = 'SENTINEL-LEGACY-TOKEN';
    const result = await runRelayCommand(
      ctxWith(
        principal({
          relayUrl: 'ws://127.0.0.1:9',
          relayRunId: RUN,
          relayToken: SENTINEL,
        }),
      ),
      ['snapshot'],
    );
    expect(JSON.stringify(result)).not.toContain(SENTINEL);
    expect(JSON.stringify(result)).not.toContain(FAKE_TOKEN);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('redacts the fetched token from a failing command', async () => {
    const result = await runRelayCommand(
      ctxWith(principal({ relayUrl: 'ws://127.0.0.1:9', relayRunId: RUN })),
      ['snapshot'],
    );
    expect(result.success).toBe(false);
    expect(JSON.stringify(result)).not.toContain(FAKE_TOKEN);
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
      ctxWith(principal({ relayRunId: RUN })),
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
    ['complete', { relayUrl: 'wss://relay.example', relayRunId: RUN }],
    ['partial', { relayRunId: RUN }],
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

describe('recovery after an extension restart', () => {
  const RESTORED =
    'The browser connection was restored. Take a new snapshot before continuing.';
  const CLOSED = 'CDP response channel closed';

  let seq = 0;
  const load = async (
    answer: (command: readonly string[], session: string) => unknown,
  ) => {
    vi.resetModules();
    const calls: Array<{ command: string[]; session: string; cdp?: string }> =
      [];
    const runCommand = vi.fn(
      async (
        command: readonly string[],
        options: { session: string; cdpUrl?: string },
      ) => {
        calls.push({
          command: [...command],
          session: options.session,
          cdp: options.cdpUrl,
        });
        return answer(command, options.session);
      },
    );
    vi.doMock('@/lib/kernel/cli', () => ({ runCommand }));
    const mod = await import('@/lib/relay/eve-relay-browser');
    const { cliSessionName } = await import('@/lib/kernel/session-store');
    seq += 1;
    const ctx = ctxWith(
      principal({ relayUrl: 'ws://127.0.0.1:9', relayRunId: `${RUN}-${seq}` }),
      null,
      `wrun_restart_${seq}`,
    );
    const base = cliSessionName('s', `wrun_restart_${seq}`);
    // Calls that ran a browser command, not the old session's close.
    const ran = () => calls.filter((c) => c.command[0] !== 'close');
    return { run: mod.runRelayCommand, ctx, base, calls, ran };
  };
  afterEach(() => {
    vi.doUnmock('@/lib/kernel/cli');
  });

  const ok = (data: unknown = 'snap') => ({ success: true, data, error: null });
  const fail = (error: string) => ({ success: false, data: null, error });

  it('does not retry a ref command and tells the agent to snapshot again', async () => {
    const { run, ctx, base, ran } = await load(() => fail(CLOSED));
    const result = await run(ctx, ['click', '@e3']);
    expect(result).toEqual({
      success: false,
      error: RESTORED,
      liveViewUrl: null,
    });
    expect(ran()).toEqual([
      { command: ['click', '@e3'], session: base, cdp: expect.any(String) },
    ]);
  });

  it('retries a ref-free command once on the next session name', async () => {
    let first = true;
    const { run, ctx, base, ran, calls } = await load((command) => {
      if (command[0] === 'close') return ok();
      if (first) {
        first = false;
        return fail('CDP error (DOM.enable): not attached');
      }
      return ok('fresh snapshot');
    });
    const result = await run(ctx, ['snapshot', '-i']);
    expect(result).toEqual({
      success: true,
      data: 'fresh snapshot',
      error: null,
      liveViewUrl: null,
    });
    expect(ran().map((c) => c.session)).toEqual([base, `${base}-r1`]);
    // The old daemon session is closed before the retry runs.
    expect(calls.map((c) => [c.command[0], c.session])).toEqual([
      ['snapshot', base],
      ['close', base],
      ['snapshot', `${base}-r1`],
    ]);
  });

  it('retries only once', async () => {
    const { run, ctx, ran } = await load((command) =>
      command[0] === 'close' ? ok() : fail(CLOSED),
    );
    const result = await run(ctx, ['snapshot', '-i']);
    expect(result).toMatchObject({ success: false, error: CLOSED });
    expect(ran()).toHaveLength(2);
  });

  it('does not retry a failure that is not a lost connection', async () => {
    const { run, ctx, base, ran } = await load(() =>
      fail('Unknown ref: e1 (element not found)'),
    );
    const result = await run(ctx, ['snapshot', '-i']);
    expect(result).toMatchObject({ success: false });
    await run(ctx, ['get', 'url']);
    expect(ran().map((c) => c.session)).toEqual([base, base]);
  });

  it('treats a ref in a frame command as a ref command', async () => {
    const { run, ctx, ran } = await load((command) =>
      command[0] === 'close' ? ok() : fail(CLOSED),
    );
    expect((await run(ctx, ['click', '@e2'])).error).toBe(RESTORED);
    expect((await run(ctx, ['frame', '@e7'])).error).toBe(RESTORED);
    // The click failed and rotated; the frame command never ran.
    expect(ran()).toHaveLength(1);
  });

  it('advances the session name once per restart', async () => {
    let down = true;
    const { run, ctx, base, ran } = await load((command, session) => {
      if (command[0] === 'close') return ok();
      if (down) {
        down = false;
        return fail(CLOSED);
      }
      return ok(session);
    });
    await run(ctx, ['snapshot', '-i']);
    await run(ctx, ['get', 'url']);
    down = true;
    await run(ctx, ['snapshot', '-i']);
    expect(ran().map((c) => c.session)).toEqual([
      base,
      `${base}-r1`,
      `${base}-r1`,
      `${base}-r1`,
      `${base}-r2`,
    ]);
  });

  it('rotates once when two commands are in flight and refuses the queued ref command', async () => {
    let down = true;
    const { run, ctx, base, ran } = await load((command) => {
      if (command[0] === 'close') return ok();
      if (down) {
        down = false;
        return fail('CDP error (Page.enable): tab or extension disconnected');
      }
      return ok();
    });
    const [snap, click, url] = await Promise.all([
      run(ctx, ['snapshot', '-i']),
      run(ctx, ['click', '@e5']),
      run(ctx, ['get', 'url']),
    ]);
    expect(snap.success).toBe(true);
    expect(click).toEqual({
      success: false,
      error: RESTORED,
      liveViewUrl: null,
    });
    expect(url.success).toBe(true);
    expect(ran().map((c) => [c.command[0], c.session])).toEqual([
      ['snapshot', base],
      ['snapshot', `${base}-r1`],
      ['get', `${base}-r1`],
    ]);
  });

  it('refuses ref commands after a rotation until a snapshot succeeds', async () => {
    let down = true;
    const { run, ctx, base, ran } = await load((command) => {
      if (command[0] === 'close') return ok();
      if (down) {
        down = false;
        return fail(CLOSED);
      }
      return ok();
    });
    // A read-only command detects the restart and is retried on -r1.
    expect((await run(ctx, ['get', 'url'])).success).toBe(true);
    // Its refs came from the old daemon: refused without running.
    expect((await run(ctx, ['fill', '@e3', 'Ada'])).error).toBe(RESTORED);
    expect((await run(ctx, ['get', 'url'])).success).toBe(true);
    expect((await run(ctx, ['click', '@e3'])).error).toBe(RESTORED);
    expect((await run(ctx, ['snapshot', '-i'])).success).toBe(true);
    expect((await run(ctx, ['fill', '@e3', 'Ada'])).success).toBe(true);
    expect(ran().map((c) => [c.command[0], c.session])).toEqual([
      ['get', base],
      ['get', `${base}-r1`],
      ['get', `${base}-r1`],
      ['snapshot', `${base}-r1`],
      ['fill', `${base}-r1`],
    ]);
  });

  it.each([
    [['press', 'Enter']],
    [['open', 'https://benefits.example.gov/next']],
    [['click', '@e9']],
    [['back']],
    [['frame', 'main']],
    [['hover', '@e3']],
  ])('does not run %j again after a lost connection', async (command) => {
    const { run, ctx, base, ran } = await load((argv) =>
      argv[0] === 'close' ? ok() : fail(CLOSED),
    );
    expect(await run(ctx, command)).toEqual({
      success: false,
      error: RESTORED,
      liveViewUrl: null,
    });
    expect(ran()).toEqual([
      { command, session: base, cdp: expect.any(String) },
    ]);
  });

  it.each([
    [['snapshot', '-i']],
    [['screenshot']],
    [['get', 'title']],
    [['get', 'url']],
    [['tab']],
    [['tab', 'list']],
  ])('retries the read-only %j once', async (command) => {
    let first = true;
    const { run, ctx, base, ran } = await load((argv) => {
      if (argv[0] === 'close') return ok();
      if (first) {
        first = false;
        return fail(CLOSED);
      }
      return ok();
    });
    expect((await run(ctx, command)).success).toBe(true);
    expect(ran().map((c) => c.session)).toEqual([base, `${base}-r1`]);
  });

  it.each([
    'CDP error (Runtime.evaluate): Error: the dialog was closed',
    'Element is not attached to the DOM',
    'CDP error (DOM.focus): not attached to the DOM',
    'CDP error (DOM.focus): Node is detached, socket closed',
    'Navigation failed: net::ERR_CONNECTION_CLOSED',
    'page said: ECONNRESET',
  ])(
    'does not treat the page-style error %j as a lost connection',
    async (error) => {
      const { run, ctx, base, ran } = await load(() => fail(error));
      expect(await run(ctx, ['snapshot', '-i'])).toMatchObject({
        success: false,
        error,
      });
      await run(ctx, ['get', 'url']);
      expect(ran().map((c) => c.session)).toEqual([base, base]);
    },
  );

  it.each([
    'CDP response channel closed',
    'CDP error (DOM.enable): not attached',
    'CDP error (Page.enable): tab or extension disconnected',
    'CDP error (Runtime.evaluate): extension restarted',
    'ECONNRESET: socket hang up',
    'EPIPE',
  ])('treats the transport error %j as a lost connection', async (error) => {
    let first = true;
    const { run, ctx, base, ran } = await load((argv) => {
      if (argv[0] === 'close') return ok();
      if (first) {
        first = false;
        return fail(error);
      }
      return ok();
    });
    expect((await run(ctx, ['snapshot', '-i'])).success).toBe(true);
    expect(ran().map((c) => c.session)).toEqual([base, `${base}-r1`]);
  });

  it('keeps the token out of the result when recovery fails', async () => {
    const { run, ctx } = await load((command) =>
      command[0] === 'close'
        ? fail(`closed ${FAKE_TOKEN}`)
        : fail(`ECONNRESET at ws://127.0.0.1:9/cdp?token=${FAKE_TOKEN}`),
    );
    for (const command of [['snapshot'], ['click', '@e1']]) {
      const result = await run(ctx, command);
      expect(JSON.stringify(result)).not.toContain(FAKE_TOKEN);
    }
  });
});

describe('command policy in extension runs', () => {
  const REFUSED =
    'This command is not available in extension runs. Verify values from the snapshot, which masks sensitive fields.';
  const relayCtx = () =>
    ctxWith(principal({ relayUrl: 'ws://127.0.0.1:9', relayRunId: RUN }));

  const load = async () => {
    vi.resetModules();
    const runCommand = vi.fn(async () => ({
      success: true,
      data: 'ok',
      error: null,
    }));
    vi.doMock('@/lib/kernel/cli', () => ({ runCommand }));
    const fetchAgentToken = vi.fn(async () => ({
      ok: true as const,
      token: FAKE_TOKEN,
    }));
    vi.doMock('@/lib/relay/agent-token', () => ({ fetchAgentToken }));
    const mod = await import('@/lib/relay/eve-relay-browser');
    return { run: mod.runRelayCommand, runCommand, fetchAgentToken };
  };
  afterEach(() => {
    vi.doUnmock('@/lib/kernel/cli');
    vi.doUnmock('@/lib/relay/agent-token');
  });

  it.each([
    [['eval', 'document.title']],
    [['get', 'text', '@e1']],
    [['get', 'value', '@e1']],
    [['batch', 'eval 1']],
    [['--cdp', 'ws://evil.example', 'snapshot']],
    [['find', 'role', 'textbox', 'text']],
    [['wait', '--text', 'SSN: 9']],
    [['wait', '--fn', '1']],
    [[]],
  ])(
    'refuses %j without minting a token or starting the CLI',
    async (command) => {
      const { run, runCommand, fetchAgentToken } = await load();
      const result = await run(relayCtx(), command);
      expect(result).toEqual({
        success: false,
        error: REFUSED,
        liveViewUrl: null,
      });
      expect(fetchAgentToken).not.toHaveBeenCalled();
      expect(runCommand).not.toHaveBeenCalled();
    },
  );

  it('refuses before the credential and URL checks, with no argv in the error', async () => {
    const { run, fetchAgentToken } = await load();
    const noCreds = await run(ctxWith(null), ['eval', 'SECRET-6789']);
    expect(noCreds.error).toBe(REFUSED);
    expect(noCreds.error).not.toContain('6789');
    const badUrl = await run(
      ctxWith(principal({ relayUrl: 'http://evil.example', relayRunId: RUN })),
      ['get', 'value', '@e1'],
    );
    expect(badUrl.error).toBe(REFUSED);
    expect(fetchAgentToken).not.toHaveBeenCalled();
  });

  it('runs an allowed command', async () => {
    const { run, runCommand } = await load();
    const result = await run(relayCtx(), ['get', 'url']);
    expect(result.success).toBe(true);
    expect(runCommand).toHaveBeenCalledTimes(1);
  });

  it('does not retry get text after a lost connection: it never runs', async () => {
    const { run, runCommand } = await load();
    runCommand.mockResolvedValue({
      success: false,
      data: null,
      error: 'CDP response channel closed',
    } as never);
    const result = await run(relayCtx(), ['get', 'text', '#status']);
    expect(result.error).toBe(REFUSED);
    expect(runCommand).not.toHaveBeenCalled();
  });

  it('leaves the Kernel path unpoliced', async () => {
    vi.resetModules();
    const runBrowserCommand = vi.fn(async () => ({
      success: true,
      data: 'kernel',
      error: null,
      liveViewUrl: null,
    }));
    vi.doMock('@/lib/kernel/eve-browser', () => ({ runBrowserCommand }));
    const tool = (await import('@/agent/tools/browser')).default;
    const result = await (
      tool as unknown as {
        execute: (a: unknown, c: unknown) => Promise<{ success: boolean }>;
      }
    ).execute({ command: ['eval', 'document.title'] }, ctxWith(null));
    expect(runBrowserCommand).toHaveBeenCalledTimes(1);
    expect(result.success).toBe(true);
    vi.doUnmock('@/lib/kernel/eve-browser');
  });
});
