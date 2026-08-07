import { expect, test } from 'vitest';
import { buildArgs, cliEnv, parseResponse } from '@/lib/kernel/cli';

test('buildArgs attaches to the Kernel browser by CDP url', () => {
  expect(
    buildArgs(['snapshot', '-i'], {
      session: 'chat-1-user-1',
      cdpUrl: 'wss://kernel.example/cdp?token=abc',
    }),
  ).toEqual([
    '--session',
    'chat-1-user-1',
    '--cdp',
    'wss://kernel.example/cdp?token=abc',
    '--json',
    'snapshot',
    '-i',
  ]);
});

test('buildArgs omits --cdp when no url is given', () => {
  expect(buildArgs(['get', 'url'], { session: 's' })).toEqual([
    '--session',
    's',
    '--json',
    'get',
    'url',
  ]);
});

test('buildArgs passes command argv through verbatim', () => {
  // Values are argv entries, never shell-interpolated, so quotes and spaces
  // in user-supplied text reach the binary intact.
  const args = buildArgs(['fill', '@e1', "O'Brien & Co; rm -rf /"], {
    session: 's',
  });
  expect(args.slice(-3)).toEqual(['fill', '@e1', "O'Brien & Co; rm -rf /"]);
});

test('buildArgs rejects an empty command', () => {
  expect(() => buildArgs([], { session: 's' })).toThrow(/must not be empty/);
});

test('parseResponse returns the CLI envelope as-is', () => {
  const stdout = JSON.stringify({
    success: true,
    data: { title: 'Example Domain' },
    error: null,
  });
  expect(parseResponse(stdout, '')).toEqual({
    success: true,
    data: { title: 'Example Domain' },
    error: null,
  });
});

test('parseResponse preserves a structured browser failure', () => {
  // Exit code is non-zero here, but the message is the useful part.
  const stdout = JSON.stringify({
    success: false,
    data: null,
    error: 'Element not found: @e9',
  });
  expect(parseResponse(stdout, '')).toEqual({
    success: false,
    data: null,
    error: 'Element not found: @e9',
  });
});

test('parseResponse wraps a batch array under data', () => {
  const stdout = JSON.stringify([
    { command: ['get', 'title'], success: true, result: { title: 'A' } },
  ]);
  const parsed = parseResponse(stdout, '');
  expect(parsed.success).toBe(true);
  expect(Array.isArray(parsed.data)).toBe(true);
});

test('parseResponse falls back to stderr when stdout is not JSON', () => {
  expect(parseResponse('', 'Kernel API error (401)')).toEqual({
    success: false,
    data: null,
    error: 'Kernel API error (401)',
  });
});

test('parseResponse reports empty output rather than throwing', () => {
  expect(parseResponse('', '')).toEqual({
    success: false,
    data: null,
    error: 'agent-browser produced no output',
  });
});

test('cliSessionName is deterministic for the same browser', async () => {
  // The tools and the lifecycle code must derive the same --session name, or
  // commands would drive a different daemon than standby/delete tear down.
  const { cliSessionName } = await import('@/lib/kernel/session-store');
  expect(cliSessionName('user-1', 'chat-1-user-1')).toBe(
    cliSessionName('user-1', 'chat-1-user-1'),
  );
});

test('cliSessionName stays short enough for a unix socket path', async () => {
  // Regression: `${userId}:${sessionId}` repeated the UUID and produced a
  // 135-byte socket path, over the ~103-byte cap, so every command failed.
  const { cliSessionName } = await import('@/lib/kernel/session-store');
  const userId = 'f272a42e-3cf1-428f-8c3a-834d83ad913b';
  const sessionId = `3cad2ba2-9857-44e2-ad02-40f3a5bfe895-${userId}`;
  expect(cliSessionName(userId, sessionId).length).toBeLessThanOrEqual(32);
});

test('cliSessionName distinguishes different chats', async () => {
  const { cliSessionName } = await import('@/lib/kernel/session-store');
  expect(cliSessionName('u1', 'chat-a-u1')).not.toBe(
    cliSessionName('u1', 'chat-b-u1'),
  );
});

test('cliEnv strips AGENT_BROWSER_PROVIDER when attaching by CDP', () => {
  // The CLI rejects `--cdp` combined with a provider ("Cannot use --cdp and
  // -p/--provider together"). An inherited env var broke every browser command
  // in production while working locally, so strip it at the source.
  const base: NodeJS.ProcessEnv = {
    AGENT_BROWSER_PROVIDER: 'kernel',
    PATH: '/usr/bin',
    NODE_ENV: 'test',
  };
  const env = cliEnv({ cdpUrl: 'wss://kernel.example/cdp' }, base);
  expect(env.AGENT_BROWSER_PROVIDER).toBeUndefined();
  expect(env.PATH).toBe('/usr/bin');
});

test('cliEnv leaves the provider in place when there is no CDP url', () => {
  const base: NodeJS.ProcessEnv = {
    AGENT_BROWSER_PROVIDER: 'kernel',
    PATH: '/usr/bin',
    NODE_ENV: 'test',
  };
  const env = cliEnv({}, base);
  expect(env.AGENT_BROWSER_PROVIDER).toBe('kernel');
});
