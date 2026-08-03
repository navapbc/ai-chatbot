import { expect, test } from 'vitest';
import { buildArgs, parseResponse } from '@/lib/kernel/cli';

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

test('cliSessionName matches the in-memory cache key', async () => {
  // The tools and the lifecycle code must derive the same --session name, or
  // commands would drive a different daemon than standby/delete tear down.
  const { cacheKey, cliSessionName } = await import(
    '@/lib/kernel/session-store'
  );
  expect(cliSessionName('user-1', 'chat-1-user-1')).toBe(
    cacheKey('user-1', 'chat-1-user-1'),
  );
});
