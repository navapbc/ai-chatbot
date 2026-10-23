import { chmodSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest';

const TOKEN = 'tok/with+chars=1';
const CDP = `wss://relay.example/cdp?token=${encodeURIComponent(TOKEN)}`;

const ends: Array<{ outcome: string; error?: string | null }> = [];
vi.mock('@/lib/observability/browser-telemetry', () => ({
  startCommandTelemetry: () => ({
    end: (r: (typeof ends)[number]) => ends.push(r),
  }),
}));

let dir: string;
let runCommand: typeof import('@/lib/kernel/cli').runCommand;

// A stand-in binary that fails the way agent-browser does, echoing the URL.
const fake = (body: string) => {
  const file = path.join(dir, 'fake-ab');
  writeFileSync(file, `#!/bin/sh\n${body}\n`);
  chmodSync(file, 0o755);
};

beforeAll(async () => {
  dir = mkdtempSync(path.join(tmpdir(), 'cli-redact-'));
  process.env.AGENT_BROWSER_BIN = path.join(dir, 'fake-ab');
  fake(':');
  ({ runCommand } = await import('@/lib/kernel/cli'));
});
afterAll(() => {
  rmSync(dir, { recursive: true, force: true });
  delete process.env.AGENT_BROWSER_BIN;
});

const leaky = `{"success":false,"error":"connect failed for ${CDP} token ${TOKEN}","data":{"nested":["see ${CDP}","raw ${TOKEN}"],"n":1}}`;

describe('runCommand redaction', () => {
  it('redacts the url and token (raw and encoded) from error, data and telemetry', async () => {
    fake(`echo '${leaky}'`);
    ends.length = 0;
    const res = await runCommand(['get', 'url'], {
      session: 's',
      cdpUrl: CDP,
      redact: [TOKEN, CDP],
    });
    const everything = JSON.stringify([res, ends]);
    expect(everything).not.toContain(TOKEN);
    expect(everything).not.toContain(encodeURIComponent(TOKEN));
    expect(everything).not.toContain('relay.example/cdp');
    expect(res.error).toBe('connect failed for [redacted] token [redacted]');
    expect(res.data).toEqual({
      nested: ['see [redacted]', 'raw [redacted]'],
      n: 1,
    });
    expect(ends[0].error).toBe(
      'connect failed for [redacted] token [redacted]',
    );
  });

  it('redacts stderr when stdout is not JSON', async () => {
    fake(`echo 'cannot reach ${CDP}' >&2; exit 1`);
    ends.length = 0;
    const res = await runCommand(['snapshot'], {
      session: 's',
      cdpUrl: CDP,
      redact: [TOKEN, CDP],
    });
    expect(res.success).toBe(false);
    expect(res.error).toBe('cannot reach [redacted]');
    expect(ends[0].error).toBe('cannot reach [redacted]');
  });

  it('negative control: without `redact` the same output leaks', async () => {
    fake(`echo '${leaky}'`);
    ends.length = 0;
    const res = await runCommand(['get', 'url'], { session: 's', cdpUrl: CDP });
    expect(res.error).toContain(TOKEN);
    expect(ends[0].error).toContain(TOKEN);
  });
});
