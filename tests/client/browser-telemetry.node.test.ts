import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import {
  logAgentStep,
  startCommandTelemetry,
  withKernelSpan,
} from '@/lib/observability/browser-telemetry';

let out: string[];
let err: string[];

beforeEach(() => {
  out = [];
  err = [];
  vi.spyOn(console, 'log').mockImplementation((line: string) => {
    out.push(line);
  });
  vi.spyOn(console, 'error').mockImplementation((line: string) => {
    err.push(line);
  });
});

afterEach(() => {
  vi.restoreAllMocks();
});

const parsed = (lines: string[]) => lines.map((l) => JSON.parse(l));

test('a start event is emitted before the command runs', () => {
  // The whole point: a subprocess that never returns still leaves a trace.
  startCommandTelemetry(['snapshot', '-i'], {
    session: 'u:c',
    remote: true,
    timeoutMs: 120_000,
  });

  const [start] = parsed(out);
  expect(start).toMatchObject({
    severity: 'INFO',
    event: 'agent_browser.command.start',
    command: 'snapshot',
    session: 'u:c',
    remote: true,
    timeoutMs: 120_000,
  });
});

test('success logs an end event with a duration', () => {
  const t = startCommandTelemetry(['click', '@e1'], {
    session: 'u:c',
    remote: true,
    timeoutMs: 1000,
  });
  t.end({ outcome: 'success' });

  const end = parsed(out).find((e) => e.event === 'agent_browser.command.end');
  expect(end).toMatchObject({
    severity: 'INFO',
    command: 'click',
    outcome: 'success',
  });
  expect(typeof end.durationMs).toBe('number');
});

test('failures log at ERROR on stderr with the outcome and message', () => {
  const t = startCommandTelemetry(['click', '@e9'], {
    session: 'u:c',
    remote: true,
    timeoutMs: 1000,
  });
  t.end({ outcome: 'timeout', error: 'Command timed out after 1000ms' });

  const [end] = parsed(err);
  expect(end).toMatchObject({
    severity: 'ERROR',
    event: 'agent_browser.command.end',
    outcome: 'timeout',
    error: 'Command timed out after 1000ms',
  });
});

test('only the command name is logged, never the arguments', () => {
  // Args carry applicant PII (names, SSNs); the command name does not.
  startCommandTelemetry(['fill', '@e1', '123-45-6789'], {
    session: 'u:c',
    remote: true,
    timeoutMs: 1000,
  });

  expect(out.join('\n')).not.toContain('123-45-6789');
  expect(parsed(out)[0].command).toBe('fill');
});

test('withKernelSpan brackets a successful call with start and end', async () => {
  const result = await withKernelSpan(
    'browsers.create',
    { a: 1 },
    async () => 'ok',
  );

  expect(result).toBe('ok');
  const events = parsed(out).map((e) => e.event);
  expect(events).toEqual(['kernel.operation.start', 'kernel.operation.end']);
});

test('withKernelSpan logs and rethrows a failure', async () => {
  await expect(
    withKernelSpan('browsers.create', {}, async () => {
      throw new Error('kernel exploded');
    }),
  ).rejects.toThrow('kernel exploded');

  const [end] = parsed(err);
  expect(end).toMatchObject({
    severity: 'ERROR',
    event: 'kernel.operation.end',
    operation: 'browsers.create',
    error: 'kernel exploded',
  });
});

test('agent step events record tool names but never arguments', () => {
  logAgentStep({
    index: 3,
    toolNames: ['browser', 'getApricotRecord'],
    finishReason: 'tool-calls',
    inputTokens: 1200,
    outputTokens: 80,
    durationMs: 4500,
  });

  const [event] = parsed(out);
  expect(event).toMatchObject({
    severity: 'INFO',
    event: 'agent.step.finish',
    step: 3,
    tools: ['browser', 'getApricotRecord'],
    finishReason: 'tool-calls',
    durationMs: 4500,
  });
});
