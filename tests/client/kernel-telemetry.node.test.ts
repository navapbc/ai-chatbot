import { expect, test } from 'vitest';
import { toTimelineEvent } from '@/lib/kernel/telemetry';

test('maps a kernel event to a named, timestamped span event', () => {
  const event = toTimelineEvent(
    {
      category: 'console',
      type: 'console_error',
      ts: 1_755_100_000_000_000, // unix microseconds
      data: { message: 'Uncaught TypeError: x is not a function' },
    },
    42,
  );

  expect(event.name).toBe('kernel.console_error');
  expect(event.timestamp.getTime()).toBe(1_755_100_000_000);
  expect(event.attributes).toMatchObject({
    'kernel.category': 'console',
    'kernel.seq': 42,
    'kernel.message': 'Uncaught TypeError: x is not a function',
  });
});

test('drops nested payload fields and keeps primitives', () => {
  const event = toTimelineEvent(
    {
      category: 'system',
      type: 'system_oom_kill',
      ts: 1_755_100_000_000_000,
      data: {
        victim_rss_bytes: 4_800_000_000,
        fatal: true,
        report: { lines: ['...'] }, // nested — must not become an attribute
      },
    },
    1,
  );

  expect(event.attributes).toMatchObject({
    'kernel.victim_rss_bytes': 4_800_000_000,
    'kernel.fatal': true,
  });
  expect(event.attributes).not.toHaveProperty('kernel.report');
});

test('truncates long page-controlled strings', () => {
  const event = toTimelineEvent(
    {
      category: 'console',
      type: 'console_error',
      ts: 1_755_100_000_000_000,
      data: { message: 'x'.repeat(2_000) },
    },
    1,
  );

  const message = event.attributes?.['kernel.message'] as string;
  expect(message.length).toBeLessThanOrEqual(501); // 500 + ellipsis
  expect(message.endsWith('…')).toBe(true);
});

test('flags events kernel itself truncated', () => {
  const event = toTimelineEvent(
    {
      category: 'console',
      type: 'console_log',
      ts: 1_755_100_000_000_000,
      truncated: true,
    },
    1,
  );

  expect(event.attributes).toMatchObject({ 'kernel.truncated': true });
});
