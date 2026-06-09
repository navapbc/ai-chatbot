import { expect, test } from 'vitest';
import { evaluateLifecycle } from '@/hooks/use-session-lifecycle';
import {
  CAP_WARNING_BEFORE_MS,
  HARD_CAP_MS,
  IDLE_DISCONNECT_AFTER_MS,
  IDLE_WARNING_AFTER_MS,
} from '@/lib/kernel/session-config';

const NOW = 1_000_000_000_000; // fixed clock; values below are relative to it

test('no action while active and well within the cap', () => {
  const action = evaluateLifecycle(NOW, NOW - 60_000, NOW - 1_000);
  expect(action.kind).toBe('none');
});

test('warns on idle once past the idle warning threshold', () => {
  const action = evaluateLifecycle(
    NOW,
    NOW - 60_000,
    NOW - (IDLE_WARNING_AFTER_MS + 1_000),
  );
  expect(action).toMatchObject({ kind: 'warn', reason: 'idle' });
  if (action.kind === 'warn') {
    // ~3-min countdown remaining minus the 1s we advanced.
    expect(action.countdownSeconds).toBeGreaterThan(0);
    expect(action.countdownSeconds).toBeLessThanOrEqual(
      IDLE_DISCONNECT_AFTER_MS / 1000,
    );
  }
});

test('standby once idle reaches the disconnect threshold', () => {
  const action = evaluateLifecycle(
    NOW,
    NOW - 60_000,
    NOW - (IDLE_DISCONNECT_AFTER_MS + 1),
  );
  expect(action.kind).toBe('standby');
});

test('hard cap warning fires inside the warning window', () => {
  const startedAt = NOW - (HARD_CAP_MS - CAP_WARNING_BEFORE_MS + 1_000);
  const action = evaluateLifecycle(NOW, startedAt, NOW - 1_000);
  expect(action).toMatchObject({ kind: 'warn', reason: 'cap' });
});

test('hard end once the cap is exceeded', () => {
  const action = evaluateLifecycle(NOW, NOW - (HARD_CAP_MS + 1), NOW - 1_000);
  expect(action.kind).toBe('hard-end');
});

test('cap takes precedence over idle', () => {
  // Both idle AND past the cap → cap (hard-end) wins.
  const action = evaluateLifecycle(
    NOW,
    NOW - (HARD_CAP_MS + 1),
    NOW - (IDLE_DISCONNECT_AFTER_MS + 1),
  );
  expect(action.kind).toBe('hard-end');
});

test('cap warning beats idle standby when both apply', () => {
  const startedAt = NOW - (HARD_CAP_MS - CAP_WARNING_BEFORE_MS + 1_000);
  const action = evaluateLifecycle(
    NOW,
    startedAt,
    NOW - (IDLE_DISCONNECT_AFTER_MS + 1),
  );
  expect(action).toMatchObject({ kind: 'warn', reason: 'cap' });
});

test('idle warning countdown shrinks as inactivity grows', () => {
  const justWarned = evaluateLifecycle(
    NOW,
    NOW - 60_000,
    NOW - (IDLE_WARNING_AFTER_MS + 1_000),
  );
  const almostOut = evaluateLifecycle(
    NOW,
    NOW - 60_000,
    NOW - (IDLE_DISCONNECT_AFTER_MS - 5_000),
  );
  if (justWarned.kind === 'warn' && almostOut.kind === 'warn') {
    expect(almostOut.countdownSeconds).toBeLessThan(
      justWarned.countdownSeconds,
    );
    expect(almostOut.countdownSeconds).toBeLessThanOrEqual(5);
  } else {
    throw new Error('expected both to be warn actions');
  }
});

test('exactly at the idle disconnect threshold triggers standby', () => {
  const action = evaluateLifecycle(
    NOW,
    NOW - 60_000,
    NOW - IDLE_DISCONNECT_AFTER_MS,
  );
  expect(action.kind).toBe('standby');
});
