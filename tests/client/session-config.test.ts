import { expect, test } from 'vitest';
import {
  CAP_WARNING_BEFORE_MS,
  HARD_CAP_MS,
  IDLE_COUNTDOWN_MS,
  IDLE_DISCONNECT_AFTER_MS,
  IDLE_WARNING_AFTER_MS,
  KERNEL_TIMEOUT_SECONDS,
} from '@/lib/kernel/session-config';

// These assert structural invariants that must hold under any timing values
// (production or the temporary short test timings), rather than literal
// minutes — so toggling the durations for testing doesn't break the suite.

test('idle disconnect is warning + countdown', () => {
  expect(IDLE_DISCONNECT_AFTER_MS).toBe(
    IDLE_WARNING_AFTER_MS + IDLE_COUNTDOWN_MS,
  );
});

test('all idle/cap durations are positive', () => {
  for (const v of [
    IDLE_WARNING_AFTER_MS,
    IDLE_COUNTDOWN_MS,
    HARD_CAP_MS,
    CAP_WARNING_BEFORE_MS,
  ]) {
    expect(v).toBeGreaterThan(0);
  }
});

test('the cap warning fits inside the hard cap', () => {
  expect(CAP_WARNING_BEFORE_MS).toBeLessThan(HARD_CAP_MS);
});

test('idle disconnect happens before the hard cap', () => {
  // Otherwise the cap would always pre-empt the idle path.
  expect(IDLE_DISCONNECT_AFTER_MS).toBeLessThan(HARD_CAP_MS);
});

test('Kernel backstop is within Kernel bounds (10s..72h)', () => {
  expect(KERNEL_TIMEOUT_SECONDS).toBeGreaterThanOrEqual(10);
  expect(KERNEL_TIMEOUT_SECONDS).toBeLessThanOrEqual(259_200);
});
