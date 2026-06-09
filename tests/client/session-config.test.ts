import { expect, test } from 'vitest';
import {
  CAP_WARNING_BEFORE_MS,
  HARD_CAP_MS,
  IDLE_COUNTDOWN_MS,
  IDLE_DISCONNECT_AFTER_MS,
  IDLE_WARNING_AFTER_MS,
  KERNEL_TIMEOUT_SECONDS,
} from '@/lib/kernel/session-config';

test('idle disconnect is warning + countdown', () => {
  expect(IDLE_DISCONNECT_AFTER_MS).toBe(
    IDLE_WARNING_AFTER_MS + IDLE_COUNTDOWN_MS,
  );
});

test('spec timings: 12-min idle warning, 3-min countdown', () => {
  expect(IDLE_WARNING_AFTER_MS).toBe(12 * 60_000);
  expect(IDLE_COUNTDOWN_MS).toBe(3 * 60_000);
});

test('hard cap is 60 min with a 5-min warning', () => {
  expect(HARD_CAP_MS).toBe(60 * 60_000);
  expect(CAP_WARNING_BEFORE_MS).toBe(5 * 60_000);
});

test('Kernel backstop timeout exceeds the hard cap so our controller governs', () => {
  expect(KERNEL_TIMEOUT_SECONDS * 1000).toBeGreaterThan(HARD_CAP_MS);
  // Kernel allows up to 72h (259200s); stay within bounds.
  expect(KERNEL_TIMEOUT_SECONDS).toBeLessThanOrEqual(259_200);
});
