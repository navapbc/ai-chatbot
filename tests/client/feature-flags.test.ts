import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';
import {
  FEATURE_FLAGS,
  getFlagOverride,
  isFeatureEnabled,
  setFlagOverride,
  subscribeToFlags,
} from '@/lib/feature-flags';

const KEY = 'declutterToolCalls';

beforeEach(() => {
  window.localStorage.clear();
});

afterEach(() => {
  window.localStorage.clear();
});

describe('flag overrides', () => {
  test('getFlagOverride returns null when unset', () => {
    expect(getFlagOverride(KEY)).toBeNull();
  });

  test('setFlagOverride(true) then getFlagOverride returns true', () => {
    setFlagOverride(KEY, true);
    expect(getFlagOverride(KEY)).toBe(true);
  });

  test('setFlagOverride(false) then getFlagOverride returns false', () => {
    setFlagOverride(KEY, false);
    expect(getFlagOverride(KEY)).toBe(false);
  });

  test('setFlagOverride(null) clears the override', () => {
    setFlagOverride(KEY, true);
    setFlagOverride(KEY, null);
    expect(getFlagOverride(KEY)).toBeNull();
  });
});

describe('isFeatureEnabled', () => {
  test('falls back to the registry default when no override (dev default is off)', () => {
    expect(isFeatureEnabled(KEY)).toBe(FEATURE_FLAGS[KEY].defaultValue);
  });

  test('honors an override over the default', () => {
    setFlagOverride(KEY, true);
    expect(isFeatureEnabled(KEY)).toBe(true);
    setFlagOverride(KEY, false);
    expect(isFeatureEnabled(KEY)).toBe(false);
  });
});

describe('subscribeToFlags', () => {
  test('notifies subscribers when a flag changes and stops after unsubscribe', () => {
    const cb = vi.fn();
    const unsubscribe = subscribeToFlags(cb);

    setFlagOverride(KEY, true);
    expect(cb).toHaveBeenCalledTimes(1);

    unsubscribe();
    setFlagOverride(KEY, false);
    expect(cb).toHaveBeenCalledTimes(1);
  });
});
