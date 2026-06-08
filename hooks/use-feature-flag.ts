'use client';

import { useSyncExternalStore } from 'react';
import {
  FEATURE_FLAGS,
  getFlagOverride,
  subscribeToFlags,
  type FeatureFlagKey,
} from '@/lib/feature-flags';

// Reactive read of a feature flag. Re-renders when the flag's localStorage
// override changes (via the dev-only feature-flags menu or another tab).
export function useFeatureFlag(key: FeatureFlagKey): boolean {
  const defaultValue = FEATURE_FLAGS[key].defaultValue;
  return useSyncExternalStore(
    subscribeToFlags,
    () => getFlagOverride(key) ?? defaultValue,
    () => defaultValue,
  );
}
