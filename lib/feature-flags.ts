import { isProductionEnvironment } from './constants';

// Feature flags for individual features. Each flag ships with an
// environment-aware default and can be overridden per-browser via localStorage
// (flipped by the dev-only feature-flags menu) so QA can preview features in
// dev/preview without a redeploy.
export type FeatureFlagKey = 'declutterToolCalls';

export interface FeatureFlagDef {
  key: FeatureFlagKey;
  label: string;
  description: string;
  defaultValue: boolean;
}

export const FEATURE_FLAGS: Record<FeatureFlagKey, FeatureFlagDef> = {
  declutterToolCalls: {
    key: 'declutterToolCalls',
    label: 'Declutter tool calls',
    description: 'Show only value-bearing tool calls (production behavior).',
    defaultValue: isProductionEnvironment,
  },
};

const STORAGE_PREFIX = 'ff:';
const CHANGE_EVENT = 'feature-flag-change';

export function getFlagOverride(key: FeatureFlagKey): boolean | null {
  if (typeof window === 'undefined') return null;
  const raw = window.localStorage.getItem(STORAGE_PREFIX + key);
  if (raw === null) return null;
  return raw === 'true';
}

export function setFlagOverride(key: FeatureFlagKey, value: boolean | null): void {
  if (typeof window === 'undefined') return;
  if (value === null) {
    window.localStorage.removeItem(STORAGE_PREFIX + key);
  } else {
    window.localStorage.setItem(STORAGE_PREFIX + key, String(value));
  }
  window.dispatchEvent(new CustomEvent(CHANGE_EVENT, { detail: { key } }));
}

export function subscribeToFlags(callback: () => void): () => void {
  if (typeof window === 'undefined') return () => {};
  window.addEventListener(CHANGE_EVENT, callback);
  window.addEventListener('storage', callback);
  return () => {
    window.removeEventListener(CHANGE_EVENT, callback);
    window.removeEventListener('storage', callback);
  };
}

export function isFeatureEnabled(key: FeatureFlagKey): boolean {
  return getFlagOverride(key) ?? FEATURE_FLAGS[key].defaultValue;
}
