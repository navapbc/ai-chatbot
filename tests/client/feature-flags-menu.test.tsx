import { afterEach, beforeEach, expect, test } from 'vitest';
import { render } from 'vitest-browser-react';
import { FeatureFlagsMenu } from '@/components/feature-flags-menu';
import { isFeatureEnabled, setFlagOverride } from '@/lib/feature-flags';

beforeEach(() => {
  window.localStorage.clear();
});

afterEach(() => {
  setFlagOverride('declutterToolCalls', null);
  window.localStorage.clear();
});

test('toggling the switch flips the feature flag', async () => {
  expect(isFeatureEnabled('declutterToolCalls')).toBe(false);

  const { getByRole } = render(<FeatureFlagsMenu />);

  await getByRole('button', { name: /flags/i }).click();
  await getByRole('switch', { name: /declutter tool calls/i }).click();

  expect(isFeatureEnabled('declutterToolCalls')).toBe(true);
});
