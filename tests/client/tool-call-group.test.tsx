import { expect, test, vi } from 'vitest';
import { render } from 'vitest-browser-react';

// Force the production path so the filtering behavior is exercised.
vi.mock('@/lib/constants', () => ({
  isProductionEnvironment: true,
  isDevelopmentEnvironment: false,
  isTestEnvironment: false,
  DUMMY_PASSWORD: 'test',
}));

import { ToolCallGroup } from '@/components/tool-call-group';

const part = (toolCallId: string, input: Record<string, unknown>) => ({
  type: 'tool-browser',
  toolCallId,
  state: 'output-available',
  input,
});

test('production: a group shows only specific tool calls when expanded', async () => {
  const { getByRole, getByText } = render(
    <ToolCallGroup
      parts={[
        part('1', { action: 'click' }),
        part('2', { action: 'fill', value: '05/05/2026' }),
        part('3', { action: 'snapshot' }),
        part('4', { action: 'select', values: ['Female'] }),
      ]}
    />,
  );

  // Expand the accordion.
  await getByRole('button').click();

  await expect.element(getByText(/Filling "05\/05\/2026"/)).toBeVisible();
  await expect.element(getByText(/Selecting "Female"/)).toBeVisible();
  await expect.element(getByText(/Clicking/)).not.toBeInTheDocument();
  await expect.element(getByText(/Reading page/)).not.toBeInTheDocument();
});

test('production: a group with only generic actions shows a summary line and no expander', async () => {
  const { getByText, getByRole } = render(
    <ToolCallGroup
      parts={[part('1', { action: 'click' }), part('2', { action: 'snapshot' })]}
    />,
  );

  await expect.element(getByText('Completed actions')).toBeVisible();
  // No expander button to open into an empty list.
  await expect.element(getByRole('button')).not.toBeInTheDocument();
});

test('production: a single generic action collapses to the summary line', async () => {
  const { getByText, getByRole } = render(
    <ToolCallGroup parts={[part('1', { action: 'click' })]} />,
  );

  await expect.element(getByText('Completed actions')).toBeVisible();
  await expect.element(getByRole('button')).not.toBeInTheDocument();
});
