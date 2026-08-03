import { afterEach, beforeEach, expect, test } from 'vitest';
import { render } from 'vitest-browser-react';
import { setFlagOverride } from '@/lib/feature-flags';
import { ToolCallGroup } from '@/components/tool-call-group';

// Enable the declutter flag so the filtering behavior is exercised in dev tests.
beforeEach(() => {
  setFlagOverride('declutterToolCalls', true);
});

afterEach(() => {
  setFlagOverride('declutterToolCalls', null);
});

const part = (toolCallId: string, input: Record<string, unknown>) => ({
  type: 'tool-browser',
  toolCallId,
  state: 'output-available',
  input,
});

test('decluttered: a group shows only specific tool calls when expanded', async () => {
  const { getByRole, getByText } = render(
    <ToolCallGroup
      parts={[
        part('1', { command: ['click', '@e1'] }),
        part('2', { command: ['fill', '@e2', '05/05/2026'] }),
        part('3', { command: ['snapshot'] }),
        part('4', { command: ['select', '@e4', 'Female'] }),
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

test('decluttered: a group with only generic actions shows a summary line and no expander', async () => {
  const { getByText, getByRole } = render(
    <ToolCallGroup
      parts={[
        part('1', { command: ['click', '@e1'] }),
        part('2', { command: ['snapshot'] }),
      ]}
    />,
  );

  await expect.element(getByText('Completed actions')).toBeVisible();
  // No expander button to open into an empty list.
  await expect.element(getByRole('button')).not.toBeInTheDocument();
});

test('decluttered: a single generic action collapses to the summary line', async () => {
  const { getByText, getByRole } = render(
    <ToolCallGroup parts={[part('1', { command: ['click', '@e1'] })]} />,
  );

  await expect.element(getByText('Completed actions')).toBeVisible();
  await expect.element(getByRole('button')).not.toBeInTheDocument();
});

test('flag off: all tool calls remain visible (dev default)', async () => {
  setFlagOverride('declutterToolCalls', false);
  const { getByRole, getByText } = render(
    <ToolCallGroup
      parts={[
        part('1', { command: ['click', '@e1'] }),
        part('2', { command: ['fill', '@e2', '05/05/2026'] }),
      ]}
    />,
  );

  await getByRole('button').click();
  await expect.element(getByText(/Clicking/)).toBeVisible();
  await expect.element(getByText(/Filling "05\/05\/2026"/)).toBeVisible();
});
