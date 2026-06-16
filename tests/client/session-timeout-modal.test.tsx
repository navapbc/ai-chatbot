import { expect, test, vi } from 'vitest';
import { render } from 'vitest-browser-react';
import { SessionTimeoutModal } from '@/components/session-timeout-modal';

function setup(overrides?: Partial<Parameters<typeof SessionTimeoutModal>[0]>) {
  const onOpenChange = vi.fn();
  const onEndSession = vi.fn();
  const onContinueSession = vi.fn();
  const utils = render(
    <SessionTimeoutModal
      open={true}
      onOpenChange={onOpenChange}
      countdownSeconds={125}
      onEndSession={onEndSession}
      onContinueSession={onContinueSession}
      {...overrides}
    />,
  );
  return { ...utils, onOpenChange, onEndSession, onContinueSession };
}

test('renders the heading, countdown, and both actions when open', async () => {
  const { getByText, getByRole } = setup();

  await expect
    .element(getByText('Your session is ending soon'))
    .toBeInTheDocument();
  // 125s → 2:05
  await expect.element(getByText('2:05')).toBeInTheDocument();
  await expect
    .element(getByRole('button', { name: 'End session' }))
    .toBeInTheDocument();
  await expect
    .element(getByRole('button', { name: 'Continue session' }))
    .toBeInTheDocument();
});

test('idle reason shows the inactivity copy', async () => {
  const { getByText } = setup({ reason: 'idle' });
  await expect
    .element(getByText(/sessions end after\s+inactivity/i))
    .toBeInTheDocument();
});

test('cap reason shows the maximum-length copy', async () => {
  const { getByText } = setup({ reason: 'cap' });
  await expect
    .element(getByText(/reached the maximum session length/i))
    .toBeInTheDocument();
});

test('clamps negative countdown to 0:00', async () => {
  const { getByText } = setup({ countdownSeconds: -5 });
  await expect.element(getByText('0:00')).toBeInTheDocument();
});

test('Continue session calls onContinueSession and closes', async () => {
  const { getByRole, onContinueSession, onEndSession, onOpenChange } = setup();

  await getByRole('button', { name: 'Continue session' }).click();

  expect(onContinueSession).toHaveBeenCalledTimes(1);
  expect(onOpenChange).toHaveBeenCalledWith(false);
  expect(onEndSession).not.toHaveBeenCalled();
});

test('End session calls onEndSession and closes', async () => {
  const { getByRole, onContinueSession, onEndSession, onOpenChange } = setup();

  await getByRole('button', { name: 'End session' }).click();

  expect(onEndSession).toHaveBeenCalledTimes(1);
  expect(onOpenChange).toHaveBeenCalledWith(false);
  expect(onContinueSession).not.toHaveBeenCalled();
});

test('does not render content when closed', async () => {
  const { container } = setup({ open: false });
  expect(container.textContent).not.toContain('Your session is ending soon');
});
