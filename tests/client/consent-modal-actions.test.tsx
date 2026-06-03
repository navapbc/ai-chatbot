import { expect, test, vi } from 'vitest';
import { render } from 'vitest-browser-react';
import { ConsentModal } from '@/components/consent-modal';

test('ConsentModal shows the confirm action label', async () => {
  const { getByText } = render(
    <ConsentModal
      open={true}
      onOpenChange={vi.fn()}
      onContinue={vi.fn()}
    />,
  );

  await expect.element(getByText('Confirm')).toBeInTheDocument();
});
