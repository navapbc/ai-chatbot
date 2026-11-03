import { useEffect, useRef } from 'react';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { messageFor } from '../lib/copy';
import { AppApiError } from '../../src/app-api';

const COPY = messageFor(new AppApiError('signed_out', 401));

type Props = {
  onOpenApp: () => void;
  onRetry: () => void;
};

export const SignedOutScreen = ({ onOpenApp, onRetry }: Props) => {
  const primary = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    primary.current?.focus();
  }, []);

  const handleOpenApp = () => onOpenApp();
  const handleRetry = () => onRetry();

  return (
    <section className="flex flex-col gap-4" aria-label="Signed out">
      <Alert>
        <AlertTitle>{COPY.title}</AlertTitle>
        <AlertDescription>{COPY.detail}</AlertDescription>
      </Alert>
      <Button ref={primary} type="button" onClick={handleOpenApp}>
        Open the app
      </Button>
      <Button type="button" variant="outline" onClick={handleRetry}>
        Try again
      </Button>
    </section>
  );
};
