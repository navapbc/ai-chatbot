import { useEffect, useRef } from 'react';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import type { ErrorMessage } from '../lib/copy';

type Props = {
  error: ErrorMessage;
  onRetry: () => void;
};

export const ErrorScreen = ({ error, onRetry }: Props) => {
  const retry = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    retry.current?.focus();
  }, []);

  const handleRetry = () => onRetry();

  return (
    <section className="flex flex-col gap-4" aria-label="Error">
      <Alert variant="destructive">
        <AlertTitle>{error.title}</AlertTitle>
        <AlertDescription>{error.detail}</AlertDescription>
      </Alert>
      {error.retryable ? (
        <Button ref={retry} type="button" onClick={handleRetry}>
          Retry
        </Button>
      ) : null}
    </section>
  );
};
