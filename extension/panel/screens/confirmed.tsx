import { useEffect, useRef } from 'react';
import { Button } from '@/components/ui/button';

export const STOP_FAILED_TEXT =
  'We could not reach the app or the extension to confirm the stop. The agent may still be working: check the page and use Stop if it is.';

type Props = {
  siteName: string;
  /** The stop could not be confirmed: the agent may still be working. */
  stopFailed: boolean;
  onStartAnother: () => void;
  onHome: () => void;
};

// Honest wording: this records what the caseworker said, not that the site
// accepted the form. No confirmation number is read or shown.
export const ConfirmedScreen = ({
  siteName,
  stopFailed,
  onStartAnother,
  onHome,
}: Props) => {
  const headingRef = useRef<HTMLHeadingElement>(null);
  useEffect(() => {
    headingRef.current?.focus();
  }, []);
  return (
    <section className="flex flex-col gap-4" aria-label="Submitted">
      <h2 ref={headingRef} tabIndex={-1} className="text-base font-semibold">
        Submitted
      </h2>
      <p role="status" aria-live="polite" className="text-sm">
        You said you submitted the form on {siteName}.
      </p>
      {stopFailed ? (
        <p role="alert" className="text-sm">
          {STOP_FAILED_TEXT}
        </p>
      ) : null}
      <div className="flex flex-col gap-2">
        <Button type="button" onClick={onStartAnother}>
          Start another form for this person
        </Button>
        <Button type="button" variant="outline" onClick={onHome}>
          Back to Home
        </Button>
      </div>
    </section>
  );
};
