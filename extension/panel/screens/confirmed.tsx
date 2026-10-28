import { useEffect, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { CONFIRMED_COPY } from '../lib/copy';
import type { ConfirmedHow } from '../lib/run-session';
import type { Confirmation } from '../lib/worker-api';

export const STOP_FAILED_TEXT =
  'We could not reach the app or the extension to confirm the stop. The agent may still be working: check the page and use Stop if it is.';

type Props = {
  siteName: string;
  /** The stop could not be confirmed: the agent may still be working. */
  stopFailed: boolean;
  /** Detected from the page (or "Yes" to the prompt), or said by hand. */
  how?: ConfirmedHow;
  /** The number the page showed, if any: shown here and nowhere else. */
  confirmation?: Confirmation;
  onStartAnother: () => void;
  onHome: () => void;
};

// Honest wording: a detected submission means the page moved on after a form
// was sent, not that the site accepted the application. The number is the
// worker's vetted token, rendered as React text only.
export const ConfirmedScreen = ({
  siteName,
  stopFailed,
  how = 'manual',
  confirmation,
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
      <div role="status" aria-live="polite" className="flex flex-col gap-2">
        {how === 'detected' ? (
          <>
            <p className="text-sm">{CONFIRMED_COPY.detected(siteName)}</p>
            <p className="text-sm">{CONFIRMED_COPY.detectedDetail}</p>
          </>
        ) : (
          <p className="text-sm">{CONFIRMED_COPY.manual(siteName)}</p>
        )}
        {confirmation ? (
          <div className="flex flex-col gap-1">
            <p className="text-sm font-medium">
              {confirmation.kind === 'confirmation'
                ? CONFIRMED_COPY.confirmationLabel
                : CONFIRMED_COPY.otherLabel}
            </p>
            <p className="break-all font-ibm-plex-mono text-base">
              {confirmation.token}
            </p>
            <p className="text-sm">{CONFIRMED_COPY.keepIt}</p>
          </div>
        ) : null}
      </div>
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
