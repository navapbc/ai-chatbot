import { Button } from '@/components/ui/button';
import type { RunProgress } from '../../../lib/relay/progress';
import { buildReviewModel, reviewActions } from '../lib/review-model';

type Props = {
  progress: RunProgress | null;
  actions: string[];
  stopping: boolean;
  /** The stop could not be confirmed (shown after "I submitted it"). */
  stopFailed: boolean;
  onSubmitted: () => void;
  onStop: () => void;
};

// The agent's turn ended with nothing pending. The panel says only that the
// agent stopped, never that the form is complete or correct. Names and the
// action log are untrusted agent text, rendered as React text only.
export const ReviewPanel = ({
  progress,
  actions,
  stopping,
  stopFailed,
  onSubmitted,
  onStop,
}: Props) => {
  const model = buildReviewModel(progress);
  const log = reviewActions(actions);
  return (
    <section
      aria-label="Ready for your review"
      className="flex flex-col gap-3 rounded-xl border border-accent bg-background p-3"
    >
      <h3 className="text-base font-semibold">Ready for your review</h3>
      <p className="text-sm">
        The agent has stopped. It has not submitted the form.
      </p>
      <div className="flex flex-col gap-1">
        <h4 className="text-sm font-medium">Left for you</h4>
        {model.left.length === 0 ? (
          <p className="text-sm">The agent did not report anything left.</p>
        ) : (
          <ul className="flex flex-col gap-1">
            {model.left.map((item, i) => (
              <li
                // Names can repeat; order is stable.
                key={`${i}-${item.name}`}
                className="flex items-center gap-2 text-sm"
              >
                <span className="min-w-0 flex-1 break-words">{item.name}</span>
                <span className="shrink-0 text-muted-foreground">
                  {item.count}
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>
      <details className="text-sm">
        <summary className="cursor-pointer font-medium">
          What the agent did
        </summary>
        {log.length === 0 ? (
          <p className="mt-1">The panel did not see the agent's steps.</p>
        ) : (
          <ul className="mt-1 flex flex-col gap-1">
            {log.map((action, i) => (
              <li
                key={`${i}-${action}`}
                className="break-words font-ibm-plex-mono text-xs text-muted-foreground"
              >
                {action}
              </li>
            ))}
          </ul>
        )}
      </details>
      <p className="text-sm">
        Check the form on the page, then submit it there yourself.
      </p>
      {stopFailed ? (
        <p role="status" className="text-sm">
          We could not reach the app or the extension to confirm the stop.
        </p>
      ) : null}
      <div className="flex flex-col gap-2">
        <Button
          type="button"
          onClick={onSubmitted}
          disabled={stopping}
          aria-busy={stopping}
        >
          I submitted it
        </Button>
        <Button
          type="button"
          variant="outline"
          onClick={onStop}
          disabled={stopping}
          aria-busy={stopping}
        >
          Stop without submitting
        </Button>
      </div>
    </section>
  );
};
