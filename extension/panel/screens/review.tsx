import { useEffect, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import type { RunProgress } from '../../../lib/relay/progress';
import { UNFILLED_COPY, unfilledStatusText } from '../lib/copy';
import { createUnfilledField, type UnfilledView } from '../lib/unfilled-field';
import { createWorkerApi, type WorkerApi } from '../lib/worker-api';
import {
  buildReviewModel,
  reviewActions,
  reviewLogNote,
} from '../lib/review-model';

type Props = {
  runId: string;
  /** The worker call; defaults to the real worker client. */
  showUnfilledField?: WorkerApi['showUnfilledField'];
  progress: RunProgress | null;
  actions: string[];
  stopping: boolean;
  /** The panel was rebuilt mid-run: the log starts empty. */
  openedMidRun?: boolean;
  onSubmitted: () => void;
  onStop: () => void;
};

const defaultShowUnfilledField: WorkerApi['showUnfilledField'] = (runId) =>
  createWorkerApi().showUnfilledField(runId);

const IDLE: UnfilledView = { busy: false, result: null };

// The agent's turn ended with nothing pending. The panel says only that the
// agent stopped, never that the form is complete or correct. Names and the
// action log are untrusted agent text, rendered as React text only.
export const ReviewPanel = ({
  runId,
  showUnfilledField = defaultShowUnfilledField,
  progress,
  actions,
  stopping,
  openedMidRun = false,
  onSubmitted,
  onStop,
}: Props) => {
  const headingRef = useRef<HTMLHeadingElement>(null);
  // Once, when the panel appears: not on every poll.
  useEffect(() => {
    headingRef.current?.focus();
  }, []);
  const [unfilled, setUnfilled] = useState<UnfilledView>(IDLE);
  const unfilledRef = useRef<ReturnType<typeof createUnfilledField> | null>(
    null,
  );
  // One controller per mounted panel: it goes with the review phase, so the
  // status line clears when the phase leaves review.
  useEffect(() => {
    const controller = createUnfilledField({ runId, showUnfilledField });
    unfilledRef.current = controller;
    // A replaced controller starts clean: no stale result or busy.
    setUnfilled(IDLE);
    const unsubscribe = controller.subscribe(setUnfilled);
    return () => {
      unsubscribe();
      controller.dispose();
      unfilledRef.current = null;
    };
  }, [runId, showUnfilledField]);
  const handleShowUnfilled = () => {
    void unfilledRef.current?.press();
  };
  const model = buildReviewModel(progress);
  const log = reviewActions(actions);
  const note = reviewLogNote(openedMidRun);
  return (
    <section
      aria-label="Ready for your review"
      className="flex flex-col gap-3 rounded-xl border border-accent bg-background p-3"
    >
      <h3 ref={headingRef} tabIndex={-1} className="text-base font-semibold">
        Ready for your review
      </h3>
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
        {note ? <p className="mt-1">{note}</p> : null}
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
      <div className="flex flex-col gap-2">
        <div className="flex flex-col gap-1">
          <Button
            type="button"
            variant="outline"
            onClick={handleShowUnfilled}
            aria-disabled={unfilled.busy}
            aria-busy={unfilled.busy}
            className="aria-disabled:pointer-events-none aria-disabled:opacity-50"
          >
            {UNFILLED_COPY.button}
          </Button>
          <p role="status" aria-live="polite" className="text-sm">
            {unfilled.result ? unfilledStatusText(unfilled.result) : null}
          </p>
        </div>
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
