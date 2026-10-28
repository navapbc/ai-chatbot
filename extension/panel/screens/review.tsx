import { useEffect, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import type { RunProgress } from '../../../lib/relay/progress';
import { SUBMIT_COPY, UNFILLED_COPY, unfilledStatusText } from '../lib/copy';
import type { SubmissionView } from '../lib/run-session';
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
  /** A submission the worker saw: checking for a number, or asking. */
  submission?: SubmissionView;
  onSubmitted: () => void;
  onStop: () => void;
  /** "Yes" to "Did you submit the form?". */
  onConfirmSubmission?: () => void;
  /** "No": the run goes on and the watch looks for the next submission. */
  onDismissSubmission?: () => void;
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
  submission,
  onSubmitted,
  onStop,
  onConfirmSubmission,
  onDismissSubmission,
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
      <p className="text-sm">{SUBMIT_COPY.reviewHint}</p>
      {/* Always mounted and empty unless checking: screen readers announce
          a change to a live region, not one inserted already filled. */}
      <p
        role="status"
        aria-live="polite"
        aria-label={SUBMIT_COPY.statusLabel}
        className="text-sm"
      >
        {submission?.stage === 'checking' ? SUBMIT_COPY.checking : null}
      </p>
      {submission?.stage === 'ask' ? (
        <SubmitPrompt
          token={submission.confirmation?.token}
          stopping={stopping}
          onYes={onConfirmSubmission}
          onNo={onDismissSubmission}
        />
      ) : null}
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

type SubmitPromptProps = {
  /** A number after a label other than "confirmation": shown, not trusted. */
  token?: string;
  stopping: boolean;
  onYes?: () => void;
  onNo?: () => void;
};

// The page moved on after a form was sent, with no confirmation number: the
// caseworker says whether it was the submission. The token is React text.
const SubmitPrompt = ({ token, stopping, onYes, onNo }: SubmitPromptProps) => {
  const headingRef = useRef<HTMLHeadingElement>(null);
  useEffect(() => {
    headingRef.current?.focus();
  }, []);
  const handleYes = () => onYes?.();
  const handleNo = () => onNo?.();
  return (
    <section
      aria-label={SUBMIT_COPY.askTitle}
      className="flex flex-col gap-2 rounded-lg border border-primary p-3"
    >
      <h4 ref={headingRef} tabIndex={-1} className="text-sm font-semibold">
        {SUBMIT_COPY.askTitle}
      </h4>
      <p className="text-sm">{SUBMIT_COPY.askBody}</p>
      {token ? (
        <p className="text-sm">
          {SUBMIT_COPY.numberOnPage}{' '}
          <span className="break-all font-ibm-plex-mono">{token}</span>
        </p>
      ) : null}
      <div className="flex gap-2">
        <Button
          type="button"
          onClick={handleYes}
          disabled={stopping}
          aria-busy={stopping}
        >
          {SUBMIT_COPY.yes}
        </Button>
        <Button
          type="button"
          variant="outline"
          onClick={handleNo}
          disabled={stopping}
        >
          {SUBMIT_COPY.no}
        </Button>
      </div>
    </section>
  );
};
