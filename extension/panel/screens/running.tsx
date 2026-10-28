import { useEffect, useMemo, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import type {
  ConfirmedHow,
  EndedReason,
  RunSession,
  RunSessionState,
  SessionRun,
} from '../lib/run-session';
import type { Confirmation } from '../lib/worker-api';
import { showAllowSite } from '../lib/allow-site';
import { createAnswerSender } from '../lib/answer-client';
import { buildProgressView } from '../lib/progress-view';
import { createControllerLifecycle } from '../lib/search-controller';
import {
  AllowSiteCard,
  type AllowSiteCardDeps,
  defaultAllowSiteDeps,
} from './allow-site';
import {
  answerableRequests,
  NeedsYouCard,
  type SendAnswers,
} from './needs-you';
import { ProgressView } from './progress-view';
import { isConfirming } from '../lib/review-model';
import { ReviewPanel } from './review';

type Props = {
  siteName: string;
  run: SessionRun;
  /** Builds the session for the run; stable for the screen's lifetime. */
  createSession: (run: SessionRun) => RunSession;
  /** Leaves the ended run's screen (dispatches runEnded). */
  onHome: () => void;
  /**
   * The run ended as submitted ("I submitted it", "Yes", or a confirmation
   * number on the page): show Confirmed. Called once.
   */
  onConfirmed: (result: {
    stopFailed: boolean;
    how: ConfirmedHow;
    confirmation?: Confirmation;
  }) => void;
  /** The build's app origin, for the default answer sender. */
  appOrigin?: string;
  /** Sends answers to the agent's questions; defaults to the real client. */
  sendAnswers?: SendAnswers;
  /** The allow-site card's worker and tab access; defaults to Chrome's. */
  allowSiteDeps?: AllowSiteCardDeps;
};

export const ENDED_TEXT: Record<EndedReason, string> = {
  user: 'You stopped this run.',
  submitted: 'The run ended as submitted.',
  stopped: 'This run was stopped.',
  'worker-ended':
    'The run ended in the browser. The tab may have closed or moved to another site, or the time ran out.',
};

/** Elapsed time as m:ss (minutes are not capped at 59). */
export const formatElapsed = (ms: number): string => {
  const total = Math.max(0, Math.floor(ms / 1000));
  const minutes = Math.floor(total / 60);
  const seconds = total % 60;
  return `${minutes}:${seconds.toString().padStart(2, '0')}`;
};

const endsInText = (secondsLeft: number): string => {
  const minutes = Math.max(1, Math.ceil(secondsLeft / 60));
  const unit = minutes === 1 ? 'minute' : 'minutes';
  return `This run ends in ${minutes} ${unit}.`;
};

/**
 * The run in progress: site, elapsed time, notices, Stop. There is no way
 * off this screen while the run is live except Stop; once it ended, Back to
 * Home. The session is built in the effect (a StrictMode remount gets a fresh
 * one) and disposed on unmount. Applicant values never reach this screen.
 */
export const RunningScreen = ({
  siteName,
  run,
  createSession,
  onHome,
  onConfirmed,
  appOrigin = '',
  sendAnswers,
  allowSiteDeps,
}: Props) => {
  const send = useMemo(
    () => sendAnswers ?? createAnswerSender(appOrigin),
    [sendAnswers, appOrigin],
  );
  const allowDeps = useMemo(
    () => allowSiteDeps ?? defaultAllowSiteDeps(),
    [allowSiteDeps],
  );
  const [state, setState] = useState<RunSessionState | null>(null);
  const session = useRef<RunSession | null>(null);
  const startedAt = useRef(Date.now());
  const [now, setNow] = useState(() => Date.now());
  const homeButton = useRef<HTMLButtonElement>(null);
  const confirmedSent = useRef(false);

  useEffect(() => {
    const lifecycle = createControllerLifecycle(() => createSession(run));
    const current = lifecycle.mount();
    session.current = current;
    current.subscribe((next) => {
      setState(next);
    });
    return () => {
      lifecycle.unmount();
      session.current = null;
    };
  }, [run, createSession]);

  const ended = state?.status === 'ended';

  useEffect(() => {
    if (ended) return;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [ended]);

  useEffect(() => {
    if (ended) homeButton.current?.focus();
  }, [ended]);

  // Confirmed follows a submitted end whether or not either stop worked (the
  // form was submitted); a failed stop shows a notice there. One place for
  // every way in: "I submitted it", "Yes", and a number on the page.
  useEffect(() => {
    if (state?.status !== 'ended' || state.endedReason !== 'submitted') return;
    if (confirmedSent.current) return;
    confirmedSent.current = true;
    onConfirmed({
      stopFailed: state.stopFailed === true,
      how: state.confirmedHow ?? 'manual',
      ...(state.confirmation ? { confirmation: state.confirmation } : {}),
    });
  }, [state, onConfirmed]);

  const handleStop = () => {
    void session.current?.stop();
  };
  const handleSubmitted = () => {
    void session.current?.stop('submitted');
  };
  const handleConfirmSubmission = () => {
    void session.current?.confirmSubmission();
  };
  const handleDismissSubmission = () => {
    session.current?.dismissSubmission();
  };
  const handleHome = () => onHome();
  // The card goes away on its own; a refresh keeps an allowed origin.
  const handleAllowed = () => {};

  const stopping = state?.stopping === true;
  const questions =
    !ended && state?.phase === 'needs_you'
      ? answerableRequests(state.pending)
      : [];
  const secondsLeft = state?.secondsLeft ?? 0;
  const heldOn = showAllowSite(state);
  // Only for an agent that stopped with nothing pending, and never during a
  // hold (the allow-site card comes first). A stale review has no follow-up
  // input anywhere in the panel.
  const showReview =
    !ended && state?.phase === 'review' && !state.pendingOrigin && !heldOn;

  // Confirmed follows at once; nothing to show in between.
  if (isConfirming(state)) return null;

  return (
    <section className="flex flex-col gap-4" aria-label="Running">
      <h2 className="text-base font-semibold">
        {ended ? 'Run ended' : 'Running'}
      </h2>
      <p className="text-sm">Site: {siteName}</p>
      <p className="text-sm">
        Elapsed: {formatElapsed(now - startedAt.current)}
      </p>
      {ended && state?.endedReason ? (
        <div role="status" aria-live="polite" className="flex flex-col gap-2">
          <p className="text-sm">{ENDED_TEXT[state.endedReason]}</p>
          {state.stopFailed ? (
            <p className="text-sm">
              We could not reach the app or the extension to confirm the stop.
              The run is shown as ended here.
            </p>
          ) : null}
          <Button ref={homeButton} type="button" onClick={handleHome}>
            Back to Home
          </Button>
        </div>
      ) : null}
      {!ended ? (
        <div aria-live="polite" className="flex flex-col gap-2">
          {state?.expiresAtEstimated ? (
            <p className="text-sm">This run may end soon.</p>
          ) : null}
          {state?.refreshFailed ? (
            // Only once a refresh failed: a successful one keeps the run
            // going, so the countdown would only flicker around it.
            <p className="text-sm">
              We could not extend this run. It ends at its original time.
              {state.expiresAtEstimated ? null : ` ${endsInText(secondsLeft)}`}
            </p>
          ) : null}
        </div>
      ) : null}
      {heldOn && state ? (
        // Above the questions and the progress: the agent is paused until
        // the caseworker decides. Keyed by the origin, so a new origin gets
        // a fresh card naming it.
        <AllowSiteCard
          key={heldOn}
          origin={heldOn}
          pendingOrigin={state.pendingOrigin}
          stopping={stopping}
          tabId={run.tabId}
          holdEndsAt={state.holdEndsAt}
          deps={allowDeps}
          onAllowed={handleAllowed}
          onStop={handleStop}
        />
      ) : null}
      {questions.length > 0 ? (
        // Answers are offered only while the agent waits on them: never
        // while it works or after it stopped (a replayed turn can briefly
        // show a stale review).
        <NeedsYouCard
          key={questions.map((p) => p.requestId).join('\n')}
          runId={run.runId}
          pending={questions}
          sendAnswers={send}
        />
      ) : null}
      {!ended ? (
        <ProgressView
          model={buildProgressView(
            state?.progress ?? null,
            state?.actions ?? [],
          )}
        />
      ) : null}
      {showReview ? (
        <ReviewPanel
          runId={run.runId}
          progress={state?.progress ?? null}
          actions={state?.actions ?? []}
          stopping={stopping}
          openedMidRun={run.expiresAtEstimated === true}
          submission={state?.submission}
          onSubmitted={handleSubmitted}
          onStop={handleStop}
          onConfirmSubmission={handleConfirmSubmission}
          onDismissSubmission={handleDismissSubmission}
        />
      ) : null}
      {!ended ? (
        <Button
          type="button"
          variant="destructive"
          onClick={handleStop}
          disabled={stopping}
          aria-busy={stopping}
        >
          {stopping ? 'Stopping...' : 'Stop'}
        </Button>
      ) : null}
    </section>
  );
};
