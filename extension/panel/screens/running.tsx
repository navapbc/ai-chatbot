import { useEffect, useMemo, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import type {
  EndedReason,
  RunSession,
  RunSessionState,
  SessionRun,
} from '../lib/run-session';
import { allowedNote, showAllowSite } from '../lib/allow-site';
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
  /** "I submitted it": the run was stopped; show Confirmed. */
  onConfirmed: (stopFailed: boolean) => void;
  /** The build's app origin, for the default answer sender. */
  appOrigin?: string;
  /** Sends answers to the agent's questions; defaults to the real client. */
  sendAnswers?: SendAnswers;
  /** The allow-site card's worker and tab access; defaults to Chrome's. */
  allowSiteDeps?: AllowSiteCardDeps;
};

const ENDED_TEXT: Record<EndedReason, string> = {
  user: 'You stopped this run.',
  submitted: 'You said you submitted the form.',
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
  // Set once the caseworker allowed a site: the run can no longer be
  // extended (a refresh is refused on an allowed origin), which stays true
  // after the card itself goes away.
  const [allowed, setAllowed] = useState(false);
  const [state, setState] = useState<RunSessionState | null>(null);
  const session = useRef<RunSession | null>(null);
  const startedAt = useRef(Date.now());
  const [now, setNow] = useState(() => Date.now());
  const latest = useRef<RunSessionState | null>(null);
  const homeButton = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    const lifecycle = createControllerLifecycle(() => createSession(run));
    const current = lifecycle.mount();
    session.current = current;
    current.subscribe((next) => {
      latest.current = next;
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

  const handleStop = () => {
    void session.current?.stop();
  };
  const handleSubmitted = async () => {
    // Confirmed follows whether or not either stop worked: the caseworker
    // did submit. A failed stop shows a notice only while this screen lives.
    await session.current?.stop('submitted');
    onConfirmed(latest.current?.stopFailed === true);
  };
  const handleHome = () => onHome();
  const handleAllowed = () => setAllowed(true);

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
          {allowed && state ? (
            <p className="text-sm">
              {allowedNote(state.expiresAt, state.expiresAtEstimated === true)}
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
          progress={state?.progress ?? null}
          actions={state?.actions ?? []}
          stopping={stopping}
          onSubmitted={handleSubmitted}
          onStop={handleStop}
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
