import { useEffect, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import type {
  EndedReason,
  RunSession,
  RunSessionState,
  SessionRun,
} from '../lib/run-session';
import { createControllerLifecycle } from '../lib/search-controller';

type Props = {
  siteName: string;
  run: SessionRun;
  /** Builds the session for the run; stable for the screen's lifetime. */
  createSession: (run: SessionRun) => RunSession;
  /** Leaves the ended run's screen (dispatches runEnded). */
  onHome: () => void;
};

const ENDED_TEXT: Record<EndedReason, string> = {
  user: 'You stopped this run.',
  stopped: 'This run was stopped.',
  'worker-ended':
    'The run ended in the browser. The tab may have closed or moved to another site, or the time ran out.',
};

export const PENDING_ORIGIN_TEXT =
  'The page moved to another site. The agent is paused until you decide.';

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
}: Props) => {
  const [state, setState] = useState<RunSessionState | null>(null);
  const session = useRef<RunSession | null>(null);
  const startedAt = useRef(Date.now());
  const [now, setNow] = useState(() => Date.now());
  const homeButton = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    const lifecycle = createControllerLifecycle(() => createSession(run));
    const current = lifecycle.mount();
    session.current = current;
    current.subscribe(setState);
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
  const handleHome = () => onHome();

  const stopping = state?.stopping === true;
  const secondsLeft = state?.secondsLeft ?? 0;

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
          {state?.pendingOrigin ? (
            <p className="text-sm">{PENDING_ORIGIN_TEXT}</p>
          ) : null}
        </div>
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
