import { useCallback, useEffect, useMemo, useReducer } from 'react';
import { APP_ORIGIN } from '../src/config';
import { createAppApi } from '../src/app-api';
import { NotConnectedError } from './lib/copy';
import { markRecentEnded } from './lib/recents';
import {
  createRunSession,
  rebuildRun,
  type SessionRun,
} from './lib/run-session';
import { startRun } from './lib/run-starter';
import { type Action, initialState, reducer, type Target } from './lib/state';
import { createWorkerApi } from './lib/worker-api';
import { DetailsScreen } from './screens/details';
import { ErrorScreen } from './screens/error';
import { HomeScreen } from './screens/home';
import { ParticipantScreen } from './screens/participant';
import { type OnStart, ReadyScreen } from './screens/ready';
import { RunningScreen } from './screens/running';
import { SignedOutScreen } from './screens/signed-out';

const probe = async () => {
  if (!APP_ORIGIN) throw new NotConnectedError();
  try {
    return await createAppApi({ origin: APP_ORIGIN }).getConfig();
  } catch (error) {
    // An unusable APP_ORIGIN is a build problem, not a network one.
    if (error instanceof Error && error.message === 'invalid app origin') {
      throw new NotConnectedError();
    }
    throw error;
  }
};

// The real starter: Chrome's APIs and the build's APP_ORIGIN. A failure is
// thrown, so the Ready screen shows its fixed message and enables Start again.
const createStarter =
  (dispatch: (action: Action) => void): OnStart =>
  async ({ tabId, siteName, request }) => {
    if (!APP_ORIGIN) throw new NotConnectedError();
    const result = await startRun(
      {
        api: createAppApi({ origin: APP_ORIGIN }),
        worker: createWorkerApi(),
        fetch: (input, init) => globalThis.fetch(input, init),
        appOrigin: APP_ORIGIN,
        newId: () => crypto.randomUUID(),
        now: () => Date.now(),
        tabs: { get: (id) => chrome.tabs.get(id) },
      },
      { tabId, siteName, request },
    );
    if (!result.ok) throw result.error;
    dispatch({
      type: 'runStarted',
      runId: result.runId,
      expiresAt: result.expiresAt,
      tabId: result.tabId,
    });
  };

// The panel reopened mid-run: the worker's live run, if the server agrees.
const rebuild = () => {
  if (!APP_ORIGIN) return Promise.resolve(null);
  return rebuildRun({
    api: createAppApi({ origin: APP_ORIGIN }),
    worker: createWorkerApi(),
    now: () => Date.now(),
  });
};

export const App = ({ onStart }: { onStart?: OnStart }) => {
  const [state, dispatch] = useReducer(reducer, initialState);
  const starter = useMemo(() => onStart ?? createStarter(dispatch), [onStart]);

  useEffect(() => {
    if (state.screen !== 'loading') return;
    let current = true;
    probe().then(
      async (config) => {
        const rebuilt = await rebuild();
        if (!current) return;
        dispatch(
          rebuilt
            ? { type: 'runRebuilt', config, ...rebuilt }
            : { type: 'probeSucceeded', config },
        );
      },
      (error) => current && dispatch({ type: 'probeFailed', error }),
    );
    return () => {
      current = false;
    };
  }, [state.screen]);

  const api = useMemo(
    () => (APP_ORIGIN ? createAppApi({ origin: APP_ORIGIN }) : null),
    [],
  );

  const createSession = useCallback(
    (run: SessionRun) => {
      if (!api) throw new NotConnectedError();
      return createRunSession(
        {
          api,
          worker: createWorkerApi(),
          now: () => Date.now(),
          setInterval: (fn, ms) => setInterval(fn, ms),
          clearInterval: (id) => clearInterval(id),
          setTimeout: (fn, ms) => setTimeout(fn, ms),
          clearTimeout: (id) => clearTimeout(id),
          markRecentEnded,
        },
        run,
      );
    },
    [api],
  );

  const handleRetry = () => dispatch({ type: 'retry' });
  const handleHome = () => dispatch({ type: 'clearTarget' });
  const handleChoose = (target: Target) =>
    dispatch({ type: 'chooseTarget', target });
  const handleOpenApp = () => {
    chrome.tabs.create({ url: APP_ORIGIN });
  };

  return (
    <main className="flex min-h-screen flex-col gap-4 bg-background p-4 text-foreground">
      <h1 className="text-lg font-semibold">Labs ASP</h1>
      {state.screen === 'loading' ? (
        <p role="status" aria-live="polite" className="text-sm">
          Checking your sign-in...
        </p>
      ) : null}
      {state.screen === 'signedOut' ? (
        <SignedOutScreen onOpenApp={handleOpenApp} onRetry={handleRetry} />
      ) : null}
      {state.screen === 'error' && state.error ? (
        <ErrorScreen error={state.error} onRetry={handleRetry} />
      ) : null}
      {state.screen === 'home' ? (
        <HomeScreen sites={state.config?.sites ?? []} onChoose={handleChoose} />
      ) : null}
      {state.screen === 'participant' && state.target && api ? (
        <ParticipantScreen
          api={api}
          siteName={state.target.siteName}
          onChoose={(participant) =>
            dispatch({ type: 'chooseParticipant', participant })
          }
          onDetails={() => dispatch({ type: 'chooseDetails' })}
          onBack={handleHome}
          onSignedOut={handleRetry}
        />
      ) : null}
      {state.screen === 'details' ? (
        <DetailsScreen
          onSubmit={(details, save) =>
            dispatch({ type: 'submitDetails', details, save })
          }
          onBack={() => dispatch({ type: 'goto', screen: 'participant' })}
        />
      ) : null}
      {state.screen === 'ready' ? (
        <ReadyScreen state={state} dispatch={dispatch} onStart={starter} />
      ) : null}
      {state.screen === 'starting' ? (
        // No way off this screen: a start always ends in running or back on
        // Ready, and a run started while the panel was elsewhere would be
        // left with nothing to stop it.
        <section className="flex flex-col gap-4" aria-label="Starting">
          <h2 className="text-base font-semibold">Starting</h2>
          <p role="status" aria-live="polite" className="text-sm">
            Starting the run...
          </p>
        </section>
      ) : null}
      {state.screen === 'running' && state.run && state.target && api ? (
        // No Back to Home while the run is live: only Stop leaves it.
        <RunningScreen
          siteName={state.target.siteName}
          run={state.run}
          createSession={createSession}
          onHome={() => dispatch({ type: 'runEnded' })}
          appOrigin={APP_ORIGIN}
        />
      ) : null}
    </main>
  );
};
