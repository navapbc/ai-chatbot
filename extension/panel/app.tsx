import { useEffect, useReducer } from 'react';
import { APP_ORIGIN } from '../src/config';
import { createAppApi } from '../src/app-api';
import { NotConnectedError } from './lib/copy';
import { initialState, reducer, type Screen } from './lib/state';
import { ErrorScreen } from './screens/error';
import { Placeholder } from './screens/placeholder';
import { SignedOutScreen } from './screens/signed-out';

const STUB_NAMES: Partial<Record<Screen, string>> = {
  home: 'Home',
  participant: 'Participant',
  details: 'Details',
  ready: 'Ready',
  starting: 'Starting',
  running: 'Running',
};

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

export const App = () => {
  const [state, dispatch] = useReducer(reducer, initialState);

  useEffect(() => {
    if (state.screen !== 'loading') return;
    let current = true;
    probe().then(
      (config) => current && dispatch({ type: 'probeSucceeded', config }),
      (error) => current && dispatch({ type: 'probeFailed', error }),
    );
    return () => {
      current = false;
    };
  }, [state.screen]);

  const handleRetry = () => dispatch({ type: 'retry' });
  const handleHome = () => dispatch({ type: 'goHome' });
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
      {STUB_NAMES[state.screen] ? (
        <Placeholder
          name={STUB_NAMES[state.screen] as string}
          onHome={handleHome}
        />
      ) : null}
    </main>
  );
};
