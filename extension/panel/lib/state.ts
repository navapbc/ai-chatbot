import type { RelayConfig } from '../../../lib/relay/config';
import { AppApiError } from '../../src/app-api';
import { type ErrorMessage, messageFor } from './copy';

// The panel's screen state machine, pure so it can be tested without a DOM.
// `loading` is the sign-in probe in flight; later start-flow tasks replace the
// stub screens.

export type Screen =
  | 'loading'
  | 'home'
  | 'participant'
  | 'details'
  | 'ready'
  | 'starting'
  | 'running'
  | 'signedOut'
  | 'error';

export type State = {
  screen: Screen;
  config?: RelayConfig;
  error?: ErrorMessage;
};

export type Action =
  | { type: 'probeSucceeded'; config: RelayConfig }
  | { type: 'probeFailed'; error: unknown }
  | { type: 'retry' }
  | { type: 'goto'; screen: Screen }
  | { type: 'goHome' };

export const initialState: State = { screen: 'loading' };

export const reducer = (state: State, action: Action): State => {
  if (action.type === 'probeSucceeded') {
    return { screen: 'home', config: action.config };
  }
  if (action.type === 'probeFailed') {
    if (
      action.error instanceof AppApiError &&
      action.error.code === 'signed_out'
    ) {
      return { screen: 'signedOut' };
    }
    return { screen: 'error', error: messageFor(action.error) };
  }
  if (action.type === 'retry') return { screen: 'loading' };
  if (action.type === 'goto') return { ...state, screen: action.screen };
  if (action.type === 'goHome') return { ...state, screen: 'home' };
  return state;
};
