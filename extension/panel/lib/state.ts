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

export type Target = { tabId: number; origin: string; siteName: string };

export type Participant =
  | { kind: 'record'; recordId: string; name: string }
  | { kind: 'details'; details: Record<string, string> };

export type State = {
  screen: Screen;
  config?: RelayConfig;
  target?: Target;
  participant?: Participant;
  modelId?: string;
  error?: ErrorMessage;
};

export type Action =
  | { type: 'probeSucceeded'; config: RelayConfig }
  | { type: 'probeFailed'; error: unknown }
  | { type: 'retry' }
  | { type: 'goto'; screen: Screen }
  | { type: 'goHome' }
  | { type: 'chooseTarget'; target: Target }
  | { type: 'clearTarget' }
  | {
      type: 'chooseParticipant';
      participant: Extract<Participant, { kind: 'record' }>;
    }
  | { type: 'chooseDetails' }
  | {
      type: 'submitDetails';
      details: Record<string, string>;
      save: boolean;
    }
  | { type: 'chooseModel'; modelId: string }
  | { type: 'startRequested' }
  | { type: 'startFailed'; error: unknown }
  | { type: 'goBack' };

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
  if (action.type === 'chooseTarget') {
    return { ...state, target: action.target, screen: 'participant' };
  }
  if (action.type === 'clearTarget') {
    const { target: _target, participant: _participant, ...rest } = state;
    return { ...rest, screen: 'home' };
  }
  if (action.type === 'chooseParticipant') {
    return { ...state, participant: action.participant, screen: 'ready' };
  }
  if (action.type === 'chooseDetails') return { ...state, screen: 'details' };
  if (action.type === 'submitDetails') {
    return {
      ...state,
      participant: { kind: 'details', details: action.details },
      screen: 'ready',
    };
  }
  if (action.type === 'chooseModel') {
    return { ...state, modelId: action.modelId };
  }
  if (action.type === 'startRequested') {
    // Only a ready screen can start, so a second press changes nothing.
    if (state.screen !== 'ready') return state;
    const { error: _error, ...rest } = state;
    return { ...rest, screen: 'starting' };
  }
  if (action.type === 'startFailed') {
    if (state.screen !== 'starting') return state;
    return { ...state, screen: 'ready', error: messageFor(action.error) };
  }
  if (action.type === 'goBack') {
    if (state.screen !== 'ready') return state;
    const { error: _error, participant, ...rest } = state;
    return {
      ...rest,
      screen: participant?.kind === 'details' ? 'details' : 'participant',
    };
  }
  return state;
};
