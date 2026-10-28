import type { RelayConfig } from '../../../lib/relay/config';
import { AppApiError } from '../../src/app-api';
import { type ErrorMessage, messageFor } from './copy';
import type { SessionRun } from './run-session';

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
  | 'confirmed'
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
  /**
   * The run this panel started (or rebuilt on reopening). `expiresAt` is the
   * extension token's expiry as the token route returns it: Unix time in
   * SECONDS, not milliseconds. `tabId` is the run's tab, which a token
   * refresh needs.
   */
  run?: SessionRun;
  error?: ErrorMessage;
  /** Set on the Confirmed screen: whether the stop could be confirmed. */
  confirmed?: { stopFailed: boolean };
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
  | { type: 'runStarted'; runId: string; expiresAt: number; tabId: number }
  /** The run ended and the caseworker left its screen. */
  | { type: 'runEnded' }
  /** The caseworker said they submitted; the run is over. */
  | { type: 'runConfirmed'; stopFailed: boolean }
  /** From Confirmed: the same person and site, back on Ready. */
  | { type: 'startAnother' }
  /** The panel reopened while the worker held a live run (see rebuildRun). */
  | {
      type: 'runRebuilt';
      config: RelayConfig;
      run: SessionRun;
      target: Target;
    }
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
    const {
      target: _target,
      participant: _participant,
      run: _run,
      confirmed: _confirmed,
      ...rest
    } = state;
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
  if (action.type === 'runStarted') {
    if (state.screen !== 'starting') return state;
    const { error: _error, ...rest } = state;
    return {
      ...rest,
      screen: 'running',
      run: {
        runId: action.runId,
        expiresAt: action.expiresAt,
        tabId: action.tabId,
      },
    };
  }
  if (action.type === 'runEnded') {
    if (state.screen !== 'running') return state;
    const {
      run: _run,
      target: _target,
      participant: _participant,
      ...rest
    } = state;
    return { ...rest, screen: 'home' };
  }
  if (action.type === 'runConfirmed') {
    if (state.screen !== 'running') return state;
    const { run: _run, error: _error, ...rest } = state;
    return {
      ...rest,
      screen: 'confirmed',
      confirmed: { stopFailed: action.stopFailed },
    };
  }
  if (action.type === 'startAnother') {
    if (state.screen !== 'confirmed') return state;
    const { run: _run, error: _error, confirmed: _confirmed, ...rest } = state;
    return { ...rest, screen: 'ready' };
  }
  if (action.type === 'runRebuilt') {
    if (state.screen !== 'loading') return state;
    return {
      screen: 'running',
      config: action.config,
      run: action.run,
      target: action.target,
    };
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
