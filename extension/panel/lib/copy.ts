import { AppApiError } from '../../src/app-api';
import { WorkerError } from './worker-api';

// Every message the panel shows for a failure comes from this table, chosen by
// the error's class and code. Nothing from an error's message, body or cause
// is ever copied into the output, so server or worker text cannot reach the
// screen.

export type ErrorMessage = {
  title: string;
  detail: string;
  retryable: boolean;
};

/** The build has no APP_ORIGIN, so the panel has no app to talk to. */
export class NotConnectedError extends Error {
  constructor() {
    super('not connected');
    this.name = 'NotConnectedError';
  }
}

const GENERIC: ErrorMessage = {
  title: 'Something went wrong',
  detail:
    'Try again. If it keeps happening, close this panel and open it again.',
  retryable: true,
};

const APP: Record<AppApiError['code'], ErrorMessage> = {
  signed_out: {
    title: 'You are signed out',
    detail: 'Sign in to the app in a browser tab, then try again.',
    retryable: true,
  },
  forbidden: {
    title: 'You do not have access',
    detail: 'Your account cannot do this. Ask your team to check your access.',
    retryable: false,
  },
  rate_limited: {
    title: 'Too many runs right now',
    detail:
      'You have reached the limit for now. Wait a few minutes, then try again.',
    retryable: true,
  },
  network: {
    title: 'Cannot reach the app',
    detail: 'Check your internet connection, then try again.',
    retryable: true,
  },
  server_error: {
    title: 'The app had a problem',
    detail: 'This is not your doing. Wait a moment, then try again.',
    retryable: true,
  },
  bad_request: {
    title: 'The app could not use that request',
    detail: 'Go back and check what you entered, then try again.',
    retryable: false,
  },
  not_found: {
    title: 'We could not find that',
    detail: 'It may have been removed. Go back to the start and try again.',
    retryable: false,
  },
};

const WORKER: Record<WorkerError['code'], ErrorMessage> = {
  worker_unreachable: {
    title: 'The extension is not responding',
    detail: 'Close this panel and open it again from the toolbar icon.',
    retryable: true,
  },
  tab_not_found: {
    title: 'The tab was closed',
    detail: 'Open the site in a tab, then start again.',
    retryable: false,
  },
  tab_out_of_scope: {
    title: 'The tab left the site',
    detail: 'Go back to the site you chose in that tab, then start again.',
    retryable: false,
  },
  token_invalid: {
    title: 'Your session timed out',
    detail: 'Start again to get a fresh one.',
    retryable: false,
  },
  relay_unreachable: {
    title: 'Cannot reach the connection service',
    detail: 'Check your internet connection, then try again.',
    retryable: true,
  },
  no_active_run: {
    title: 'No run is active',
    detail: 'Go back to the start and begin a new run.',
    retryable: false,
  },
  wrong_run: {
    title: 'That run has ended',
    detail: 'Go back to the start and begin a new run.',
    retryable: false,
  },
  origin_not_pending: {
    title: 'That site is no longer waiting',
    detail: 'Nothing is needed from you for it. Carry on with the run.',
    retryable: false,
  },
  worker_error: GENERIC,
};

const NOT_CONNECTED: ErrorMessage = {
  title: 'This build is not connected to an app',
  detail: 'Install the extension from your team, then open this panel again.',
  retryable: false,
};

export const messageFor = (error: unknown): ErrorMessage => {
  if (error instanceof NotConnectedError) return { ...NOT_CONNECTED };
  if (error instanceof AppApiError) return { ...(APP[error.code] ?? GENERIC) };
  if (error instanceof WorkerError) {
    return { ...(WORKER[error.code] ?? GENERIC) };
  }
  return { ...GENERIC };
};

/** Fixed messages for the Home screen's tab choices. */
export const HOME_COPY = {
  httpsOnly:
    'The agent works on https pages only. Open the form page in this tab first.',
  cannotOpen: 'Could not open that site. Check the address and try again.',
} as const;
