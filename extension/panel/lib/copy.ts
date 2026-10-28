import { AppApiError } from '../../src/app-api';
import {
  type TypeSensitiveCode,
  type UnfilledFieldCode,
  type UnfilledFieldResult,
  WorkerError,
} from './worker-api';

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

/** The chosen tab is gone (closed, or unreadable) when a run starts. */
export class TabGoneError extends Error {
  constructor() {
    super('tab closed');
    this.name = 'TabGoneError';
  }
}

/** The chosen tab is no longer on the run's origin when a run starts. */
export class TabMovedError extends Error {
  constructor() {
    super('tab left the origin');
    this.name = 'TabMovedError';
  }
}

/** A start was asked for while another one is still in flight. */
export class StartInFlightError extends Error {
  constructor() {
    super('start in flight');
    this.name = 'StartInFlightError';
  }
}

/** The server says these questions were already answered (or replaced). */
export class AnswerAlreadyGivenError extends Error {
  constructor() {
    super('already answered');
    this.name = 'AnswerAlreadyGivenError';
  }
}

/** The server refused an answer's text (a sensitive value, or too long). */
export class AnswerRefusedError extends Error {
  constructor() {
    super('cannot send that here');
    this.name = 'AnswerRefusedError';
  }
}

/** The server refused an answer turn as too large. */
export class AnswerTooLongError extends Error {
  constructor() {
    super('answer too long');
    this.name = 'AnswerTooLongError';
  }
}

/** The server says the run has ended. */
export class RunEndedError extends Error {
  constructor() {
    super('run ended');
    this.name = 'RunEndedError';
  }
}

/** The panel refused to send an answer set (empty, too long, duplicated). */
export class AnswerInvalidError extends Error {
  constructor() {
    super('invalid answer');
    this.name = 'AnswerInvalidError';
  }
}

/** An answer for one of these questions is already on its way. */
export class AnswerInFlightError extends Error {
  constructor() {
    super('answer in flight');
    this.name = 'AnswerInFlightError';
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
  too_many_origins: {
    title: 'You have already allowed the most sites for this run',
    detail: 'Stop the run, or bring the tab back to the form.',
    retryable: false,
  },
  worker_error: GENERIC,
};

const NOT_CONNECTED: ErrorMessage = {
  title: 'This build is not connected to an app',
  detail: 'Install the extension from your team, then open this panel again.',
  retryable: false,
};

const START_IN_FLIGHT: ErrorMessage = {
  title: 'A run is already starting',
  detail: 'Wait for it to finish starting.',
  retryable: false,
};

const ANSWER_COPY = {
  alreadyAnswered: {
    title: 'This question was already answered',
    detail: 'Wait a moment for the agent to catch up.',
    retryable: false,
  },
  refused: {
    title: "We can't send that here",
    detail: 'Type that value in the page yourself, then press Skip.',
    retryable: true,
  },
  tooLong: {
    title: 'That answer is too long',
    detail: 'Shorten it, then press Send again.',
    retryable: true,
  },
  runEnded: {
    title: 'This run has ended',
    detail: 'Nothing more can be sent to it.',
    retryable: false,
  },
  invalid: {
    title: 'Check your answers',
    detail:
      'Every question needs an answer of 2000 characters or fewer, or Skip.',
    retryable: true,
  },
  inFlight: {
    title: 'Your answers are being sent',
    detail: 'Wait for them to finish sending.',
    retryable: false,
  },
} satisfies Record<string, ErrorMessage>;

export const messageFor = (error: unknown): ErrorMessage => {
  if (error instanceof AnswerAlreadyGivenError) {
    return { ...ANSWER_COPY.alreadyAnswered };
  }
  if (error instanceof AnswerRefusedError) return { ...ANSWER_COPY.refused };
  if (error instanceof AnswerTooLongError) return { ...ANSWER_COPY.tooLong };
  if (error instanceof RunEndedError) return { ...ANSWER_COPY.runEnded };
  if (error instanceof AnswerInvalidError) return { ...ANSWER_COPY.invalid };
  if (error instanceof AnswerInFlightError) return { ...ANSWER_COPY.inFlight };
  if (error instanceof NotConnectedError) return { ...NOT_CONNECTED };
  if (error instanceof TabGoneError) return { ...WORKER.tab_not_found };
  if (error instanceof TabMovedError) return { ...WORKER.tab_out_of_scope };
  if (error instanceof StartInFlightError) return { ...START_IN_FLIGHT };
  if (error instanceof AppApiError) return { ...(APP[error.code] ?? GENERIC) };
  if (error instanceof WorkerError) {
    return { ...(WORKER[error.code] ?? GENERIC) };
  }
  return { ...GENERIC };
};

/** The app refused an answer for a reason the card has no copy for. */
const ANSWER_BAD_REQUEST: ErrorMessage = {
  title: 'The app could not use that answer',
  detail: 'Wait a moment and try again.',
  retryable: true,
};

/** messageFor, with the Needs You card's own copy for a plain bad request. */
export const answerMessageFor = (error: unknown): ErrorMessage =>
  error instanceof AppApiError && error.code === 'bad_request'
    ? { ...ANSWER_BAD_REQUEST }
    : messageFor(error);

/** Fixed copy for the Needs You card. */
export const NEEDS_YOU_COPY = {
  title: 'The agent needs you',
  sensitive:
    'This one is sensitive. Type it in the page yourself first, then choose "I typed it in the page" and press Send. Or type it here and press Fill it in.',
  cannotAnswer:
    'This question cannot be answered here. The agent is waiting; you can Stop the run.',
  noNumbers:
    'Do not type Social Security, bank, card or ID numbers here. Type those in the page.',
  typeMyself: 'I typed it in the page',
  fillIt: 'Fill it in',
  fillInstruction: 'Click the field in the page, then press Fill it in.',
  fillInputLabel: 'Value to fill in',
  showTyped: 'Show what I typed',
  filled: 'Filled in.',
  skip: 'Skip',
  skipped: 'Skipped.',
  send: 'Send',
  sending: 'Sending...',
  sent: 'Sent. Waiting for the agent...',
} as const;

/**
 * Fixed copy for each way "Fill it in" can fail, chosen by code. The worker's
 * text and the value are never part of it.
 */
export const TYPE_SENSITIVE_COPY: Record<TypeSensitiveCode, string> = {
  not_attached:
    'The extension is not connected to this tab right now. Type it in the page yourself, then choose "I typed it in the page".',
  run_held:
    'The run is paused while the tab is on another site. Go back to the form page, or type it in the page yourself.',
  value_not_accepted:
    'The extension could not type that value here. Type it in the page yourself.',
  no_focused_field: 'Click the field in the page first, then press Fill it in.',
  field_not_sensitive:
    'The extension does not treat that field as sensitive, so it will not type there. Type it in the page yourself.',
  field_out_of_scope:
    'That field is outside the page the agent is working on. Click a field on the form page, or type it yourself.',
  type_failed:
    'Typing did not work. Click the field again and press Fill it in, or type it in the page yourself.',
  worker_error:
    'The extension could not type that. Type it in the page yourself.',
};

/**
 * Review's "Show me the unfilled field" result. Fixed text and one integer;
 * the worker's text, and any field's name, label or value, are never part of
 * it. The highlight may be only a scroll and a focus ring.
 */
export const UNFILLED_COPY = {
  button: 'Show me the unfilled field',
  look: 'Look for the field in your page that is highlighted or focused.',
  sensitive: 'It is a sensitive field: fill it in yourself in the page.',
  capped: 'There are many; showing the first.',
  none: 'No empty required fields found. Check the page before you submit.',
} as const;

export const UNFILLED_REFUSAL_COPY: Record<UnfilledFieldCode, string> = {
  run_held:
    'The run is paused while the tab is on another site. Go back to the form page and try again.',
  not_attached:
    'The extension is not connected to this tab right now. Look at the page yourself.',
  field_out_of_scope:
    'The field is outside the page the agent is working on. Look at the form page yourself.',
  registration_pending:
    'The extension is still getting ready. Wait a moment and try again.',
  scan_failed:
    'Could not check the page. Look at it yourself before you submit.',
  worker_error:
    'Could not check the page. Look at it yourself before you submit.',
};

/** The status line for a result: fixed copy and the integer only. */
export const unfilledStatusText = (result: UnfilledFieldResult): string => {
  if (!result.ok) return UNFILLED_REFUSAL_COPY[result.code];
  if (!result.found) return UNFILLED_COPY.none;
  const parts = [
    `Showing the next field that is still empty. ${result.remaining} left.`,
    UNFILLED_COPY.look,
  ];
  if (result.sensitive) parts.push(UNFILLED_COPY.sensitive);
  if (result.capped) parts.push(UNFILLED_COPY.capped);
  return parts.join(' ');
};

/** Fixed messages for the Home screen's tab choices. */
export const HOME_COPY = {
  httpsOnly:
    'The agent works on https pages only. Open the form page in this tab first.',
  cannotOpen: 'Could not open that site. Check the address and try again.',
  /** After a recent's site and time, for any end (a swept run included). */
  ended: ' (ended)',
} as const;

/**
 * Fixed copy for the allow-this-site card. The origin and the tab text are
 * passed in (a worker-validated https origin; the tab's origin plus path) and
 * rendered as text. Nothing here says the site is safe: allowing it is the
 * caseworker's decision for this run only.
 */
export const ALLOW_SITE_COPY = {
  title: 'Allow this site?',
  moved: (origin: string) =>
    `The page moved to ${origin}. The agent is paused until you decide.`,
  tabNow: (text: string) => `The tab is now on: ${text}`,
  hint: 'Allow it only if you expected the form to go there. This applies to this run only.',
  secondsLeft: (seconds: number) =>
    `About ${seconds} ${seconds === 1 ? 'second' : 'seconds'} left`,
  decideSoon: 'Decide soon: the agent stops waiting after about a minute.',
  holdMayHaveEnded: 'The hold may have ended',
  holdEnded: 'The hold ended. The agent has stopped waiting.',
  tooMany: 'You have already allowed the most sites for this run.',
  unreachable: 'The extension did not answer. Try again.',
  allow: 'Allow this site for this run',
  allowing: 'Allowing...',
  stop: 'Stop the run',
} as const;
