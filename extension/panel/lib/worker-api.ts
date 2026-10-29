import type { BridgeMessage, RunStatus } from '../../src/bridge';

// The panel's client for the extension's service worker. Every method sends
// one BridgeMessage and resolves with the worker's reply. A refusal becomes a
// WorkerError whose code is chosen from the worker's fixed error strings; the
// worker's text is never kept, so nothing it says can reach the screen.

export type WorkerErrorCode =
  | 'worker_unreachable'
  | 'tab_not_found'
  | 'token_invalid'
  | 'tab_out_of_scope'
  | 'no_active_run'
  | 'wrong_run'
  | 'origin_not_pending'
  | 'too_many_origins'
  | 'relay_unreachable'
  | 'worker_error';

const MESSAGES: Record<WorkerErrorCode, string> = {
  worker_unreachable: 'extension worker unreachable',
  tab_not_found: 'tab not found',
  token_invalid: 'token invalid',
  tab_out_of_scope: 'tab outside the allowed origin',
  no_active_run: 'no active run',
  wrong_run: 'wrong run',
  origin_not_pending: 'origin not pending',
  too_many_origins: 'too many allowed origins',
  relay_unreachable: 'relay unreachable',
  worker_error: 'worker error',
};

export class WorkerError extends Error {
  readonly code: WorkerErrorCode;
  constructor(code: WorkerErrorCode) {
    super(MESSAGES[code]);
    this.name = 'WorkerError';
    this.code = code;
  }
}

// The worker's own error strings (extension/src/bridge.ts), by exact text.
const CODE_FOR_TEXT: Record<string, WorkerErrorCode> = {
  'tab not found': 'tab_not_found',
  'token is invalid or expired': 'token_invalid',
  'tab is outside the allowed origin': 'tab_out_of_scope',
  'no active run': 'no_active_run',
  'wrong run': 'wrong_run',
  'origin not pending': 'origin_not_pending',
  'too many allowed origins': 'too_many_origins',
  'relay URL is invalid': 'relay_unreachable',
  'relay URL must use wss: (ws: only on 127.0.0.1 or localhost)':
    'relay_unreachable',
  'relay URL must not carry credentials': 'relay_unreachable',
};

export type TypeSensitiveCode =
  | 'not_attached'
  | 'run_held'
  | 'value_not_accepted'
  | 'no_focused_field'
  | 'field_not_sensitive'
  | 'field_out_of_scope'
  | 'type_failed'
  | 'worker_error';

export type TypeSensitiveResult =
  | { ok: true }
  | { ok: false; code: TypeSensitiveCode };

// typeSensitive's refusals (extension/src/bridge.ts) by exact text. 'no active
// run' and 'wrong run' are absent on purpose: they fall to worker_error.
const TYPE_CODE_FOR_TEXT: Record<string, TypeSensitiveCode> = {
  'not attached': 'not_attached',
  'run is held': 'run_held',
  'value not accepted': 'value_not_accepted',
  'no focused field': 'no_focused_field',
  'field not sensitive': 'field_not_sensitive',
  'field out of scope': 'field_out_of_scope',
  'type failed': 'type_failed',
};

export type UnfilledFieldCode =
  | 'not_attached'
  | 'run_held'
  | 'field_out_of_scope'
  | 'registration_pending'
  | 'scan_failed'
  | 'worker_error';

export type UnfilledFieldResult =
  | {
      ok: true;
      found: boolean;
      remaining: number;
      sensitive?: boolean;
      capped?: boolean;
    }
  | { ok: false; code: UnfilledFieldCode };

// showUnfilledField's refusals (extension/src/bridge.ts) by exact text. 'no
// active run' and 'wrong run' are absent on purpose: they fall to worker_error.
const UNFILLED_CODE_FOR_TEXT: Record<string, UnfilledFieldCode> = {
  'not attached': 'not_attached',
  'run is held': 'run_held',
  'field out of scope': 'field_out_of_scope',
  'registration pending': 'registration_pending',
  'scan failed': 'scan_failed',
};

const isRecord = (v: unknown): v is Record<string, unknown> =>
  typeof v === 'object' && v !== null && !Array.isArray(v);

export type SubmitWatchCode =
  | 'no_active_run'
  | 'wrong_run'
  | 'run_held'
  | 'not_attached'
  | 'worker_error';

export type SubmitWatchResult =
  | { ok: true }
  | { ok: false; code: SubmitWatchCode };

// armSubmitWatch's and disarmSubmitWatch's refusals (extension/src/bridge.ts)
// by exact text; disarm answers only the first two.
const WATCH_CODE_FOR_TEXT: Record<string, SubmitWatchCode> = {
  'no active run': 'no_active_run',
  'wrong run': 'wrong_run',
  'run is held': 'run_held',
  'not attached': 'not_attached',
};

/** A confirmation number the worker read off the page, and its label kind. */
export type Confirmation = { token: string; kind: 'confirmation' | 'other' };

/** What the panel keeps of a submission the worker saw. */
export type Submitted = { at: number; confirmation?: Confirmation };

/** The longest token the panel shows (the worker's own limit). */
export const MAX_TOKEN = 32;
const TOKEN_SHAPE = /^[A-Za-z0-9\-_/.]+$/;

/**
 * Reduces a confirmation to exactly `{token, kind}`, or undefined: a token
 * that is not a string of 1 to 32 characters from a fixed set is dropped,
 * never cut (a shortened number is a wrong number), and so is any kind but
 * the two known ones.
 */
export const reduceConfirmation = (raw: unknown): Confirmation | undefined => {
  if (!isRecord(raw)) return undefined;
  const { token, kind } = raw;
  if (typeof token !== 'string') return undefined;
  if (token.length < 1 || token.length > MAX_TOKEN) return undefined;
  if (!TOKEN_SHAPE.test(token)) return undefined;
  if (kind !== 'confirmation' && kind !== 'other') return undefined;
  return { token, kind };
};

/** Reduces a submission to `{at}` plus a reduced confirmation, or undefined. */
export const reduceSubmitted = (raw: unknown): Submitted | undefined => {
  if (!isRecord(raw)) return undefined;
  if (typeof raw.at !== 'number' || !Number.isFinite(raw.at)) return undefined;
  const confirmation = reduceConfirmation(raw.confirmation);
  return confirmation ? { at: raw.at, confirmation } : { at: raw.at };
};

/** The most `remaining` the panel will carry. */
export const MAX_REMAINING = 9999;

type Send = (message: unknown) => Promise<unknown>;

const chromeSend: Send = (message) =>
  new Promise((resolve, reject) => {
    try {
      chrome.runtime.sendMessage(message, (reply: unknown) => {
        if (chrome.runtime.lastError) {
          reject(new Error('unreachable'));
          return;
        }
        resolve(reply);
      });
    } catch {
      reject(new Error('unreachable'));
    }
  });

export interface WorkerApi {
  startRun(args: {
    relayUrl: string;
    token: string;
    tabId: number;
    allowedFrameOrigins?: string[];
  }): Promise<void>;
  /**
   * Stops the worker's run. With `runId`, only if that run is the current one
   * (another run is left alone and the call still succeeds).
   */
  stopRun(runId?: string): Promise<void>;
  addKnownValue(runId: string, value: string): Promise<void>;
  registrationComplete(runId: string, count: number): Promise<void>;
  allowOrigin(origin: string): Promise<void>;
  getRunStatus(): Promise<RunStatus>;
  /**
   * Types a sensitive value into the focused field in the page. The result
   * carries a code only: never the value and never the worker's text.
   */
  typeSensitive(runId: string, value: string): Promise<TypeSensitiveResult>;
  /**
   * Highlights the next required field still empty in the page. The result
   * is booleans and a count, or a code: never a label, a value or the
   * worker's text.
   */
  showUnfilledField(runId: string): Promise<UnfilledFieldResult>;
  /**
   * Arms the worker's submit watch for this run (Review only). With
   * `reset`, a submission it already saw is forgotten first. A code only.
   */
  armSubmitWatch(
    runId: string,
    options?: { reset?: boolean },
  ): Promise<SubmitWatchResult>;
  /** Disarms the submit watch and forgets what it saw. A code only. */
  disarmSubmitWatch(runId: string): Promise<SubmitWatchResult>;
}

export const createWorkerApi = (sendMessage: Send = chromeSend): WorkerApi => {
  const call = async (
    message: BridgeMessage,
  ): Promise<Record<string, unknown>> => {
    let reply: unknown;
    try {
      reply = await sendMessage(message);
    } catch {
      throw new WorkerError('worker_unreachable');
    }
    if (!isRecord(reply)) throw new WorkerError('worker_unreachable');
    if (reply.ok !== true) {
      // Own keys only: a prototype name ('constructor') is no code.
      const code =
        typeof reply.error === 'string' &&
        Object.hasOwn(CODE_FOR_TEXT, reply.error)
          ? CODE_FOR_TEXT[reply.error]
          : undefined;
      throw new WorkerError(code ?? 'worker_error');
    }
    return reply;
  };
  const send = async (message: BridgeMessage): Promise<void> => {
    await call(message);
  };

  const typeSensitive = async (
    runId: string,
    value: string,
  ): Promise<TypeSensitiveResult> => {
    let reply: unknown;
    try {
      reply = await sendMessage({ type: 'typeSensitive', runId, value });
    } catch {
      return { ok: false, code: 'worker_error' };
    }
    if (!isRecord(reply)) return { ok: false, code: 'worker_error' };
    if (reply.ok === true) return { ok: true };
    const code =
      typeof reply.error === 'string' &&
      Object.hasOwn(TYPE_CODE_FOR_TEXT, reply.error)
        ? TYPE_CODE_FOR_TEXT[reply.error]
        : 'worker_error';
    return { ok: false, code };
  };

  const showUnfilledField = async (
    runId: string,
  ): Promise<UnfilledFieldResult> => {
    let reply: unknown;
    try {
      reply = await sendMessage({ type: 'showUnfilledField', runId });
    } catch {
      return { ok: false, code: 'worker_error' };
    }
    if (!isRecord(reply)) return { ok: false, code: 'worker_error' };
    if (reply.ok === true) {
      const remaining =
        typeof reply.remaining === 'number' && Number.isFinite(reply.remaining)
          ? Math.min(MAX_REMAINING, Math.max(0, Math.floor(reply.remaining)))
          : 0;
      const result: UnfilledFieldResult = {
        ok: true,
        found: reply.found === true,
        remaining,
      };
      if (reply.sensitive === true) result.sensitive = true;
      if (reply.capped === true) result.capped = true;
      return result;
    }
    const code =
      typeof reply.error === 'string' &&
      Object.hasOwn(UNFILLED_CODE_FOR_TEXT, reply.error)
        ? UNFILLED_CODE_FOR_TEXT[reply.error]
        : 'worker_error';
    return { ok: false, code };
  };

  const watchCall = async (
    message: BridgeMessage,
  ): Promise<SubmitWatchResult> => {
    let reply: unknown;
    try {
      reply = await sendMessage(message);
    } catch {
      return { ok: false, code: 'worker_error' };
    }
    if (!isRecord(reply)) return { ok: false, code: 'worker_error' };
    if (reply.ok === true) return { ok: true };
    const code =
      typeof reply.error === 'string' &&
      Object.hasOwn(WATCH_CODE_FOR_TEXT, reply.error)
        ? WATCH_CODE_FOR_TEXT[reply.error]
        : 'worker_error';
    return { ok: false, code };
  };

  const getRunStatus = async (): Promise<RunStatus> => {
    const reply = await call({ type: 'getRunStatus' });
    // `armed` is a flag and `submitted` is reduced to its fixed shape:
    // nothing else the worker might add rides along with the page's token.
    const { armed: _armed, submitted: rawSubmitted, ...rest } = reply;
    const status = rest as unknown as RunStatus;
    if (reply.active === true) status.armed = reply.armed === true;
    const submitted = reduceSubmitted(rawSubmitted);
    if (submitted) status.submitted = submitted;
    return status;
  };

  return {
    typeSensitive,
    showUnfilledField,
    armSubmitWatch: (runId, options) =>
      watchCall(
        options?.reset === true
          ? { type: 'armSubmitWatch', runId, reset: true }
          : { type: 'armSubmitWatch', runId },
      ),
    disarmSubmitWatch: (runId) =>
      watchCall({ type: 'disarmSubmitWatch', runId }),
    startRun: (args) => send({ type: 'startRun', ...args }),
    stopRun: (runId) =>
      send(
        runId === undefined ? { type: 'stopRun' } : { type: 'stopRun', runId },
      ),
    addKnownValue: (runId, value) =>
      send({ type: 'addKnownValue', runId, value }),
    registrationComplete: (runId, count) =>
      send({ type: 'registrationComplete', runId, count }),
    allowOrigin: (origin) => send({ type: 'allowOrigin', origin }),
    getRunStatus,
  };
};
