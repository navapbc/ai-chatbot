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
  'relay URL is invalid': 'relay_unreachable',
  'relay URL must use wss: (ws: only on 127.0.0.1 or localhost)':
    'relay_unreachable',
  'relay URL must not carry credentials': 'relay_unreachable',
};

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
  stopRun(): Promise<void>;
  addKnownValue(runId: string, value: string): Promise<void>;
  registrationComplete(runId: string, count: number): Promise<void>;
  allowOrigin(origin: string): Promise<void>;
  getRunStatus(): Promise<RunStatus>;
}

const isRecord = (v: unknown): v is Record<string, unknown> =>
  typeof v === 'object' && v !== null && !Array.isArray(v);

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
      const code =
        typeof reply.error === 'string'
          ? CODE_FOR_TEXT[reply.error]
          : undefined;
      throw new WorkerError(code ?? 'worker_error');
    }
    return reply;
  };
  const send = async (message: BridgeMessage): Promise<void> => {
    await call(message);
  };

  return {
    startRun: (args) => send({ type: 'startRun', ...args }),
    stopRun: () => send({ type: 'stopRun' }),
    addKnownValue: (runId, value) =>
      send({ type: 'addKnownValue', runId, value }),
    registrationComplete: (runId, count) =>
      send({ type: 'registrationComplete', runId, count }),
    allowOrigin: (origin) => send({ type: 'allowOrigin', origin }),
    getRunStatus: async () =>
      (await call({ type: 'getRunStatus' })) as unknown as RunStatus,
  };
};
