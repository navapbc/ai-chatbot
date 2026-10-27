import type { PendingInputRequest } from '../../lib/ai/eve/input-requests';
import type { ParticipantSummary } from '../../lib/data/participant-source';
import type { RelayConfig } from '../../lib/relay/config';
import type { RunProgress } from '../../lib/relay/progress';
import type { FieldDescriptor } from '../../lib/relay/sensitive-fields';

// A typed client for the web app's participant and relay routes, called from
// the extension with the caseworker's session cookie. Type-only imports from
// lib/: nothing server-side is bundled. Nothing here logs, and no method
// takes or returns a sensitive value: classifyFields sends descriptor keys
// only, and an error carries a status and a code, never response text.

export type AppApiErrorCode =
  | 'signed_out'
  | 'forbidden'
  | 'not_found'
  | 'bad_request'
  | 'rate_limited'
  | 'server_error'
  | 'network';

const MESSAGES: Record<AppApiErrorCode, string> = {
  signed_out: 'not signed in',
  forbidden: 'forbidden',
  not_found: 'not found',
  bad_request: 'bad request',
  rate_limited: 'rate limited',
  server_error: 'server error',
  network: 'network error',
};

export class AppApiError extends Error {
  readonly status: number;
  readonly code: AppApiErrorCode;
  constructor(code: AppApiErrorCode, status: number) {
    super(MESSAGES[code]);
    this.name = 'AppApiError';
    this.status = status;
    this.code = code;
  }
}

const codeForStatus = (status: number): AppApiErrorCode => {
  if (status === 401) return 'signed_out';
  if (status === 403) return 'forbidden';
  if (status === 404) return 'not_found';
  if (status === 400) return 'bad_request';
  if (status === 429) return 'rate_limited';
  return 'server_error';
};

export type RunState = {
  runId: string;
  origin: string;
  ended: boolean;
  endedAt: string | null;
  progress: RunProgress | null;
  pending: PendingInputRequest[];
};

export type ExtensionToken = {
  extensionToken: string;
  relayUrl: string;
  expiresAt: number;
};

export interface AppApi {
  searchParticipants(
    q: string,
    signal?: AbortSignal,
  ): Promise<ParticipantSummary[]>;
  getParticipant(recordId: string): Promise<ParticipantSummary>;
  getConfig(): Promise<RelayConfig>;
  getRunState(runId: string): Promise<RunState>;
  stopRun(runId: string): Promise<{ ended: boolean; alreadyEnded: boolean }>;
  getExtensionToken(runId: string): Promise<ExtensionToken>;
  classifyFields(
    runId: string,
    fields: readonly FieldDescriptor[],
  ): Promise<boolean[]>;
}

// Kept equal to DESCRIPTOR_KEYS in lib/jev/questions.ts by a test: that module
// imports the Jev SDK at runtime, which the extension must not bundle.
export const DESCRIPTOR_KEYS = [
  'role',
  'inputType',
  'autocomplete',
  'name',
  'label',
  'placeholder',
  'description',
] as const satisfies readonly (keyof FieldDescriptor)[];

const DEFAULT_TIMEOUT_MS = 10_000;
// The classify-field route refuses more than 50 fields in one request.
const MAX_FIELDS_PER_REQUEST = 50;
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const LOOPBACK = new Set(['localhost', '127.0.0.1', '[::1]']);

const validateOrigin = (origin: string): string => {
  let url: URL;
  try {
    url = new URL(origin);
  } catch {
    throw new Error('invalid app origin');
  }
  const secure =
    url.protocol === 'https:' ||
    (url.protocol === 'http:' && LOOPBACK.has(url.hostname));
  if (!secure || url.origin !== origin) throw new Error('invalid app origin');
  return url.origin;
};

const isRecord = (v: unknown): v is Record<string, unknown> =>
  typeof v === 'object' && v !== null && !Array.isArray(v);
const isStr = (v: unknown): v is string => typeof v === 'string';

const isSummary = (v: unknown): v is ParticipantSummary =>
  isRecord(v) && isStr(v.recordId) && isStr(v.name) && isStr(v.county);

const isConfig = (v: unknown): v is RelayConfig =>
  isRecord(v) &&
  Array.isArray(v.sites) &&
  Array.isArray(v.models) &&
  isStr(v.defaultModel);

const isRunState = (v: unknown): v is RunState =>
  isRecord(v) &&
  isStr(v.runId) &&
  isStr(v.origin) &&
  typeof v.ended === 'boolean' &&
  (v.endedAt === null || isStr(v.endedAt)) &&
  (v.progress === null || isRecord(v.progress)) &&
  Array.isArray(v.pending);

const isStop = (v: unknown): v is { ended: boolean; alreadyEnded: boolean } =>
  isRecord(v) &&
  typeof v.ended === 'boolean' &&
  typeof v.alreadyEnded === 'boolean';

const isToken = (v: unknown): v is ExtensionToken =>
  isRecord(v) &&
  isStr(v.extensionToken) &&
  isStr(v.relayUrl) &&
  typeof v.expiresAt === 'number';

const serverError = (status = 200) => new AppApiError('server_error', status);

const requireRunId = (runId: string): string => {
  if (!UUID.test(runId)) throw new AppApiError('bad_request', 0);
  return runId;
};

const toDescriptor = (raw: unknown): FieldDescriptor => {
  const out: FieldDescriptor = {};
  if (!isRecord(raw)) return out;
  for (const key of DESCRIPTOR_KEYS) {
    const v = raw[key];
    if (typeof v === 'string') out[key] = v;
  }
  return out;
};

export const createAppApi = (opts: {
  origin: string;
  fetch?: typeof fetch;
  timeoutMs?: number;
}): AppApi => {
  const origin = validateOrigin(opts.origin);
  const timeoutMs = opts.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const doFetch: typeof fetch =
    opts.fetch ?? ((input, init) => globalThis.fetch(input, init));

  // One request: fixed options, a timeout combined with the caller's signal,
  // status mapped to a code, the body parsed as JSON. Failures never carry
  // the response body or the underlying error text.
  const request = async (
    method: 'GET' | 'POST',
    path: string,
    body?: unknown,
    signal?: AbortSignal,
  ): Promise<unknown> => {
    if (signal?.aborted) throw new AppApiError('network', 0);
    const controller = new AbortController();
    const abort = () => controller.abort();
    signal?.addEventListener('abort', abort, { once: true });
    const timer = setTimeout(abort, timeoutMs);
    try {
      const headers: Record<string, string> = { Accept: 'application/json' };
      if (method === 'POST') headers['Content-Type'] = 'application/json';
      let response: Response;
      try {
        response = await doFetch(`${origin}${path}`, {
          method,
          headers,
          body: body === undefined ? undefined : JSON.stringify(body),
          credentials: 'include',
          cache: 'no-store',
          redirect: 'error',
          signal: controller.signal,
        });
      } catch {
        throw new AppApiError('network', 0);
      }
      if (!response.ok) {
        throw new AppApiError(codeForStatus(response.status), response.status);
      }
      try {
        return await response.json();
      } catch {
        // An abort mid-body is a timeout; anything else is a bad body.
        if (controller.signal.aborted) throw new AppApiError('network', 0);
        throw serverError(response.status);
      }
    } finally {
      clearTimeout(timer);
      signal?.removeEventListener('abort', abort);
    }
  };

  const parsed = async <T>(
    guard: (v: unknown) => v is T,
    method: 'GET' | 'POST',
    path: string,
    body?: unknown,
    signal?: AbortSignal,
  ): Promise<T> => {
    const value = await request(method, path, body, signal);
    if (!guard(value)) throw serverError();
    return value;
  };

  const classifyChunk = async (
    runId: string,
    chunk: FieldDescriptor[],
  ): Promise<boolean[]> => {
    const value = await request('POST', '/api/relay/classify-field', {
      runId,
      fields: chunk,
    });
    const sensitive = isRecord(value) ? value.sensitive : undefined;
    if (
      !Array.isArray(sensitive) ||
      sensitive.length !== chunk.length ||
      !sensitive.every((s) => typeof s === 'boolean')
    ) {
      throw serverError();
    }
    return sensitive as boolean[];
  };

  return {
    async searchParticipants(q, signal) {
      const value = await request(
        'GET',
        `/api/participants?q=${encodeURIComponent(q)}`,
        undefined,
        signal,
      );
      const list = isRecord(value) ? value.participants : undefined;
      if (!Array.isArray(list) || !list.every(isSummary)) throw serverError();
      return list;
    },
    async getParticipant(recordId) {
      // encodeURIComponent leaves "." and ".." alone, and a URL parser would
      // resolve them as path segments.
      if (!recordId || recordId === '.' || recordId === '..') {
        throw new AppApiError('bad_request', 0);
      }
      return parsed(
        isSummary,
        'GET',
        `/api/participants/${encodeURIComponent(recordId)}`,
      );
    },
    getConfig: () => parsed(isConfig, 'GET', '/api/relay/config'),
    getRunState: async (runId) =>
      parsed(isRunState, 'GET', `/api/relay/runs/${requireRunId(runId)}`),
    stopRun: async (runId) =>
      parsed(isStop, 'POST', `/api/relay/runs/${requireRunId(runId)}/stop`, {}),
    getExtensionToken: async (runId) =>
      parsed(isToken, 'POST', '/api/relay/token', {
        runId: requireRunId(runId),
      }),
    async classifyFields(runId, fields) {
      requireRunId(runId);
      const descriptors = fields.map(toDescriptor);
      const results: boolean[] = [];
      for (let i = 0; i < descriptors.length; i += MAX_FIELDS_PER_REQUEST) {
        const chunk = descriptors.slice(i, i + MAX_FIELDS_PER_REQUEST);
        results.push(...(await classifyChunk(runId, chunk)));
      }
      return results;
    },
  };
};
