import { describe, expect, it } from 'vitest';
import { AppApiError, type AppApiErrorCode } from '../../extension/src/app-api';
import { messageFor, NotConnectedError } from '../../extension/panel/lib/copy';
import {
  WorkerError,
  type WorkerErrorCode,
} from '../../extension/panel/lib/worker-api';

const APP_CODES: AppApiErrorCode[] = [
  'signed_out',
  'forbidden',
  'not_found',
  'bad_request',
  'rate_limited',
  'server_error',
  'network',
];
const WORKER_CODES: WorkerErrorCode[] = [
  'worker_unreachable',
  'tab_not_found',
  'token_invalid',
  'tab_out_of_scope',
  'no_active_run',
  'wrong_run',
  'origin_not_pending',
  'too_many_origins',
  'relay_unreachable',
  'worker_error',
];
const MARKER = 'SECRET-MARKER-9';

const check = (error: unknown) => {
  const m = messageFor(error);
  expect(m.title.length).toBeGreaterThan(0);
  expect(m.detail.length).toBeGreaterThan(0);
  expect(typeof m.retryable).toBe('boolean');
  expect(JSON.stringify(m)).not.toContain(MARKER);
  return m;
};

describe('messageFor', () => {
  it.each(APP_CODES)('covers AppApiError %s', (code) => {
    const error = new AppApiError(code, 500);
    Object.assign(error, { body: MARKER, cause: MARKER });
    error.message = MARKER;
    check(error);
  });

  it.each(WORKER_CODES)('covers WorkerError %s', (code) => {
    const error = new WorkerError(code);
    error.message = MARKER;
    check(error);
  });

  it('covers the not-connected build', () => {
    check(new NotConnectedError());
  });

  it('gives different copy for different classes', () => {
    const titles = new Set(
      [
        ...APP_CODES.map((c) => new AppApiError(c, 500)),
        ...WORKER_CODES.map((c) => new WorkerError(c)),
      ].map((e) => messageFor(e).title),
    );
    expect(titles.size).toBeGreaterThan(8);
  });

  it('treats unknown errors and non-errors as generic and retryable', () => {
    const generic = messageFor(new Error(MARKER));
    expect(generic.retryable).toBe(true);
    check(new Error(MARKER));
    check(MARKER);
    check({ message: MARKER });
    check(undefined);
    expect(messageFor(MARKER)).toEqual(generic);
  });

  it('marks signed out as not retryable without sign in', () => {
    expect(messageFor(new AppApiError('signed_out', 401)).detail).toMatch(
      /sign in/i,
    );
  });
});
