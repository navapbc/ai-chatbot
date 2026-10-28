import { describe, expect, it, vi } from 'vitest';
import {
  createWorkerApi,
  WorkerError,
} from '../../extension/panel/lib/worker-api';

const fake = (reply: unknown) => vi.fn(async (_m: unknown) => reply);

describe('createWorkerApi message shapes', () => {
  it('sends each message exactly', async () => {
    const send = fake({ ok: true });
    const api = createWorkerApi(send);
    await api.startRun({ relayUrl: 'wss://r', token: 't', tabId: 4 });
    await api.startRun({
      relayUrl: 'wss://r',
      token: 't',
      tabId: 4,
      allowedFrameOrigins: ['https://a.example'],
    });
    await api.stopRun();
    await api.stopRun('run1');
    await api.addKnownValue('run1', 'v');
    await api.registrationComplete('run1', 2);
    await api.allowOrigin('https://a.example');
    expect(send.mock.calls.map((c) => c[0])).toEqual([
      { type: 'startRun', relayUrl: 'wss://r', token: 't', tabId: 4 },
      {
        type: 'startRun',
        relayUrl: 'wss://r',
        token: 't',
        tabId: 4,
        allowedFrameOrigins: ['https://a.example'],
      },
      { type: 'stopRun' },
      { type: 'stopRun', runId: 'run1' },
      { type: 'addKnownValue', runId: 'run1', value: 'v' },
      { type: 'registrationComplete', runId: 'run1', count: 2 },
      { type: 'allowOrigin', origin: 'https://a.example' },
    ]);
  });

  it('returns the typed status', async () => {
    const status = { ok: true, active: true, runId: 'r', attached: true };
    const send = fake(status);
    expect(await createWorkerApi(send).getRunStatus()).toEqual(status);
    expect(send.mock.calls[0][0]).toEqual({ type: 'getRunStatus' });
  });
});

describe('worker errors', () => {
  it.each([
    ['wrong run', 'wrong_run'],
    ['no active run', 'no_active_run'],
    ['tab not found', 'tab_not_found'],
    ['token is invalid or expired', 'token_invalid'],
    ['tab is outside the allowed origin', 'tab_out_of_scope'],
    ['origin not pending', 'origin_not_pending'],
    ['too many allowed origins', 'too_many_origins'],
    ['relay URL is invalid', 'relay_unreachable'],
    ['something new', 'worker_error'],
    // Exact own-key lookups: a prototype name is no code.
    ['constructor', 'worker_error'],
    ['toString', 'worker_error'],
    ['__proto__', 'worker_error'],
  ])('maps %s to %s with a fixed message', async (text, code) => {
    const api = createWorkerApi(fake({ ok: false, error: text }));
    const error = await api.stopRun().catch((e) => e);
    expect(error).toBeInstanceOf(WorkerError);
    expect(error.code).toBe(code);
    expect(error.message.length).toBeGreaterThan(0);
  });

  it('never keeps the worker text', async () => {
    const api = createWorkerApi(fake({ ok: false, error: 'MARKER-123' }));
    const error = await api.stopRun().catch((e) => e);
    expect(`${error.message} ${JSON.stringify(error)}`).not.toContain('MARKER');
  });

  it('maps a rejection and an empty reply to worker_unreachable', async () => {
    const rejecting = createWorkerApi(async () => {
      throw new Error('MARKER boom');
    });
    const a = await rejecting.stopRun().catch((e) => e);
    expect(a.code).toBe('worker_unreachable');
    expect(a.message).not.toContain('MARKER');
    const b = await createWorkerApi(fake(undefined))
      .getRunStatus()
      .catch((e) => e);
    expect(b.code).toBe('worker_unreachable');
  });
});
