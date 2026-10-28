import { describe, expect, it, vi } from 'vitest';
import {
  createWorkerApi,
  WorkerError,
} from '../../extension/panel/lib/worker-api';
import { TYPE_SENSITIVE_COPY } from '../../extension/panel/lib/copy';

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

describe('typeSensitive', () => {
  const VALUE = 'SECRET-VALUE-9';

  it('sends one message and resolves ok', async () => {
    const send = fake({ ok: true });
    expect(await createWorkerApi(send).typeSensitive('run1', VALUE)).toEqual({
      ok: true,
    });
    expect(send.mock.calls[0][0]).toEqual({
      type: 'typeSensitive',
      runId: 'run1',
      value: VALUE,
    });
  });

  it.each([
    ['not attached', 'not_attached'],
    ['run is held', 'run_held'],
    ['value not accepted', 'value_not_accepted'],
    ['no focused field', 'no_focused_field'],
    ['field not sensitive', 'field_not_sensitive'],
    ['field out of scope', 'field_out_of_scope'],
    ['type failed', 'type_failed'],
    ['no active run', 'worker_error'],
    ['wrong run', 'worker_error'],
    ['something new', 'worker_error'],
    ['constructor', 'worker_error'],
    ['__proto__', 'worker_error'],
  ])('maps %s to %s', async (text, code) => {
    const api = createWorkerApi(fake({ ok: false, error: text }));
    expect(await api.typeSensitive('run1', VALUE)).toEqual({
      ok: false,
      code,
    });
  });

  it('maps a malformed reply to worker_error', async () => {
    const api = createWorkerApi(fake(undefined));
    expect(await api.typeSensitive('run1', VALUE)).toEqual({
      ok: false,
      code: 'worker_error',
    });
  });

  it('drops a thrown error and never echoes the value', async () => {
    const api = createWorkerApi(async () => {
      throw new Error(`boom ${VALUE}`);
    });
    const result = await api.typeSensitive('run1', VALUE);
    expect(result).toEqual({ ok: false, code: 'worker_error' });
    expect(JSON.stringify(result)).not.toContain(VALUE);
    expect(JSON.stringify(result)).not.toContain('boom');
  });

  it('has no value property on any result', async () => {
    const ok = await createWorkerApi(
      fake({ ok: true, value: VALUE }),
    ).typeSensitive('r', VALUE);
    const bad = await createWorkerApi(
      fake({ ok: false, error: 'type failed', value: VALUE }),
    ).typeSensitive('r', VALUE);
    for (const r of [ok, bad]) {
      expect(r).not.toHaveProperty('value');
      expect(JSON.stringify(r)).not.toContain(VALUE);
    }
  });

  it('has fixed non-empty copy for every code', () => {
    const codes = [
      'not_attached',
      'run_held',
      'value_not_accepted',
      'no_focused_field',
      'field_not_sensitive',
      'field_out_of_scope',
      'type_failed',
      'worker_error',
    ] as const;
    for (const code of codes) {
      expect(TYPE_SENSITIVE_COPY[code].length).toBeGreaterThan(10);
    }
    expect(Object.keys(TYPE_SENSITIVE_COPY).sort()).toEqual([...codes].sort());
  });
});
