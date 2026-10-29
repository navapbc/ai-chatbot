import { describe, expect, it, vi } from 'vitest';
import {
  createWorkerApi,
  WorkerError,
} from '../../extension/panel/lib/worker-api';
import {
  TYPE_SENSITIVE_COPY,
  UNFILLED_REFUSAL_COPY,
} from '../../extension/panel/lib/copy';

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
    expect(await createWorkerApi(send).getRunStatus()).toEqual({
      ...status,
      armed: false,
    });
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

describe('showUnfilledField', () => {
  const MARK = 'Marker-Label-Value-42';
  const fake = (reply: unknown) =>
    vi.fn(async (message: unknown) => {
      void message;
      return reply;
    });

  it('sends the message with the run id', async () => {
    const send = fake({ ok: true, found: false, remaining: 0 });
    await createWorkerApi(send).showUnfilledField('run-9');
    expect(send).toHaveBeenCalledWith({
      type: 'showUnfilledField',
      runId: 'run-9',
    });
  });

  it.each([
    ['not attached', 'not_attached'],
    ['run is held', 'run_held'],
    ['field out of scope', 'field_out_of_scope'],
    ['registration pending', 'registration_pending'],
    ['scan failed', 'scan_failed'],
    ['no active run', 'worker_error'],
    ['wrong run', 'worker_error'],
    ['something else', 'worker_error'],
    ['constructor', 'worker_error'],
  ])('maps %s to %s', async (error, code) => {
    const r = await createWorkerApi(
      fake({ ok: false, error }),
    ).showUnfilledField('r');
    expect(r).toEqual({ ok: false, code });
  });

  it('maps unreachable, thrown and malformed replies to worker_error without the text', async () => {
    const thrown = await createWorkerApi(async () => {
      throw new Error(MARK);
    }).showUnfilledField('r');
    expect(thrown).toEqual({ ok: false, code: 'worker_error' });
    expect(JSON.stringify(thrown)).not.toContain(MARK);
    for (const reply of [undefined, null, 'x', [1]]) {
      expect(await createWorkerApi(fake(reply)).showUnfilledField('r')).toEqual(
        {
          ok: false,
          code: 'worker_error',
        },
      );
    }
    const unknown = await createWorkerApi(
      fake({ ok: false, error: MARK }),
    ).showUnfilledField('r');
    expect(JSON.stringify(unknown)).not.toContain(MARK);
  });

  it('reduces a success to the fixed fields', async () => {
    const r = await createWorkerApi(
      fake({
        ok: true,
        found: true,
        remaining: 3,
        sensitive: true,
        capped: true,
        label: MARK,
        value: MARK,
        extra: { a: MARK },
      }),
    ).showUnfilledField('r');
    expect(r).toEqual({
      ok: true,
      found: true,
      remaining: 3,
      sensitive: true,
      capped: true,
    });
    expect(JSON.stringify(r)).not.toContain(MARK);
  });

  it('omits false flags and treats a non-true found as not found', async () => {
    const r = await createWorkerApi(
      fake({ ok: true, found: 'yes', remaining: 0, sensitive: 'x', capped: 1 }),
    ).showUnfilledField('r');
    expect(r).toEqual({ ok: true, found: false, remaining: 0 });
  });

  it.each([
    [-5, 0],
    [2.9, 2],
    [123456, 9999],
    [Number.NaN, 0],
    [Number.POSITIVE_INFINITY, 0],
    ['7', 0],
    [undefined, 0],
  ])('clamps remaining %s to %s', async (input, expected) => {
    const r = await createWorkerApi(
      fake({ ok: true, found: true, remaining: input }),
    ).showUnfilledField('r');
    expect(r).toMatchObject({ ok: true, remaining: expected });
  });

  it('has fixed copy for every code', () => {
    expect(Object.keys(UNFILLED_REFUSAL_COPY).sort()).toEqual(
      [
        'field_out_of_scope',
        'not_attached',
        'registration_pending',
        'run_held',
        'scan_failed',
        'worker_error',
      ].sort(),
    );
    for (const text of Object.values(UNFILLED_REFUSAL_COPY)) {
      expect(text.length).toBeGreaterThan(10);
    }
  });
});

describe('the submit watch', () => {
  const MARK = 'Marker-Label-Value-42';
  it('sends arm (with reset only when asked) and disarm with the run id', async () => {
    const send = fake({ ok: true });
    const api = createWorkerApi(send);
    expect(await api.armSubmitWatch('run-1')).toEqual({ ok: true });
    expect(await api.armSubmitWatch('run-1', { reset: true })).toEqual({
      ok: true,
    });
    expect(await api.armSubmitWatch('run-1', { reset: false })).toEqual({
      ok: true,
    });
    expect(await api.disarmSubmitWatch('run-1')).toEqual({ ok: true });
    expect(send.mock.calls.map((c) => c[0])).toEqual([
      { type: 'armSubmitWatch', runId: 'run-1' },
      { type: 'armSubmitWatch', runId: 'run-1', reset: true },
      { type: 'armSubmitWatch', runId: 'run-1' },
      { type: 'disarmSubmitWatch', runId: 'run-1' },
    ]);
  });

  it.each([
    ['no active run', 'no_active_run'],
    ['wrong run', 'wrong_run'],
    ['run is held', 'run_held'],
    ['not attached', 'not_attached'],
    ['constructor', 'worker_error'],
    [MARK, 'worker_error'],
  ])(
    'maps %s to %s, never throwing or keeping the text',
    async (error, code) => {
      const api = createWorkerApi(fake({ ok: false, error, extra: MARK }));
      const arm = await api.armSubmitWatch('r');
      const disarm = await api.disarmSubmitWatch('r');
      expect(arm).toEqual({ ok: false, code });
      expect(disarm).toEqual({ ok: false, code });
      expect(JSON.stringify([arm, disarm])).not.toContain(MARK);
    },
  );

  it('maps a rejection or a malformed reply to worker_error', async () => {
    const thrown = createWorkerApi(async () => {
      throw new Error(MARK);
    });
    expect(await thrown.armSubmitWatch('r')).toEqual({
      ok: false,
      code: 'worker_error',
    });
    expect(await thrown.disarmSubmitWatch('r')).toEqual({
      ok: false,
      code: 'worker_error',
    });
    for (const reply of [undefined, null, 'x', [1]]) {
      expect(await createWorkerApi(fake(reply)).armSubmitWatch('r')).toEqual({
        ok: false,
        code: 'worker_error',
      });
    }
  });
});

describe('getRunStatus: armed and submitted', () => {
  const status = (extra: Record<string, unknown>) =>
    createWorkerApi(
      fake({ ok: true, active: true, runId: 'r', ...extra }),
    ).getRunStatus();

  it('keeps armed only as true or false', async () => {
    expect((await status({ armed: true })).armed).toBe(true);
    expect((await status({ armed: 'yes' })).armed).toBe(false);
    expect((await status({})).armed).toBe(false);
  });

  it('reduces submitted to its time and a vetted number, dropping anything else', async () => {
    const s = await status({
      submitted: {
        at: 5,
        url: 'https://x/submitted?ssn=1',
        confirmation: { token: 'AB-1234', kind: 'confirmation', label: 'x' },
      },
    });
    expect(s.submitted).toEqual({
      at: 5,
      confirmation: { token: 'AB-1234', kind: 'confirmation' },
    });
    expect(
      (await status({ submitted: { at: 5, confirmation: OTHER_OK } }))
        .submitted,
    ).toEqual({ at: 5, confirmation: OTHER_OK });
  });

  it.each([
    [{ token: 'AB-1234', kind: 'case' }],
    [{ token: 'AB-1234' }],
    [{ token: '', kind: 'confirmation' }],
    [{ token: 'A'.repeat(33), kind: 'confirmation' }],
    [{ token: 'AB 1234', kind: 'confirmation' }],
    [{ token: '<b>1</b>', kind: 'confirmation' }],
    [{ token: 1234, kind: 'confirmation' }],
    ['AB-1234'],
    [null],
  ])('drops an invalid confirmation %j and keeps the time', async (c) => {
    const s = await status({ submitted: { at: 5, confirmation: c } });
    expect(s.submitted).toEqual({ at: 5 });
  });

  it('keeps a 32-character token and drops a submitted without a finite time', async () => {
    const token = `${'A'.repeat(30)}-1`;
    expect(
      (
        await status({
          submitted: { at: 5, confirmation: { token, kind: 'confirmation' } },
        })
      ).submitted?.confirmation?.token,
    ).toBe(token);
    for (const submitted of [
      { at: Number.NaN },
      { at: '5' },
      {},
      'x',
      { confirmation: { token: 'AB-1', kind: 'confirmation' } },
    ]) {
      expect((await status({ submitted })).submitted).toBeUndefined();
    }
  });
});

const OTHER_OK = { token: '20240017', kind: 'other' };
