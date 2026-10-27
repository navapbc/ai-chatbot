import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { InputAnswer } from '../../lib/ai/eve/input-requests';
import { AppApiError } from '../../extension/src/app-api';
import {
  answerRequests,
  ANSWER_TIMEOUT_MS,
} from '../../extension/panel/lib/answer-client';
import {
  AnswerAlreadyGivenError,
  AnswerInFlightError,
  AnswerInvalidError,
  AnswerRefusedError,
  messageFor,
  RunEndedError,
} from '../../extension/panel/lib/copy';

const APP = 'https://app.example.org';
const RUN = { runId: '11111111-2222-4333-8444-555555555555' };
const SECRET = 'Zelda-Quill-Secret-Answer';
const SECRET_OPTION = 'opt-secret-77';

type Plan =
  | { kind: 'chunk' }
  | { kind: 'empty' }
  | { kind: 'reject' }
  | { kind: 'hang' }
  | { kind: 'status'; status: number; body?: string }
  | { kind: 'hangingBody'; status: number };

const makeFetch = (plan: Plan) => {
  const state = {
    calls: 0,
    url: '',
    init: undefined as RequestInit | undefined,
    reads: 0,
    cancelled: false,
  };
  const fetchFn = vi.fn(async (url: string, init?: RequestInit) => {
    state.calls += 1;
    state.url = url;
    state.init = init;
    const signal = init?.signal;
    if (plan.kind === 'reject') throw new TypeError('Failed to fetch');
    if (plan.kind === 'hang') {
      return new Promise<Response>((_resolve, reject) => {
        signal?.addEventListener('abort', () =>
          reject(new DOMException('aborted', 'AbortError')),
        );
      });
    }
    if (plan.kind === 'status') {
      return new Response(plan.body ?? null, { status: plan.status });
    }
    if (plan.kind === 'hangingBody') {
      // Headers arrive, the body never does.
      const body = new ReadableStream<Uint8Array>({ start() {} });
      return new Response(body, { status: plan.status });
    }
    let reads = 0;
    const reader = {
      read: vi.fn(() => {
        state.reads += 1;
        reads += 1;
        if (plan.kind === 'empty') {
          return Promise.resolve({ done: true, value: undefined });
        }
        if (reads === 1) {
          return Promise.resolve({
            done: false,
            value: new TextEncoder().encode('data: {"type":"start"}\n\n'),
          });
        }
        return new Promise(() => {});
      }),
      cancel: vi.fn(async () => {
        state.cancelled = true;
      }),
    };
    return {
      ok: true,
      status: 200,
      body: { getReader: () => reader },
    } as unknown as Response;
  });
  return { fetchFn, state };
};

const deps = (plan: Plan) => {
  const { fetchFn, state } = makeFetch(plan);
  return {
    state,
    deps: { fetch: fetchFn as unknown as typeof fetch, appOrigin: APP },
  };
};

const TEXT: InputAnswer = { requestId: 'r-text', text: `  ${SECRET}  ` };
const OPTION: InputAnswer = { requestId: 'r-opt', optionId: SECRET_OPTION };

const consoleSpies: ReturnType<typeof vi.spyOn>[] = [];

beforeEach(() => {
  for (const method of ['log', 'info', 'warn', 'error', 'debug'] as const) {
    consoleSpies.push(vi.spyOn(console, method));
  }
});

afterEach(() => {
  for (const spy of consoleSpies) expect(spy).not.toHaveBeenCalled();
  for (const spy of consoleSpies.splice(0)) spy.mockRestore();
  vi.useRealTimers();
});

const expectNoAnswer = (value: unknown) => {
  const text = JSON.stringify(value, Object.getOwnPropertyNames(value ?? {}));
  expect(text).not.toContain(SECRET);
  expect(text).not.toContain(SECRET_OPTION);
  if (value instanceof Error) {
    expect(value.message).not.toContain(SECRET);
    expect(String(value.stack)).not.toContain(SECRET);
  }
};

const failure = (result: { ok: boolean; error?: unknown }) => {
  expect(result.ok).toBe(false);
  const error = (result as { error: unknown }).error;
  expectNoAnswer(error);
  expectNoAnswer(messageFor(error));
  return error;
};

describe('answerRequests: the request', () => {
  it('posts exactly { id, inputResponses } for a text answer, trimmed', async () => {
    const { deps: d, state } = deps({ kind: 'chunk' });
    const result = await answerRequests(d, RUN, [TEXT]);
    expect(result).toEqual({ ok: true });
    expect(state.url).toBe(`${APP}/api/eve-chat`);
    expect(JSON.parse(state.init?.body as string)).toEqual({
      id: RUN.runId,
      inputResponses: [{ requestId: 'r-text', text: SECRET }],
    });
    expect(Object.keys(JSON.parse(state.init?.body as string))).toEqual([
      'id',
      'inputResponses',
    ]);
  });

  it('posts an option answer and a mixed batch in order', async () => {
    const { deps: d, state } = deps({ kind: 'chunk' });
    expect(await answerRequests(d, RUN, [OPTION])).toEqual({ ok: true });
    expect(JSON.parse(state.init?.body as string)).toEqual({
      id: RUN.runId,
      inputResponses: [{ requestId: 'r-opt', optionId: SECRET_OPTION }],
    });
    expect(await answerRequests(d, RUN, [TEXT, OPTION])).toEqual({ ok: true });
    expect(JSON.parse(state.init?.body as string)).toEqual({
      id: RUN.runId,
      inputResponses: [
        { requestId: 'r-text', text: SECRET },
        { requestId: 'r-opt', optionId: SECRET_OPTION },
      ],
    });
  });

  it('drops unknown keys on an answer', async () => {
    const { deps: d, state } = deps({ kind: 'chunk' });
    const extra = { ...OPTION, note: 'x' } as InputAnswer;
    await answerRequests(d, RUN, [extra]);
    expect(
      (JSON.parse(state.init?.body as string) as { inputResponses: unknown })
        .inputResponses,
    ).toEqual([{ requestId: 'r-opt', optionId: SECRET_OPTION }]);
  });

  it('uses the starter options and reads only to the first chunk', async () => {
    const { deps: d, state } = deps({ kind: 'chunk' });
    await answerRequests(d, RUN, [OPTION]);
    const init = state.init as RequestInit;
    expect(init.method).toBe('POST');
    expect(init.headers).toEqual({ 'Content-Type': 'application/json' });
    expect(init.credentials).toBe('include');
    expect(init.cache).toBe('no-store');
    expect(init.redirect).toBe('error');
    expect(init.signal).toBeInstanceOf(AbortSignal);
    expect(state.reads).toBe(1);
    await vi.waitFor(() => expect(state.cancelled).toBe(true));
  });

  it('does not mutate the answers array or its entries', async () => {
    const { deps: d } = deps({ kind: 'chunk' });
    const answers = [{ ...TEXT }, { ...OPTION }];
    const copy = JSON.parse(JSON.stringify(answers));
    await answerRequests(d, RUN, answers);
    expect(answers).toEqual(copy);
  });
});

describe('answerRequests: failures', () => {
  const statusCases: [number, string][] = [
    [401, 'signed_out'],
    [403, 'forbidden'],
    [429, 'rate_limited'],
    [500, 'server_error'],
    [503, 'server_error'],
  ];
  for (const [status, code] of statusCases) {
    it(`maps ${status} to ${code}`, async () => {
      const { deps: d } = deps({
        kind: 'status',
        status,
        body: `{"cause":"answers_pending","echo":"${SECRET}"}`,
      });
      const error = failure(await answerRequests(d, RUN, [TEXT]));
      expect(error).toBeInstanceOf(AppApiError);
      expect((error as AppApiError).code).toBe(code);
    });
  }

  const body400 = (cause: unknown) =>
    JSON.stringify({ code: 'bad_request:api', message: 'm', cause });

  const reasonCases: [string, new () => Error][] = [
    ['unknown_request', AnswerAlreadyGivenError],
    ['answers_pending', AnswerAlreadyGivenError],
    ['sensitive_value_in_message', AnswerRefusedError],
    ['message_too_large', AnswerRefusedError],
    ['run_ended', RunEndedError],
  ];
  for (const [reason, cls] of reasonCases) {
    it(`maps 400 ${reason} to its fixed error`, async () => {
      const { deps: d } = deps({
        kind: 'status',
        status: 400,
        body: body400(reason),
      });
      expect(failure(await answerRequests(d, RUN, [TEXT]))).toBeInstanceOf(cls);
    });
  }

  it('maps any other 400 reason, HTML or non-JSON body to bad_request without keeping it', async () => {
    for (const body of [
      body400('unknown_option'),
      body400(`${SECRET}`),
      body400({ nested: 'run_ended' }),
      `<html><b>${SECRET}</b></html>`,
      'not json',
      '',
      JSON.stringify({ reason: 'run_ended' }),
      JSON.stringify(['run_ended']),
      'x'.repeat(100_000),
    ]) {
      const { deps: d } = deps({ kind: 'status', status: 400, body });
      const error = failure(await answerRequests(d, RUN, [TEXT]));
      expect(error).toBeInstanceOf(AppApiError);
      expect((error as AppApiError).code).toBe('bad_request');
      expect(JSON.stringify(error, ['message', 'name', 'code'])).not.toContain(
        'html',
      );
    }
  });

  it('maps a rejected fetch and an empty stream to network', async () => {
    for (const plan of [{ kind: 'reject' }, { kind: 'empty' }] as Plan[]) {
      const { deps: d } = deps(plan);
      const error = failure(await answerRequests(d, RUN, [OPTION]));
      expect((error as AppApiError).code).toBe('network');
    }
  });

  it('times out after 15 s', async () => {
    vi.useFakeTimers();
    const { deps: d } = deps({ kind: 'hang' });
    const pending = answerRequests(d, RUN, [OPTION]);
    await vi.advanceTimersByTimeAsync(ANSWER_TIMEOUT_MS);
    const error = failure(await pending);
    expect((error as AppApiError).code).toBe('network');
  });

  it('times out on a 400 whose body never arrives', async () => {
    vi.useFakeTimers();
    const { deps: d } = deps({ kind: 'hangingBody', status: 400 });
    const pending = answerRequests(d, RUN, [OPTION]);
    await vi.advanceTimersByTimeAsync(ANSWER_TIMEOUT_MS);
    const error = failure(await pending);
    expect((error as AppApiError).code).toBe('bad_request');
  });

  it('uses injected timers', async () => {
    const { fetchFn } = makeFetch({ kind: 'hang' });
    let fire: (() => void) | undefined;
    const clear = vi.fn();
    const pending = answerRequests(
      {
        fetch: fetchFn as unknown as typeof fetch,
        appOrigin: APP,
        setTimeout: (fn: () => void) => {
          fire = fn;
          return 1 as unknown as ReturnType<typeof setTimeout>;
        },
        clearTimeout: clear,
      },
      RUN,
      [OPTION],
    );
    await vi.waitFor(() => expect(fire).toBeDefined());
    fire?.();
    expect(((await pending) as { error: AppApiError }).error.code).toBe(
      'network',
    );
    expect(clear).toHaveBeenCalled();
  });
});

describe('answerRequests: refused before any request', () => {
  const invalid: [string, unknown][] = [
    ['an empty list', []],
    ['not a list', 'x'],
    ['an empty text', [{ requestId: 'r', text: '   ' }]],
    ['over 2000 chars after trim', [{ requestId: 'r', text: 'a'.repeat(2001) }]],
    ['both optionId and text', [{ requestId: 'r', optionId: 'a', text: 'b' }]],
    ['neither', [{ requestId: 'r' }]],
    ['an empty optionId', [{ requestId: 'r', optionId: '' }]],
    ['a missing requestId', [{ text: 'a' }]],
    ['an empty requestId', [{ requestId: '', text: 'a' }]],
    [
      'duplicate ids',
      [
        { requestId: 'r', text: 'a' },
        { requestId: 'r', optionId: 'b' },
      ],
    ],
    ['a non-object entry', [null]],
    ['a non-string text', [{ requestId: 'r', text: 5 }]],
  ];
  for (const [name, answers] of invalid) {
    it(`refuses ${name}`, async () => {
      const { deps: d, state } = deps({ kind: 'chunk' });
      const error = failure(
        await answerRequests(d, RUN, answers as InputAnswer[]),
      );
      expect(error).toBeInstanceOf(AnswerInvalidError);
      expect(state.calls).toBe(0);
    });
  }

  it('accepts exactly 2000 chars after trim', async () => {
    const { deps: d } = deps({ kind: 'chunk' });
    expect(
      await answerRequests(d, RUN, [
        { requestId: 'r', text: ` ${'a'.repeat(2000)} ` },
      ]),
    ).toEqual({ ok: true });
  });

  it('refuses a second call for a request id in flight, then allows it after', async () => {
    vi.useFakeTimers();
    const { deps: d, state } = deps({ kind: 'hang' });
    const first = answerRequests(d, RUN, [TEXT, OPTION]);
    const again = await answerRequests(d, RUN, [OPTION]);
    expect(failure(again)).toBeInstanceOf(AnswerInFlightError);
    const same = await answerRequests(d, RUN, [TEXT, OPTION]);
    expect(failure(same)).toBeInstanceOf(AnswerInFlightError);
    expect(state.calls).toBe(1);
    await vi.advanceTimersByTimeAsync(ANSWER_TIMEOUT_MS);
    failure(await first);
    vi.useRealTimers();

    // Cleared after a failure.
    const ok = deps({ kind: 'chunk' });
    expect(await answerRequests(ok.deps, RUN, [TEXT, OPTION])).toEqual({
      ok: true,
    });
    // Cleared after a success.
    expect(await answerRequests(ok.deps, RUN, [TEXT, OPTION])).toEqual({
      ok: true,
    });
    expect(ok.state.calls).toBe(2);
  });

  it('an invalid call leaves nothing in flight', async () => {
    const { deps: d } = deps({ kind: 'chunk' });
    await answerRequests(d, RUN, [
      { requestId: 'r-text', text: 'a' },
      { requestId: 'r-text', text: 'b' },
    ]);
    expect(await answerRequests(d, RUN, [TEXT])).toEqual({ ok: true });
  });

  it('every new error class has fixed copy', () => {
    for (const error of [
      new AnswerAlreadyGivenError(),
      new AnswerRefusedError(),
      new RunEndedError(),
      new AnswerInvalidError(),
      new AnswerInFlightError(),
    ]) {
      const m = messageFor(error);
      expect(m.title).not.toBe('Something went wrong');
      expect(m.detail.length).toBeGreaterThan(0);
    }
  });
});
