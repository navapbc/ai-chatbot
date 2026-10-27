import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { AppApi } from '../../extension/src/app-api';
import { createJevClassifier } from '../../extension/src/jev-classifier';
import type { FieldDescriptor } from '../../lib/relay/sensitive-fields';

const RUN = '123e4567-e89b-42d3-a456-426614174000';

type Call = {
  runId: string;
  fields: FieldDescriptor[];
  signal?: AbortSignal;
};

/**
 * A fake AppApi whose classifyFields records each call and answers with
 * `respond` (default: sensitive when the label contains "secret").
 */
const fakeApi = (
  respond: (call: Call) => Promise<(boolean | null)[]> | (boolean | null)[] = (
    c,
  ) => c.fields.map((f) => (f.label ?? '').includes('secret')),
) => {
  const calls: Call[] = [];
  const classifyFields = vi.fn(
    async (
      runId: string,
      fields: readonly FieldDescriptor[],
      signal?: AbortSignal,
    ) => {
      const call = {
        runId,
        fields: JSON.parse(JSON.stringify(fields)),
        signal,
      };
      calls.push(call);
      return respond(call);
    },
  );
  return { calls, api: { classifyFields } as unknown as AppApi };
};

/** Settles `p` without letting a rejection go unhandled. */
const settle = <T>(p: Promise<T>) =>
  p.then(
    (value) => ({ ok: true as const, value }),
    (error: unknown) => ({ ok: false as const, error }),
  );

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

describe('batching', () => {
  it('sends descriptors arriving within the window in one request, in order', async () => {
    const { api, calls } = fakeApi();
    const jev = createJevClassifier({ api, runId: RUN });
    const a = jev.classify({ label: 'a secret' });
    const b = jev.classify({ label: 'plain' });
    await vi.advanceTimersByTimeAsync(2);
    const c = jev.classify({ label: 'another secret' });
    await vi.advanceTimersByTimeAsync(10);
    expect(await Promise.all([a, b, c])).toEqual([true, false, true]);
    expect(calls).toHaveLength(1);
    expect(calls[0].runId).toBe(RUN);
    expect(calls[0].fields.map((f) => f.label)).toEqual([
      'a secret',
      'plain',
      'another secret',
    ]);
    expect(jev.requests()).toBe(1);
    expect(jev.failures()).toBe(0);
  });

  it('descriptors in separate windows go in separate requests', async () => {
    const { api, calls } = fakeApi();
    const jev = createJevClassifier({ api, runId: RUN });
    const a = jev.classify({ label: 'one' });
    await vi.advanceTimersByTimeAsync(10);
    const b = jev.classify({ label: 'two secret' });
    await vi.advanceTimersByTimeAsync(10);
    expect(await Promise.all([a, b])).toEqual([false, true]);
    expect(calls.map((c) => c.fields.map((f) => f.label))).toEqual([
      ['one'],
      ['two secret'],
    ]);
    expect(jev.requests()).toBe(2);
  });

  it('windowMs is configurable', async () => {
    const { api, calls } = fakeApi();
    const jev = createJevClassifier({ api, runId: RUN, windowMs: 50 });
    const a = jev.classify({ label: 'one' });
    await vi.advanceTimersByTimeAsync(40);
    const b = jev.classify({ label: 'two' });
    expect(calls).toHaveLength(0);
    await vi.advanceTimersByTimeAsync(20);
    await Promise.all([a, b]);
    expect(calls).toHaveLength(1);
    expect(calls[0].fields).toHaveLength(2);
  });

  it('splits more than 50 descriptors into requests of at most 50, results in order', async () => {
    const { api, calls } = fakeApi((c) =>
      c.fields.map((f) => Number(f.name) % 3 === 0),
    );
    const jev = createJevClassifier({ api, runId: RUN });
    const ps = Array.from({ length: 120 }, (_, i) =>
      jev.classify({ name: String(i) }),
    );
    await vi.advanceTimersByTimeAsync(10);
    const results = await Promise.all(ps);
    expect(results).toEqual(Array.from({ length: 120 }, (_, i) => i % 3 === 0));
    expect(calls.map((c) => c.fields.length)).toEqual([50, 50, 20]);
    expect(calls.flatMap((c) => c.fields.map((f) => f.name))).toEqual(
      Array.from({ length: 120 }, (_, i) => String(i)),
    );
    expect(jev.requests()).toBe(3);
  });

  it('identical descriptors in one window share one slot of the batch', async () => {
    const { api, calls } = fakeApi();
    const jev = createJevClassifier({ api, runId: RUN });
    const a = jev.classify({ label: 'x secret', name: 'n' });
    // Same keys and values in another order: the same descriptor.
    const b = jev.classify({ name: 'n', label: 'x secret' });
    const c = jev.classify({ label: 'y' });
    await vi.advanceTimersByTimeAsync(10);
    expect(await Promise.all([a, b, c])).toEqual([true, true, false]);
    expect(calls).toHaveLength(1);
    expect(calls[0].fields).toEqual([
      { name: 'n', label: 'x secret' },
      { label: 'y' },
    ]);
  });
});

describe('what is sent', () => {
  it('never sends a value key or the value, even when a caller passes one', async () => {
    const { api, calls } = fakeApi();
    const jev = createJevClassifier({ api, runId: RUN });
    const p = jev.classify({
      label: 'Reference code',
      value: 'QX-4417',
      other: 'QX-4417',
      name: 5,
    } as unknown as FieldDescriptor);
    await vi.advanceTimersByTimeAsync(10);
    await p;
    expect(calls[0].fields).toEqual([{ label: 'Reference code' }]);
    expect(JSON.stringify(calls[0].fields)).not.toContain('QX-4417');
    expect(Object.keys(calls[0].fields[0])).not.toContain('value');
  });
});

describe('failure rejects and is counted, never false', () => {
  it('an api rejection rejects every waiter in the batch', async () => {
    const { api } = fakeApi(() => Promise.reject(new Error('boom')));
    const jev = createJevClassifier({ api, runId: RUN });
    const a = settle(jev.classify({ label: 'a' }));
    const b = settle(jev.classify({ label: 'b' }));
    await vi.advanceTimersByTimeAsync(10);
    const results = await Promise.all([a, b]);
    expect(results.map((r) => r.ok)).toEqual([false, false]);
    expect(jev.failures()).toBe(2);
    // The rejection carries no descriptor text.
    for (const r of results) {
      if (r.ok) throw new Error('expected rejection');
      expect(String((r.error as Error).message)).not.toMatch(/\ba\b|\bb\b/);
    }
  });

  it('a wrong-length answer rejects', async () => {
    const { api } = fakeApi(() => [true]);
    const jev = createJevClassifier({ api, runId: RUN });
    const a = settle(jev.classify({ label: 'a' }));
    const b = settle(jev.classify({ label: 'b' }));
    await vi.advanceTimersByTimeAsync(10);
    expect((await Promise.all([a, b])).map((r) => r.ok)).toEqual([
      false,
      false,
    ]);
    expect(jev.failures()).toBe(2);
  });

  it('a non-boolean or non-array answer rejects', async () => {
    for (const answer of [['yes'], null, { sensitive: [true] }]) {
      const { api } = fakeApi(() => answer as unknown as boolean[]);
      const jev = createJevClassifier({ api, runId: RUN });
      const a = settle(jev.classify({ label: 'a' }));
      await vi.advanceTimersByTimeAsync(10);
      expect((await a).ok).toBe(false);
      expect(jev.failures()).toBe(1);
    }
  });

  it('a failure does not affect a later request for the same descriptor', async () => {
    let fail = true;
    const { api } = fakeApi((c) => {
      if (fail) return Promise.reject(new Error('down'));
      return c.fields.map(() => true);
    });
    const jev = createJevClassifier({ api, runId: RUN });
    const first = settle(jev.classify({ label: 'a' }));
    await vi.advanceTimersByTimeAsync(10);
    expect((await first).ok).toBe(false);
    fail = false;
    const second = jev.classify({ label: 'a' });
    await vi.advanceTimersByTimeAsync(10);
    expect(await second).toBe(true);
    expect(jev.failures()).toBe(1);
  });
});

describe('a null (no verdict) answer', () => {
  it('rejects that descriptor and counts it; booleans in the batch resolve', async () => {
    const { api } = fakeApi(() => [true, null, false]);
    const jev = createJevClassifier({ api, runId: RUN });
    const a = settle(jev.classify({ label: 'a' }));
    const b = settle(jev.classify({ label: 'b' }));
    const c = settle(jev.classify({ label: 'c' }));
    await vi.advanceTimersByTimeAsync(10);
    const [ra, rb, rc] = await Promise.all([a, b, c]);
    expect(ra).toEqual({ ok: true, value: true });
    expect(rb.ok).toBe(false);
    expect(rc).toEqual({ ok: true, value: false });
    expect(jev.failures()).toBe(1);
  });

  it('counts every waiter sharing the null descriptor', async () => {
    const { api } = fakeApi(() => [null]);
    const jev = createJevClassifier({ api, runId: RUN });
    const a = settle(jev.classify({ label: 'same' }));
    const b = settle(jev.classify({ label: 'same' }));
    await vi.advanceTimersByTimeAsync(10);
    expect((await Promise.all([a, b])).map((r) => r.ok)).toEqual([
      false,
      false,
    ]);
    expect(jev.failures()).toBe(2);
  });

  it('a later boolean answer for the same descriptor resolves', async () => {
    let verdict: boolean | null = null;
    const { api } = fakeApi((c) => c.fields.map(() => verdict));
    const jev = createJevClassifier({ api, runId: RUN });
    const first = settle(jev.classify({ label: 'a' }));
    await vi.advanceTimersByTimeAsync(10);
    expect((await first).ok).toBe(false);
    verdict = true;
    const second = jev.classify({ label: 'a' });
    await vi.advanceTimersByTimeAsync(10);
    expect(await second).toBe(true);
  });
});

describe('settles on its own', () => {
  it('passes a signal that aborts at 1800ms, not before', async () => {
    const { api, calls } = fakeApi(() => [false]);
    const jev = createJevClassifier({ api, runId: RUN });
    const p = jev.classify({ label: 'a' });
    await vi.advanceTimersByTimeAsync(5);
    await p;
    const signal = calls[0].signal;
    expect(signal).toBeInstanceOf(AbortSignal);
    // The request settled, so its timer is cleared and the signal stays.
    await vi.advanceTimersByTimeAsync(5000);
    expect(signal?.aborted).toBe(false);
  });

  it('a hung request rejects by 1800ms after it starts, and the signal aborts', async () => {
    const { api, calls } = fakeApi(() => new Promise<boolean[]>(() => {}));
    const jev = createJevClassifier({ api, runId: RUN });
    const p = settle(jev.classify({ label: 'a' }));
    await vi.advanceTimersByTimeAsync(5);
    expect(calls).toHaveLength(1);
    const signal = calls[0].signal;
    let done = false;
    void p.then(() => {
      done = true;
    });
    // The request started at the window (5ms); its deadline is 1800ms later.
    await vi.advanceTimersByTimeAsync(1799);
    expect(signal?.aborted).toBe(false);
    expect(done).toBe(false);
    await vi.advanceTimersByTimeAsync(1);
    expect(signal?.aborted).toBe(true);
    expect((await p).ok).toBe(false);
    expect(jev.failures()).toBe(1);
  });

  it('a late answer after the abort is ignored and unhandled rejections do not escape', async () => {
    let rejectLate: (e: Error) => void = () => {};
    const { api } = fakeApi(
      () =>
        new Promise<boolean[]>((_, reject) => {
          rejectLate = reject;
        }),
    );
    const jev = createJevClassifier({ api, runId: RUN });
    const p = settle(jev.classify({ label: 'a' }));
    await vi.advanceTimersByTimeAsync(1805);
    expect((await p).ok).toBe(false);
    rejectLate(new Error('late'));
    await vi.advanceTimersByTimeAsync(1);
    expect(jev.failures()).toBe(1);
  });
});

describe('budget', () => {
  it('counts requests; once exhausted every call rejects without a request', async () => {
    const { api, calls } = fakeApi();
    const jev = createJevClassifier({ api, runId: RUN, budgetPerRun: 2 });
    for (const label of ['a', 'b']) {
      const p = jev.classify({ label });
      await vi.advanceTimersByTimeAsync(10);
      await p;
    }
    expect(jev.requests()).toBe(2);
    const r = await settle(jev.classify({ label: 'c' }));
    expect(r.ok).toBe(false);
    await vi.advanceTimersByTimeAsync(10);
    expect(calls).toHaveLength(2);
    expect(jev.failures()).toBe(1);
  });

  it('chunks past the budget in one window reject, earlier ones are sent', async () => {
    const { api, calls } = fakeApi(() => Array(50).fill(true));
    const jev = createJevClassifier({ api, runId: RUN, budgetPerRun: 1 });
    const ps = Array.from({ length: 60 }, (_, i) =>
      settle(jev.classify({ name: String(i) })),
    );
    await vi.advanceTimersByTimeAsync(10);
    const results = await Promise.all(ps);
    expect(results.slice(0, 50).every((r) => r.ok)).toBe(true);
    expect(results.slice(50).every((r) => !r.ok)).toBe(true);
    expect(calls).toHaveLength(1);
    expect(jev.requests()).toBe(1);
    expect(jev.failures()).toBe(10);
  });

  it('defaults to 200 requests', async () => {
    const { api, calls } = fakeApi();
    const jev = createJevClassifier({ api, runId: RUN });
    for (let i = 0; i < 201; i++) {
      const p = settle(jev.classify({ name: String(i) }));
      await vi.advanceTimersByTimeAsync(6);
      await p;
    }
    expect(calls).toHaveLength(200);
    expect(jev.failures()).toBe(1);
  });
});

describe('stop', () => {
  it('rejects the waiters of an open window, sends nothing, and leaves no timer', async () => {
    const { api, calls } = fakeApi();
    const jev = createJevClassifier({ api, runId: RUN });
    const a = settle(jev.classify({ label: 'a secret' }));
    jev.stop();
    expect(vi.getTimerCount()).toBe(0);
    expect((await a).ok).toBe(false);
    await vi.advanceTimersByTimeAsync(50);
    expect(calls).toHaveLength(0);
  });

  it('aborts an in-flight request, rejects its waiters, and clears its deadline', async () => {
    const { api, calls } = fakeApi(() => new Promise<boolean[]>(() => {}));
    const jev = createJevClassifier({ api, runId: RUN });
    const a = settle(jev.classify({ label: 'x' }));
    await vi.advanceTimersByTimeAsync(10);
    expect(calls).toHaveLength(1);
    jev.stop();
    expect(calls[0].signal?.aborted).toBe(true);
    expect(vi.getTimerCount()).toBe(0);
    expect((await a).ok).toBe(false);
  });

  it('a late answer after stop resolves nothing', async () => {
    let answer: (r: boolean[]) => void = () => {};
    const { api } = fakeApi(
      () =>
        new Promise<boolean[]>((resolve) => {
          answer = resolve;
        }),
    );
    const jev = createJevClassifier({ api, runId: RUN });
    const a = settle(jev.classify({ label: 'x' }));
    await vi.advanceTimersByTimeAsync(10);
    jev.stop();
    answer([true]);
    expect((await a).ok).toBe(false);
  });

  it('rejects every later classify without a request or a timer', async () => {
    const { api, calls } = fakeApi();
    const jev = createJevClassifier({ api, runId: RUN });
    jev.stop();
    const a = settle(jev.classify({ label: 'a secret' }));
    expect(vi.getTimerCount()).toBe(0);
    expect((await a).ok).toBe(false);
    await vi.advanceTimersByTimeAsync(50);
    expect(calls).toHaveLength(0);
    // Idempotent.
    jev.stop();
  });
});
