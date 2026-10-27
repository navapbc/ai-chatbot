import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { AppApiError } from '../../extension/src/app-api';
import { messageFor } from '../../extension/panel/lib/copy';
import {
  createSearchController,
  type SearchState,
} from '../../extension/panel/lib/search-controller';

type Call = {
  q: string;
  signal?: AbortSignal;
  resolve: (r: { recordId: string; name: string; county: string }[]) => void;
  reject: (e: unknown) => void;
};

const setup = () => {
  const calls: Call[] = [];
  const api = {
    searchParticipants: (q: string, signal?: AbortSignal) =>
      new Promise<{ recordId: string; name: string; county: string }[]>(
        (resolve, reject) => {
          calls.push({ q, signal, resolve, reject });
        },
      ),
  };
  const states: SearchState[] = [];
  const controller = createSearchController(api, (s) => states.push(s));
  return { calls, states, controller };
};

const row = (n: string) => ({ recordId: `r-${n}`, name: n, county: 'Kern' });
const flush = () => vi.advanceTimersByTimeAsync(0);

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

describe('search controller', () => {
  it('debounces rapid typing into one request with the final query', async () => {
    const { calls, controller } = setup();
    controller.setQuery('a');
    await vi.advanceTimersByTimeAsync(100);
    controller.setQuery('al');
    await vi.advanceTimersByTimeAsync(100);
    controller.setQuery('ali');
    expect(calls).toHaveLength(0);
    await vi.advanceTimersByTimeAsync(250);
    expect(calls.map((c) => c.q)).toEqual(['ali']);
  });

  it('shows loading, then results replacing earlier results', async () => {
    const { calls, states, controller } = setup();
    controller.setQuery('ali');
    await vi.advanceTimersByTimeAsync(250);
    expect(states.at(-1)).toEqual({ status: 'loading', query: 'ali' });
    calls[0].resolve([row('a'), row('b')]);
    await flush();
    controller.setQuery('bob');
    await vi.advanceTimersByTimeAsync(250);
    calls[1].resolve([row('c')]);
    await flush();
    expect(states.at(-1)).toEqual({
      status: 'results',
      query: 'bob',
      results: [row('c')],
    });
  });

  it('aborts the earlier request when a new one starts', async () => {
    const { calls, controller } = setup();
    controller.setQuery('one');
    await vi.advanceTimersByTimeAsync(250);
    expect(calls[0].signal?.aborted).toBe(false);
    controller.setQuery('two');
    await vi.advanceTimersByTimeAsync(250);
    expect(calls[0].signal?.aborted).toBe(true);
    expect(calls[1].signal?.aborted).toBe(false);
  });

  it.each([
    ['early first', true],
    ['late first', false],
  ])('ignores a stale response (%s)', async (_n, earlyFirst) => {
    const { calls, states, controller } = setup();
    controller.setQuery('one');
    await vi.advanceTimersByTimeAsync(250);
    controller.setQuery('two');
    await vi.advanceTimersByTimeAsync(250);
    if (earlyFirst) {
      calls[0].resolve([row('stale')]);
      await flush();
      calls[1].resolve([row('fresh')]);
    } else {
      calls[1].resolve([row('fresh')]);
      await flush();
      calls[0].resolve([row('stale')]);
    }
    await flush();
    expect(states.at(-1)).toEqual({
      status: 'results',
      query: 'two',
      results: [row('fresh')],
    });
    expect(JSON.stringify(states)).not.toContain('stale');
  });

  it('ignores an aborted request that rejects', async () => {
    const { calls, states, controller } = setup();
    controller.setQuery('one');
    await vi.advanceTimersByTimeAsync(250);
    controller.setQuery('two');
    await vi.advanceTimersByTimeAsync(250);
    calls[0].reject(new DOMException('aborted', 'AbortError'));
    await flush();
    expect(states.some((s) => s.status === 'error')).toBe(false);
    expect(states.at(-1)).toEqual({ status: 'loading', query: 'two' });
  });

  it.each(['', '   \t'])('goes idle on %j without a request', async (q) => {
    const { calls, states, controller } = setup();
    controller.setQuery('abc');
    await vi.advanceTimersByTimeAsync(100);
    controller.setQuery(q);
    expect(states.at(-1)).toEqual({ status: 'idle' });
    await vi.advanceTimersByTimeAsync(1000);
    expect(calls).toHaveLength(0);
  });

  it('empty query aborts an in-flight request and ignores its response', async () => {
    const { calls, states, controller } = setup();
    controller.setQuery('abc');
    await vi.advanceTimersByTimeAsync(250);
    controller.setQuery('');
    expect(calls[0].signal?.aborted).toBe(true);
    calls[0].resolve([row('late')]);
    await flush();
    expect(states.at(-1)).toEqual({ status: 'idle' });
  });

  it('trims and cuts the query to 100 characters', async () => {
    const { calls, controller } = setup();
    controller.setQuery(`  ${'x'.repeat(150)}  `);
    await vi.advanceTimersByTimeAsync(250);
    expect(calls[0].q).toBe('x'.repeat(100));
  });

  it('passes % and emoji through unchanged', async () => {
    const { calls, controller } = setup();
    controller.setQuery('50%_\u{1F600}');
    await vi.advanceTimersByTimeAsync(250);
    expect(calls[0].q).toBe('50%_\u{1F600}');
  });

  it('maps an error through messageFor with no server text', async () => {
    const { calls, states, controller } = setup();
    controller.setQuery('abc');
    await vi.advanceTimersByTimeAsync(250);
    const err = new AppApiError('server_error', 500);
    err.message = 'SERVER-MARKER';
    calls[0].reject(err);
    await flush();
    const last = states.at(-1);
    expect(last).toMatchObject({
      status: 'error',
      query: 'abc',
      error: messageFor(err),
      code: 'server_error',
    });
    expect(JSON.stringify(last)).not.toContain('SERVER-MARKER');
  });

  it('surfaces signed_out', async () => {
    const { calls, states, controller } = setup();
    controller.setQuery('abc');
    await vi.advanceTimersByTimeAsync(250);
    calls[0].reject(new AppApiError('signed_out', 401));
    await flush();
    expect(states.at(-1)).toMatchObject({
      status: 'error',
      code: 'signed_out',
    });
  });

  it('dispose stops callbacks, aborts and clears the timer', async () => {
    const { calls, states, controller } = setup();
    controller.setQuery('abc');
    await vi.advanceTimersByTimeAsync(250);
    controller.setQuery('abcd');
    const before = states.length;
    controller.dispose();
    await vi.advanceTimersByTimeAsync(1000);
    expect(calls).toHaveLength(1);
    expect(calls[0].signal?.aborted).toBe(true);
    calls[0].resolve([row('x')]);
    await flush();
    controller.setQuery('zzz');
    await vi.advanceTimersByTimeAsync(1000);
    expect(states).toHaveLength(before);
    expect(calls).toHaveLength(1);
    expect(vi.getTimerCount()).toBe(0);
  });
});
