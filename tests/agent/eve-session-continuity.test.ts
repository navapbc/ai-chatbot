import { describe, it, expect } from 'vitest';
import { getContinuity, setContinuity, clearContinuity } from '@/lib/ai/eve/session-continuity';

describe('session-continuity', () => {
  it('returns undefined for an unknown chat', () => {
    expect(getContinuity('u1', 'c-unknown')).toBeUndefined();
  });
  it('stores and retrieves per (user, chat)', () => {
    setContinuity('u1', 'c1', {
      eveSessionId: 's1',
      continuationToken: 't1',
      streamIndex: 0,
    });
    expect(getContinuity('u1', 'c1')).toEqual({
      eveSessionId: 's1',
      continuationToken: 't1',
      streamIndex: 0,
    });
    // isolation: same chatId, different user is separate
    expect(getContinuity('u2', 'c1')).toBeUndefined();
  });
  it('overwrites on repeated set (new continuation token)', () => {
    setContinuity('u1', 'c2', {
      eveSessionId: 's2',
      continuationToken: 't2',
      streamIndex: 0,
    });
    setContinuity('u1', 'c2', {
      eveSessionId: 's2',
      continuationToken: 't2b',
      streamIndex: 29,
    });
    expect(getContinuity('u1', 'c2')?.continuationToken).toBe('t2b');
  });
  // The stream cursor is what keeps a follow-up turn from replaying the
  // previous one and stopping at its `session.waiting` boundary.
  it('carries the stream cursor forward across turns', () => {
    setContinuity('u1', 'c4', {
      eveSessionId: 's4',
      continuationToken: 't4',
      streamIndex: 0,
    });
    expect(getContinuity('u1', 'c4')?.streamIndex).toBe(0);

    // Turn 0 parked after 29 events (indexes 0..28, session.waiting at 28).
    setContinuity('u1', 'c4', {
      eveSessionId: 's4',
      continuationToken: 't4b',
      streamIndex: 29,
    });
    // The next read resumes at 29 — turn 1's first event, not a replay of 0.
    expect(getContinuity('u1', 'c4')?.streamIndex).toBe(29);
  });
  it('clears an entry', () => {
    setContinuity('u1', 'c3', {
      eveSessionId: 's3',
      continuationToken: 't3',
      streamIndex: 0,
    });
    clearContinuity('u1', 'c3');
    expect(getContinuity('u1', 'c3')).toBeUndefined();
  });
});
