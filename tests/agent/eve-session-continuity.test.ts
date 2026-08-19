import { describe, it, expect } from 'vitest';
import { getContinuity, setContinuity, clearContinuity } from '@/lib/ai/eve/session-continuity';

describe('session-continuity', () => {
  it('returns undefined for an unknown chat', () => {
    expect(getContinuity('u1', 'c-unknown')).toBeUndefined();
  });
  it('stores and retrieves per (user, chat)', () => {
    setContinuity('u1', 'c1', { eveSessionId: 's1', continuationToken: 't1' });
    expect(getContinuity('u1', 'c1')).toEqual({ eveSessionId: 's1', continuationToken: 't1' });
    // isolation: same chatId, different user is separate
    expect(getContinuity('u2', 'c1')).toBeUndefined();
  });
  it('overwrites on repeated set (new continuation token)', () => {
    setContinuity('u1', 'c2', { eveSessionId: 's2', continuationToken: 't2' });
    setContinuity('u1', 'c2', { eveSessionId: 's2', continuationToken: 't2b' });
    expect(getContinuity('u1', 'c2')?.continuationToken).toBe('t2b');
  });
  it('clears an entry', () => {
    setContinuity('u1', 'c3', { eveSessionId: 's3', continuationToken: 't3' });
    clearContinuity('u1', 'c3');
    expect(getContinuity('u1', 'c3')).toBeUndefined();
  });
});
