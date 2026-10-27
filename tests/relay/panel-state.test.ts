import { describe, expect, it } from 'vitest';
import { AppApiError } from '../../extension/src/app-api';
import { initialState, reducer } from '../../extension/panel/lib/state';

const config = {
  sites: [],
  models: [{ id: 'claude-opus-5' }],
  defaultModel: 'claude-opus-5',
};

describe('panel reducer', () => {
  it('starts probing', () => {
    expect(initialState.screen).toBe('loading');
  });

  it('probe success goes home and keeps the config', () => {
    const s = reducer(initialState, { type: 'probeSucceeded', config });
    expect(s.screen).toBe('home');
    expect(s.config).toEqual(config);
  });

  it('a 401 goes to signedOut', () => {
    const s = reducer(initialState, {
      type: 'probeFailed',
      error: new AppApiError('signed_out', 401),
    });
    expect(s.screen).toBe('signedOut');
  });

  it('another error goes to error with the retryable flag', () => {
    const s = reducer(initialState, {
      type: 'probeFailed',
      error: new AppApiError('network', 0),
    });
    expect(s.screen).toBe('error');
    expect(s.error?.retryable).toBe(true);
    const nc = reducer(initialState, {
      type: 'probeFailed',
      error: new AppApiError('forbidden', 403),
    });
    expect(nc.error?.retryable).toBe(false);
  });

  it('retry resets to loading and clears the error', () => {
    const failed = reducer(initialState, {
      type: 'probeFailed',
      error: new AppApiError('network', 0),
    });
    const s = reducer(failed, { type: 'retry' });
    expect(s.screen).toBe('loading');
    expect(s.error).toBeUndefined();
  });

  it('goHome returns home from a stub screen, keeping the config', () => {
    const home = reducer(initialState, { type: 'probeSucceeded', config });
    const s = reducer(reducer(home, { type: 'goto', screen: 'details' }), {
      type: 'goHome',
    });
    expect(s.screen).toBe('home');
    expect(s.config).toEqual(config);
  });
});
