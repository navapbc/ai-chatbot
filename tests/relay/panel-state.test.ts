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

  it('chooseTarget stores the target and moves to participant', () => {
    const home = reducer(initialState, { type: 'probeSucceeded', config });
    const target = {
      tabId: 7,
      origin: 'https://example.gov',
      siteName: 'Example',
    };
    const s = reducer(home, { type: 'chooseTarget', target });
    expect(s.screen).toBe('participant');
    expect(s.target).toEqual(target);
    expect(s.config).toEqual(config);
  });

  it('clearTarget drops the target and goes home', () => {
    const home = reducer(initialState, { type: 'probeSucceeded', config });
    const chosen = reducer(home, {
      type: 'chooseTarget',
      target: { tabId: 7, origin: 'https://example.gov', siteName: 'E' },
    });
    const s = reducer(chosen, { type: 'clearTarget' });
    expect(s.screen).toBe('home');
    expect(s.target).toBeUndefined();
    expect(s.config).toEqual(config);
  });

  it('chooseParticipant stores the record and goes ready; clearTarget clears it', () => {
    const target = { tabId: 1, origin: 'https://a.gov', siteName: 'A' };
    const at = reducer(
      { screen: 'participant', target },
      {
        type: 'chooseParticipant',
        participant: { kind: 'record', recordId: 'r1', name: 'Ann' },
      },
    );
    expect(at.screen).toBe('ready');
    expect(at.participant).toEqual({
      kind: 'record',
      recordId: 'r1',
      name: 'Ann',
    });
    expect(at.target).toEqual(target);
    const home = reducer(at, { type: 'clearTarget' });
    expect(home.screen).toBe('home');
    expect(home.participant).toBeUndefined();
    expect(home.target).toBeUndefined();
  });

  it('chooseDetails goes to the details screen', () => {
    const s = reducer({ screen: 'participant' }, { type: 'chooseDetails' });
    expect(s.screen).toBe('details');
  });

  it('submitDetails stores the details and goes to ready', () => {
    const s = reducer(
      { screen: 'details' },
      { type: 'submitDetails', details: { city: 'Fresno' }, save: true },
    );
    expect(s.screen).toBe('ready');
    expect(s.participant).toEqual({
      kind: 'details',
      details: { city: 'Fresno' },
    });
    expect(JSON.stringify(s)).not.toContain('save');
  });

  const RUN = {
    runId: '11111111-2222-4333-8444-555555555555',
    expiresAt: 1_800_001_800,
    tabId: 7,
  };
  const target = {
    tabId: 7,
    origin: 'https://example.gov',
    siteName: 'example.gov',
  };

  it('runEnded clears the run, target and participant and goes home', () => {
    const running = {
      screen: 'running' as const,
      config,
      target,
      participant: { kind: 'record' as const, recordId: 'r', name: 'n' },
      modelId: 'claude-opus-5',
      run: RUN,
    };
    const s = reducer(running, { type: 'runEnded' });
    expect(s).toEqual({
      screen: 'home',
      config,
      modelId: 'claude-opus-5',
    });
  });

  it('runEnded is ignored unless running', () => {
    const home = { screen: 'home' as const, config };
    expect(reducer(home, { type: 'runEnded' })).toBe(home);
  });

  it('runRebuilt goes from loading to running with the config, run and target', () => {
    const s = reducer(initialState, {
      type: 'runRebuilt',
      config,
      run: { ...RUN, expiresAtEstimated: true },
      target,
    });
    expect(s).toEqual({
      screen: 'running',
      config,
      run: { ...RUN, expiresAtEstimated: true },
      target,
    });
  });

  it('runRebuilt is ignored unless loading', () => {
    const home = { screen: 'home' as const, config };
    expect(
      reducer(home, { type: 'runRebuilt', config, run: RUN, target }),
    ).toBe(home);
  });
});
