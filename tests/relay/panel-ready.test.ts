import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import {
  DEBUGGER_BAR_COPY,
  POLICY_RESTRICTED_COPY,
} from '../../extension/panel/lib/copy';
import { ReadyScreen } from '../../extension/panel/screens/ready';
import { PolicyNotice } from '../../extension/panel/screens/running';
import { AppApiError } from '../../extension/src/app-api';
import { DETAIL_FIELDS } from '../../extension/panel/lib/saved-details';
import {
  buildStartRequest,
  StartRequestError,
  type ReadyModel,
} from '../../extension/panel/lib/start-request';
import { reducer, type State } from '../../extension/panel/lib/state';

const models = [{ id: 'claude-opus-5' }, { id: 'claude-haiku-4-5' }];

const record = (over: Partial<ReadyModel> = {}): ReadyModel => ({
  siteName: 'Example Benefits',
  origin: 'https://example.gov',
  participant: { kind: 'record', recordId: 'rec-1', name: 'Zelda Quill' },
  models,
  defaultModel: 'claude-opus-5',
  modelId: 'claude-opus-5',
  ...over,
});

const details = (
  d: Record<string, string>,
  over: Partial<ReadyModel> = {},
): ReadyModel => ({
  ...record(),
  participant: { kind: 'details', details: d },
  ...over,
});

describe('buildStartRequest', () => {
  it('builds a record request with no model when it is the default', () => {
    expect(buildStartRequest(record())).toEqual({
      origin: 'https://example.gov',
      participant: { recordId: 'rec-1' },
    });
  });

  it('builds a details request', () => {
    const r = buildStartRequest(details({ first_name: 'Zelda', city: 'Reno' }));
    expect(r).toEqual({
      origin: 'https://example.gov',
      participant: { details: { first_name: 'Zelda', city: 'Reno' } },
    });
    expect(Object.keys(r).sort()).toEqual(['origin', 'participant']);
  });

  it('includes the model only when it differs from the default and is listed', () => {
    expect(buildStartRequest(record({ modelId: 'claude-haiku-4-5' }))).toEqual({
      origin: 'https://example.gov',
      participant: { recordId: 'rec-1' },
      model: 'claude-haiku-4-5',
    });
  });

  it('refuses a model that is not in the list', () => {
    expect(() => buildStartRequest(record({ modelId: 'other' }))).toThrow(
      StartRequestError,
    );
  });

  it.each([
    'http://example.gov',
    'HTTPS://EXAMPLE.GOV',
    'https://example.gov/',
    'https://example.gov/path',
    'https://example.gov?x=1',
    'example.gov',
    '',
  ])('refuses the origin %s', (origin) => {
    expect(() => buildStartRequest(record({ origin }))).toThrow(
      StartRequestError,
    );
  });

  it('refuses an empty, blank or too long record id', () => {
    for (const recordId of ['', '   ', 'x'.repeat(129)]) {
      expect(() =>
        buildStartRequest(
          record({ participant: { kind: 'record', recordId, name: 'N' } }),
        ),
      ).toThrow(StartRequestError);
    }
    expect(
      buildStartRequest(
        record({
          participant: { kind: 'record', recordId: 'x'.repeat(128), name: 'N' },
        }),
      ).participant,
    ).toEqual({ recordId: 'x'.repeat(128) });
  });

  it('drops keys outside DETAIL_FIELDS and empty values, and trims', () => {
    const r = buildStartRequest(
      details({
        first_name: '  Zelda ',
        last_name: '   ',
        ssn: '123-45-6789',
        extra: 'x',
        city: 'Reno',
      }),
    );
    expect(r.participant).toEqual({
      details: { first_name: 'Zelda', city: 'Reno' },
    });
  });

  it('drops non-string values', () => {
    const r = buildStartRequest(
      details({ city: 'Reno', zip: 12345 as unknown as string }),
    );
    expect(r.participant).toEqual({ details: { city: 'Reno' } });
  });

  it('refuses details with no value left', () => {
    expect(() => buildStartRequest(details({}))).toThrow(StartRequestError);
    expect(() => buildStartRequest(details({ city: ' ', ssn: 'x' }))).toThrow(
      StartRequestError,
    );
  });

  it('refuses a value over 500 characters and accepts exactly 500', () => {
    expect(() => buildStartRequest(details({ city: 'x'.repeat(501) }))).toThrow(
      StartRequestError,
    );
    expect(
      buildStartRequest(details({ city: 'x'.repeat(500) })).participant,
    ).toEqual({ details: { city: 'x'.repeat(500) } });
  });

  it('can never exceed 12 keys', () => {
    const all = Object.fromEntries(
      DETAIL_FIELDS.map((f) => [f.key, 'v']),
    ) as Record<string, string>;
    for (let i = 0; i < 60; i++) all[`extra_${i}`] = 'v';
    const r = buildStartRequest(details(all));
    expect(
      Object.keys((r.participant as { details: object }).details),
    ).toHaveLength(12);
  });

  it('never carries the site name or the participant name', () => {
    for (const r of [record(), record({ modelId: 'claude-haiku-4-5' })]) {
      const json = JSON.stringify(buildStartRequest(r));
      expect(json).not.toContain('Example Benefits');
      expect(json).not.toContain('Zelda');
      expect(json).not.toContain('siteName');
      expect(json).not.toContain('name');
    }
  });

  it('uses a fixed error message that carries no input', () => {
    try {
      buildStartRequest(details({ city: 'Secretville'.repeat(60) }));
      throw new Error('should have thrown');
    } catch (error) {
      expect(error).toBeInstanceOf(StartRequestError);
      expect((error as Error).message).not.toContain('Secretville');
    }
    try {
      buildStartRequest(record({ origin: 'http://leak.example' }));
    } catch (error) {
      expect((error as Error).message).not.toContain('leak');
    }
  });
});

describe('ready screen reducer actions', () => {
  const ready: State = {
    screen: 'ready',
    target: { tabId: 1, origin: 'https://a.gov', siteName: 'A' },
    participant: { kind: 'record', recordId: 'r1', name: 'Ann' },
  };

  it('chooseModel sets modelId', () => {
    expect(reducer(ready, { type: 'chooseModel', modelId: 'm2' }).modelId).toBe(
      'm2',
    );
  });

  it('startRequested moves ready to starting and clears the error', () => {
    const s = reducer(
      { ...ready, error: { title: 't', detail: 'd', retryable: true } },
      { type: 'startRequested' },
    );
    expect(s.screen).toBe('starting');
    expect(s.error).toBeUndefined();
    expect(s.participant).toEqual(ready.participant);
  });

  it('startRequested is ignored unless on ready (double press)', () => {
    const starting = reducer(ready, { type: 'startRequested' });
    expect(reducer(starting, { type: 'startRequested' })).toBe(starting);
  });

  it('startFailed returns to ready with a fixed message', () => {
    const starting = reducer(ready, { type: 'startRequested' });
    const s = reducer(starting, {
      type: 'startFailed',
      error: new AppApiError('rate_limited', 429),
    });
    expect(s.screen).toBe('ready');
    expect(s.error?.title).toBe('Too many runs right now');
    expect(s.participant).toEqual(ready.participant);
  });

  it('runStarted moves starting to running and keeps the run', () => {
    const starting = reducer(ready, { type: 'startRequested' });
    const runId = '11111111-2222-4333-8444-555555555555';
    const s = reducer(starting, {
      type: 'runStarted',
      runId,
      expiresAt: 1_800_001_800,
      tabId: 7,
    });
    expect(s.screen).toBe('running');
    expect(s.run).toEqual({ runId, expiresAt: 1_800_001_800, tabId: 7 });
    expect(s.target).toEqual(ready.target);
    expect(s.error).toBeUndefined();
  });

  it('runStarted is ignored unless starting', () => {
    const action = {
      type: 'runStarted',
      runId: '11111111-2222-4333-8444-555555555555',
      expiresAt: 1,
      tabId: 7,
    } as const;
    expect(reducer(ready, action)).toBe(ready);
    const home: State = { screen: 'home' };
    expect(reducer(home, action)).toBe(home);
  });

  it('startFailed is ignored unless starting', () => {
    expect(reducer(ready, { type: 'startFailed', error: new Error('x') })).toBe(
      ready,
    );
  });

  it('goBack leads to participant for a record and details for details', () => {
    const r = reducer(ready, { type: 'goBack' });
    expect(r.screen).toBe('participant');
    expect(r.participant).toBeUndefined();
    expect(r.target).toEqual(ready.target);
    const d = reducer(
      { ...ready, participant: { kind: 'details', details: { city: 'x' } } },
      { type: 'goBack' },
    );
    expect(d.screen).toBe('details');
  });

  it('goBack clears the error and is ignored off ready', () => {
    const e = reducer(
      { ...ready, error: { title: 't', detail: 'd', retryable: true } },
      { type: 'goBack' },
    );
    expect(e.error).toBeUndefined();
    const home: State = { screen: 'home' };
    expect(reducer(home, { type: 'goBack' })).toBe(home);
  });
});

describe('ReadyScreen: the debugger bar sentence', () => {
  it('says Chrome will show a being-debugged bar and that it is expected', () => {
    const out = renderToStaticMarkup(
      createElement(ReadyScreen, {
        state: {
          screen: 'ready',
          config: { models, defaultModel: 'claude-opus-5' },
          target: {
            tabId: 1,
            origin: 'https://example.gov',
            siteName: 'Example Benefits',
          },
          participant: {
            kind: 'record',
            recordId: 'rec-1',
            name: 'Zelda Quill',
          },
        } as never,
        dispatch: () => {},
        onStart: () => undefined,
      }),
    );
    expect(DEBUGGER_BAR_COPY).toBe(
      'While the agent works, Chrome may show a bar saying the browser is being debugged; that is expected.',
    );
    expect(out).toContain(DEBUGGER_BAR_COPY);
  });
});

describe('PolicyNotice', () => {
  const html = (state: unknown) =>
    renderToStaticMarkup(
      createElement(PolicyNotice, { state: state as never }),
    );
  it('shows the fixed explanation as an alert only while live and restricted', () => {
    const live = html({ status: 'running', policyRestricted: true });
    expect(live).toContain('role="alert"');
    expect(live).toContain(POLICY_RESTRICTED_COPY.replace("'", '&#x27;'));
    expect(html({ status: 'running' })).toBe('');
    expect(html({ status: 'ended', policyRestricted: true })).toBe('');
    expect(html(null)).toBe('');
  });
});
