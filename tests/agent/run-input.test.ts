import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  allowedRelayModels,
  isRelayModelAllowed,
} from '@/lib/ai/eve/model-map';
import { PARTICIPANTS, buildApplicationPrompt } from '@/lib/data/participants';
import { WITHHELD } from '@/lib/data/sensitive-record';
import { extractParticipantFromText } from '@/lib/jev/participant';
import {
  buildRelayOpeningMessage,
  maxRelayRunsPerHour,
  parseRelayStart,
} from '@/lib/relay/run-input';

const ORIGIN = 'https://apply.example.gov';

describe('parseRelayStart', () => {
  it('accepts a record id', () => {
    expect(
      parseRelayStart({ origin: ORIGIN, participant: { recordId: '339619' } }),
    ).toEqual({
      ok: true,
      origin: ORIGIN,
      participant: { recordId: '339619' },
    });
  });

  it('accepts details and a model', () => {
    expect(
      parseRelayStart({
        origin: ORIGIN,
        participant: { details: { 'First name': 'Ada' } },
        model: 'claude-opus-5',
      }),
    ).toEqual({
      ok: true,
      origin: ORIGIN,
      participant: { details: { 'First name': 'Ada' } },
      model: 'claude-opus-5',
    });
  });

  it('never returns fields the caller did not validate', () => {
    const parsed = parseRelayStart({
      origin: ORIGIN,
      participant: { recordId: '1', extra: 'x' },
      extra: 'y',
    });
    expect(parsed).toEqual({
      ok: true,
      origin: ORIGIN,
      participant: { recordId: '1' },
    });
  });

  it.each([
    ['a non-object', 'https://a.gov'],
    ['null', null],
    ['an array', [ORIGIN]],
  ])('refuses %s', (_n, raw) => {
    expect(parseRelayStart(raw).ok).toBe(false);
  });

  it.each([
    ['a path', 'https://a.gov/x'],
    ['http', 'http://a.gov'],
    ['a trailing slash', 'https://a.gov/'],
    ['an uppercase host', 'https://A.gov'],
    ['a non-string', 5],
  ])('refuses an origin with %s', (_n, origin) => {
    expect(parseRelayStart({ origin, participant: { recordId: '1' } })).toEqual(
      { ok: false, reason: 'invalid_origin' },
    );
  });

  it.each([
    ['no participant', undefined],
    ['a string participant', '339619'],
    ['both a record id and details', { recordId: '1', details: { a: 'b' } }],
    ['neither', {}],
    ['an empty record id', { recordId: '  ' }],
    ['a numeric record id', { recordId: 339619 }],
    ['an overlong record id', { recordId: 'x'.repeat(129) }],
  ])('refuses %s', (_n, participant) => {
    expect(parseRelayStart({ origin: ORIGIN, participant })).toEqual({
      ok: false,
      reason: 'invalid_participant',
    });
  });

  it('refuses a non-string model', () => {
    expect(
      parseRelayStart({
        origin: ORIGIN,
        participant: { recordId: '1' },
        model: 5,
      }),
    ).toEqual({ ok: false, reason: 'model_not_allowed' });
  });

  describe('details', () => {
    const parse = (details: unknown) =>
      parseRelayStart({ origin: ORIGIN, participant: { details } });

    it.each([
      ['a nested object value', { Address: { city: 'X' } }],
      ['a number value', { Age: 5 }],
      ['a null value', { Note: null }],
      ['an array', ['a']],
      ['null', null],
      ['a string', 'Ada'],
      ['no keys', {}],
      ['an empty key', { '': 'x' }],
    ])('refuses %s as details_invalid', (_n, details) => {
      expect(parse(details)).toEqual({ ok: false, reason: 'details_invalid' });
    });

    it('accepts 40 keys and refuses 41', () => {
      const keys = (n: number) =>
        Object.fromEntries(
          Array.from({ length: n }, (_, i) => [`Field ${i}`, 'v']),
        );
      expect(parse(keys(40)).ok).toBe(true);
      expect(parse(keys(41))).toEqual({
        ok: false,
        reason: 'details_too_large',
      });
    });

    it('accepts a 500-character value and refuses 501', () => {
      expect(parse({ Note: 'a'.repeat(500) }).ok).toBe(true);
      expect(parse({ Note: 'a'.repeat(501) })).toEqual({
        ok: false,
        reason: 'details_too_large',
      });
    });

    it('refuses an overlong key', () => {
      expect(parse({ ['k'.repeat(501)]: 'v' })).toEqual({
        ok: false,
        reason: 'details_too_large',
      });
    });

    it.each([
      'ssn',
      'SSN',
      'Social security number',
      'calworks_id',
      'password',
    ])('refuses the sensitive key %s rather than stripping it', (key) => {
      expect(parse({ 'First name': 'Ada', [key]: 'x' })).toEqual({
        ok: false,
        reason: 'sensitive_detail',
      });
    });
  });
});

describe('buildRelayOpeningMessage', () => {
  const record = PARTICIPANTS[0];
  const SECRET = record.participant.calworks_id;

  it('builds the application prompt for the origin with the record stripped', () => {
    const msg = buildRelayOpeningMessage({
      origin: ORIGIN,
      participant: record,
    });
    expect(
      msg.startsWith(`Fill out ${ORIGIN} for the participant below.`),
    ).toBe(true);
    expect(msg).toContain(record.participant.name.first);
    expect(msg).not.toContain(SECRET);
    expect(extractParticipantFromText(msg)?.participant.calworks_id).toBe(
      WITHHELD,
    );
  });

  it('does not modify the record it was given', () => {
    const copy = structuredClone(record);
    buildRelayOpeningMessage({ origin: ORIGIN, participant: copy });
    expect(copy).toEqual(record);
  });

  it('uses the same prefix and JSON format for details', () => {
    const details = { 'First name': 'Ada', City: 'Riverside' };
    const msg = buildRelayOpeningMessage({
      origin: ORIGIN,
      participant: details,
    });
    const prefix = buildApplicationPrompt({
      participant: record,
      target: ORIGIN,
    }).split('\n\n')[0];
    expect(msg).toBe(`${prefix}\n\n${JSON.stringify(details, null, 2)}`);
  });
});

describe('relay model allowlist', () => {
  it('defaults to claude-opus-5', () => {
    expect(allowedRelayModels({})).toEqual(['claude-opus-5']);
    expect(allowedRelayModels({ RELAY_ALLOWED_MODELS: '  ' })).toEqual([
      'claude-opus-5',
    ]);
    expect(isRelayModelAllowed('claude-opus-5', {})).toBe(true);
    expect(isRelayModelAllowed('gpt-5.4', {})).toBe(false);
  });

  it('reads a comma-separated list and drops ids outside MODEL_MAP', () => {
    const env = {
      RELAY_ALLOWED_MODELS: ' claude-opus-5 , claude-sonnet-5,not-a-model',
    };
    expect(allowedRelayModels(env)).toEqual([
      'claude-opus-5',
      'claude-sonnet-5',
    ]);
    expect(isRelayModelAllowed('claude-sonnet-5', env)).toBe(true);
    expect(isRelayModelAllowed('not-a-model', env)).toBe(false);
  });

  it('falls back to the default when nothing in the list is a known model', () => {
    expect(
      allowedRelayModels({ RELAY_ALLOWED_MODELS: 'nope,also-nope' }),
    ).toEqual(['claude-opus-5']);
  });

  it('reads process.env by default', () => {
    vi.stubEnv('RELAY_ALLOWED_MODELS', 'claude-haiku-4-5');
    expect(isRelayModelAllowed('claude-haiku-4-5')).toBe(true);
    vi.unstubAllEnvs();
  });
});

describe('maxRelayRunsPerHour', () => {
  afterEach(() => vi.unstubAllEnvs());

  it.each([
    [undefined, 20],
    ['', 20],
    ['5', 5],
    ['0', 20],
    ['-3', 20],
    ['20abc', 20],
    ['1.5', 20],
    ['100', 100],
  ])('%s -> %i', (value, expected) => {
    expect(
      maxRelayRunsPerHour(
        value === undefined ? {} : { RELAY_MAX_RUNS_PER_HOUR: value },
      ),
    ).toBe(expected);
  });
});
