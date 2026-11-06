import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { isJevEnabled } from '@/lib/jev/client';
import {
  describeFieldForJev,
  NOUL_THRESHOLD,
  sensitiveFieldIsSensitive,
  sensitiveFieldQuestions,
} from '@/lib/jev/questions';

type Outcome = Parameters<typeof sensitiveFieldIsSensitive>[0];

const noulOutcome = (noul: number): Outcome => ({
  ok: true,
  answers: { sensitive: { type: 'noul', noul } },
  model: 'm',
  usage: { inputTokens: 1, outputTokens: 1 },
});

describe('sensitiveFieldQuestions', () => {
  it('state carries only the descriptor keys that are present', () => {
    const { state } = sensitiveFieldQuestions({
      inputType: 'text',
      label: 'Account number',
    });
    expect(Object.keys(state).sort()).toEqual(['inputType', 'label']);
  });

  it('asks one noul question named sensitive', () => {
    const { questions } = sensitiveFieldQuestions({ label: 'SSN' });
    expect(Object.keys(questions)).toEqual(['sensitive']);
  });

  it('omits undefined descriptor keys', () => {
    const { state } = sensitiveFieldQuestions({
      label: 'SSN',
      name: undefined,
    });
    expect(Object.keys(state)).toEqual(['label']);
  });
});

describe('describeFieldForJev', () => {
  it('drops keys outside the allowlist', () => {
    const field = {
      label: 'SSN',
      value: 'secret',
      url: 'https://x.test',
      text: 'page',
    };
    const state = describeFieldForJev(field);
    expect(Object.keys(state)).toEqual(['label']);
    expect(Object.keys(sensitiveFieldQuestions(field).state)).toEqual([
      'label',
    ]);
  });
  it('caps each string at 200 characters', () => {
    expect(describeFieldForJev({ label: 'a'.repeat(500) }).label).toHaveLength(
      200,
    );
  });
  it('replaces every digit with 9 and keeps the shape', () => {
    expect(
      describeFieldForJev({ placeholder: '123-45-6789 / 0' }).placeholder,
    ).toBe('999-99-9999 / 9');
  });
});

describe('sensitiveFieldIsSensitive', () => {
  it('fails toward sensitive when the call failed', () => {
    expect(sensitiveFieldIsSensitive({ ok: false, reason: 'timeout' })).toBe(
      true,
    );
  });
  it('is true at the threshold and above', () => {
    expect(sensitiveFieldIsSensitive(noulOutcome(NOUL_THRESHOLD))).toBe(true);
    expect(sensitiveFieldIsSensitive(noulOutcome(0.99))).toBe(true);
  });
  it.each([
    ['NaN', Number.NaN],
    ['undefined', undefined],
    ['a string', '0.9'],
    ['Infinity', Number.POSITIVE_INFINITY],
  ])('fails toward sensitive for a noul of %s', (_n, bad) => {
    expect(sensitiveFieldIsSensitive(noulOutcome(bad as number))).toBe(true);
  });
  it('fails toward sensitive when answers are missing', () => {
    expect(
      sensitiveFieldIsSensitive({
        ok: true,
        answers: {},
      } as unknown as Outcome),
    ).toBe(true);
    expect(
      sensitiveFieldIsSensitive({
        ok: true,
        answers: undefined,
      } as unknown as Outcome),
    ).toBe(true);
  });
  it('is false below the threshold', () => {
    expect(sensitiveFieldIsSensitive(noulOutcome(NOUL_THRESHOLD - 0.01))).toBe(
      false,
    );
  });
});

describe('sensitive-field feature flag', () => {
  const saved = { ...process.env };
  beforeEach(() => {
    process.env.TYPESAFE_API_KEY = 'test-key';
  });
  afterEach(() => {
    process.env = { ...saved };
  });

  it('parses from JEV_FEATURES by name', () => {
    process.env.JEV_FEATURES = 'sensitive-field';
    expect(isJevEnabled('sensitive-field')).toBe(true);
    expect(isJevEnabled('gap-triage')).toBe(false);
  });
  it('is included by all', () => {
    process.env.JEV_FEATURES = 'all';
    expect(isJevEnabled('sensitive-field')).toBe(true);
  });
});
