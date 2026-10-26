import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { isJevEnabled } from '@/lib/jev/client';
import {
  NOUL_THRESHOLD,
  sensitiveFieldIsSensitive,
  sensitiveFieldQuestions,
} from '@/lib/jev/questions';

const noulOutcome = (noul: number) =>
  ({
    ok: true,
    answers: { sensitive: { noul } },
    model: 'm',
    usage: { inputTokens: 1, outputTokens: 1 },
  }) as never;

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
