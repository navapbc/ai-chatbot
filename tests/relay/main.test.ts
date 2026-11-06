import { describe, expect, it } from 'vitest';
import { checkRelayEnv } from '../../relay/env';

const SECRET = 'test-secret-that-is-at-least-32-characters-long';
const REDIS = 'redis://user:hunter2@redis.internal:6379';

describe('checkRelayEnv', () => {
  it('accepts a secret with no REDIS_URL', () => {
    expect(checkRelayEnv({ RELAY_TOKEN_SECRET: SECRET })).toEqual({ ok: true });
  });

  it('refuses REDIS_URL without the acknowledgement, without echoing it', () => {
    const result = checkRelayEnv({
      RELAY_TOKEN_SECRET: SECRET,
      REDIS_URL: REDIS,
    });
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.message).toContain('RELAY_MULTI_INSTANCE_ACKNOWLEDGED=1');
    expect(result.message).not.toContain('hunter2');
    expect(result.message).not.toContain('redis.internal');
  });

  it('refuses an acknowledgement that is not exactly 1', () => {
    const result = checkRelayEnv({
      RELAY_TOKEN_SECRET: SECRET,
      REDIS_URL: REDIS,
      RELAY_MULTI_INSTANCE_ACKNOWLEDGED: 'true',
    });
    expect(result.ok).toBe(false);
  });

  it('accepts REDIS_URL with the acknowledgement', () => {
    expect(
      checkRelayEnv({
        RELAY_TOKEN_SECRET: SECRET,
        REDIS_URL: REDIS,
        RELAY_MULTI_INSTANCE_ACKNOWLEDGED: '1',
      }),
    ).toEqual({ ok: true });
  });

  it('refuses a missing or short secret without echoing it', () => {
    for (const env of [{}, { RELAY_TOKEN_SECRET: 'abcdefg' }]) {
      const result = checkRelayEnv(env);
      expect(result.ok).toBe(false);
      if (!result.ok) expect(result.message).not.toContain('abcdefg');
    }
  });

  it('refuses empty entries in the host lists', () => {
    for (const name of ['RELAY_ALLOWED_HOST_SUFFIXES', 'RELAY_ALLOWED_HOSTS']) {
      for (const value of ['a.test,', ',a.test', 'a.test,,b.test']) {
        const result = checkRelayEnv({
          RELAY_TOKEN_SECRET: SECRET,
          [name]: value,
        });
        expect(result.ok).toBe(false);
      }
      expect(
        checkRelayEnv({ RELAY_TOKEN_SECRET: SECRET, [name]: 'a.test,b.test' }),
      ).toEqual({ ok: true });
    }
  });
});
