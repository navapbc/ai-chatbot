import { createHmac } from 'node:crypto';
import { describe, expect, it } from 'vitest';
import {
  RELAY_TOKEN_TTL_SECONDS,
  mintRelayToken,
  verifyRelayToken,
} from '@/lib/relay/token';

const SECRET = 'test-secret-that-is-at-least-32-characters-long';
const NOW = 1_800_000_000;
const claims = {
  runId: 'run-1',
  role: 'extension' as const,
  origin: 'https://example.com',
  sub: 'user-1',
};

const b64url = (value: unknown) =>
  Buffer.from(JSON.stringify(value)).toString('base64url');

// Signs an arbitrary payload the way mintRelayToken does, so verify can be
// tested against claims that mint itself refuses to produce.
const signRaw = (value: unknown) => {
  const payload = b64url(value);
  const sig = createHmac('sha256', SECRET).update(payload).digest('base64url');
  return `${payload}.${sig}`;
};

const flip = (part: string) =>
  `${part.slice(0, 2)}${part[2] === 'A' ? 'B' : 'A'}${part.slice(3)}`;

describe('relay token', () => {
  it('round-trips claims with exp = now + ttl', () => {
    const token = mintRelayToken(claims, SECRET, { now: NOW });
    expect(RELAY_TOKEN_TTL_SECONDS).toBe(1800);
    expect(verifyRelayToken(token, SECRET, { now: NOW })).toEqual({
      ...claims,
      exp: NOW + 1800,
    });
  });

  it('honors a custom ttl', () => {
    const token = mintRelayToken(claims, SECRET, { now: NOW, ttlSeconds: 60 });
    expect(verifyRelayToken(token, SECRET, { now: NOW })?.exp).toBe(NOW + 60);
  });

  it('rejects the wrong secret', () => {
    const token = mintRelayToken(claims, SECRET, { now: NOW });
    expect(verifyRelayToken(token, `${SECRET}-other`, { now: NOW })).toBeNull();
  });

  it('rejects a tampered payload', () => {
    const [payload, sig] = mintRelayToken(claims, SECRET, { now: NOW }).split(
      '.',
    );
    expect(
      verifyRelayToken(`${flip(payload)}.${sig}`, SECRET, { now: NOW }),
    ).toBeNull();
  });

  it('rejects a tampered signature', () => {
    const [payload, sig] = mintRelayToken(claims, SECRET, { now: NOW }).split(
      '.',
    );
    expect(
      verifyRelayToken(`${payload}.${flip(sig)}`, SECRET, { now: NOW }),
    ).toBeNull();
  });

  it('rejects an expired token (now = exp)', () => {
    const token = mintRelayToken(claims, SECRET, { now: NOW });
    expect(verifyRelayToken(token, SECRET, { now: NOW + 1800 })).toBeNull();
    expect(verifyRelayToken(token, SECRET, { now: NOW + 1799 })).not.toBeNull();
  });

  it('rejects a token with no dot', () => {
    expect(verifyRelayToken('nodot', SECRET, { now: NOW })).toBeNull();
  });

  it('rejects a validly signed non-JSON payload', () => {
    const payload = Buffer.from('not json').toString('base64url');
    const sig = createHmac('sha256', SECRET)
      .update(payload)
      .digest('base64url');
    expect(
      verifyRelayToken(`${payload}.${sig}`, SECRET, { now: NOW }),
    ).toBeNull();
  });

  it('keeps sub through a round trip', () => {
    const token = mintRelayToken(claims, SECRET, { now: NOW });
    expect(verifyRelayToken(token, SECRET, { now: NOW })?.sub).toBe('user-1');
  });

  it('accepts a hand-signed token with the same claims (signRaw matches mint)', () => {
    const token = signRaw({ ...claims, exp: NOW + 60 });
    expect(verifyRelayToken(token, SECRET, { now: NOW })).toEqual({
      ...claims,
      exp: NOW + 60,
    });
  });

  it.each([
    ['role admin', { ...claims, role: 'admin' }],
    ['http origin', { ...claims, origin: 'http://example.com' }],
    ['empty runId', { ...claims, runId: '' }],
    [
      'no sub',
      { runId: claims.runId, role: claims.role, origin: claims.origin },
    ],
    ['empty sub', { ...claims, sub: '' }],
    ['numeric sub', { ...claims, sub: 42 }],
  ])('rejects a validly signed token with %s', (_name, bad) => {
    const token = signRaw({ ...bad, exp: NOW + 60 });
    expect(verifyRelayToken(token, SECRET, { now: NOW })).toBeNull();
  });

  it.each([
    'https://a.gov/path',
    'https://a.gov?x=1',
    'https://user:pw@a.gov',
    'https://a.gov/',
    'https://a.gov:443',
    'https://A.gov',
  ])('rejects non-exact origin %s', (origin) => {
    const token = signRaw({ ...claims, origin, exp: NOW + 60 });
    expect(verifyRelayToken(token, SECRET, { now: NOW })).toBeNull();
  });

  it.each(['https://a.gov', 'https://a.gov:8443'])(
    'accepts exact origin %s',
    (origin) => {
      const token = mintRelayToken({ ...claims, origin }, SECRET, { now: NOW });
      expect(verifyRelayToken(token, SECRET, { now: NOW })?.origin).toBe(
        origin,
      );
    },
  );

  describe('mint refuses', () => {
    it.each([
      ['an empty secret', claims, ''],
      ['a 31-character secret', claims, 'x'.repeat(31)],
      ['an empty sub', { ...claims, sub: '' }, SECRET],
      ['a missing sub', { ...claims, sub: undefined }, SECRET],
      ['an http origin', { ...claims, origin: 'http://example.com' }, SECRET],
      ['a non-exact origin', { ...claims, origin: 'https://a.gov/' }, SECRET],
      ['an empty runId', { ...claims, runId: '' }, SECRET],
      ['an unknown role', { ...claims, role: 'admin' }, SECRET],
    ])('%s, without echoing values', (_name, bad, secret) => {
      let message = '';
      try {
        mintRelayToken(bad as never, secret);
      } catch (error) {
        message = (error as Error).message;
      }
      expect(message).not.toBe('');
      expect(message).not.toContain(SECRET);
      expect(message).not.toContain('user-1');
      expect(message).not.toContain('example.com');
    });

    it('accepts a 32-character secret', () => {
      expect(() => mintRelayToken(claims, 'x'.repeat(32))).not.toThrow();
    });
  });
});
