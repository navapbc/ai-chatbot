import { describe, expect, it } from 'vitest';
import {
  RELAY_TOKEN_TTL_SECONDS,
  mintRelayToken,
  verifyRelayToken,
} from '@/lib/relay/token';

const SECRET = 'test-secret';
const NOW = 1_800_000_000;
const claims = {
  runId: 'run-1',
  role: 'extension' as const,
  origin: 'https://example.com',
};

const b64url = (value: unknown) =>
  Buffer.from(JSON.stringify(value)).toString('base64url');

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
    expect(verifyRelayToken(token, 'other', { now: NOW })).toBeNull();
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

  it('rejects a validly signed non-JSON payload', async () => {
    const { createHmac } = await import('node:crypto');
    const payload = Buffer.from('not json').toString('base64url');
    const sig = createHmac('sha256', SECRET)
      .update(payload)
      .digest('base64url');
    expect(
      verifyRelayToken(`${payload}.${sig}`, SECRET, { now: NOW }),
    ).toBeNull();
  });

  it.each([
    ['role admin', { ...claims, role: 'admin' }],
    ['http origin', { ...claims, origin: 'http://example.com' }],
    ['empty runId', { ...claims, runId: '' }],
  ])('rejects a validly signed token with %s', (_name, bad) => {
    const token = mintRelayToken(bad as never, SECRET, { now: NOW });
    expect(verifyRelayToken(token, SECRET, { now: NOW })).toBeNull();
  });
});
