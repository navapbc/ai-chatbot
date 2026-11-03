import { describe, expect, it } from 'vitest';
import {
  internalSecret,
  internalSecretMisconfigured,
} from '@/lib/relay/internal-secret';

describe('internalSecret', () => {
  it.each([undefined, '', 'x'.repeat(31)])('treats %j as unset', (value) => {
    expect(internalSecret(value)).toBeUndefined();
  });

  it.each([32, 33])('accepts %i characters', (n) => {
    expect(internalSecret('x'.repeat(n))).toBe('x'.repeat(n));
  });

  it('flags only a set-but-short secret as misconfigured', () => {
    expect(internalSecretMisconfigured(undefined)).toBe(false);
    expect(internalSecretMisconfigured('')).toBe(false);
    expect(internalSecretMisconfigured('x'.repeat(31))).toBe(true);
    expect(internalSecretMisconfigured('x'.repeat(32))).toBe(false);
  });
});
