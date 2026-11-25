import { generateKeyPairSync, sign } from 'node:crypto';
import { describe, expect, it } from 'vitest';
import { verifyPreparation, GSN_SIGNED_PATH } from '../../lib/gsn/signature';
const keys = generateKeyPairSync('ed25519');
const publicKey = keys.publicKey
  .export({ type: 'spki', format: 'der' })
  .toString('base64url');
const now = 1795550000000;
const nonce = '00000000-0000-4000-8000-000000000000';
const body = JSON.stringify({
  requestId: nonce,
  section: 'personal',
  data: { firstName: 'Fictional' },
});
const headers = new Headers({
  'x-gsn-time': String(now),
  'x-gsn-nonce': nonce,
  'x-gsn-signature': sign(
    null,
    Buffer.from(`POST\n${GSN_SIGNED_PATH}\n${now}\n${nonce}\n${body}`),
    keys.privateKey,
  ).toString('base64url'),
});
describe('GSN service request signatures', () => {
  it('requires the configured service identity, exact body and a recent timestamp', () => {
    expect(verifyPreparation(body, headers, publicKey, now)).toBe(true);
    expect(verifyPreparation(`${body} `, headers, publicKey, now)).toBe(false);
    expect(verifyPreparation(body, headers, publicKey, now + 90001)).toBe(
      false,
    );
    expect(verifyPreparation(body, headers, undefined, now)).toBe(false);
    expect(verifyPreparation(body, new Headers(), publicKey, now)).toBe(false);
    const other = generateKeyPairSync('ed25519')
      .publicKey.export({ type: 'spki', format: 'der' })
      .toString('base64url');
    expect(verifyPreparation(body, headers, other, now)).toBe(false);
  });
});
