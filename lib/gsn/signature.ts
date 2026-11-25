import { createPublicKey, verify } from 'node:crypto';
export const GSN_SIGNED_PATH = '/api/gsn/prepare';
export function verifyPreparation(
  body: string,
  headers: Headers,
  publicKey: string | undefined,
  now = Date.now(),
): boolean {
  if (!publicKey || !/^[A-Za-z0-9_-]{40,200}$/.test(publicKey)) return false;
  const timestamp = headers.get('x-gsn-time') ?? '';
  const nonce = headers.get('x-gsn-nonce') ?? '';
  const signature = headers.get('x-gsn-signature') ?? '';
  if (
    !/^\d{13}$/.test(timestamp) ||
    Math.abs(now - Number(timestamp)) > 90_000 ||
    !/^[0-9a-f-]{36}$/.test(nonce) ||
    !/^[A-Za-z0-9_-]{86}$/.test(signature)
  )
    return false;
  try {
    const key = createPublicKey({
      key: Buffer.from(publicKey, 'base64url'),
      type: 'spki',
      format: 'der',
    });
    return (
      key.asymmetricKeyType === 'ed25519' &&
      verify(
        null,
        Buffer.from(
          `POST\n${GSN_SIGNED_PATH}\n${timestamp}\n${nonce}\n${body}`,
        ),
        key,
        Buffer.from(signature, 'base64url'),
      )
    );
  } catch {
    return false;
  }
}
