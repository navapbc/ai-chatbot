import { createHash, createHmac, timingSafeEqual } from 'node:crypto';

export type RelayRole = 'extension' | 'agent';

export interface RelayClaims {
  runId: string;
  role: RelayRole;
  origin: string;
  /** Expiry, epoch seconds. */
  exp: number;
}

export const RELAY_TOKEN_TTL_SECONDS = 1800;

const ROLES: readonly string[] = ['extension', 'agent'];

const sign = (payload: string, secret: string): Buffer =>
  createHmac('sha256', secret).update(payload).digest();

const nowSeconds = () => Math.floor(Date.now() / 1000);

export function mintRelayToken(
  claims: Omit<RelayClaims, 'exp'>,
  secret: string,
  opts: { now?: number; ttlSeconds?: number } = {},
): string {
  const now = opts.now ?? nowSeconds();
  const full: RelayClaims = {
    runId: claims.runId,
    role: claims.role,
    origin: claims.origin,
    exp: now + (opts.ttlSeconds ?? RELAY_TOKEN_TTL_SECONDS),
  };
  const payload = Buffer.from(JSON.stringify(full)).toString('base64url');
  return `${payload}.${sign(payload, secret).toString('base64url')}`;
}

const isHttpsOrigin = (value: string): boolean => {
  try {
    const url = new URL(value);
    return url.protocol === 'https:' && url.origin === value;
  } catch {
    return false;
  }
};

/** Returns the claims, or null for anything invalid. Never throws. */
export function verifyRelayToken(
  token: string,
  secret: string,
  opts: { now?: number } = {},
): RelayClaims | null {
  try {
    const parts = token.split('.');
    if (parts.length !== 2) return null;
    const [payload, signature] = parts;

    // Hash both sides so timingSafeEqual always gets equal-length buffers.
    const expected = createHash('sha256')
      .update(sign(payload, secret))
      .digest();
    const actual = createHash('sha256')
      .update(Buffer.from(signature, 'base64url'))
      .digest();
    if (!timingSafeEqual(expected, actual)) return null;

    const parsed: unknown = JSON.parse(
      Buffer.from(payload, 'base64url').toString('utf8'),
    );
    if (typeof parsed !== 'object' || parsed === null) return null;
    const { runId, role, origin, exp } = parsed as Record<string, unknown>;

    if (typeof runId !== 'string' || runId === '') return null;
    if (typeof role !== 'string' || !ROLES.includes(role)) return null;
    if (typeof origin !== 'string' || !isHttpsOrigin(origin)) return null;
    if (typeof exp !== 'number' || !Number.isFinite(exp)) return null;
    if ((opts.now ?? nowSeconds()) >= exp) return null;

    return { runId, role: role as RelayRole, origin, exp };
  } catch {
    return null;
  }
}
