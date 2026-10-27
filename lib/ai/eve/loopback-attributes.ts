import { createHash, timingSafeEqual } from 'node:crypto';
import { internalSecret as usableSecret } from '@/lib/relay/internal-secret';

// Pure core of the Eve channel's header-to-attribute auth (agent/channels/
// eve.ts), kept free of Eve imports so it can be unit tested.

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

const digest = (value: string) => createHash('sha256').update(value).digest();

/** Constant-time compare of two secrets via fixed-length digests. */
export function secretMatches(
  provided: string | null,
  expected: string | undefined,
) {
  const secret = usableSecret(expected);
  if (!provided || !secret) return false;
  return timingSafeEqual(digest(provided), digest(secret));
}

export type RelayHeaderGrant = 'absent' | 'granted' | 'refused';

/**
 * Whether a request's relay run reference is granted. `absent` when neither
 * `x-eve-relay-url` nor `x-eve-relay-run` is sent (the retired
 * `x-eve-relay-token` alone counts as absent: it is ignored, never read).
 * `granted` needs both headers, loopback, a UUID run id and the shared
 * internal secret. Anything else is `refused`, and the channel must reject
 * the request: dropping the reference and carrying on would start a session
 * that drives a Kernel browser instead of the caseworker's tab.
 */
export function relayHeaderGrant(
  headers: Headers,
  loopback: boolean,
  internalSecret: string | undefined = usableSecret(),
): RelayHeaderGrant {
  const relayUrl = headers.get('x-eve-relay-url');
  const relayRunId = headers.get('x-eve-relay-run');
  if (relayUrl === null && relayRunId === null) return 'absent';
  return loopback &&
    relayUrl &&
    relayRunId &&
    UUID.test(relayRunId) &&
    secretMatches(headers.get('x-eve-internal-secret'), internalSecret)
    ? 'granted'
    : 'refused';
}

/**
 * Attributes a request may set on its session, or null when none.
 *
 * Loopback is not a trust boundary on its own (withEve() proxies the public
 * origin to 127.0.0.1), so the relay run reference additionally requires the
 * shared internal secret, and both relay headers together (see
 * `relayHeaderGrant`, which also decides when the request must be refused).
 * Neither is secret (the token is fetched at call time, see
 * lib/relay/agent-token.ts); the retired `x-eve-relay-token` header is never
 * read. `x-eve-model` keeps its loopback-only rule.
 */
export function attributesFromHeaders(
  headers: Headers,
  loopback: boolean,
  internalSecret: string | undefined = usableSecret(),
): Record<string, string> | null {
  if (!loopback) return null;
  const attributes: Record<string, string> = {};

  const model = headers.get('x-eve-model');
  if (model) attributes.eveModel = model;

  if (relayHeaderGrant(headers, loopback, internalSecret) === 'granted') {
    attributes.relayUrl = headers.get('x-eve-relay-url') as string;
    attributes.relayRunId = headers.get('x-eve-relay-run') as string;
  }
  return Object.keys(attributes).length > 0 ? attributes : null;
}
