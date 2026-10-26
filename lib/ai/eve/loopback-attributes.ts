import { createHash, timingSafeEqual } from 'node:crypto';

// Pure core of the Eve channel's header-to-attribute auth (agent/channels/
// eve.ts), kept free of Eve imports so it can be unit tested.

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

const digest = (value: string) => createHash('sha256').update(value).digest();

/** Constant-time compare of two secrets via fixed-length digests. */
export function secretMatches(provided: string | null, expected: string | undefined) {
  if (!provided || !expected) return false;
  return timingSafeEqual(digest(provided), digest(expected));
}

/**
 * Attributes a request may set on its session, or null when none.
 *
 * Loopback is not a trust boundary on its own (withEve() proxies the public
 * origin to 127.0.0.1), so the relay run reference additionally requires the
 * shared internal secret, and both relay headers together. Neither is secret
 * (the token is fetched at call time, see lib/relay/agent-token.ts); the
 * retired `x-eve-relay-token` header is never read. `x-eve-model` keeps
 * its loopback-only rule.
 */
export function attributesFromHeaders(
  headers: Headers,
  loopback: boolean,
  internalSecret: string | undefined = process.env.EVE_INTERNAL_SECRET,
): Record<string, string> | null {
  if (!loopback) return null;
  const attributes: Record<string, string> = {};

  const model = headers.get('x-eve-model');
  if (model) attributes.eveModel = model;

  const relayUrl = headers.get('x-eve-relay-url');
  const relayRunId = headers.get('x-eve-relay-run');
  if (
    relayUrl &&
    relayRunId &&
    UUID.test(relayRunId) &&
    secretMatches(headers.get('x-eve-internal-secret'), internalSecret)
  ) {
    attributes.relayUrl = relayUrl;
    attributes.relayRunId = relayRunId;
  }
  return Object.keys(attributes).length > 0 ? attributes : null;
}
