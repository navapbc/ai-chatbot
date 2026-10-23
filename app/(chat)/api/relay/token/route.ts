import { z } from 'zod';
import { auth } from '@/app/(auth)/auth';
import { ChatSDKError } from '@/lib/errors';
import { normalizeOrigin } from '@/lib/relay/policy';
import { mintRelayToken, RELAY_TOKEN_TTL_SECONDS } from '@/lib/relay/token';

const MIN_SECRET_LENGTH = 32;
const LOOPBACK_HOSTS = new Set(['localhost', '127.0.0.1', '[::1]']);

const bodySchema = z.object({
  runId: z.string().uuid(),
  // Exact https origin only: no path, query, default port or trailing slash.
  origin: z.string().refine((value) => normalizeOrigin(value) === value),
});

/**
 * Turns RELAY_PUBLIC_URL into the websocket URL handed to clients: https
 * becomes wss; http is accepted for loopback dev only and becomes ws.
 * Returns null for anything else so the route fails closed.
 */
function toRelayUrl(value: string | undefined): string | null {
  if (!value) return null;
  try {
    const url = new URL(value);
    if (url.username || url.password || url.search || url.hash) return null;
    const loopback = LOOPBACK_HOSTS.has(url.hostname);
    if (url.protocol === 'https:' || url.protocol === 'wss:') {
      url.protocol = 'wss:';
    } else if (
      (url.protocol === 'http:' || url.protocol === 'ws:') &&
      loopback
    ) {
      url.protocol = 'ws:';
    } else {
      return null;
    }
    const path = url.pathname.replace(/\/+$/, '');
    return `${url.protocol === 'wss:' ? 'wss' : 'ws'}://${url.host}${path}`;
  } catch {
    return null;
  }
}

export async function POST(request: Request) {
  // Disabled by default: this mints tokens for any runId without an ownership
  // check. It stays disabled until plan 4 binds runs to users (server-created
  // runs, a `sub` claim and a hub peer check) and must not be enabled in
  // deployment config before then. Checked first, before auth and parsing.
  if (process.env.RELAY_TOKEN_ENDPOINT_ENABLED !== '1') {
    return new Response(null, { status: 404 });
  }

  const session = await auth();
  if (!session?.user) {
    return new ChatSDKError('unauthorized:chat').toResponse();
  }

  let parsed: z.infer<typeof bodySchema>;
  try {
    parsed = bodySchema.parse(await request.json());
  } catch {
    return new ChatSDKError('bad_request:api').toResponse();
  }

  const secret = process.env.RELAY_TOKEN_SECRET;
  const relayUrl = toRelayUrl(process.env.RELAY_PUBLIC_URL);
  if (!secret || secret.length < MIN_SECRET_LENGTH || !relayUrl) {
    // Deliberately no cause: the body must not hint at env values.
    return new ChatSDKError('internal_server_error:api').toResponse();
  }

  // One clock read so both tokens share the same exp.
  const now = Math.floor(Date.now() / 1000);
  const expiresAt = now + RELAY_TOKEN_TTL_SECONDS;
  const claims = { runId: parsed.runId, origin: parsed.origin };
  const extensionToken = mintRelayToken(
    { ...claims, role: 'extension' },
    secret,
    { now },
  );
  const agentToken = mintRelayToken({ ...claims, role: 'agent' }, secret, {
    now,
  });

  return Response.json(
    { extensionToken, agentToken, relayUrl, expiresAt },
    { headers: { 'Cache-Control': 'no-store' } },
  );
}
