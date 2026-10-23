import { z } from 'zod';
import { auth } from '@/app/(auth)/auth';
import { ChatSDKError } from '@/lib/errors';
import { normalizeOrigin } from '@/lib/relay/policy';
import { mintRelayToken, RELAY_TOKEN_TTL_SECONDS } from '@/lib/relay/token';
import { toRelayWsUrl } from '@/lib/relay/url';

const MIN_SECRET_LENGTH = 32;

const bodySchema = z.object({
  runId: z.string().uuid(),
  // Exact https origin only: no path, query, default port or trailing slash.
  origin: z.string().refine((value) => normalizeOrigin(value) === value),
});

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
  // RELAY_PUBLIC_URL (https) becomes the wss URL handed to clients.
  const publicUrl = process.env.RELAY_PUBLIC_URL;
  const relayUrl = publicUrl ? toRelayWsUrl(publicUrl) : null;
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
