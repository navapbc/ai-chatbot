import { z } from 'zod';
import { getRunStore } from '@/lib/ai/eve/run-store';
import { ChatSDKError } from '@/lib/errors';
import { guardRelayRoute } from '@/lib/relay/route-guard';
import {
  MIN_RELAY_SECRET_LENGTH,
  mintRelayToken,
  RELAY_TOKEN_TTL_SECONDS,
} from '@/lib/relay/token';
import { toRelayWsUrl } from '@/lib/relay/url';

// Mints the extension's relay token for a run the caller owns. The run, its
// origin and its owner all come from the run store: the body names the run
// and nothing else. Unknown keys (an `origin`, a `role`, a `sub`) are stripped
// by the schema and ignored, so a caller cannot steer the token's claims. The
// agent's token is minted separately, by the internal agent-token route, for
// the Eve session bound to the run.
//
// Gated by RELAY_TOKEN_ENDPOINT_ENABLED (checked first, before auth) and by
// RELAY_RUNS_ENABLED (through guardRelayRoute). The endpoint flag stays unset
// in deployment config until the owner decides to turn it on.

const NO_STORE = 'no-store';

const bodySchema = z.object({ runId: z.string().uuid() });

const noStore = (response: Response): Response => {
  response.headers.set('Cache-Control', NO_STORE);
  return response;
};

// Bodyless, so an unknown run, another user's run and a web run look alike.
const notFound = () =>
  new Response(null, { status: 404, headers: { 'Cache-Control': NO_STORE } });

// Deliberately no cause: the body must not hint at env or stored values.
const internalError = () =>
  noStore(new ChatSDKError('internal_server_error:api').toResponse());

export async function POST(request: Request) {
  if (process.env.RELAY_TOKEN_ENDPOINT_ENABLED !== '1') return notFound();

  const guard = await guardRelayRoute();
  if (!guard.ok) return noStore(guard.response);
  const { userId } = guard;

  let runId: string;
  try {
    ({ runId } = bodySchema.parse(await request.json()));
  } catch {
    return noStore(new ChatSDKError('bad_request:api').toResponse());
  }

  let relayOrigin: string | null;
  try {
    const run = await getRunStore().get(userId, runId);
    if (!run) return notFound();
    relayOrigin = run.relayOrigin;
  } catch {
    return internalError();
  }
  if (relayOrigin === null) return notFound();

  const secret = process.env.RELAY_TOKEN_SECRET;
  // RELAY_PUBLIC_URL (https) becomes the wss URL handed to clients.
  const publicUrl = process.env.RELAY_PUBLIC_URL;
  const relayUrl = publicUrl ? toRelayWsUrl(publicUrl) : null;
  if (!secret || secret.length < MIN_RELAY_SECRET_LENGTH || !relayUrl) {
    console.error('[relay-token] misconfigured');
    return internalError();
  }

  const now = Math.floor(Date.now() / 1000);
  let extensionToken: string;
  try {
    extensionToken = mintRelayToken(
      { runId, role: 'extension', origin: relayOrigin, sub: userId },
      secret,
      { now },
    );
  } catch {
    return internalError();
  }

  return Response.json(
    { extensionToken, relayUrl, expiresAt: now + RELAY_TOKEN_TTL_SECONDS },
    { headers: { 'Cache-Control': NO_STORE } },
  );
}
