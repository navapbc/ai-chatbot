import { z } from 'zod';
import { getRunStore } from '@/lib/ai/eve/run-store';
import { secretMatches } from '@/lib/ai/eve/loopback-attributes';
import { mintRelayToken, RELAY_TOKEN_TTL_SECONDS } from '@/lib/relay/token';

// Internal only: the Eve process calls this over loopback to get a fresh relay
// agent token for a run (lib/relay/agent-token.ts). There is no user session
// (proxy.ts never blocks a request), so this route is its own gate: the flag,
// the shared secret, then the run lookup. Every refusal is a bodyless 404 so
// the route is indistinguishable from an absent one. Nothing here may put the
// token, the secret, env values or the stored origin into a log or an error.

const MIN_SECRET_LENGTH = 32;
const NO_STORE = { 'Cache-Control': 'no-store' };

const bodySchema = z.object({ runId: z.string().uuid() });

const notFound = () => new Response(null, { status: 404, headers: NO_STORE });
const failure = (status: number, error: string) =>
  Response.json({ error }, { status, headers: NO_STORE });

export async function POST(request: Request) {
  if (process.env.RELAY_RUNS_ENABLED !== '1') return notFound();

  if (
    !secretMatches(
      request.headers.get('x-eve-internal-secret'),
      process.env.EVE_INTERNAL_SECRET,
    )
  ) {
    return notFound();
  }

  let runId: string;
  try {
    ({ runId } = bodySchema.parse(await request.json()));
  } catch {
    return failure(400, 'Invalid request.');
  }

  let run: Awaited<ReturnType<ReturnType<typeof getRunStore>['getByChatId']>>;
  try {
    run = await getRunStore().getByChatId(runId);
  } catch {
    return failure(500, 'Internal error.');
  }
  // The origin is the one the run was created with; no origin, no token.
  if (!run || run.relayOrigin === null) return notFound();

  const secret = process.env.RELAY_TOKEN_SECRET;
  if (!secret || secret.length < MIN_SECRET_LENGTH) {
    return failure(500, 'Internal error.');
  }

  const now = Math.floor(Date.now() / 1000);
  const token = mintRelayToken(
    { runId, role: 'agent', origin: run.relayOrigin },
    secret,
    { now },
  );
  return Response.json(
    { token, expiresAt: now + RELAY_TOKEN_TTL_SECONDS },
    { headers: NO_STORE },
  );
}
