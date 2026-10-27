import { z } from 'zod';
import { getRunStore } from '@/lib/ai/eve/run-store';
import { secretMatches } from '@/lib/ai/eve/loopback-attributes';
import {
  MIN_RELAY_SECRET_LENGTH,
  mintRelayToken,
  RELAY_TOKEN_TTL_SECONDS,
} from '@/lib/relay/token';

// Internal only: the Eve process calls this over loopback to get a fresh relay
// agent token for a run (lib/relay/agent-token.ts). There is no user session
// (proxy.ts never blocks a request), so this route is its own gate: the flag,
// the shared secret, then the run lookup, which must name the Eve session the
// run was created with. That binding matters: two racing first turns each
// create an Eve session carrying the same run id, and the loser's orphan
// session must not get a token that lets it take the hub's agent slot. Every
// refusal is a bodyless 404 so the route is indistinguishable from an absent
// one. Nothing here may put the token, the secret, env values or the stored
// origin into a log or an error.

const NO_STORE = { 'Cache-Control': 'no-store' };

// Eve session ids are short (`wrun_` plus a ULID); the bound only keeps an
// oversized value out of the comparison.
const MAX_SESSION_ID_LENGTH = 200;

const bodySchema = z.object({
  runId: z.string().uuid(),
  sessionId: z.string().min(1).max(MAX_SESSION_ID_LENGTH),
});

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
  let sessionId: string;
  try {
    ({ runId, sessionId } = bodySchema.parse(await request.json()));
  } catch {
    return failure(400, 'Invalid request.');
  }

  let run: Awaited<ReturnType<ReturnType<typeof getRunStore>['getByChatId']>>;
  try {
    run = await getRunStore().getByChatId(runId);
  } catch {
    return failure(500, 'Internal error.');
  }
  // The origin is the one the run was created with; no origin, no token. Only
  // the run's own Eve session gets one: an orphan from a lost race, or a
  // subagent's child session, is refused like an unknown run. Between Eve
  // creating the session and the run row landing this also refuses; the
  // caller does not cache refusals, so the next tool call asks again.
  if (!run || run.relayOrigin === null || run.eveSessionId !== sessionId) {
    return notFound();
  }

  const secret = process.env.RELAY_TOKEN_SECRET;
  if (!secret || secret.length < MIN_RELAY_SECRET_LENGTH) {
    return failure(500, 'Internal error.');
  }

  const now = Math.floor(Date.now() / 1000);
  let token: string;
  try {
    // The run's owner is the token's sub, so the hub pairs this agent only
    // with an extension that holds a token minted for the same user.
    token = mintRelayToken(
      { runId, role: 'agent', origin: run.relayOrigin, sub: run.userId },
      secret,
      { now },
    );
  } catch {
    return failure(500, 'Internal error.');
  }
  return Response.json(
    { token, expiresAt: now + RELAY_TOKEN_TTL_SECONDS },
    { headers: NO_STORE },
  );
}
