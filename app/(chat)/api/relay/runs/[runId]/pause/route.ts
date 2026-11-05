import { cancelEveTurn } from '@/lib/ai/eve/eve-client';
import { getRunStore } from '@/lib/ai/eve/run-store';
import { ChatSDKError } from '@/lib/errors';
import { guardRelayRoute } from '@/lib/relay/route-guard';

// Server-side Pause for a relay run. Cancels the Eve turn in flight and
// leaves the row live: the run keeps its session, progress and token, so a
// follow-up turn (the panel's Resume) continues it. The extension worker
// refuses the agent's page commands first (pauseAgent), so this only stops
// the agent from spending more steps. A run that ended answers `run_ended`
// (400, as eve-chat does), and an unknown run, another user's run and a web
// chat all get the same bodyless 404 as the stop route. A cancel failure is
// logged without values and answers 502, so the panel can say the pause did
// not take.

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

const noStore = (response: Response): Response => {
  response.headers.set('Cache-Control', 'no-store');
  return response;
};
const notFound = () => noStore(new Response(null, { status: 404 }));
// Deliberately no cause: the body must not hint at stored values.
const internalError = () =>
  noStore(new ChatSDKError('internal_server_error:api').toResponse());

export async function POST(
  request: Request,
  { params }: { params: Promise<{ runId: string }> },
) {
  const guard = await guardRelayRoute(request, { requireOrigin: true });
  if (!guard.ok) return noStore(guard.response);
  const { userId } = guard;

  const { runId } = await params;
  if (!UUID.test(runId)) {
    return noStore(new ChatSDKError('bad_request:api').toResponse());
  }

  let eveSessionId: string;
  try {
    const run = await getRunStore().get(userId, runId);
    if (!run || run.relayOrigin === null) return notFound();
    if (run.endedAt !== null) {
      return noStore(
        new ChatSDKError('bad_request:api', 'run_ended').toResponse(),
      );
    }
    eveSessionId = run.eveSessionId;
  } catch {
    return internalError();
  }

  try {
    await cancelEveTurn(eveSessionId);
  } catch {
    // One fixed line: no session id, run id or error text.
    console.error('[relay-pause] cancel failed');
    return internalError();
  }

  return noStore(Response.json({ paused: true }));
}
