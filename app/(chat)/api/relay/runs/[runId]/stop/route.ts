import { cancelEveTurn } from '@/lib/ai/eve/eve-client';
import { getRunStore } from '@/lib/ai/eve/run-store';
import { ChatSDKError } from '@/lib/errors';
import { guardRelayRoute } from '@/lib/relay/route-guard';

// Server-side Stop for a relay run (P4b-c). Ends the row first, so the run is
// closed to tokens and turns whatever happens next, then cancels the Eve turn
// in flight, once: only the call that ended the run cancels. A cancel failure
// is logged without values and does not undo the end. There is no resume; the
// caseworker starts a new run.

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

  const store = getRunStore();
  let eveSessionId: string;
  let ended: boolean;
  try {
    // Read first: an unknown run, another user's run or a web chat is a 404
    // before anything is written.
    const run = await store.get(userId, runId);
    if (!run || run.relayOrigin === null) return notFound();
    eveSessionId = run.eveSessionId;
    ended = await store.end(userId, runId);
  } catch {
    return internalError();
  }

  if (ended) {
    try {
      await cancelEveTurn(eveSessionId);
    } catch {
      // One fixed line: no session id, run id or error text.
      console.error('[relay-stop] cancel failed');
    }
  }

  return noStore(Response.json({ ended: true, alreadyEnded: !ended }));
}
