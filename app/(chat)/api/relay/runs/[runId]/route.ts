import { getRunStore } from '@/lib/ai/eve/run-store';
import { ChatSDKError } from '@/lib/errors';
import { guardRelayRoute } from '@/lib/relay/route-guard';

// The run state the extension panel reads: its own relay run's origin,
// progress, pending questions and whether it has ended. Only what the stream
// already carried; never the Eve session id, the continuation token or the
// stream cursor. An unknown run, another user's run and a web chat all get
// the same bodyless 404.

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

const noStore = (response: Response): Response => {
  response.headers.set('Cache-Control', 'no-store');
  return response;
};
const notFound = () => noStore(new Response(null, { status: 404 }));

export async function GET(
  request: Request,
  { params }: { params: Promise<{ runId: string }> },
) {
  const guard = await guardRelayRoute(request);
  if (!guard.ok) return noStore(guard.response);

  const { runId } = await params;
  if (!UUID.test(runId)) {
    return noStore(new ChatSDKError('bad_request:api').toResponse());
  }

  let run: Awaited<ReturnType<ReturnType<typeof getRunStore>['get']>>;
  try {
    run = await getRunStore().get(guard.userId, runId);
  } catch {
    // Deliberately no cause: the body must not hint at stored values.
    return noStore(new ChatSDKError('internal_server_error:api').toResponse());
  }
  if (!run || run.relayOrigin === null) return notFound();

  // An explicit allowlist, never a spread of the record.
  return noStore(
    Response.json({
      runId,
      origin: run.relayOrigin,
      ended: run.endedAt !== null,
      endedAt: run.endedAt ? run.endedAt.toISOString() : null,
      progress: run.progress,
      // An ended run takes no answers (P4b-c), so it has nothing to ask; the
      // row may still hold questions persisted by a read that outlived Stop.
      pending: run.endedAt !== null ? [] : run.pendingRequests,
    }),
  );
}
