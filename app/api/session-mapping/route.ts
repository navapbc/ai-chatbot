import { auth } from '@/app/(auth)/auth';
import { getChatById, upsertSessionMapping } from '@/lib/db/queries';

// Build the PostHog-hosted replay deep-link for a session. PostHog has no video
// export, so we store the link rather than re-recording. Requires the numeric
// project id; returns undefined if it isn't configured so we still record the id.
function buildPosthogReplayUrl(posthogSessionId: string): string | undefined {
  const projectId = process.env.POSTHOG_PROJECT_ID;
  if (!projectId) return undefined;

  // App-facing host (us.posthog.com), distinct from the ingest host
  // (us.i.posthog.com) used by NEXT_PUBLIC_POSTHOG_HOST.
  const host = (
    process.env.POSTHOG_APP_HOST || 'https://us.posthog.com'
  ).replace(/\/$/, '');
  return `${host}/project/${projectId}/replay/${posthogSessionId}`;
}

// Receives the PostHog session id from the client (the only place it is
// knowable) and records it — plus a derived replay deep-link — against the
// chat. The kernel session/replay ids and video are filled in server-side when
// the browser is created/torn down — see lib/kernel/browser.ts.
export async function POST(request: Request) {
  const session = await auth();
  if (!session?.user) {
    return Response.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const userId = session.user.id;

  try {
    const { chatId, posthogSessionId } = await request.json();

    if (!chatId || !posthogSessionId) {
      return Response.json(
        { error: 'chatId and posthogSessionId are required' },
        { status: 400 },
      );
    }

    // Ownership check: the chat must exist and belong to the caller. This also
    // guarantees the chatId FK is satisfiable before we insert the mapping.
    const chat = await getChatById({ id: chatId });
    if (!chat || chat.userId !== userId) {
      return Response.json(
        { error: 'Forbidden: chat does not belong to user' },
        { status: 403 },
      );
    }

    await upsertSessionMapping({
      chatId,
      userId,
      posthogSessionId,
      posthogReplayUrl: buildPosthogReplayUrl(posthogSessionId),
    });

    return Response.json({ success: true });
  } catch (error) {
    console.error('Session mapping API error:', error);
    return Response.json(
      { error: 'Failed to record session mapping' },
      { status: 500 },
    );
  }
}
