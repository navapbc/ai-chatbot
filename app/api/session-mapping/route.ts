import { auth } from '@/app/(auth)/auth';
import { getChatById, upsertSessionMapping } from '@/lib/db/queries';

// Receives the PostHog session id from the client (the only place it is
// knowable) and records it against the chat. The kernel session/replay ids are
// filled in server-side when the browser is created — see lib/kernel/browser.ts.
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

    await upsertSessionMapping({ chatId, userId, posthogSessionId });

    return Response.json({ success: true });
  } catch (error) {
    console.error('Session mapping API error:', error);
    return Response.json(
      { error: 'Failed to record session mapping' },
      { status: 500 },
    );
  }
}
