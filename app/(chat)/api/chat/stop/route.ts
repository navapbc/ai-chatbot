import { auth } from '@/app/(auth)/auth';
import { signalAbort } from '@/lib/ai/abort-registry';
import { ChatSDKError } from '@/lib/errors';

export async function POST(request: Request) {
  try {
    const { chatId } = await request.json();

    if (!chatId) {
      return new ChatSDKError('bad_request:api', 'chatId is required').toResponse();
    }

    const session = await auth();

    if (!session?.user) {
      return new ChatSDKError('unauthorized:chat').toResponse();
    }

    console.log(`[chat/stop] Stopping chat ${chatId} for user ${session.user.id}`);

    // Signal abort for this chat (works across distributed instances)
    await signalAbort(chatId);

    return Response.json({ success: true });
  } catch (error) {
    console.error('Error stopping chat:', error);
    return new ChatSDKError('internal_server_error:api').toResponse();
  }
}
