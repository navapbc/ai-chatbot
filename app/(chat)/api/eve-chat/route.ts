import { createUIMessageStream, JsonToSseTransformStream } from 'ai';
import { auth } from '@/app/(auth)/auth';
import { generateUUID } from '@/lib/utils';
import { ChatSDKError } from '@/lib/errors';
import { getContinuity, setContinuity } from '@/lib/ai/eve/session-continuity';
import {
  createEveSession,
  continueEveSession,
  openEveStream,
  parseNdjson,
} from '@/lib/ai/eve/eve-client';
import {
  translateEveEvent,
  extractLatestUserText,
} from '@/lib/ai/eve/stream-adapter';

export const maxDuration = 300; // 5 min for long web-automation turns

export async function POST(request: Request) {
  const session = await auth();
  if (!session?.user?.id) {
    return new ChatSDKError('unauthorized:chat').toResponse();
  }
  const userId = session.user.id;

  let body: { id?: string; message?: { role?: string } };
  try {
    body = await request.json();
  } catch {
    return new ChatSDKError('bad_request:api').toResponse();
  }
  const chatId = body.id;
  // Defense-in-depth: Eve owns its own agent loop, so only a genuine
  // user-initiated turn may ever be forwarded to it. Never scrape/forward
  // an assistant (or other non-user) message as if it were a new user turn.
  if (body.message?.role !== 'user') {
    return new ChatSDKError('bad_request:api').toResponse();
  }
  const text = extractLatestUserText(body.message);
  if (!chatId || !text) {
    return new ChatSDKError('bad_request:api').toResponse();
  }

  // Resolve or create the Eve session for this chat.
  let sessionId: string;
  try {
    const existing = getContinuity(userId, chatId);
    if (existing) {
      const { continuationToken } = await continueEveSession(
        existing.eveSessionId,
        existing.continuationToken,
        text,
      );
      sessionId = existing.eveSessionId;
      setContinuity(userId, chatId, {
        eveSessionId: sessionId,
        continuationToken,
      });
    } else {
      const created = await createEveSession(text);
      sessionId = created.sessionId;
      setContinuity(userId, chatId, {
        eveSessionId: sessionId,
        continuationToken: created.continuationToken,
      });
    }
  } catch (err) {
    // Eve server unreachable or errored on session create/continue.
    console.error('[eve-chat] session error:', err);
    return new ChatSDKError('offline:chat').toResponse();
  }

  const stream = createUIMessageStream({
    execute: async ({ writer }) => {
      const res = await openEveStream(sessionId);
      let ctx = { textId: null as string | null, generateId: generateUUID };
      for await (const event of parseNdjson(res.body!)) {
        const r = translateEveEvent(event, writer, ctx);
        ctx = { ...ctx, textId: r.textId };
        if (r.continuationToken) {
          setContinuity(userId, chatId, {
            eveSessionId: sessionId,
            continuationToken: r.continuationToken,
          });
        }
        if (r.done) break; // r.done is only true on session.waiting (after the token is captured above) — turn.completed always returns done:false. Eve's stream stays open past session.waiting, so this break is required to avoid hanging.
      }
    },
    generateId: generateUUID,
    onError: () => 'Oops, an error occurred running the agent.',
  });

  return new Response(stream.pipeThrough(new JsonToSseTransformStream()));
}
