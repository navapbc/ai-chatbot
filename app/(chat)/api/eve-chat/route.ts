import { createUIMessageStream, JsonToSseTransformStream } from 'ai';
import { auth } from '@/app/(auth)/auth';
import { generateUUID } from '@/lib/utils';
import { ChatSDKError } from '@/lib/errors';
import { getContinuity, setContinuity } from '@/lib/ai/eve/session-continuity';
import { setLiveViewUrl } from '@/lib/ai/eve/live-view-store';
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
import { toVertexModelId } from '@/lib/ai/eve/model-map';
import { isProductionEnvironment } from '@/lib/constants';

export const maxDuration = 300; // 5 min for long web-automation turns

export async function POST(request: Request) {
  const session = await auth();
  if (!session?.user?.id) {
    return new ChatSDKError('unauthorized:chat').toResponse();
  }
  const userId = session.user.id;

  let body: {
    id?: string;
    message?: { role?: string };
    modelOverride?: string;
  };
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
  // Where to resume this chat's durable event stream. A continued session picks
  // up past the previous turn's `session.waiting`; a fresh one starts at 0.
  let startIndex = 0;
  try {
    const existing = getContinuity(userId, chatId);
    // [eve-chat-debug] TEMP diagnostic — remove after debugging the gap round-trip.
    console.log(
      `[eve-chat-debug] userId=${userId} chatId=${chatId} decision=${existing ? 'CONTINUE' : 'CREATE'} existingSession=${existing?.eveSessionId ?? 'none'} resumeFrom=${existing?.streamIndex ?? 0} msg=${JSON.stringify(text.slice(0, 60))}`,
    );
    if (existing) {
      const { continuationToken } = await continueEveSession(
        existing.eveSessionId,
        existing.continuationToken,
        text,
      );
      sessionId = existing.eveSessionId;
      startIndex = existing.streamIndex;
      setContinuity(userId, chatId, {
        eveSessionId: sessionId,
        continuationToken,
        streamIndex: startIndex,
      });
    } else {
      // Model override is a dev/eval feature — never honor it in production
      // (server-side parity with the legacy route; the client also only sends
      // modelOverride in non-prod).
      const model = isProductionEnvironment
        ? undefined
        : toVertexModelId(body.modelOverride);
      const created = await createEveSession(text, model);
      sessionId = created.sessionId;
      setContinuity(userId, chatId, {
        eveSessionId: sessionId,
        continuationToken: created.continuationToken,
        streamIndex: 0,
      });
    }
  } catch (err) {
    // Eve server unreachable or errored on session create/continue.
    console.error('[eve-chat] session error:', err);
    return new ChatSDKError('offline:chat').toResponse();
  }

  const stream = createUIMessageStream({
    execute: async ({ writer }) => {
      const res = await openEveStream(sessionId, startIndex);
      if (!res.body) throw new Error('Eve stream returned no body');
      let ctx = { textId: null as string | null, generateId: generateUUID };
      // Absolute stream position, counted from where this read started, so the
      // cursor stays meaningful across turns.
      let consumed = startIndex;
      for await (const event of parseNdjson(res.body)) {
        consumed++;
        const r = translateEveEvent(event, writer, ctx);
        ctx = { ...ctx, textId: r.textId };
        if (r.liveViewUrl) {
          // Hand the Kernel live-view URL to /api/kernel-browser, which the
          // browser artifact polls. It can't come from eve-browser.ts directly:
          // that map lives in the `eve dev` process, not this one.
          setLiveViewUrl(userId, chatId, r.liveViewUrl);
        }
        if (r.continuationToken) {
          // Persist the cursor with the token: `consumed` already counts this
          // `session.waiting`, so the next turn resumes on the event after it.
          // Only advanced at the turn boundary — a connection dropped mid-turn
          // replays that turn rather than skipping it.
          setContinuity(userId, chatId, {
            eveSessionId: sessionId,
            continuationToken: r.continuationToken,
            streamIndex: consumed,
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
