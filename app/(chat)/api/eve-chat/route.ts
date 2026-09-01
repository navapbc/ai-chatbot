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

/**
 * How long to keep resuming a turn's stream before giving up.
 *
 * Node's fetch (undici) caps an idle response body at 300s — `bodyTimeout` —
 * and eve's ROOT stream goes completely silent for the whole of a subagent
 * run: it emits `subagent.called` and then nothing until the child finishes.
 * A `requirements_research` step that takes >5 min therefore killed the read
 * with `TypeError: terminated` / `UND_ERR_BODY_TIMEOUT` even though the
 * session was healthy and still working — the user got a generic error while
 * eve went on to finish the turn, which is why the traces looked clean.
 *
 * Eve's stream is durable and replayable from an absolute index, so the fix is
 * to re-open at the cursor rather than fail the turn. The timeout is kept as a
 * liveness signal (removing it would turn a recoverable stall into an
 * unbounded hang); this budget bounds the recovery. Sized under Cloud Run's
 * request timeout (`chatbot_timeout`, 3600s in terraform/variables.tf).
 */
const RECONNECT_BUDGET_MS = 45 * 60 * 1000;

/**
 * Secondary bound on the same loop. Elapsed time is the honest limit, but a
 * body timeout takes ~300s to fire, so at most ~9 fit in the budget above — a
 * run that burns through more than this is failing fast for some other reason
 * and must not hot-loop against the eve server for 45 minutes.
 */
const RECONNECT_MAX_ATTEMPTS = 12;

/** Only undici's idle-body timeout is recoverable; the code sits on `cause`. */
function isBodyTimeout(err: unknown): boolean {
  return (
    (err as { cause?: { code?: string } } | null)?.cause?.code ===
    'UND_ERR_BODY_TIMEOUT'
  );
}

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
      // MUST outlive the reconnect loop below. `consumed` is the resume cursor,
      // and `ctx` carries the open text/reasoning block ids plus the set of
      // announced tool calls — rebuilding any of it per connection would
      // re-open at a stale index, orphan a half-written block, or (worst)
      // forget a call and let the adapter synthesize an empty input over the
      // real one, blanking the card it was streamed for.
      const ctx = {
        textId: null as string | null,
        reasoningId: null as string | null,
        generateId: generateUUID,
        emittedToolCallIds: new Set<string>(),
      };
      // Absolute stream position, counted from where this read started, so the
      // cursor stays meaningful across turns.
      let consumed = startIndex;
      let done = false;
      const deadline = Date.now() + RECONNECT_BUDGET_MS;
      let reconnects = 0;

      while (!done) {
        try {
          const res = await openEveStream(sessionId, consumed);
          if (!res.body) throw new Error('Eve stream returned no body');
          for await (const event of parseNdjson(res.body)) {
            consumed++;
            const r = translateEveEvent(event, writer, ctx);
            ctx.textId = r.textId;
            ctx.reasoningId = r.reasoningId;
            if (r.liveViewUrl) {
              // Hand the Kernel live-view URL to /api/kernel-browser, which the
              // browser artifact polls. It can't come from eve-browser.ts
              // directly: that map lives in the `eve dev` process, not this one.
              setLiveViewUrl(userId, chatId, r.liveViewUrl);
            }
            if (r.continuationToken) {
              // Persist the cursor with the token: `consumed` already counts
              // this `session.waiting`, so the next turn resumes on the event
              // after it. Only advanced at the turn boundary — a connection
              // dropped mid-turn replays that turn rather than skipping it.
              setContinuity(userId, chatId, {
                eveSessionId: sessionId,
                continuationToken: r.continuationToken,
                streamIndex: consumed,
              });
            }
            if (r.done) {
              // Only true on session.waiting (after the token is captured
              // above) — turn.completed always returns done:false. Eve's stream
              // stays open past session.waiting, so this break is required to
              // avoid hanging.
              done = true;
              break;
            }
          }
          // Body ended cleanly without session.waiting. Eve normally holds the
          // stream open past the turn boundary, so treat this as terminal
          // rather than reconnecting — re-opening a closed stream would spin.
          if (!done) break;
        } catch (err) {
          // Anything that is NOT the idle-body timeout must still reject, or we
          // go blind to the next distinct failure all over again.
          if (
            !isBodyTimeout(err) ||
            Date.now() > deadline ||
            reconnects >= RECONNECT_MAX_ATTEMPTS
          ) {
            throw err;
          }
          reconnects += 1;
          console.warn('[eve-chat] body timeout — resuming stream', {
            sessionId,
            chatId,
            resumeFrom: consumed,
            reconnects,
          });
        }
      }
    },
    generateId: generateUUID,
    onError: (error) => {
      // The client only ever sees the generic string below, so without this the
      // actual exception is discarded and every distinct failure looks
      // identical from the outside. Log the real cause: the candidates behave
      // very differently and need telling apart — an undici body/inactivity
      // timeout on the loopback read (`UND_ERR_BODY_TIMEOUT`), a dropped
      // connection (`ECONNRESET`), an AI SDK `UIMessageStreamError` (a
      // tool-output chunk whose tool-input never arrived, e.g. on a replayed
      // startIndex), or a throw out of setContinuity/setLiveViewUrl.
      console.error('[eve-chat] stream error:', {
        sessionId,
        chatId,
        startIndex,
        name: (error as { name?: string } | null)?.name,
        code: (error as { code?: unknown } | null)?.code,
        cause: (error as { cause?: unknown } | null)?.cause,
        error,
      });
      return 'Oops, an error occurred running the agent.';
    },
  });

  return new Response(stream.pipeThrough(new JsonToSseTransformStream()));
}
