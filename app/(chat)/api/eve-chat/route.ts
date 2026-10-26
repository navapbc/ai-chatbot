import { createUIMessageStream, JsonToSseTransformStream } from 'ai';
import { auth } from '@/app/(auth)/auth';
import { generateUUID } from '@/lib/utils';
import { ChatSDKError } from '@/lib/errors';
import { EveRunExistsError, getRunStore } from '@/lib/ai/eve/run-store';
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
import { type InputAnswer, validateAnswers } from '@/lib/ai/eve/input-requests';
import { isProductionEnvironment } from '@/lib/constants';
import { normalizeOrigin } from '@/lib/relay/policy';
import {
  MAX_RELAY_TEXT_CHARS,
  mustRefuseRelayText,
  stripSensitiveText,
} from '@/lib/data/sensitive-record';
import { buildExtensionRunMessage } from '@/lib/relay/run-prompt';
import { toRelayWsUrl } from '@/lib/relay/url';

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

/** Placeholder stored while a follow-up turn is in flight; never sent to Eve. */
const CLAIM_PREFIX = 'claim:';

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

// The text of a relay turn, made safe to forward. Size first: the scanners
// are linear but their input is attacker-controlled. Then strip, then refuse
// what is still a sensitive label followed by a value (nothing is echoed).
type RelayText =
  | { ok: true; text: string }
  | { ok: false; reason: 'message_too_large' | 'sensitive_value_in_message' };
const relayText = (text: string): RelayText => {
  if (text.length > MAX_RELAY_TEXT_CHARS) {
    return { ok: false, reason: 'message_too_large' };
  }
  const stripped = stripSensitiveText(text);
  if (mustRefuseRelayText(stripped)) {
    return { ok: false, reason: 'sensitive_value_in_message' };
  }
  return { ok: true, text: stripped };
};

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
    relay?: unknown;
    /** Relay runs only: answers to the pending `ask_question` requests. */
    inputResponses?: unknown;
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
  const isUserMessage = body.message?.role === 'user';
  const text = isUserMessage ? extractLatestUserText(body.message) : '';
  // A body with `inputResponses` may carry no message: whether it is an
  // answer turn depends on the stored run, so its checks wait for the row.
  const hasAnswers = body.inputResponses !== undefined;
  if (!chatId || (!hasAnswers && (!isUserMessage || !text))) {
    return new ChatSDKError('bad_request:api').toResponse();
  }

  // A relay run is opt-in and flag-gated. With the flag off the field is not
  // silently dropped: the caller asked for something this server won't do.
  const relayRunsEnabled = process.env.RELAY_RUNS_ENABLED === '1';
  if (body.relay !== undefined && !relayRunsEnabled) {
    return new ChatSDKError('bad_request:api').toResponse();
  }

  // Resolve or create the Eve session for this chat.
  const store = getRunStore();
  let sessionId: string;
  // Where to resume this chat's durable event stream. A continued session picks
  // up past the previous turn's `session.waiting`; a fresh one starts at 0.
  let startIndex = 0;
  // Relay runs get structured questions; web chats keep the prose rendering.
  let structuredInput = false;
  // True once this request's answer turn has cleared pending itself, so the
  // session.waiting heal below has nothing to do.
  let pendingCleared = false;
  // A follow-up's claim and the token Eve returned for it. At this turn's
  // session.waiting the claim, if still stored (the final swap failed), is
  // replaced by that token. The stream's own token is never stored.
  let heal: { claim: string; token: string } | undefined;
  try {
    const existing = await store.get(userId, chatId);
    // A web chat never becomes a relay run: a relay field on its follow-up is
    // refused rather than ignored. On a relay run's follow-up it is ignored
    // (the stored origin governs).
    if (existing && existing.relayOrigin === null && body.relay !== undefined) {
      return new ChatSDKError('bad_request:api').toResponse();
    }
    // What a follow-up forwards to Eve. Decided, and every rejection
    // returned, before the token is claimed: a rejected turn never touches
    // the token or reaches Eve.
    let input: string | { inputResponses: InputAnswer[] } = text;
    let answering = false;
    if (existing && existing.relayOrigin !== null) {
      structuredInput = true;
      // A relay run never forwards a participant record's sensitive values,
      // on any turn. Text without a record is returned unchanged.
      const safe = relayText(text);
      if (!safe.ok)
        return new ChatSDKError('bad_request:api', safe.reason).toResponse();
      input = safe.text;
      const pending = existing.pendingRequests;
      if (hasAnswers) {
        if (pending.length === 0) {
          return new ChatSDKError(
            'bad_request:api',
            'unknown_request',
          ).toResponse();
        }
        const result = validateAnswers(pending, body.inputResponses);
        if (!result.ok) {
          return new ChatSDKError('bad_request:api', result.error).toResponse();
        }
        // Only the validated responses go to Eve: never the client's body,
        // and never a `message` sent alongside the answers.
        input = { inputResponses: result.responses };
        answering = true;
      } else if (pending.length > 0) {
        return new ChatSDKError(
          'bad_request:api',
          'answers_pending',
        ).toResponse();
      }
    }
    // Web chats and first turns ignore `inputResponses` and need a message.
    if (!answering && (!isUserMessage || !text)) {
      return new ChatSDKError('bad_request:api').toResponse();
    }
    // Session-continuity diagnostic. This is the only place that records
    // whether a turn continued an Eve session or started a new one, and the
    // resume cursor here is what a stream stall has to be read against.
    //
    // It deliberately logs NO message content. It used to append
    // `text.slice(0, 60)`, which on a real run was cut mid-SSN at "Social
    // Security Nu" — a shorter preamble would have written the digits into
    // the dev log and, in deployment, into Cloud Logging. Applicant PII must
    // not reach logs: it is the same reason `lib/kernel/telemetry.ts` leaves
    // Kernel's `network` category off. Log the length instead; if a future
    // bug needs the text, read it from the Eve session, which already holds
    // it under access control.
    console.log(
      `[eve-chat] turn userId=${userId} chatId=${chatId} decision=${existing ? 'CONTINUE' : 'CREATE'} existingSession=${existing?.eveSessionId ?? 'none'} resumeFrom=${existing?.streamIndex ?? 0} ${typeof input === 'string' ? `msgChars=${text.length}` : `answers=${input.inputResponses.length}`}`,
    );
    if (existing) {
      // Claim the stored token before talking to Eve: of turns that arrive
      // together, only one wins the compare-and-set and reaches Eve. The
      // claim is released at the final swap, right after continue returns,
      // not at session.waiting: a follow-up sent while Eve is still running
      // the turn can reach Eve, which folds it into the next turn, and that
      // request's read replays from the old cursor (Eve 0.27 keeps one token
      // per session, so the token alone does not stop it). Holding the claim
      // until session.waiting is an open decision (todo). The claim is a
      // random placeholder; the real token replaces it. A crash
      // between the claim and that final swap leaves the chat on a claim until
      // recovery is built (no stale-claim takeover exists yet).
      if (existing.continuationToken.startsWith(CLAIM_PREFIX)) {
        // Another turn holds the claim. Never hand a placeholder to Eve.
        console.error('[eve-chat] turn in flight; stored token is a claim', {
          chatId,
        });
        return new ChatSDKError('offline:chat').toResponse();
      }
      const claim = `${CLAIM_PREFIX}${generateUUID()}`;
      if (
        !(await store.swapToken(
          userId,
          chatId,
          existing.continuationToken,
          claim,
        ))
      ) {
        return new ChatSDKError('offline:chat').toResponse();
      }
      let nextToken: string;
      try {
        let landedOn: string | undefined;
        ({ continuationToken: nextToken, sessionId: landedOn } =
          await continueEveSession(
            existing.eveSessionId,
            existing.continuationToken,
            input,
          ));
        // When the stored session's run has ended, Eve's send path starts a
        // fresh session with this request's auth, which carries no relay
        // attributes: its browser would be Kernel. Fail the turn instead (the
        // claim is rolled back below, as for any continue failure). Web chats
        // keep the previous behavior.
        if (
          existing.relayOrigin !== null &&
          landedOn !== existing.eveSessionId
        ) {
          console.error('[eve-chat] continue landed on another session', {
            chatId,
          });
          throw new Error('Eve continue landed on a different session');
        }
        // Eve 0.27's continue response carries no token (`{ ok, sessionId }`)
        // and the eve channel keeps one token for the whole session, so the
        // token just sent stays the valid one. Never store an empty token: the
        // next turn would send it and Eve rejects an empty one with a 400.
        if (!nextToken) nextToken = existing.continuationToken;
      } catch (err) {
        // Eve never accepted the message, so put the stored token back and the
        // run is unchanged and retryable. A failed continue never falls back to
        // a fresh session; the turn fails as it did before.
        try {
          await store.swapToken(
            userId,
            chatId,
            claim,
            existing.continuationToken,
          );
        } catch (rollbackErr) {
          console.error('[eve-chat] token rollback failed', {
            chatId,
            rollbackErr,
          });
        }
        throw err;
      }
      // From here the claim must never be rolled back: Eve accepted the turn.
      // If this swap fails the row keeps the claim, and `advance` at this
      // turn's session.waiting replaces it with nextToken.
      heal = { claim, token: nextToken };
      if (answering) {
        // Eve accepted the answers. Clear them while the claim is still held,
        // so no other turn can see the real token next to answered questions
        // and forward them a second time.
        try {
          await store.setPending(userId, chatId, []);
          pendingCleared = true;
        } catch (clearErr) {
          console.error('[eve-chat] pending clear failed', {
            chatId,
            clearErr,
          });
        }
      }
      try {
        if (!(await store.swapToken(userId, chatId, claim, nextToken))) {
          console.error('[eve-chat] final token swap returned false', {
            chatId,
          });
        }
      } catch (swapErr) {
        console.error('[eve-chat] final token swap failed', {
          chatId,
          swapErr,
        });
      }
      sessionId = existing.eveSessionId;
      startIndex = existing.streamIndex;
    } else {
      // Model override is a dev/eval feature — never honor it in production
      // (server-side parity with the legacy route; the client also only sends
      // modelOverride in non-prod).
      const model = isProductionEnvironment
        ? undefined
        : toVertexModelId(body.modelOverride);
      // The relay field is read on the first turn only. A follow-up turn
      // ignores it (the stored origin governs), so it is never validated there.
      let relayOrigin: string | null = null;
      let relayRef: { url: string; runId: string } | undefined;
      if (body.relay !== undefined) {
        const origin = (body.relay as { origin?: unknown } | null)?.origin;
        const url = process.env.RELAY_PUBLIC_URL
          ? toRelayWsUrl(process.env.RELAY_PUBLIC_URL)
          : null;
        if (
          typeof origin !== 'string' ||
          normalizeOrigin(origin) !== origin ||
          !url ||
          !UUID.test(chatId)
        ) {
          return new ChatSDKError('bad_request:api').toResponse();
        }
        relayOrigin = origin;
        structuredInput = true;
        // The relay run id is the chat id.
        relayRef = { url, runId: chatId };
      }
      let safe: ReturnType<typeof relayText> = { ok: true, text };
      if (relayRef) safe = relayText(text);
      if (!safe.ok)
        return new ChatSDKError('bad_request:api', safe.reason).toResponse();
      const created = relayRef
        ? await createEveSession(
            // Stripped BEFORE the marker is added, and before anything is sent.
            buildExtensionRunMessage(safe.text),
            model,
            relayRef,
          )
        : await createEveSession(text, model);
      sessionId = created.sessionId;
      try {
        await store.create(userId, chatId, {
          eveSessionId: sessionId,
          continuationToken: created.continuationToken,
          ...(relayOrigin ? { relayOrigin } : {}),
        });
      } catch (err) {
        if (!(err instanceof EveRunExistsError)) {
          // A real store failure, not a lost race: do not report it as the
          // conflict. The Eve session above is likewise unreferenced.
          console.error('[eve-chat] run create failed; orphan eve session', {
            sessionId,
            chatId,
            err,
          });
          return new ChatSDKError('bad_request:database').toResponse();
        }
        // Two first turns raced; the other one owns the run. The Eve session
        // created above is now unreferenced (Eve has no delete call here), so
        // report it rather than leave it silent.
        console.error('[eve-chat] orphan eve session after create conflict', {
          sessionId,
          chatId,
        });
        return new ChatSDKError('offline:chat').toResponse();
      }
    }
  } catch (err) {
    // Eve server unreachable or errored on session create/continue, or the run
    // store failed.
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
        structuredInput,
      };
      // Absolute stream position, counted from where this read started, so the
      // cursor stays meaningful across turns.
      let consumed = startIndex;
      let done = false;
      // Survives reconnects: whether this turn's stream raised questions.
      let sawInputRequests = false;
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
            if (r.progress) {
              // The adapter only returns progress on relay runs, already
              // sanitized. Best-effort status: a failed write must not kill
              // the stream. Log the error only, never the progress.
              try {
                await store.setProgress(userId, chatId, r.progress);
              } catch (err) {
                console.error('[eve-chat] setProgress failed', err);
              }
            }
            if (r.inputRequests) {
              // Persisted before session.waiting is handled, so the answer
              // turn validates against them even after a restart. An answer
              // sent before this write lands is rejected (nothing pending).
              await store.setPending(userId, chatId, r.inputRequests);
              sawInputRequests = true;
            }
            if (
              r.continuationToken &&
              structuredInput &&
              !sawInputRequests &&
              !pendingCleared
            ) {
              // The turn ended without new questions, so nothing may be
              // pending. Heals a list a failed clear left behind.
              await store.setPending(userId, chatId, []);
            }
            if (r.continuationToken) {
              // Persist the cursor: `consumed` already counts this
              // `session.waiting`, so the next turn resumes on the event after
              // it. Only advanced at a turn boundary — a connection dropped
              // mid-turn replays that turn rather than skipping it.
              //
              // The event's token is deliberately not stored. A read that
              // resumes from a stale cursor (an earlier turn's read died
              // before its session.waiting) reaches that earlier boundary
              // first, and its token may be one a later turn already used.
              // Tokens come only from Eve's create/continue responses; the
              // one write here is healing this turn's own claim.
              await store.advance(userId, chatId, {
                streamIndex: consumed,
                ...(heal ? { heal } : {}),
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
      // startIndex), or a throw out of store.advance/setLiveViewUrl.
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
