// Pure translator: Eve NDJSON events -> AI SDK v7 UIMessage stream chunks.
// No `eve` import, no server-only — maps plain objects, so it is unit-testable
// and safe in a Next route. Chunk shapes match AI SDK v7 (text-start/delta/end,
// tool-input-available, tool-output-available, data-* transient).

export const EVE_TOOL_NAME_MAP: Record<string, string> = {
  gap_analysis: 'gapAnalysis',
  form_summary: 'formSummary',
};

export function mapToolName(eveName: string): string {
  return EVE_TOOL_NAME_MAP[eveName] ?? eveName;
}

interface Writer {
  write(chunk: any): void;
}
interface Ctx {
  textId: string | null;
  /**
   * Open reasoning-block id, tracked separately from `textId` because eve
   * streams reasoning and message text as independent blocks that can
   * interleave. Optional for callers that predate it.
   */
  reasoningId?: string | null;
  generateId: () => string;
  /**
   * callIds for which a `tool-input-available` chunk has already been written
   * on THIS stream.
   *
   * The AI SDK reducer resolves a `tool-output-available` chunk by looking up
   * its callId among the current message's parts, and THROWS
   * (`UIMessageStreamError`) when it finds none — killing the whole stream. A
   * result therefore must never be forwarded on its own. That happens for real
   * on any HITL round trip: the call is announced on the turn that parks, but
   * its `action.result` only arrives on the LATER turn that answers it, i.e. a
   * different HTTP request and a different UI message. Optional so a caller
   * that omits it keeps the old behavior rather than silently losing inputs.
   */
  emittedToolCallIds?: Set<string>;
}

// Render one pending HITL request as readable assistant text. Eve resolves a
// plain follow-up message whose text matches an option id, label, or index
// (see node_modules/eve/docs/tools/human-in-the-loop.md), so the normal chat
// input is already a working answer channel — the prompt just has to be
// visible for the user to know what to type.
function formatInputRequest(request: any): string {
  const prompt = typeof request?.prompt === 'string' ? request.prompt : '';
  const options = Array.isArray(request?.options) ? request.options : [];
  const lines = [prompt.trim()];
  if (options.length > 0) {
    lines.push(
      '',
      ...options.map((o: any) => {
        const label = typeof o?.label === 'string' ? o.label : String(o?.id ?? '');
        const description =
          typeof o?.description === 'string' && o.description
            ? ` — ${o.description}`
            : '';
        return `- \`${o?.id}\` ${label}${description}`;
      }),
      '',
      `Reply with one of: ${options.map((o: any) => o?.id).join(', ')}`,
    );
  }
  return lines.join('\n');
}

// Pull the user's text out of the AI SDK UIMessage the client sends as body.message.
export function extractLatestUserText(message: unknown): string {
  const m = message as { parts?: Array<{ type?: string; text?: string }> } | null;
  if (!m?.parts) return '';
  return m.parts
    .filter((p) => p.type === 'text' && typeof p.text === 'string')
    .map((p) => p.text as string)
    .join('')
    .trim();
}

// Apply ONE Eve event to the writer. Returns updated text-block id, whether the
// turn is finished, and any continuation token seen.
export function translateEveEvent(
  event: any,
  writer: Writer,
  ctx: Ctx,
): {
  textId: string | null;
  reasoningId: string | null;
  done: boolean;
  continuationToken?: string;
  liveViewUrl?: string;
} {
  let textId = ctx.textId;
  let reasoningId = ctx.reasoningId ?? null;
  // Reported by the browser tool so the chat UI's live panel can find the
  // Kernel browser that `eve dev` created in its own process.
  let liveViewUrl: string | undefined;
  switch (event?.type) {
    case 'message.appended': {
      const delta = event.data?.messageDelta ?? '';
      if (!delta) break;
      if (textId === null) {
        textId = ctx.generateId();
        writer.write({ type: 'text-start', id: textId });
      }
      writer.write({ type: 'text-delta', id: textId, delta });
      break;
    }
    case 'message.completed': {
      if (textId !== null) {
        writer.write({ type: 'text-end', id: textId });
        textId = null;
      }
      break;
    }
    // Reasoning streams as its own block alongside message text. Dropped before,
    // so extended-thinking output never reached the UI at all.
    case 'reasoning.appended': {
      const delta = event.data?.reasoningDelta ?? '';
      if (!delta) break;
      if (reasoningId === null) {
        reasoningId = ctx.generateId();
        writer.write({ type: 'reasoning-start', id: reasoningId });
      }
      writer.write({ type: 'reasoning-delta', id: reasoningId, delta });
      break;
    }
    case 'reasoning.completed': {
      if (reasoningId !== null) {
        writer.write({ type: 'reasoning-end', id: reasoningId });
        reasoningId = null;
      }
      break;
    }
    case 'actions.requested': {
      for (const a of event.data?.actions ?? []) {
        if (a?.kind !== 'tool-call') continue;
        writer.write({
          type: 'tool-input-available',
          toolCallId: a.callId,
          toolName: mapToolName(a.toolName),
          input: a.input ?? {},
        });
        ctx.emittedToolCallIds?.add(a.callId);
      }
      break;
    }
    case 'input.requested': {
      // The agent parked for a person: a tool approval, or an `ask_question`.
      // Without this case the batch is dropped and the turn ends on the
      // `session.waiting` that follows, so the user gets a silently EMPTY
      // assistant message and the session sits parked with nothing on screen.
      // `ask_question` is the worst of it: eve excludes it from
      // `actions.requested` (excludedActionToolNames in harness/tool-loop.js),
      // so this event is the only place it ever surfaces.
      //
      // Deliberately no tool part here. It would render a row stuck at
      // `input-available` forever — the matching result lands on a later turn's
      // stream — and it would not prevent the throw that guard in
      // `action.result` handles, since that later stream is a different UI
      // message either way.
      const requests = event.data?.requests ?? [];
      const text = requests
        .map((r: any) => formatInputRequest(r))
        .filter((t: string) => t.length > 0)
        .join('\n\n');
      if (text) {
        if (textId === null) {
          textId = ctx.generateId();
          writer.write({ type: 'text-start', id: textId });
        }
        writer.write({ type: 'text-delta', id: textId, delta: text });
        writer.write({ type: 'text-end', id: textId });
        textId = null;
      }
      break;
    }
    case 'action.result': {
      const r = event.data?.result;
      if (r?.kind === 'tool-result') {
        // Never forward a result whose call was not announced on this stream —
        // the reducer throws on the lookup miss and the stream dies. Synthesize
        // the missing input first (empty, so a real input already streamed is
        // never clobbered — hence the set rather than doing this every time).
        if (
          ctx.emittedToolCallIds !== undefined &&
          !ctx.emittedToolCallIds.has(r.callId)
        ) {
          writer.write({
            type: 'tool-input-available',
            toolCallId: r.callId,
            toolName: mapToolName(r.toolName),
            input: {},
          });
          ctx.emittedToolCallIds.add(r.callId);
        }
        writer.write({ type: 'tool-output-available', toolCallId: r.callId, output: r.output });
        // Only the browser tool reports this field, so no callId->toolName
        // correlation is needed to recognize it.
        const url = (r.output as { liveViewUrl?: unknown } | null)?.liveViewUrl;
        if (typeof url === 'string' && url) liveViewUrl = url;
      }
      break;
    }
    case 'step.completed': {
      const u = event.data?.usage;
      if (u) {
        writer.write({
          type: 'data-token-usage',
          data: {
            inputTokens: u.inputTokens ?? 0,
            outputTokens: u.outputTokens ?? 0,
            cachedInputTokens: u.cacheReadTokens ?? 0,
          },
          transient: true,
        });
      }
      break;
    }
    case 'session.waiting':
      return {
        textId,
        reasoningId,
        done: true,
        continuationToken: event.data?.continuationToken,
      };
    case 'turn.completed':
      // turn.completed fires BEFORE session.waiting and carries no continuationToken.
      // Only session.waiting may signal done, or a route loop that breaks on
      // r.done would stop here and never read the continuationToken.
      return { textId, reasoningId, done: false };
    case 'step.started':
      // Restores per-step tool grouping in the UI, and is defense-in-depth:
      // a start-step chunk between steps helps keep the client from ever
      // mistaking a mid-turn tool step for a fully "complete" assistant
      // message (see sendAutomaticallyWhen gating in components/chat.tsx).
      writer.write({ type: 'start-step' });
      break;
    case 'session.started':
    case 'turn.started':
    case 'message.received':
      // Known lifecycle echoes with nothing for the client to render.
      break;
    case 'subagent.called':
    case 'subagent.started':
    case 'subagent.completed':
      // Control-plane only. The child runs on its own session stream, which
      // this route does not subscribe to, so there is nothing to forward. Named
      // explicitly so the unhandled-type warning below stays meaningful.
      //
      // Worth knowing: the parent stream emits `subagent.called` and then goes
      // SILENT until the child finishes. That silence is what trips undici's
      // 300s body timeout on a long research step — see the reconnect loop in
      // app/(chat)/api/eve-chat/route.ts.
      break;
    default:
      // An event type this translator has no case for is silently dropped
      // otherwise — the exact failure mode that made a completed gap_analysis
      // tool call vanish client-side while Braintrust showed it succeeding.
      console.warn('[eve-stream-adapter] unhandled Eve event type:', event?.type, event?.data);
      break;
  }
  return { textId, reasoningId, done: false, liveViewUrl };
}
