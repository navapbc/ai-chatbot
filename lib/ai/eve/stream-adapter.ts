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
  generateId: () => string;
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
): { textId: string | null; done: boolean; continuationToken?: string } {
  let textId = ctx.textId;
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
    case 'actions.requested': {
      for (const a of event.data?.actions ?? []) {
        if (a?.kind !== 'tool-call') continue;
        writer.write({
          type: 'tool-input-available',
          toolCallId: a.callId,
          toolName: mapToolName(a.toolName),
          input: a.input ?? {},
        });
      }
      break;
    }
    case 'action.result': {
      const r = event.data?.result;
      if (r?.kind === 'tool-result') {
        writer.write({ type: 'tool-output-available', toolCallId: r.callId, output: r.output });
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
      return { textId, done: true, continuationToken: event.data?.continuationToken };
    case 'turn.completed':
      return { textId, done: true };
    // session.started / turn.started / message.received / step.started: ignored.
    default:
      break;
  }
  return { textId, done: false };
}
