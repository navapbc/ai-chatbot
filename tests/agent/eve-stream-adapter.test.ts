import { describe, it, expect } from 'vitest';
import { mapToolName, extractLatestUserText, translateEveEvent } from '@/lib/ai/eve/stream-adapter';

function collect() {
  const chunks: any[] = [];
  return { writer: { write: (c: any) => chunks.push(c) }, chunks };
}
const gen = () => 'txt-1';

describe('mapToolName', () => {
  it('maps snake_case card tools to camelCase for message.tsx renderers', () => {
    expect(mapToolName('gap_analysis')).toBe('gapAnalysis');
    expect(mapToolName('form_summary')).toBe('formSummary');
  });
  it('passes through unmapped tools', () => {
    expect(mapToolName('browser')).toBe('browser');
    expect(mapToolName('check_submit_gate')).toBe('check_submit_gate');
  });
});

describe('extractLatestUserText', () => {
  it('pulls text from an AI SDK UIMessage parts array', () => {
    const msg = { role: 'user', parts: [{ type: 'text', text: 'apply for WIC' }] };
    expect(extractLatestUserText(msg)).toBe('apply for WIC');
  });
  it('falls back to empty string when no text part', () => {
    expect(extractLatestUserText({ role: 'user', parts: [] })).toBe('');
  });
});

describe('translateEveEvent', () => {
  it('streams text: message.appended -> text-start + text-delta, message.completed -> text-end', () => {
    const { writer, chunks } = collect();
    let ctx = { textId: null as string | null, generateId: gen };
    let r = translateEveEvent({ type: 'message.appended', data: { messageDelta: 'Hel' } }, writer, ctx);
    ctx = { ...ctx, textId: r.textId };
    r = translateEveEvent({ type: 'message.appended', data: { messageDelta: 'lo' } }, writer, ctx);
    ctx = { ...ctx, textId: r.textId };
    translateEveEvent({ type: 'message.completed', data: {} }, writer, ctx);
    expect(chunks).toEqual([
      { type: 'text-start', id: 'txt-1' },
      { type: 'text-delta', id: 'txt-1', delta: 'Hel' },
      { type: 'text-delta', id: 'txt-1', delta: 'lo' },
      { type: 'text-end', id: 'txt-1' },
    ]);
  });
  it('maps a tool call + result to AI SDK tool chunks with the camelCase name', () => {
    const { writer, chunks } = collect();
    const ctx = { textId: null, generateId: gen };
    translateEveEvent(
      { type: 'actions.requested', data: { actions: [{ kind: 'tool-call', toolName: 'gap_analysis', input: { formName: 'WIC' }, callId: 'call-1' }] } },
      writer, ctx,
    );
    translateEveEvent(
      { type: 'action.result', data: { result: { kind: 'tool-result', callId: 'call-1', toolName: 'gap_analysis', output: { rendered: true } } } },
      writer, ctx,
    );
    expect(chunks).toEqual([
      { type: 'tool-input-available', toolCallId: 'call-1', toolName: 'gapAnalysis', input: { formName: 'WIC' } },
      { type: 'tool-output-available', toolCallId: 'call-1', output: { rendered: true } },
    ]);
  });
  it('maps step.completed.usage to a transient data-token-usage event', () => {
    const { writer, chunks } = collect();
    translateEveEvent(
      { type: 'step.completed', data: { usage: { inputTokens: 100, outputTokens: 20, cacheReadTokens: 40 } } },
      writer, { textId: null, generateId: gen },
    );
    expect(chunks).toEqual([
      { type: 'data-token-usage', data: { inputTokens: 100, outputTokens: 20, cachedInputTokens: 40 }, transient: true },
    ]);
  });
  it('signals done + captures continuationToken on session.waiting', () => {
    const { writer } = collect();
    const r = translateEveEvent({ type: 'session.waiting', data: { continuationToken: 'tok-9' } }, writer, { textId: null, generateId: gen });
    expect(r.done).toBe(true);
    expect(r.continuationToken).toBe('tok-9');
  });
  it('does NOT signal done on turn.completed (it precedes session.waiting and carries no continuationToken)', () => {
    const { writer } = collect();
    const r = translateEveEvent({ type: 'turn.completed', data: {} }, writer, { textId: null, generateId: gen });
    expect(r.done).toBe(false);
  });
  it('ignores lifecycle/echo events without writing', () => {
    const { writer, chunks } = collect();
    for (const t of ['session.started', 'turn.started', 'message.received']) {
      translateEveEvent({ type: t, data: {} }, writer, { textId: null, generateId: gen });
    }
    expect(chunks).toEqual([]);
  });
  // A dropped input.requested batch left the turn silently empty: the agent
  // parked for a person, session.waiting ended the turn, and nothing rendered.
  describe('input.requested (HITL)', () => {
    const askQuestion = {
      type: 'input.requested',
      data: {
        requests: [
          {
            requestId: 'req-1',
            prompt: 'Which county issued the card?',
            display: 'select',
            options: [
              { id: 'riverside', label: 'Riverside' },
              { id: 'orange', label: 'Orange', description: 'Orange County' },
            ],
            action: { kind: 'tool-call', callId: 'call-q', toolName: 'ask_question', input: {} },
          },
        ],
      },
    };

    it('renders the prompt and its option ids as assistant text', () => {
      const { writer, chunks } = collect();
      translateEveEvent(askQuestion, writer, {
        textId: null,
        generateId: gen,
        emittedToolCallIds: new Set(),
      });
      expect(chunks[0]).toEqual({ type: 'text-start', id: 'txt-1' });
      expect(chunks[2]).toEqual({ type: 'text-end', id: 'txt-1' });
      const delta = chunks[1].delta as string;
      expect(delta).toContain('Which county issued the card?');
      // Eve resolves a follow-up matching an option id, so ids must be visible.
      expect(delta).toContain('riverside');
      expect(delta).toContain('Orange County');
      expect(delta).toContain('Reply with one of: riverside, orange');
    });

    it('closes its text block so it never swallows later deltas', () => {
      const { writer } = collect();
      const r = translateEveEvent(askQuestion, writer, {
        textId: null,
        generateId: gen,
        emittedToolCallIds: new Set(),
      });
      expect(r.textId).toBeNull();
      expect(r.done).toBe(false);
    });

    it('writes nothing for an empty batch', () => {
      const { writer, chunks } = collect();
      translateEveEvent({ type: 'input.requested', data: { requests: [] } }, writer, {
        textId: null,
        generateId: gen,
        emittedToolCallIds: new Set(),
      });
      expect(chunks).toEqual([]);
    });

    // An approval's result arrives on the turn that ANSWERS it — a different
    // stream and UI message from the one that announced the call. The AI SDK
    // reducer throws on a tool result it cannot match to a call, so the adapter
    // must synthesize the missing input rather than emit the result alone.
    it('synthesizes the missing call when a result arrives unannounced', () => {
      const { writer, chunks } = collect();
      translateEveEvent(
        { type: 'action.result', data: { result: { kind: 'tool-result', callId: 'call-x', toolName: 'gap_analysis', output: { rendered: true } } } },
        writer,
        { textId: null, generateId: gen, emittedToolCallIds: new Set() },
      );
      expect(chunks).toEqual([
        { type: 'tool-input-available', toolCallId: 'call-x', toolName: 'gapAnalysis', input: {} },
        { type: 'tool-output-available', toolCallId: 'call-x', output: { rendered: true } },
      ]);
    });

    it('does NOT synthesize (and so never clobbers real input) when the call was announced', () => {
      const { writer, chunks } = collect();
      const ctx = { textId: null, generateId: gen, emittedToolCallIds: new Set<string>() };
      translateEveEvent(
        { type: 'actions.requested', data: { actions: [{ kind: 'tool-call', toolName: 'gap_analysis', input: { formName: 'WIC' }, callId: 'call-1' }] } },
        writer, ctx,
      );
      translateEveEvent(
        { type: 'action.result', data: { result: { kind: 'tool-result', callId: 'call-1', toolName: 'gap_analysis', output: { rendered: true } } } },
        writer, ctx,
      );
      expect(chunks).toEqual([
        { type: 'tool-input-available', toolCallId: 'call-1', toolName: 'gapAnalysis', input: { formName: 'WIC' } },
        { type: 'tool-output-available', toolCallId: 'call-1', output: { rendered: true } },
      ]);
    });
  });

  it('emits start-step on step.started (restores per-step tool grouping; not done)', () => {
    const { writer, chunks } = collect();
    const r = translateEveEvent({ type: 'step.started', data: {} }, writer, { textId: null, generateId: gen });
    expect(chunks).toEqual([{ type: 'start-step' }]);
    expect(r.done).toBe(false);
  });

  // The live-view URL can only reach the Next process on a tool result: the
  // Kernel browser is created inside the separate `eve dev` process.
  it('reports liveViewUrl off a browser tool result', () => {
    const { writer } = collect();
    const r = translateEveEvent(
      {
        type: 'action.result',
        data: {
          result: {
            kind: 'tool-result',
            callId: 'call-9',
            output: { success: true, output: 'ok', liveViewUrl: 'https://live.example/v/abc' },
          },
        },
      },
      writer,
      { textId: null, generateId: gen },
    );
    expect(r.liveViewUrl).toBe('https://live.example/v/abc');
  });
  it('still forwards the tool output when a liveViewUrl rides along', () => {
    const { writer, chunks } = collect();
    translateEveEvent(
      {
        type: 'action.result',
        data: {
          result: {
            kind: 'tool-result',
            callId: 'call-9',
            output: { success: true, output: 'ok', liveViewUrl: 'https://live.example/v/abc' },
          },
        },
      },
      writer,
      { textId: null, generateId: gen },
    );
    expect(chunks).toEqual([
      {
        type: 'tool-output-available',
        toolCallId: 'call-9',
        output: { success: true, output: 'ok', liveViewUrl: 'https://live.example/v/abc' },
      },
    ]);
  });
  it('leaves liveViewUrl undefined for other tools and for a headless browser', () => {
    const { writer } = collect();
    const other = translateEveEvent(
      { type: 'action.result', data: { result: { kind: 'tool-result', callId: 'c1', output: { rendered: true } } } },
      writer,
      { textId: null, generateId: gen },
    );
    expect(other.liveViewUrl).toBeUndefined();
    // A headless session reports null rather than a URL — must not be stored.
    const headless = translateEveEvent(
      { type: 'action.result', data: { result: { kind: 'tool-result', callId: 'c2', output: { success: true, liveViewUrl: null } } } },
      writer,
      { textId: null, generateId: gen },
    );
    expect(headless.liveViewUrl).toBeUndefined();
  });
});
