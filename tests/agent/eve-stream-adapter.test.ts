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
    for (const t of ['session.started', 'turn.started', 'message.received', 'step.started']) {
      translateEveEvent({ type: t, data: {} }, writer, { textId: null, generateId: gen });
    }
    expect(chunks).toEqual([]);
  });
});
