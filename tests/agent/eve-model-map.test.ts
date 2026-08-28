import { describe, it, expect } from 'vitest';
import {
  VERTEX_CONTEXT_WINDOW_TOKENS,
  isVertexModelId,
  toVertexModelId,
} from '@/lib/ai/eve/model-map';

describe('toVertexModelId', () => {
  it('maps Claude picker ids to Vertex model ids', () => {
    expect(toVertexModelId('claude-opus-5')).toBe('claude-opus-5');
    expect(toVertexModelId('claude-sonnet-5')).toBe('claude-sonnet-5');
    expect(toVertexModelId('claude-opus-4-8')).toBe('claude-opus-4-8');
    expect(toVertexModelId('claude-opus-4-7')).toBe('claude-opus-4-7');
    expect(toVertexModelId('claude-sonnet-4-6')).toBe('claude-sonnet-4-6');
    expect(toVertexModelId('claude-haiku-4-5')).toBe('claude-haiku-4-5');
  });
  it('drops the gpt-5.4 family — Vertex does not serve OpenAI models', () => {
    expect(toVertexModelId('gpt-5.4')).toBeUndefined();
    expect(toVertexModelId('gpt-5.4-pro')).toBeUndefined();
    expect(toVertexModelId('gpt-5.4-mini')).toBeUndefined();
    expect(toVertexModelId('gpt-5.4-nano')).toBeUndefined();
  });
  it('returns undefined for unmapped / base / empty ids', () => {
    expect(toVertexModelId('chat-model')).toBeUndefined();
    expect(toVertexModelId('chat-model-reasoning')).toBeUndefined();
    expect(toVertexModelId('')).toBeUndefined();
    expect(toVertexModelId(undefined)).toBeUndefined();
    expect(toVertexModelId(null)).toBeUndefined();
    expect(toVertexModelId('something-unknown')).toBeUndefined();
  });
});

describe('isVertexModelId', () => {
  it('accepts allowlisted Vertex model ids', () => {
    expect(isVertexModelId('claude-opus-5')).toBe(true);
    expect(isVertexModelId('claude-sonnet-5')).toBe(true);
    expect(isVertexModelId('claude-sonnet-4-6')).toBe(true);
    expect(isVertexModelId('claude-opus-4-7')).toBe(true);
  });
  it('rejects gateway slugs, unknown ids, and non-strings', () => {
    // The old gateway form must not slip through as a Vertex model id.
    expect(isVertexModelId('anthropic/claude-sonnet-4.6')).toBe(false);
    expect(isVertexModelId('openai/gpt-5.4-mini')).toBe(false);
    expect(isVertexModelId('claude-sonnet-4.6')).toBe(false);
    expect(isVertexModelId('')).toBe(false);
    expect(isVertexModelId(null)).toBe(false);
    expect(isVertexModelId(undefined)).toBe(false);
    expect(isVertexModelId(['claude-sonnet-4-6'])).toBe(false);
  });
});

describe('VERTEX_CONTEXT_WINDOW_TOKENS', () => {
  it('is the conservative 200K default rather than the tier-gated 1M', () => {
    expect(VERTEX_CONTEXT_WINDOW_TOKENS).toBe(200_000);
  });
});
