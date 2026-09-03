import { describe, it, expect } from 'vitest';
import {
  OPENAI_CONTEXT_WINDOW_TOKENS,
  VERTEX_CONTEXT_WINDOW_TOKENS,
  contextWindowTokensFor,
  isOpenAIModelId,
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
  it('maps the gpt-5.4 family to OpenAI model ids', () => {
    expect(toVertexModelId('gpt-5.4')).toBe('gpt-5.4');
    expect(toVertexModelId('gpt-5.4-pro')).toBe('gpt-5.4-pro');
    expect(toVertexModelId('gpt-5.4-mini')).toBe('gpt-5.4-mini');
    expect(toVertexModelId('gpt-5.4-nano')).toBe('gpt-5.4-nano');
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
  it('accepts allowlisted OpenAI model ids', () => {
    expect(isVertexModelId('gpt-5.4')).toBe(true);
    expect(isVertexModelId('gpt-5.4-pro')).toBe(true);
    expect(isVertexModelId('gpt-5.4-mini')).toBe(true);
    expect(isVertexModelId('gpt-5.4-nano')).toBe(true);
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

describe('isOpenAIModelId', () => {
  it('accepts the allowlisted gpt-5.4 family', () => {
    expect(isOpenAIModelId('gpt-5.4')).toBe(true);
    expect(isOpenAIModelId('gpt-5.4-pro')).toBe(true);
    expect(isOpenAIModelId('gpt-5.4-mini')).toBe(true);
    expect(isOpenAIModelId('gpt-5.4-nano')).toBe(true);
  });
  it('rejects Claude ids and unknown values', () => {
    expect(isOpenAIModelId('claude-opus-5')).toBe(false);
    expect(isOpenAIModelId('gpt-4o')).toBe(false);
    expect(isOpenAIModelId(null)).toBe(false);
    expect(isOpenAIModelId(undefined)).toBe(false);
  });
});

describe('contextWindowTokensFor', () => {
  it('gives the gpt-5.4 family the 1.05M window and Claude ids the 200K window', () => {
    expect(contextWindowTokensFor('gpt-5.4')).toBe(OPENAI_CONTEXT_WINDOW_TOKENS);
    expect(contextWindowTokensFor('gpt-5.4-nano')).toBe(
      OPENAI_CONTEXT_WINDOW_TOKENS,
    );
    expect(contextWindowTokensFor('claude-opus-5')).toBe(
      VERTEX_CONTEXT_WINDOW_TOKENS,
    );
  });
});
