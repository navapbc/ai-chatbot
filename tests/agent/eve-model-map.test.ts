import { describe, it, expect } from 'vitest';
import { toGatewaySlug } from '@/lib/ai/eve/model-map';

describe('toGatewaySlug', () => {
  it('maps Claude picker ids to gateway slugs', () => {
    expect(toGatewaySlug('claude-opus-4-8')).toBe('anthropic/claude-opus-4.8');
    expect(toGatewaySlug('claude-opus-4-7')).toBe('anthropic/claude-opus-4.7');
    expect(toGatewaySlug('claude-sonnet-4-6')).toBe('anthropic/claude-sonnet-4.6');
    expect(toGatewaySlug('claude-haiku-4-5')).toBe('anthropic/claude-haiku-4.5');
  });
  it('maps the gpt-5.4 family to gateway slugs', () => {
    expect(toGatewaySlug('gpt-5.4')).toBe('openai/gpt-5.4');
    expect(toGatewaySlug('gpt-5.4-pro')).toBe('openai/gpt-5.4-pro');
    expect(toGatewaySlug('gpt-5.4-mini')).toBe('openai/gpt-5.4-mini');
    expect(toGatewaySlug('gpt-5.4-nano')).toBe('openai/gpt-5.4-nano');
  });
  it('returns undefined for unmapped / base / empty ids', () => {
    expect(toGatewaySlug('chat-model')).toBeUndefined();
    expect(toGatewaySlug('chat-model-reasoning')).toBeUndefined();
    expect(toGatewaySlug('')).toBeUndefined();
    expect(toGatewaySlug(undefined)).toBeUndefined();
    expect(toGatewaySlug(null)).toBeUndefined();
    expect(toGatewaySlug('something-unknown')).toBeUndefined();
  });
});
