// Maps the dev model-override picker ids (lib/ai/providers.ts customProvider +
// lib/ai/models.ts) to AI Gateway model slugs Eve routes through. Unmapped ids
// return undefined so the caller sends no model header and Eve uses its
// fallback (anthropic/claude-sonnet-4.6). Slugs are dot-versioned gateway ids
// (verify against the AI Gateway model catalog).
const MODEL_MAP: Record<string, string> = {
  'claude-opus-4-8': 'anthropic/claude-opus-4.8',
  'claude-opus-4-7': 'anthropic/claude-opus-4.7',
  'claude-sonnet-4-6': 'anthropic/claude-sonnet-4.6',
  'claude-haiku-4-5': 'anthropic/claude-haiku-4.5',
  'gpt-5.4': 'openai/gpt-5.4',
  'gpt-5.4-pro': 'openai/gpt-5.4-pro',
  'gpt-5.4-mini': 'openai/gpt-5.4-mini',
  'gpt-5.4-nano': 'openai/gpt-5.4-nano',
};

export function toGatewaySlug(
  modelOverrideId?: string | null,
): string | undefined {
  if (!modelOverrideId) return undefined;
  return MODEL_MAP[modelOverrideId];
}
