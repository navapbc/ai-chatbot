import type { ModelMessage } from 'ai';

/**
 * Add an Anthropic ephemeral cache breakpoint on the last message in the
 * history. Combined with the static system-prompt breakpoint, this lets
 * every step after the first read the whole prefix (system + tools +
 * message history up to the last turn) from cache.
 *
 * The breakpoint "slides" naturally because it's computed per request on
 * the then-last message — the previous turn's messages become cached
 * prefix for the next request, and only the new turn is uncached.
 *
 * No-op if the model isn't Anthropic-family (Vertex routes other providers
 * too). The cacheControl field is ignored by non-Anthropic providers.
 */
export function withSlidingCacheBreakpoint(
  messages: ModelMessage[],
): ModelMessage[] {
  if (messages.length === 0) return messages;
  const lastIdx = messages.length - 1;
  const last = messages[lastIdx];
  const existing = (last as any).providerOptions ?? {};
  const withBp: ModelMessage = {
    ...last,
    providerOptions: {
      ...existing,
      anthropic: {
        ...(existing.anthropic ?? {}),
        cacheControl: { type: 'ephemeral' as const },
      },
    },
  };
  return [...messages.slice(0, lastIdx), withBp];
}
