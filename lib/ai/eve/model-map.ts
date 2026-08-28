// Maps the dev model-override picker ids (lib/ai/providers.ts customProvider +
// lib/ai/models.ts) to the Vertex AI Anthropic model ids Eve calls directly.
// `agent/agent.ts` resolves the mapped id back into a `vertexAnthropic(...)`
// instance from its `step.started` model resolver, so the picker no longer
// depends on the Vercel AI Gateway. Unmapped ids return undefined so the caller
// sends no model header and Eve uses its fallback (claude-sonnet-4-6).
//
// The picker's `gpt-5.4*` entries are deliberately absent: Vertex does not serve
// OpenAI models, and with the gateway gone there is nothing left to route them
// through. Selecting one sends no header and lands on the fallback.
//
// The mapping is identity today because the picker ids already use Anthropic's
// native hyphenated version format, which is also Vertex's. It stays a map
// rather than a passthrough so the two can diverge, and so `isVertexModelId`
// has an allowlist to validate the untrusted `x-eve-model` header against.
const MODEL_MAP: Record<string, string> = {
  'claude-opus-5': 'claude-opus-5',
  'claude-opus-4-8': 'claude-opus-4-8',
  'claude-opus-4-7': 'claude-opus-4-7',
  'claude-sonnet-5': 'claude-sonnet-5',
  'claude-sonnet-4-6': 'claude-sonnet-4-6',
  'claude-haiku-4-5': 'claude-haiku-4-5',
};

const ALLOWED_MODEL_IDS = new Set(Object.values(MODEL_MAP));

/**
 * Context window Eve should assume for every model in `MODEL_MAP`, in tokens.
 *
 * Hand-maintained: a direct `vertexAnthropic(...)` instance reports its
 * provider as `googleVertex.anthropic.messages`, which does not match the AI
 * Gateway catalog's `vertex` provider slug, so Eve cannot look the window up
 * and refuses to compile without an explicit override. 200K is Claude's default
 * window on Vertex; the 1M window the gateway catalog advertises is tier-gated,
 * and guessing high would push the compaction trigger past the point where
 * Vertex hard-errors on context length. Raise this only once 1M is confirmed
 * enabled for the project/region in terraform/ENV_MAPPING.md.
 */
export const VERTEX_CONTEXT_WINDOW_TOKENS = 200_000;

/** Picker id -> Vertex model id, or undefined when the picker id is unmapped. */
export function toVertexModelId(
  modelOverrideId?: string | null,
): string | undefined {
  if (!modelOverrideId) return undefined;
  return MODEL_MAP[modelOverrideId];
}

/** Whether an untrusted value names a Vertex model the picker is allowed to select. */
export function isVertexModelId(value: unknown): value is string {
  return typeof value === 'string' && ALLOWED_MODEL_IDS.has(value);
}
