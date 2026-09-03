// Maps the dev model-override picker ids (lib/ai/providers.ts customProvider +
// lib/ai/models.ts) to the model ids Eve calls directly — Claude via
// `vertexAnthropic(...)` on Vertex AI, GPT via `openai(...)` on OpenAI's own
// API (there is no OpenAI-on-Vertex path; it's a separate direct provider
// call, keyed off OPENAI_API_KEY). `agent/agent.ts` resolves the mapped id
// back into the right provider instance from its `step.started` model
// resolver, so the picker no longer depends on the Vercel AI Gateway.
// Unmapped ids return undefined so the caller sends no model header and Eve
// uses its fallback (claude-sonnet-4-6).
//
// The mapping is identity today because the picker ids already use each
// provider's own native id format. It stays a map rather than a passthrough
// so the two can diverge, and so `isVertexModelId` has an allowlist to
// validate the untrusted `x-eve-model` header against.
const MODEL_MAP: Record<string, string> = {
  'claude-opus-5': 'claude-opus-5',
  'claude-opus-4-8': 'claude-opus-4-8',
  'claude-opus-4-7': 'claude-opus-4-7',
  'claude-sonnet-5': 'claude-sonnet-5',
  'claude-sonnet-4-6': 'claude-sonnet-4-6',
  'claude-haiku-4-5': 'claude-haiku-4-5',
  'gpt-5.4': 'gpt-5.4',
  'gpt-5.4-pro': 'gpt-5.4-pro',
  'gpt-5.4-mini': 'gpt-5.4-mini',
  'gpt-5.4-nano': 'gpt-5.4-nano',
};

const ALLOWED_MODEL_IDS = new Set(Object.values(MODEL_MAP));

/**
 * Context window Eve should assume for every Claude model in `MODEL_MAP`, in
 * tokens.
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

/**
 * Context window for the GPT-5.4 family in `MODEL_MAP`, in tokens — 1.05M
 * total (≈922K input + up to 128K output) per OpenAI's docs, shared across
 * gpt-5.4/-pro/-mini/-nano (unconfirmed for mini/nano specifically; OpenAI's
 * size variants have historically matched the flagship's context window).
 * Needed for the same reason as `VERTEX_CONTEXT_WINDOW_TOKENS`: a direct
 * `openai(...)` instance isn't necessarily in the Gateway catalog Eve
 * otherwise looks the window up from.
 */
export const OPENAI_CONTEXT_WINDOW_TOKENS = 1_050_000;

const OPENAI_MODEL_IDS = new Set([
  'gpt-5.4',
  'gpt-5.4-pro',
  'gpt-5.4-mini',
  'gpt-5.4-nano',
]);

/** Whether a (validated) model id names a GPT model, not Claude. */
export function isOpenAIModelId(value: unknown): value is string {
  return typeof value === 'string' && OPENAI_MODEL_IDS.has(value);
}

/** The context window to declare for a given (validated) model id. */
export function contextWindowTokensFor(modelId: string): number {
  return isOpenAIModelId(modelId)
    ? OPENAI_CONTEXT_WINDOW_TOKENS
    : VERTEX_CONTEXT_WINDOW_TOKENS;
}

/** Picker id -> Vertex/OpenAI model id, or undefined when the picker id is unmapped. */
export function toVertexModelId(
  modelOverrideId?: string | null,
): string | undefined {
  if (!modelOverrideId) return undefined;
  return MODEL_MAP[modelOverrideId];
}

/** Whether an untrusted value names a model the picker is allowed to select. */
export function isVertexModelId(value: unknown): value is string {
  return typeof value === 'string' && ALLOWED_MODEL_IDS.has(value);
}
