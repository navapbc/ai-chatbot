/**
 * Per-model pricing for eval cost estimation, in USD per 1M tokens.
 *
 * ⚠️ TODO(verify): these rates are best-effort estimates entered when the eval
 * matrix was set up. Confirm each against the provider's current pricing page
 * before trusting the `estimated_cost_usd` metric in Braintrust:
 *   - OpenAI:    https://openai.com/api/pricing/
 *   - Anthropic: https://www.anthropic.com/pricing
 *   - Google:    https://ai.google.dev/gemini-api/docs/pricing
 *
 * `cachedInput` is the discounted rate for cache-read (cachedInputTokens). When
 * omitted, cached tokens are billed at the full `input` rate.
 *
 * Keys are the EVAL_MODEL ids used by the CI matrix and getEvalModel().
 */
export interface ModelPrice {
  /** USD per 1M input (prompt) tokens. */
  input: number;
  /** USD per 1M output (completion) tokens. */
  output: number;
  /** USD per 1M cached-input (cache-read) tokens. Defaults to `input`. */
  cachedInput?: number;
}

export const MODEL_PRICING: Record<string, ModelPrice> = {
  // TODO(verify) — OpenAI
  "gpt-5.1": { input: 1.25, output: 10, cachedInput: 0.125 },
  "gpt-5-mini": { input: 0.25, output: 2, cachedInput: 0.025 },
  // TODO(verify) — Anthropic (historical Opus tier: $15 in / $75 out)
  "claude-opus-4-7": { input: 15, output: 75, cachedInput: 1.5 },
  "claude-opus-4-8": { input: 15, output: 75, cachedInput: 1.5 },
  // TODO(verify) — Opus 5 / Sonnet 5 list at first-party rates; Vertex bills
  // at Google partner rates, which may differ.
  "claude-opus-5": { input: 5, output: 25, cachedInput: 0.5 },
  "claude-sonnet-5": { input: 2, output: 10, cachedInput: 0.2 },
  // TODO(verify) — Google (Gemini 2.5 Pro was ~$1.25 in / $10 out under 200k)
  "gemini-3-pro": { input: 2, output: 12, cachedInput: 0.2 },
};

export interface CostResult {
  /** Estimated cost in USD, or null when the model has no pricing entry. */
  costUsd: number | null;
  /** True when MODEL_PRICING has an entry for the model. */
  pricingKnown: boolean;
}

export interface UsageTotals {
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
  cachedInputTokens: number;
}

/**
 * Estimate USD cost for a token usage total. Cached input tokens are billed at
 * the discounted `cachedInput` rate; the remaining input tokens at `input`.
 * Returns `costUsd: null` (not 0) for unpriced models so a missing price reads
 * as "unknown" rather than "free" in aggregates.
 */
export function computeCostUsd(
  modelId: string,
  usage: UsageTotals,
): CostResult {
  const price = MODEL_PRICING[modelId];
  if (!price) return { costUsd: null, pricingKnown: false };

  const cachedInput = Math.min(usage.cachedInputTokens, usage.inputTokens);
  const uncachedInput = usage.inputTokens - cachedInput;
  const cachedRate = price.cachedInput ?? price.input;

  const costUsd =
    (uncachedInput * price.input +
      cachedInput * cachedRate +
      usage.outputTokens * price.output) /
    1_000_000;

  return { costUsd, pricingKnown: true };
}
