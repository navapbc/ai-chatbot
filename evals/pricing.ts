/**
 * Per-model pricing for eval cost estimation, in USD per 1M tokens.
 *
 * Verified 2026-09-15 against OpenAI (https://openai.com/api/pricing/),
 * Anthropic first-party rates, and Google (https://ai.google.dev/gemini-api/docs/pricing).
 * Re-check before trusting `estimated_cost_usd` for anything long after that date —
 * these are list prices and providers do change them.
 *
 * Re-verified 2026-10-05: every Anthropic row and the Google row confirmed
 * unchanged; Opus 5.5 / Sonnet 5.5 and the Jev row added. The two OpenAI rows
 * were NOT re-checked that day — openai.com/api/pricing returned HTTP 403 — so
 * they still rest on the 2026-09-15 check.
 *
 * `cachedInput` is the discounted rate for cache-read (cachedInputTokens).
 * `cacheWrite` is the premium rate for cache-write (cachedWriteTokens) — the
 * one-time cost of populating a new cache entry. Anthropic charges ~1.25x the
 * `input` rate for a standard (5-minute TTL) cache write; OpenAI and Google do
 * not charge a write premium (a cache write there just costs the normal
 * `input` rate), so both `cachedInput` and `cacheWrite` default to `input`
 * when omitted. This repo's evals don't set an explicit cache TTL, so the
 * 5-minute rate is assumed for every Anthropic cache write — if a 1-hour TTL
 * is ever configured (billed at ~2x `input`), that write cost would need its
 * own rate, which this table doesn't distinguish.
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
  /** USD per 1M cache-write tokens. Defaults to `input` (no write premium). */
  cacheWrite?: number;
}

export const MODEL_PRICING: Record<string, ModelPrice> = {
  // OpenAI — confirmed against openai.com/api/pricing.
  "gpt-5.1": { input: 1.25, output: 10, cachedInput: 0.125 },
  "gpt-5-mini": { input: 0.25, output: 2, cachedInput: 0.025 },
  // Anthropic — first-party API rates (these are the rates actually billed:
  // getEvalModel() resolves claude-* ids straight to @ai-sdk/anthropic, not
  // Vertex). Opus 4.7 and 4.8 previously carried the *Opus 4.1-era* rate
  // ($15 in / $75 out) here by mistake — Anthropic cut Opus to $5/$25 with the
  // 4.7 release, a 67% price drop the table was never updated for. That bug
  // overstated estimated_cost_usd by 3x for every CI matrix run on these two
  // models (evals.yml runs both) until this fix; historical Braintrust traces
  // logged before this change still carry the inflated figure.
  "claude-opus-4-7": { input: 5, output: 25, cachedInput: 0.5, cacheWrite: 6.25 },
  "claude-opus-4-8": { input: 5, output: 25, cachedInput: 0.5, cacheWrite: 6.25 },
  "claude-opus-5": { input: 5, output: 25, cachedInput: 0.5, cacheWrite: 6.25 },
  "claude-sonnet-5": { input: 2, output: 10, cachedInput: 0.2, cacheWrite: 2.5 },
  // Current generation (Opus 5.5 / Sonnet 5.5). Every figure below — including
  // both cacheWrite rates — is read directly from Anthropic's published
  // pricing table, not derived from a multiplier. Verified 2026-10-05.
  //
  // NOTE: Opus 5.5's cache read is a published $0.20 — that is 0.05x its $4
  // input rate, NOT the 0.1x that every other Anthropic row here follows
  // (Anthropic documents the 0.05x exception explicitly). Do not "correct" it
  // to 0.40.
  "claude-opus-5-5": { input: 4, output: 20, cachedInput: 0.2, cacheWrite: 5 },
  "claude-sonnet-5-5": { input: 2, output: 10, cachedInput: 0.1, cacheWrite: 2.5 },
  // Haiku 5.5 is tiered by prompt size: the row below is the ≤100k-token tier
  // ($0.10 in / $0.50 out, 5m write $0.125, read $0.01). Prompts over 100k are
  // re-rated to $0.50 / $2.50 (write $0.625, read $0.05) — not modeled here, so
  // cost is under-counted on any eval task with a very large prompt. Read from
  // platform.claude.com/docs/en/about-claude/pricing on 2026-10-08, the same
  // day Sonnet 5.5's cache read was corrected from $0.20 to the published $0.10.
  "claude-haiku-5-5": { input: 0.1, output: 0.5, cachedInput: 0.01, cacheWrite: 0.125 },
  // Not in the CI matrix or DEFAULT_EVAL_MODEL today, but selectable via
  // EVAL_MODEL=claude-haiku-4-5 (getEvalModel() accepts any claude-* id), and
  // it's the legacy route's prepareStepModel (lib/ai/providers.ts) — though
  // that path runs on Vertex, not this first-party rate.
  "claude-haiku-4-5": { input: 1, output: 5, cachedInput: 0.1, cacheWrite: 1.25 },
  // Google — ai.google.dev lists this model as "Gemini 3.1 Pro Preview" now
  // (no separate "Gemini 3 Pro" listing); rate below is its ≤200k-token-prompt
  // tier. Prompts over 200k input tokens are re-rated to $4/$18 (cache read
  // $0.40) for the whole request — not modeled here, so cost is under-counted
  // on any eval task with a very large prompt.
  "gemini-3-pro": { input: 2, output: 12, cachedInput: 0.2 },
  // TypeSafe (Jev) — not an AI-SDK model and never an EVAL_MODEL; keyed by the
  // model id `askJev` reports back at runtime (lib/jev/client.ts). TypeSafe
  // bills INPUT TOKENS ONLY at $42/Btok = $0.042/Mtok; output tokens are free,
  // so `output: 0` here is the real published rate, not a missing value.
  // Jev exposes no cache tier (its Usage has only input_tokens/output_tokens),
  // so the cache buckets are always 0 and the cache rates never apply.
  "jev-1.13.0": { input: 0.042, output: 0 },
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
  /** Cache-write (cache-creation) tokens — the AI SDK's `cacheWriteTokens`. */
  cachedWriteTokens: number;
}

/**
 * Estimate USD cost for a token usage total. `inputTokens` is the AI SDK's
 * total input count, which already includes cache-read and cache-write tokens
 * (it splits three ways: `noCacheTokens` + `cacheReadTokens` +
 * `cacheWriteTokens`) — so the plain-`input`-rate bucket here is whatever is
 * left after subtracting both cache buckets, not just the cache-read one.
 * Returns `costUsd: null` (not 0) for unpriced models so a missing price reads
 * as "unknown" rather than "free" in aggregates.
 */
export function computeCostUsd(
  modelId: string,
  usage: UsageTotals,
): CostResult {
  const price = MODEL_PRICING[modelId];
  if (!price) return { costUsd: null, pricingKnown: false };

  const cachedRead = Math.min(usage.cachedInputTokens, usage.inputTokens);
  const cachedWrite = Math.min(
    usage.cachedWriteTokens,
    usage.inputTokens - cachedRead,
  );
  const noCacheInput = usage.inputTokens - cachedRead - cachedWrite;
  const cachedReadRate = price.cachedInput ?? price.input;
  const cachedWriteRate = price.cacheWrite ?? price.input;

  const costUsd =
    (noCacheInput * price.input +
      cachedRead * cachedReadRate +
      cachedWrite * cachedWriteRate +
      usage.outputTokens * price.output) /
    1_000_000;

  return { costUsd, pricingKnown: true };
}
