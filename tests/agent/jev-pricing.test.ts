import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { MODEL_PRICING, computeCostUsd } from '@/evals/pricing';

const ONE_MILLION = 1_000_000;

/** A usage total with no cache activity — Jev's shape. */
const plainUsage = (inputTokens: number, outputTokens: number) => ({
  inputTokens,
  outputTokens,
  totalTokens: inputTokens + outputTokens,
  cachedInputTokens: 0,
  cachedWriteTokens: 0,
});

describe('Jev pricing', () => {
  it('bills input tokens only — output is free', () => {
    // TypeSafe publishes $42/Btok = $0.042/Mtok on input, output free. An
    // output rate that drifts above 0 would silently overstate every Jev row.
    const { costUsd, pricingKnown } = computeCostUsd(
      'jev-1.13.0',
      plainUsage(ONE_MILLION, ONE_MILLION),
    );
    expect(pricingKnown).toBe(true);
    expect(costUsd).toBeCloseTo(0.042, 10);
  });

  it('costs nothing for an output-only usage total', () => {
    const { costUsd } = computeCostUsd('jev-1.13.0', plainUsage(0, 500_000));
    expect(costUsd).toBe(0);
  });

  it('is far cheaper per input token than the cheapest Claude row', () => {
    // Guards the decimal place: 0.42 or 4.2 would still look plausible in the
    // table but would put Jev on par with a frontier model.
    const jev = MODEL_PRICING['jev-1.13.0'];
    const haiku = MODEL_PRICING['claude-haiku-4-5'];
    expect(jev.input).toBeLessThan(haiku.input / 10);
  });
});

describe('current-generation Anthropic rows', () => {
  it('prices Opus 5.5 cache reads at the published $0.20, not 0.1x input', () => {
    // Opus 5.5 is the one row where cache read is NOT 0.1x input ($0.40).
    // Anyone "fixing" it to match the other rows doubles the cache cost.
    const opus55 = MODEL_PRICING['claude-opus-5-5'];
    expect(opus55.input).toBe(4);
    expect(opus55.output).toBe(20);
    expect(opus55.cachedInput).toBe(0.2);
    expect(opus55.cachedInput).not.toBe(opus55.input * 0.1);
  });

  it('prices Sonnet 5.5 at the published rates', () => {
    const sonnet55 = MODEL_PRICING['claude-sonnet-5-5'];
    expect(sonnet55.input).toBe(2);
    expect(sonnet55.output).toBe(10);
    expect(sonnet55.cachedInput).toBe(0.2);
  });

  // NOTE: logJevUsageAndCost's `::warning::` on an unpriced Jev model is not
  // unit-tested here — evals/helpers.ts reaches `server-only` via
  // lib/ai/tools/browser, so it cannot be imported in node mode (the same
  // constraint CLAUDE.md describes for lib/jev/*). The condition that triggers
  // the warning is what the test below pins.
  it('still reports unknown models as unpriced rather than free', () => {
    const { costUsd, pricingKnown } = computeCostUsd(
      'not-a-model',
      plainUsage(ONE_MILLION, 0),
    );
    expect(pricingKnown).toBe(false);
    expect(costUsd).toBeNull();
  });
});

describe('askJev usage reporting', () => {
  const systemOne = vi.fn();

  beforeEach(() => {
    vi.resetModules();
    systemOne.mockReset();
    vi.doMock('@typesafe-ai/sdk', () => ({
      TypeSafeClient: class {
        systemOne = systemOne;
      },
    }));
    vi.stubEnv('TYPESAFE_API_KEY', 'test-key');
  });

  afterEach(() => {
    vi.doUnmock('@typesafe-ai/sdk');
    vi.unstubAllEnvs();
  });

  it('surfaces the token usage TypeSafe reported', async () => {
    systemOne.mockResolvedValue({
      model: 'jev-1.13.0',
      answers: { verdict: { choice: 'yes' } },
      usage: { input_tokens: 296, output_tokens: 20 },
    });
    const { askJev } = await import('@/lib/jev/client');

    const result = await askJev({
      state: {},
      questions: {} as never,
      feature: 'gap-triage',
    });

    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.model).toBe('jev-1.13.0');
    expect(result.usage).toEqual({ inputTokens: 296, outputTokens: 20 });
  });

  it('reports no usage on failure, and still does not throw', async () => {
    // The never-throws contract is what lets callers treat Jev as an
    // enrichment. A failed call bills nothing, so it must not report a
    // zeroed usage that a caller could mistake for a real measurement.
    systemOne.mockRejectedValue(new Error('rate limited'));
    const { askJev } = await import('@/lib/jev/client');

    const result = await askJev({
      state: {},
      questions: {} as never,
      feature: 'gap-triage',
    });

    expect(result.ok).toBe(false);
    expect(result).not.toHaveProperty('usage');
  });
});
