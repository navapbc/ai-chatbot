import { describe, expect, test, vi } from 'vitest';
import type { ModelMessage } from 'ai';
import { __test__, createMessageCompressor } from '@/lib/ai/context-compression';

const { maxRecentToolResultTokens, estimateMessageTokens } = __test__;

function toolMsg(payload: unknown): ModelMessage {
  return {
    role: 'tool',
    content: [
      {
        type: 'tool-result',
        toolCallId: 'tc_test',
        toolName: 'browser',
        output: { type: 'json', value: payload },
      },
    ],
  } as ModelMessage;
}

function userMsg(text: string): ModelMessage {
  return { role: 'user', content: text } as ModelMessage;
}

function assistantMsg(text: string): ModelMessage {
  return { role: 'assistant', content: text } as ModelMessage;
}

describe('estimateMessageTokens', () => {
  test('returns ceil(content-chars / 4) for a tool message', () => {
    const m = toolMsg({ data: 'a'.repeat(400) });
    const result = estimateMessageTokens(m);
    // Payload is ~500 chars after JSON wrapping: { data: "aaa...400" } plus
    // tool-result envelope. Expect ~125 tokens; bounds catch off-by-4x errors
    // in either direction.
    expect(result).toBeGreaterThanOrEqual(120);
    expect(result).toBeLessThanOrEqual(200);
  });
});

describe('maxRecentToolResultTokens', () => {
  test('returns 0 when there are no tool-role messages', () => {
    expect(maxRecentToolResultTokens([userMsg('hi'), userMsg('there')])).toBe(0);
  });

  test('returns the max estimated size among the last 5 tool messages', () => {
    const big = toolMsg({ snapshot: 'x'.repeat(60_000 * 4) }); // ~60K tokens
    const small = toolMsg({ ok: true });
    const messages = [userMsg('start'), big, small, small, small, small];
    const result = maxRecentToolResultTokens(messages);
    expect(result).toBeGreaterThan(50_000);
  });

  test('window of 5 means an older big result falls out', () => {
    const big = toolMsg({ snapshot: 'x'.repeat(60_000 * 4) });
    const small = toolMsg({ ok: true });
    const messages = [big, small, small, small, small, small];
    const result = maxRecentToolResultTokens(messages);
    expect(result).toBeLessThan(1_000);
  });

  test('window counts tool-role messages only, not all messages', () => {
    const big = toolMsg({ snapshot: 'x'.repeat(60_000 * 4) });
    const small = toolMsg({ ok: true });
    // 6 tool messages total. If implementation correctly filters first then takes
    // the last 5 tool messages, `big` is among them (positions 2-6 of the tool
    // messages) and the max is large. If implementation incorrectly slices first
    // (last 5 of all 11 messages), `big` would fall outside the slice and the
    // max would be small.
    const messages = [
      small,
      userMsg('u1'),
      big,
      userMsg('u2'),
      small,
      userMsg('u3'),
      small,
      userMsg('u4'),
      small,
      userMsg('u5'),
      small,
    ];
    const result = maxRecentToolResultTokens(messages);
    expect(result).toBeGreaterThan(50_000);
  });
});

// 200K window; 75% threshold = 150K
const UNDER_THRESHOLD_TOKENS = 130_000;
const ABOVE_THRESHOLD_TOKENS = 160_000;

const hasApi =
  !!process.env.ANTHROPIC_API_KEY ||
  !!process.env.GOOGLE_VERTEX_PROJECT ||
  !!process.env.GOOGLE_APPLICATION_CREDENTIALS;

describe('createMessageCompressor — proactive projection', () => {
  test('does NOT compact when projection stays under threshold', async () => {
    const compressor = createMessageCompressor();
    const small = toolMsg({ ok: true });
    const filler: ModelMessage[] = Array.from({ length: 20 }, (_, i) =>
      i % 2 === 0 ? userMsg(`u${i}`) : assistantMsg(`a${i}`),
    );
    const messages = [...filler, small, small, small, small, small];

    // 130K + ~1K headroom = ~131K → below 150K threshold, no compaction.
    const result = await compressor(messages, UNDER_THRESHOLD_TOKENS);

    expect(result.compacted).toBe(false);
  });

  test('projection triggers compaction without a Haiku call', async () => {
    const stubSummary = {
      summary: 'test summary',
      workingMemory: null,
      recentMessages: [] as ModelMessage[],
      splitAt: 5,
    };
    const summarizeSpy = vi.fn().mockResolvedValue(stubSummary);
    const compressor = createMessageCompressor(summarizeSpy);
    const big = toolMsg({ snapshot: 'x'.repeat(60_000 * 4) }); // ~60K headroom
    const filler: ModelMessage[] = Array.from({ length: 20 }, (_, i) =>
      i % 2 === 0 ? userMsg(`u${i}`) : assistantMsg(`a${i}`),
    );
    const messages = [...filler, big];

    // 130K (65% — under reactive threshold) + 60K headroom = 190K projected (95%)
    // → projection trigger fires. With the old reactive logic this would NOT have
    // compacted.
    const result = await compressor(messages, UNDER_THRESHOLD_TOKENS);

    expect(summarizeSpy).toHaveBeenCalledTimes(1);
    expect(result.compacted).toBe(true);
    expect(result.summary).toBe('test summary');
  });

  test.skipIf(!hasApi)(
    'compacts when lastInputTokens + headroom crosses threshold',
    async () => {
      const compressor = createMessageCompressor();
      const big = toolMsg({ snapshot: 'x'.repeat(60_000 * 4) }); // ~60K tokens
      const filler: ModelMessage[] = Array.from({ length: 20 }, (_, i) =>
        i % 2 === 0 ? userMsg(`u${i}`) : assistantMsg(`a${i}`),
      );
      const messages = [...filler, big];

      // 130K + 60K headroom = 190K (95%) → compaction triggers even though
      // lastInputTokens alone (130K = 65%) was below the 75% threshold.
      const result = await compressor(messages, UNDER_THRESHOLD_TOKENS);

      expect(result.compacted).toBe(true);
      expect(result.messages.length).toBeLessThan(messages.length);
    },
    60_000,
  );

  test.skipIf(!hasApi)(
    'still compacts when lastInputTokens alone crosses threshold',
    async () => {
      const compressor = createMessageCompressor();
      const filler: ModelMessage[] = Array.from({ length: 20 }, (_, i) =>
        i % 2 === 0 ? userMsg(`u${i}`) : assistantMsg(`a${i}`),
      );

      const result = await compressor(filler, ABOVE_THRESHOLD_TOKENS);

      expect(result.compacted).toBe(true);
    },
    60_000,
  );
});
