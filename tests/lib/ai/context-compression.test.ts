import { describe, expect, test } from 'vitest';
import type { ModelMessage } from 'ai';
import { __test__ } from '@/lib/ai/context-compression';

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
