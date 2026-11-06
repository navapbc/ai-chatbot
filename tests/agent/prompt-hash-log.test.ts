import { MockLanguageModelV4 } from 'ai/test';
import { describe, expect, it } from 'vitest';
import {
  describePrompt,
  firstDifference,
  promptHashLoggingEnabled,
  withPromptHashLogging,
} from '@/lib/ai/eve/prompt-hash-log';

const SECRET = 'Maria Gonzalez SSN 123-45-6789 DOB 1990-02-03';

type Prompt = Parameters<typeof describePrompt>[0];
const user = (text: string) => ({
  role: 'user' as const,
  content: [{ type: 'text' as const, text }],
});
const assistant = (text: string) => ({
  role: 'assistant' as const,
  content: [{ type: 'text' as const, text }],
});
const system = (content: string) => ({ role: 'system' as const, content });

const streamOf = (usage: unknown) =>
  new ReadableStream({
    start(controller) {
      controller.enqueue({ type: 'stream-start', warnings: [] });
      controller.enqueue({
        type: 'finish',
        finishReason: { unified: 'stop', raw: 'stop' },
        usage,
      });
      controller.close();
    },
  });

const usage = (total: number, read: number, write: number) => ({
  inputTokens: { total, noCache: 0, cacheRead: read, cacheWrite: write },
  outputTokens: { total: 5, text: 5, reasoning: 0 },
});

const setup = (env: Record<string, string | undefined>) => {
  const lines: Record<string, unknown>[] = [];
  const calls: unknown[] = [];
  const base = new MockLanguageModelV4({
    provider: 'googleVertex.anthropic.messages',
    modelId: 'claude-test',
    doStream: async (options) => {
      calls.push(options);
      return { stream: streamOf(usage(900, 800, 100)) } as never;
    },
  });
  const model = withPromptHashLogging(base, {
    env,
    emit: (line) => lines.push(JSON.parse(line)),
  });
  return { base, model, lines, calls };
};

const run = async (model: MockLanguageModelV4, prompt: Prompt) => {
  const { stream } = await model.doStream({ prompt } as never);
  const reader = stream.getReader();
  for (;;) {
    const { done } = await reader.read();
    if (done) break;
  }
};

describe('withPromptHashLogging', () => {
  it('returns the model untouched unless the flag is exactly 1', () => {
    for (const env of [
      {},
      { EVE_LOG_PROMPT_HASHES: '0' },
      { EVE_LOG_PROMPT_HASHES: 'true' },
    ]) {
      const { base, model } = setup(env);
      expect(model).toBe(base);
    }
    expect(promptHashLoggingEnabled({ EVE_LOG_PROMPT_HASHES: '1' })).toBe(true);
  });

  it('keeps the provider name, which Eve reads to place cache markers', () => {
    const { model } = setup({ EVE_LOG_PROMPT_HASHES: '1' });
    expect(model.provider).toBe('googleVertex.anthropic.messages');
    expect(model.modelId).toBe('claude-test');
  });

  it('logs a call line and a usage line, and passes the call through', async () => {
    const { model, lines, calls } = setup({ EVE_LOG_PROMPT_HASHES: '1' });
    const prompt = [system('rules'), user('hello'), assistant('hi')];
    await run(model as never, prompt);
    expect(calls).toHaveLength(1);
    expect((calls[0] as { prompt: unknown }).prompt).toEqual(prompt);
    expect(lines.map((l) => l.kind)).toEqual(['call', 'usage']);
    expect(lines[0]).toMatchObject({
      event: 'eve_prompt_hash',
      n: 1,
      messageCount: 3,
      firstDiff: 'first',
    });
    expect(lines[1]).toMatchObject({
      n: 1,
      input: 900,
      cacheRead: 800,
      cacheWrite: 100,
    });
  });

  it('never logs message content', async () => {
    const { model, lines } = setup({ EVE_LOG_PROMPT_HASHES: '1' });
    await run(model as never, [
      system(SECRET),
      user(SECRET),
      assistant(SECRET),
    ]);
    const text = JSON.stringify(lines);
    expect(text).not.toContain('Maria');
    expect(text).not.toContain('123-45-6789');
    expect(text).not.toContain('1990-02-03');
  });

  it('says append when only messages were added and names a rewritten message', async () => {
    const { model, lines } = setup({ EVE_LOG_PROMPT_HASHES: '1' });
    const a = [system('s'), user('task'), assistant('one')];
    await run(model as never, a);
    await run(model as never, [...a, user('result 1')]);
    await run(model as never, [
      system('s'),
      user('task'),
      assistant('CHANGED'),
      user('result 1'),
      assistant('two'),
    ]);
    const calls = lines.filter((l) => l.kind === 'call');
    expect(calls.map((l) => l.firstDiff)).toEqual(['first', 'append', 2]);
  });

  it('reports shrunk and tracks sessions separately', async () => {
    const { model, lines } = setup({ EVE_LOG_PROMPT_HASHES: '1' });
    await run(model as never, [
      system('s'),
      user('task A'),
      assistant('1'),
      user('r'),
    ]);
    await run(model as never, [system('s'), user('task B'), assistant('1')]);
    await run(model as never, [system('s'), user('task A'), assistant('1')]);
    const calls = lines.filter((l) => l.kind === 'call');
    expect(calls.map((l) => l.firstDiff)).toEqual(['first', 'first', 'shrunk']);
    expect(calls[0]?.session).not.toBe(calls[1]?.session);
  });

  it('flags a changed tool list and system block through their hashes', async () => {
    const { model, lines } = setup({ EVE_LOG_PROMPT_HASHES: '1' });
    const base = { prompt: [system('s'), user('t')] };
    await model.doStream({
      ...base,
      tools: [{ type: 'function', name: 'a', inputSchema: {} }],
    } as never);
    await model.doStream({
      ...base,
      tools: [{ type: 'function', name: 'b', inputSchema: {} }],
    } as never);
    const calls = lines.filter((l) => l.kind === 'call');
    expect(calls[0]?.toolsHash).not.toBe(calls[1]?.toolsHash);
    expect(calls[0]?.systemHash).toBe(calls[1]?.systemHash);
  });
});

describe('firstDifference', () => {
  const m = (...hs: string[]) =>
    hs.map((h, i) => ({ i, role: 'user', h, len: 1 }));
  it('compares hashes in order', () => {
    expect(firstDifference(undefined, m('a'))).toBe('first');
    expect(firstDifference(m('a', 'b'), m('a', 'b', 'c'))).toBe('append');
    expect(firstDifference(m('a', 'b'), m('a', 'x', 'c'))).toBe(1);
    expect(firstDifference(m('a', 'b', 'c'), m('a', 'b'))).toBe('shrunk');
  });
});
