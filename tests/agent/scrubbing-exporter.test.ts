import { SpanKind, SpanStatusCode, trace } from '@opentelemetry/api';
import { registerTelemetry, generateText, stepCountIs, tool } from 'ai';
import { MockLanguageModelV4 } from 'ai/test';
import { OpenTelemetry } from '@ai-sdk/otel';
import {
  BasicTracerProvider,
  InMemorySpanExporter,
  SimpleSpanProcessor,
  type ReadableSpan,
  type SpanExporter,
} from '@opentelemetry/sdk-trace-base';
import { describe, expect, it, vi } from 'vitest';
import { z } from 'zod';
import {
  SAFE_SPAN_ATTRIBUTES,
  ScrubbingSpanExporter,
  scrubSpan,
} from '@/lib/observability/scrubbing-exporter';

function makeSpan(overrides: Partial<Record<string, unknown>> = {}) {
  return {
    name: 'chat model-x',
    kind: SpanKind.CLIENT,
    spanContext: () => ({
      traceId: 'a'.repeat(32),
      spanId: 'b'.repeat(16),
      traceFlags: 1,
    }),
    parentSpanContext: {
      traceId: 'a'.repeat(32),
      spanId: 'c'.repeat(16),
      traceFlags: 1,
    },
    startTime: [10, 5],
    endTime: [11, 5],
    duration: [1, 0],
    ended: true,
    status: { code: SpanStatusCode.ERROR, message: 'failed on 123-45-6789' },
    attributes: {
      'gen_ai.request.model': 'claude-x',
      'gen_ai.usage.input_tokens': 12,
      'gen_ai.input.messages': 'SENTINEL-PROMPT-123',
      'gen_ai.tool.call.arguments': '{"ssn":"SENTINEL-ARG"}',
      'gen_ai.tool.call.result': 'SENTINEL-RESULT',
      'ai.prompt': 'SENTINEL-PROMPT-123',
      'ai.toolCall.args': 'SENTINEL-ARG',
      'ai.toolCall.result': 'SENTINEL-RESULT',
      'ai.prompt.messages': 'SENTINEL-PROMPT-123',
      'ai.toolCall.name': 'browser',
      'some.unknown.key': 'ok',
    },
    links: [
      {
        context: {
          traceId: 'd'.repeat(32),
          spanId: 'e'.repeat(16),
          traceFlags: 1,
        },
        attributes: { secret: 'SENTINEL-LINK' },
      },
    ],
    events: [
      {
        name: 'exception',
        time: [10, 6],
        attributes: {
          'exception.message': 'SENTINEL-EXC',
          'exception.stacktrace': 'SENTINEL-STACK',
        },
      },
    ],
    resource: {
      attributes: {
        'service.name': 'labs-asp',
        'process.command_args': 'SENTINEL-ARGV',
        'host.name': 'SENTINEL-HOST',
      },
      asyncAttributesPending: false,
    },
    instrumentationScope: { name: 'gen_ai', version: '1' },
    droppedAttributesCount: 0,
    droppedEventsCount: 0,
    droppedLinksCount: 0,
    ...overrides,
  } as unknown as ReadableSpan;
}

const allText = (span: ReadableSpan) => JSON.stringify(span);

describe('scrubSpan', () => {
  it('keeps only allowlisted attributes', () => {
    const out = scrubSpan(makeSpan());
    expect(out.attributes).toEqual({
      'gen_ai.request.model': 'claude-x',
      'gen_ai.usage.input_tokens': 12,
      'ai.toolCall.name': 'browser',
    });
  });

  it('drops ai.prompt, ai.toolCall.args, ai.toolCall.result and message attributes', () => {
    const out = scrubSpan(makeSpan());
    for (const key of [
      'ai.prompt',
      'ai.toolCall.args',
      'ai.toolCall.result',
      'ai.prompt.messages',
      'gen_ai.input.messages',
      'gen_ai.tool.call.arguments',
      'gen_ai.tool.call.result',
    ]) {
      expect(out.attributes).not.toHaveProperty(key);
    }
    expect(allText(out)).not.toContain('SENTINEL');
  });

  it('drops an attribute whose name is not on the list even when its value looks harmless', () => {
    const out = scrubSpan(makeSpan());
    expect(out.attributes).not.toHaveProperty('some.unknown.key');
  });

  it('the original span object is not mutated', () => {
    const span = makeSpan();
    const before = JSON.stringify(span);
    const out = scrubSpan(span);
    expect(out).not.toBe(span);
    expect(JSON.stringify(span)).toBe(before);
    expect(span.attributes['ai.prompt']).toBe('SENTINEL-PROMPT-123');
  });

  it('keeps identity, timing, kind, parent and scope', () => {
    const span = makeSpan();
    const out = scrubSpan(span);
    expect(out.spanContext()).toEqual(span.spanContext());
    expect(out.parentSpanContext).toEqual(span.parentSpanContext);
    expect(out.startTime).toEqual([10, 5]);
    expect(out.endTime).toEqual([11, 5]);
    expect(out.duration).toEqual([1, 0]);
    expect(out.kind).toBe(SpanKind.CLIENT);
    expect(out.instrumentationScope.name).toBe('gen_ai');
    expect(out.ended).toBe(true);
    expect(out.name).toBe('chat model-x');
  });

  it('drops the status message and keeps the code', () => {
    const out = scrubSpan(makeSpan());
    expect(out.status).toEqual({ code: SpanStatusCode.ERROR });
  });

  it('drops all span events, including exception events', () => {
    const out = scrubSpan(makeSpan());
    expect(out.events).toEqual([]);
  });

  it('strips link attributes but keeps the link context', () => {
    const out = scrubSpan(makeSpan());
    expect(out.links).toHaveLength(1);
    expect(out.links[0].attributes).toEqual({});
    expect(out.links[0].context.spanId).toBe('e'.repeat(16));
  });

  it('filters resource attributes to service identity', () => {
    const out = scrubSpan(makeSpan());
    expect(out.resource.attributes).toEqual({ 'service.name': 'labs-asp' });
  });

  it.each([
    ['ssn text 123-45-6789 here'],
    ['look 4111111111111111'],
    ['has\nnewline'],
    ['jane@example.com'],
    ['a'.repeat(130)],
    [`token ${'x'.repeat(41)}`],
    [''],
  ])('replaces a value-shaped span name (%#)', (name) => {
    expect(scrubSpan(makeSpan({ name })).name).toBe('scrubbed');
  });

  it.each([
    ['agent-browser snapshot'],
    ['execute_tool browser'],
    ['invoke_agent form_review'],
    ['step 2'],
    ['agent.turn'],
    ['jev gap-triage batch'],
  ])('keeps a conventional span name: %s', (name) => {
    expect(scrubSpan(makeSpan({ name })).name).toBe(name);
  });

  it.each([['page said: 123 Main St'], ['x'.repeat(65)], ['']])(
    'drops an error.type that is not a type token (%#)',
    (value) => {
      const out = scrubSpan(makeSpan({ attributes: { 'error.type': value } }));
      expect(out.attributes).toEqual({});
    },
  );

  it('keeps a type-token error.type', () => {
    const out = scrubSpan(
      makeSpan({ attributes: { 'error.type': 'jev_call_failed' } }),
    );
    expect(out.attributes).toEqual({ 'error.type': 'jev_call_failed' });
  });

  it.each([['unknown thing'], ['Jane Doe form'], ['execute_tool has space!x']])(
    'replaces a span name outside the operation prefixes (%s)',
    (name) => {
      expect(scrubSpan(makeSpan({ name })).name).toBe('scrubbed');
    },
  );

  it('replaces an execute_tool name whose tool part is not a tool token', () => {
    expect(scrubSpan(makeSpan({ name: 'execute_tool Jane Doe' })).name).toBe(
      'scrubbed',
    );
    expect(scrubSpan(makeSpan({ name: 'execute_tool browser' })).name).toBe(
      'execute_tool browser',
    );
  });

  it('does not reuse the span allowlist for resource attributes', () => {
    const out = scrubSpan(
      makeSpan({
        resource: {
          attributes: {
            'service.version': '1',
            'gen_ai.request.model': 'm',
            'agent.name': 'a',
          },
          asyncAttributesPending: false,
        },
      }),
    );
    expect(out.resource.attributes).toEqual({ 'service.version': '1' });
  });

  it('drops an allowlisted attribute whose value is an overlong string', () => {
    const out = scrubSpan(
      makeSpan({ attributes: { 'ai.toolCall.name': 'x'.repeat(500) } }),
    );
    expect(out.attributes).toEqual({});
  });

  it('drops an allowlisted attribute whose value is an object', () => {
    const out = scrubSpan(
      makeSpan({ attributes: { 'ai.toolCall.name': { a: 'SENTINEL' } } }),
    );
    expect(out.attributes).toEqual({});
  });

  it('allowlist holds no wildcard entries and no content-bearing keys', () => {
    for (const key of SAFE_SPAN_ATTRIBUTES) {
      expect(key).not.toMatch(/[*?]/);
      expect(key).not.toMatch(
        /prompt|message|args|arguments|result|text(?!Tokens)|input\.messages|output\.messages|instructions|definitions|exception|url|header/i,
      );
    }
  });
});

describe('ScrubbingSpanExporter', () => {
  function inner() {
    const exported: ReadableSpan[][] = [];
    const exporter = {
      export: vi.fn(
        (spans: ReadableSpan[], cb: (r: { code: number }) => void) => {
          exported.push(spans);
          cb({ code: 0 });
        },
      ),
      shutdown: vi.fn(async () => undefined),
      forceFlush: vi.fn(async () => undefined),
    };
    return { exported, exporter: exporter as unknown as SpanExporter };
  }

  it('passes spans through untouched when disabled', () => {
    const { exported, exporter } = inner();
    const span = makeSpan();
    new ScrubbingSpanExporter(exporter, { enabled: () => false }).export(
      [span],
      () => undefined,
    );
    expect(exported[0][0]).toBe(span);
  });

  it('scrubs every span when enabled, evaluated at export time', () => {
    const { exported, exporter } = inner();
    let on = false;
    const sut = new ScrubbingSpanExporter(exporter, { enabled: () => on });
    sut.export([makeSpan()], () => undefined);
    on = true;
    sut.export([makeSpan(), makeSpan()], () => undefined);
    expect(exported[0][0].attributes['ai.prompt']).toBe('SENTINEL-PROMPT-123');
    expect(exported[1]).toHaveLength(2);
    for (const s of exported[1]) expect(allText(s)).not.toContain('SENTINEL');
  });

  it('a span whose scrub throws is not exported', () => {
    const { exported, exporter } = inner();
    const bad = makeSpan();
    Object.defineProperty(bad, 'attributes', {
      get() {
        throw new Error('boom SENTINEL-BOOM');
      },
    });
    const sut = new ScrubbingSpanExporter(exporter, { enabled: () => true });
    sut.export([makeSpan(), bad, makeSpan()], () => undefined);
    expect(exported[0]).toHaveLength(2);
    expect(allText(exported[0][0])).not.toContain('SENTINEL');
  });

  it('treats a throwing enabled() as enabled', () => {
    const { exported, exporter } = inner();
    new ScrubbingSpanExporter(exporter, {
      enabled: () => {
        throw new Error('no');
      },
    }).export([makeSpan()], () => undefined);
    expect(exported[0][0].attributes).not.toHaveProperty('ai.prompt');
  });

  it('passes the result callback through', () => {
    const { exporter } = inner();
    const cb = vi.fn();
    new ScrubbingSpanExporter(exporter, { enabled: () => true }).export(
      [makeSpan()],
      cb,
    );
    expect(cb).toHaveBeenCalledWith({ code: 0 });
  });

  it('delegates shutdown and forceFlush', async () => {
    const { exporter } = inner();
    const sut = new ScrubbingSpanExporter(exporter, { enabled: () => true });
    await sut.shutdown();
    await sut.forceFlush();
    expect(exporter.shutdown).toHaveBeenCalledOnce();
    expect(exporter.forceFlush).toHaveBeenCalledOnce();
  });

  it('never logs span content', () => {
    const { exporter } = inner();
    const spies = (['log', 'warn', 'error', 'info'] as const).map((m) =>
      vi.spyOn(console, m).mockImplementation(() => undefined),
    );
    const bad = makeSpan();
    Object.defineProperty(bad, 'attributes', {
      get() {
        throw new Error('SENTINEL-BOOM');
      },
    });
    new ScrubbingSpanExporter(exporter, { enabled: () => true }).export(
      [bad],
      () => undefined,
    );
    for (const s of spies) {
      expect(JSON.stringify(s.mock.calls)).not.toContain('SENTINEL');
      s.mockRestore();
    }
  });
});

describe('real AI SDK spans', () => {
  it('scrubbed spans of a tool-calling generateText carry no prompt, args or results', async () => {
    const memory = new InMemorySpanExporter();
    const provider = new BasicTracerProvider({
      spanProcessors: [new SimpleSpanProcessor(memory)],
    });
    trace.setGlobalTracerProvider(provider);
    registerTelemetry(new OpenTelemetry({ runtimeContext: true }));

    let calls = 0;
    const usage = { inputTokens: { total: 3 }, outputTokens: { total: 4 } };
    const model = new MockLanguageModelV4({
      doGenerate: async () =>
        (calls++ === 0
          ? {
              content: [
                {
                  type: 'tool-call',
                  toolCallId: 'c1',
                  toolName: 'echo',
                  input: JSON.stringify({ v: 'SENTINEL-ARG-123' }),
                },
              ],
              finishReason: { unified: 'tool-calls', raw: 'x' },
              usage,
              warnings: [],
            }
          : {
              content: [{ type: 'text', text: 'SENTINEL-RESPONSE-123' }],
              finishReason: { unified: 'stop', raw: 'x' },
              usage,
              warnings: [],
            }) as never,
    });
    await generateText({
      model,
      system: 'SENTINEL-SYSTEM-123',
      prompt: 'SENTINEL-PROMPT-123',
      stopWhen: stepCountIs(3),
      runtimeContext: { note: 'SENTINEL-CTX-123' } as never,
      tools: {
        echo: tool({
          description: 'SENTINEL-TOOLDESC-123',
          inputSchema: z.object({ v: z.string() }),
          execute: async ({ v }) => `SENTINEL-RESULT-${v}`,
        }),
      },
    });

    const raw = memory.getFinishedSpans();
    expect(raw.length).toBeGreaterThan(3);
    // Sanity: the unscrubbed spans do carry the sentinels.
    expect(JSON.stringify(raw.map((s) => s.attributes))).toContain(
      'SENTINEL-PROMPT-123',
    );

    const out: ReadableSpan[] = [];
    const capture = {
      export: (s: ReadableSpan[], cb: (r: { code: number }) => void) => {
        out.push(...s);
        cb({ code: 0 });
      },
      shutdown: async () => undefined,
      forceFlush: async () => undefined,
    } as unknown as SpanExporter;
    new ScrubbingSpanExporter(capture, { enabled: () => true }).export(
      raw,
      () => undefined,
    );

    expect(out).toHaveLength(raw.length);
    expect(JSON.stringify(out)).not.toContain('SENTINEL');
    const toolSpan = out.find((s) => s.name.startsWith('execute_tool'));
    expect(toolSpan?.attributes['gen_ai.tool.name']).toBe('echo');
    const chat = out.find((s) => s.name.startsWith('chat'));
    expect(chat?.attributes['gen_ai.usage.input_tokens']).toBe(3);
  });
});
