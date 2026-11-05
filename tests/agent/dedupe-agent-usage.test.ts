import { SpanKind } from '@opentelemetry/api';
import {
  BasicTracerProvider,
  InMemorySpanExporter,
  type ReadableSpan,
  SimpleSpanProcessor,
  type SpanExporter,
} from '@opentelemetry/sdk-trace-base';
import { describe, expect, it, vi } from 'vitest';
import {
  DedupeAgentUsageExporter,
  USAGE_ATTRIBUTES,
  withoutDuplicateUsage,
} from '@/lib/observability/dedupe-agent-usage';

const ROOT = 'ai-chatbot';

const USAGE = {
  'gen_ai.usage.input_tokens': 121775,
  'gen_ai.usage.output_tokens': 807,
  'gen_ai.usage.cache_read.input_tokens': 116541,
  'gen_ai.usage.cache_creation.input_tokens': 5232,
};

/** Real SDK spans: their getters read private fields, which a copy must survive. */
function finishedSpans(
  specs: { name: string; attributes: Record<string, string | number> }[],
): ReadableSpan[] {
  const exporter = new InMemorySpanExporter();
  const provider = new BasicTracerProvider({
    spanProcessors: [new SimpleSpanProcessor(exporter)],
  });
  const tracer = provider.getTracer('test');
  for (const spec of specs) {
    const span = tracer.startSpan(spec.name, {
      kind: SpanKind.CLIENT,
      attributes: spec.attributes,
    });
    span.end();
  }
  return exporter.getFinishedSpans();
}

const rootInvoke = {
  name: 'invoke_agent claude-opus-5',
  attributes: {
    'gen_ai.operation.name': 'invoke_agent',
    'gen_ai.agent.name': ROOT,
    'gen_ai.request.model': 'claude-opus-5',
    ...USAGE,
  },
};
const chat = {
  name: 'chat claude-opus-5',
  attributes: {
    'gen_ai.operation.name': 'chat',
    'gen_ai.request.model': 'claude-opus-5',
    ...USAGE,
  },
};
const subagentInvoke = {
  name: 'invoke_agent claude-opus-5',
  attributes: {
    'gen_ai.operation.name': 'invoke_agent',
    'gen_ai.agent.name': 'form_review',
    'gen_ai.request.model': 'claude-opus-5',
    ...USAGE,
  },
};

describe('withoutDuplicateUsage', () => {
  it("drops the four usage counts from the root agent's invoke_agent span", () => {
    const [span] = finishedSpans([rootInvoke]);
    const out = withoutDuplicateUsage(span as ReadableSpan, ROOT);
    for (const key of USAGE_ATTRIBUTES)
      expect(out.attributes).not.toHaveProperty(key);
    expect(out.attributes['gen_ai.request.model']).toBe('claude-opus-5');
    expect(out.attributes['gen_ai.agent.name']).toBe(ROOT);
  });

  it('keeps identity, timing and scope readable through the copy', () => {
    const [span] = finishedSpans([rootInvoke]);
    const out = withoutDuplicateUsage(span as ReadableSpan, ROOT);
    expect(out.name).toBe('invoke_agent claude-opus-5');
    expect(out.spanContext().spanId).toBe(
      (span as ReadableSpan).spanContext().spanId,
    );
    expect(out.startTime).toEqual((span as ReadableSpan).startTime);
    expect(out.duration).toEqual((span as ReadableSpan).duration);
    expect(out.instrumentationScope.name).toBe('test');
    expect(out.ended).toBe(true);
  });

  it('never modifies the original span', () => {
    const [span] = finishedSpans([rootInvoke]);
    withoutDuplicateUsage(span as ReadableSpan, ROOT);
    expect((span as ReadableSpan).attributes['gen_ai.usage.input_tokens']).toBe(
      121775,
    );
  });

  it("leaves chat spans and a subagent's invoke_agent span alone", () => {
    const [c, sub] = finishedSpans([chat, subagentInvoke]);
    expect(withoutDuplicateUsage(c as ReadableSpan, ROOT)).toBe(c);
    expect(withoutDuplicateUsage(sub as ReadableSpan, ROOT)).toBe(sub);
  });

  it('leaves an invoke_agent span with no agent name alone', () => {
    const [span] = finishedSpans([
      {
        name: 'invoke_agent x',
        attributes: { 'gen_ai.operation.name': 'invoke_agent', ...USAGE },
      },
    ]);
    expect(withoutDuplicateUsage(span as ReadableSpan, ROOT)).toBe(span);
  });
});

describe('DedupeAgentUsageExporter', () => {
  const capture = () => {
    const exported: ReadableSpan[][] = [];
    const inner: SpanExporter = {
      export: (spans, cb) => {
        exported.push(spans);
        cb({ code: 0 });
      },
      shutdown: vi.fn(async () => {}),
      forceFlush: vi.fn(async () => {}),
    };
    return { exported, inner };
  };

  it('exports the summed usage of a trace once', () => {
    const { exported, inner } = capture();
    const exporter = new DedupeAgentUsageExporter(inner, ROOT);
    const spans = finishedSpans([rootInvoke, chat, rootInvoke, chat]);
    exporter.export(spans, () => {});
    const sum = (list: ReadableSpan[]) =>
      list.reduce(
        (total, s) =>
          total + Number(s.attributes['gen_ai.usage.input_tokens'] ?? 0),
        0,
      );
    expect(sum(spans)).toBe(121775 * 4);
    expect(sum(exported[0] ?? [])).toBe(121775 * 2);
    expect(exported[0]).toHaveLength(4);
  });

  it("keeps a subagent's usage in the sum", () => {
    const { exported, inner } = capture();
    new DedupeAgentUsageExporter(inner, ROOT).export(
      finishedSpans([rootInvoke, chat, subagentInvoke]),
      () => {},
    );
    const kept = (exported[0] ?? []).filter(
      (s) => s.attributes['gen_ai.usage.input_tokens'] !== undefined,
    );
    expect(kept.map((s) => s.name)).toEqual([
      'chat claude-opus-5',
      'invoke_agent claude-opus-5',
    ]);
  });

  it('exports a span it cannot inspect as it was, and delegates shutdown and flush', async () => {
    const { exported, inner } = capture();
    const broken = {
      get attributes(): never {
        throw new Error('unreadable');
      },
    } as unknown as ReadableSpan;
    const exporter = new DedupeAgentUsageExporter(inner, ROOT);
    exporter.export([broken], () => {});
    expect(exported[0]?.[0]).toBe(broken);
    await exporter.forceFlush();
    await exporter.shutdown();
    expect(inner.forceFlush).toHaveBeenCalled();
    expect(inner.shutdown).toHaveBeenCalled();
  });
});
