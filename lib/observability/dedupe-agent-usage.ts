import type { Attributes } from '@opentelemetry/api';
import type { ReadableSpan, SpanExporter } from '@opentelemetry/sdk-trace-base';

// Eve emits an `invoke_agent <model>` span for every model step of the root
// agent, and the AI SDK emits a `chat <model>` span for the same call. Both
// carry the step's `gen_ai.usage.*` token counts, and Braintrust prices every
// span that has a model and tokens, so a trace's summed cost came out about
// 2x the real spend (a 2026-11-04 dev trace showed $113.72; the chat spans
// summed to $56.85). The `chat` span is the one that matches what the provider
// billed, so the root agent's `invoke_agent` span keeps its identity, timing
// and model but loses its token counts.
//
// Only the root agent's spans are touched. Eve also emits `invoke_agent` for a
// subagent's usage (CHANGELOG 99c2380), under the subagent's own
// `gen_ai.agent.name`; that usage is not duplicated by a chat span in the
// parent's trace, so it is left alone.

export const USAGE_ATTRIBUTES: readonly string[] = [
  'gen_ai.usage.input_tokens',
  'gen_ai.usage.output_tokens',
  'gen_ai.usage.cache_read.input_tokens',
  'gen_ai.usage.cache_creation.input_tokens',
];

const isRootInvokeAgent = (
  attributes: Attributes | undefined,
  rootAgentName: string,
): boolean =>
  attributes?.['gen_ai.operation.name'] === 'invoke_agent' &&
  attributes['gen_ai.agent.name'] === rootAgentName;

/**
 * The span without usage counts when it is the root agent's `invoke_agent`
 * span, otherwise the span itself. The original span is never modified.
 */
export function withoutDuplicateUsage(
  span: ReadableSpan,
  rootAgentName: string,
): ReadableSpan {
  if (!isRootInvokeAgent(span.attributes, rootAgentName)) return span;
  const attributes: Attributes = {};
  for (const [key, value] of Object.entries(span.attributes)) {
    if (!USAGE_ATTRIBUTES.includes(key)) attributes[key] = value;
  }
  // ReadableSpan is a class with getters over private fields, so a copy with
  // one field changed has to read the rest through the original.
  return new Proxy(span, {
    get(target, property) {
      if (property === 'attributes') return attributes;
      const value = Reflect.get(target, property, target);
      return typeof value === 'function' ? value.bind(target) : value;
    },
  });
}

/**
 * Wraps a span exporter so the root agent's `invoke_agent` spans leave the
 * process without token counts. Everything else passes through unchanged. A
 * span that cannot be inspected is exported as it was.
 */
export class DedupeAgentUsageExporter implements SpanExporter {
  constructor(
    private readonly inner: SpanExporter,
    private readonly rootAgentName: string,
  ) {}

  export(
    spans: ReadableSpan[],
    resultCallback: Parameters<SpanExporter['export']>[1],
  ): void {
    this.inner.export(
      spans.map((span) => {
        try {
          return withoutDuplicateUsage(span, this.rootAgentName);
        } catch {
          return span;
        }
      }),
      resultCallback,
    );
  }

  shutdown(): Promise<void> {
    return this.inner.shutdown();
  }

  forceFlush(): Promise<void> {
    return this.inner.forceFlush?.() ?? Promise.resolve();
  }
}
