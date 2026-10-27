import type { Attributes } from '@opentelemetry/api';
import type { ReadableSpan, SpanExporter } from '@opentelemetry/sdk-trace-base';

/**
 * Span attribute allowlist for relay runs.
 *
 * Eve and the AI SDK record prompts, message history, tool arguments, tool
 * results and model output on spans by default. On a relay run those carry
 * values from fields the caseworker marked sensitive, so the exporter keeps
 * only the keys below: operation names, model and provider ids, token counts,
 * durations, tool NAMES, finish reasons and opaque ids.
 *
 * Exact key names only. Never add a prefix wildcard, and never add a key that
 * carries content (`*.messages`, `*.arguments`, `*.result`, `*.prompt`,
 * `*.text`, `gen_ai.system_instructions`, `gen_ai.tool.definitions`,
 * `exception.*`, URLs, headers).
 */
export const SAFE_SPAN_ATTRIBUTES: ReadonlySet<string> = new Set([
  // OpenTelemetry GenAI semantic conventions (AI SDK OpenTelemetry integration)
  'gen_ai.operation.name',
  'gen_ai.provider.name',
  'gen_ai.system',
  'gen_ai.agent.name',
  'gen_ai.request.model',
  'gen_ai.request.temperature',
  'gen_ai.request.top_p',
  'gen_ai.request.top_k',
  'gen_ai.request.max_tokens',
  'gen_ai.response.model',
  'gen_ai.response.id',
  'gen_ai.response.finish_reasons',
  'gen_ai.tool.name',
  'gen_ai.tool.call.id',
  'gen_ai.tool.type',
  'gen_ai.usage.input_tokens',
  'gen_ai.usage.output_tokens',
  'gen_ai.usage.cache_read.input_tokens',
  'gen_ai.usage.cache_creation.input_tokens',
  'gen_ai.client.operation.duration',
  'gen_ai.execute_tool.duration',
  // AI SDK legacy `ai.*` attributes (names, ids, counts, timings only)
  'operation.name',
  'ai.operationId',
  'ai.telemetry.functionId',
  'ai.model.id',
  'ai.model.provider',
  'ai.toolCall.name',
  'ai.toolCall.id',
  'ai.response.id',
  'ai.response.model',
  'ai.response.finishReason',
  'ai.response.msToFirstChunk',
  'ai.response.msToFinish',
  'ai.response.avgOutputTokensPerSecond',
  'ai.stream.msToFirstChunk',
  'ai.stream.msToFinish',
  'ai.stream.avgOutputTokensPerSecond',
  'ai.usage.inputTokens',
  'ai.usage.outputTokens',
  'ai.usage.totalTokens',
  'ai.usage.reasoningTokens',
  'ai.usage.cachedInputTokens',
  'ai.usage.inputTokenDetails.noCacheTokens',
  'ai.usage.inputTokenDetails.cacheReadTokens',
  'ai.usage.inputTokenDetails.cacheWriteTokens',
  'ai.usage.outputTokenDetails.textTokens',
  'ai.usage.outputTokenDetails.reasoningTokens',
  // Eve's own agent spans (agent-otel-provider)
  'agent.framework.name',
  'agent.framework.version',
  'agent.name',
  'agent.root.session.id',
  'agent.session.id',
  'agent.turn.id',
  'agent.turn.sequence',
  'agent.step.index',
  'agent.step.attempt',
  'agent.action.call_id',
  'agent.action.kind',
  'agent.action.name',
  'agent.model.id',
  'agent.model.provider',
  'agent.usage.input_tokens',
  'agent.usage.output_tokens',
  'eve.session.id',
  'eve.turn.id',
  'eve.turn.sequence',
  'eve.step.index',
  'eve.channel.kind',
  'eve.environment',
  'eve.version',
  // lib/observability/browser-telemetry.ts and lib/jev/telemetry.ts: command
  // names, counts, durations and outcomes. `agent_browser.session` (carries a
  // user id) and `jev.field`/`jev.reason`/`jev.verdict` are left off.
  'agent_browser.command',
  'agent_browser.arg_count',
  'agent_browser.remote',
  'agent_browser.duration_ms',
  'agent_browser.timeout_ms',
  'agent_browser.outcome',
  'process.exit.code',
  'jev.feature',
  'jev.enabled',
  'jev.ok',
  'jev.rows_attempted',
  'jev.rows_annotated',
  'jev.duration_ms',
  'kernel.event_count',
  // Error flags
  'error.type',
]);

/** The only resource attributes kept (the span allowlist is not reused). */
const SAFE_RESOURCE_ATTRIBUTES: ReadonlySet<string> = new Set([
  'service.name',
  'service.version',
  'deployment.environment',
]);

const MAX_STRING_VALUE = 200;
const MAX_NAME_LENGTH = 120;
const MAX_NAME_TOKEN = 40;
const SCRUBBED_NAME = 'scrubbed';

const NAME_PATTERN = /^[A-Za-z0-9_. :/-]{1,120}$/;
// Span names are operation names. Only these families are kept; the rest is
// replaced, since an unknown span could be named after a value.
const NAME_PREFIX =
  /^(chat|execute_tool|invoke_agent|step|jev|agent_browser|agent|ai|kernel)\b/;
const TOOL_NAME = /^[A-Za-z0-9_.-]{1,64}$/;
const EXECUTE_TOOL_PREFIX = 'execute_tool ';
const ERROR_TYPE = /^[A-Za-z0-9_.:-]{1,64}$/;
// Value shapes that must never ride in a span name.
const SSN_SHAPE = /\b\d{3}-?\d{2}-?\d{4}\b/;
const LONG_DIGITS = /\d{6,}/;
const DATE_SHAPE = /\b\d{1,4}[-/]\d{1,2}[-/]\d{1,4}\b/;

function safeName(name: unknown): string {
  if (typeof name !== 'string') return SCRUBBED_NAME;
  if (name.length > MAX_NAME_LENGTH || !NAME_PATTERN.test(name)) {
    return SCRUBBED_NAME;
  }
  if (!NAME_PREFIX.test(name)) return SCRUBBED_NAME;
  if (
    name.startsWith(EXECUTE_TOOL_PREFIX) &&
    !TOOL_NAME.test(name.slice(EXECUTE_TOOL_PREFIX.length))
  ) {
    return SCRUBBED_NAME;
  }
  if (SSN_SHAPE.test(name) || LONG_DIGITS.test(name) || DATE_SHAPE.test(name)) {
    return SCRUBBED_NAME;
  }
  if (name.split(/[ :/]/).some((token) => token.length > MAX_NAME_TOKEN)) {
    return SCRUBBED_NAME;
  }
  return name;
}

function isSafeValue(value: unknown): boolean {
  if (typeof value === 'string') return value.length <= MAX_STRING_VALUE;
  if (typeof value === 'number' || typeof value === 'boolean') return true;
  if (Array.isArray(value)) {
    return value.length <= 20 && value.every(isSafeValue);
  }
  return false;
}

function filterAttributes(
  attributes: Attributes | undefined,
  allowed: (key: string) => boolean,
): Attributes {
  const out: Attributes = {};
  for (const key of Object.keys(attributes ?? {})) {
    if (!allowed(key)) continue;
    const value = (attributes as Attributes)[key];
    if (!isSafeValue(value)) continue;
    if (
      key === 'error.type' &&
      !(typeof value === 'string' && ERROR_TYPE.test(value))
    ) {
      continue;
    }
    out[key] = value;
  }
  return out;
}

/**
 * Structural copy of `span` that keeps identity, timing, kind, parent, scope
 * and status code, and only allowlisted attributes.
 *
 * Dropped outright: span events (exceptions and `ai.*` content events carry
 * page values and prompts), the status message (an exception message),
 * link attributes, and any resource attribute outside the allowlist. A span
 * name that is not conservative text is replaced with `scrubbed`.
 *
 * The span object is a class instance with getters, so every field is read
 * explicitly. Throws if the span cannot be read; the caller drops it.
 */
export function scrubSpan(span: ReadableSpan): ReadableSpan {
  const resourceAttributes = filterAttributes(
    span.resource?.attributes,
    (key) => SAFE_RESOURCE_ATTRIBUTES.has(key),
  );
  const resource = {
    attributes: resourceAttributes,
    asyncAttributesPending: false,
    getRawAttributes: () => Object.entries(resourceAttributes),
    merge() {
      return resource;
    },
  };

  const scrubbed = {
    name: safeName(span.name),
    kind: span.kind,
    spanContext: () => span.spanContext(),
    parentSpanContext: span.parentSpanContext,
    startTime: span.startTime,
    endTime: span.endTime,
    status: { code: span.status.code },
    attributes: filterAttributes(span.attributes, (key) =>
      SAFE_SPAN_ATTRIBUTES.has(key),
    ),
    links: span.links.map((link) => ({
      context: link.context,
      attributes: {},
    })),
    events: [],
    duration: span.duration,
    ended: span.ended,
    resource,
    instrumentationScope: span.instrumentationScope,
    droppedAttributesCount: span.droppedAttributesCount,
    droppedEventsCount: span.droppedEventsCount,
    droppedLinksCount: span.droppedLinksCount,
  };
  return scrubbed as unknown as ReadableSpan;
}

/**
 * Wraps a span exporter so that, while `enabled()` is true, only allowlisted
 * attributes leave the process. A span that cannot be scrubbed is dropped, not
 * exported raw. `enabled()` is evaluated on every export, and a throwing
 * `enabled()` counts as enabled. Nothing here logs span content.
 */
export class ScrubbingSpanExporter implements SpanExporter {
  constructor(
    private readonly inner: SpanExporter,
    private readonly opts: { enabled: () => boolean },
  ) {}

  export(
    spans: ReadableSpan[],
    resultCallback: Parameters<SpanExporter['export']>[1],
  ): void {
    if (!this.isEnabled()) {
      this.inner.export(spans, resultCallback);
      return;
    }
    const safe: ReadableSpan[] = [];
    for (const span of spans) {
      try {
        safe.push(scrubSpan(span));
      } catch {
        // Fail closed: drop the span rather than export it raw.
      }
    }
    this.inner.export(safe, resultCallback);
  }

  shutdown(): Promise<void> {
    return this.inner.shutdown();
  }

  forceFlush(): Promise<void> {
    return this.inner.forceFlush?.() ?? Promise.resolve();
  }

  private isEnabled(): boolean {
    try {
      return this.opts.enabled();
    } catch {
      return true;
    }
  }
}
