import { describe, it, expect, beforeAll, afterEach } from 'vitest';
import { AsyncLocalStorage } from 'node:async_hooks';
import {
  context,
  ROOT_CONTEXT,
  trace,
  type Context,
  type ContextManager,
} from '@opentelemetry/api';
import {
  BasicTracerProvider,
  InMemorySpanExporter,
  SimpleSpanProcessor,
  type ReadableSpan,
} from '@opentelemetry/sdk-trace-base';
import {
  ATTR,
  TRACER_NAME,
  annotateVerdict,
  withJevBatchSpan,
  withJevCallSpan,
} from '@/lib/jev/telemetry';

// A real provider, not a mock: the thing under test is that spans are created
// with the right scope and attributes, and a no-op tracer would pass silently.
const memory = new InMemorySpanExporter();

/**
 * Minimal context manager over AsyncLocalStorage.
 *
 * The default manager is a no-op, so `trace.getActiveSpan()` returns undefined
 * and `annotateVerdict` would silently write nowhere — the test would pass for
 * the wrong reason. @vercel/otel installs a real one in both server processes;
 * this is the test-local equivalent, written out rather than pulling in
 * @opentelemetry/context-async-hooks as a new dependency.
 */
class AlsContextManager implements ContextManager {
  private readonly als = new AsyncLocalStorage<Context>();
  active = () => this.als.getStore() ?? ROOT_CONTEXT;
  with = <A extends unknown[], F extends (...args: A) => ReturnType<F>>(
    ctx: Context,
    fn: F,
    thisArg?: ThisParameterType<F>,
    ...args: A
  ): ReturnType<F> =>
    this.als.run(ctx, () => fn.call(thisArg as never, ...args));
  bind = <T>(ctx: Context, target: T): T =>
    typeof target === 'function'
      ? (((...args: unknown[]) =>
          this.with(ctx, target as never, undefined, ...args)) as T)
      : target;
  enable = () => this;
  disable = () => this;
}

beforeAll(() => {
  const provider = new BasicTracerProvider({
    spanProcessors: [new SimpleSpanProcessor(memory)],
  });
  trace.setGlobalTracerProvider(provider);
  context.setGlobalContextManager(new AlsContextManager());
});

afterEach(() => memory.reset());

const spanNamed = (name: string): ReadableSpan => {
  const found = memory.getFinishedSpans().find((s) => s.name === name);
  if (!found) {
    throw new Error(
      `no span named "${name}"; got ${memory
        .getFinishedSpans()
        .map((s) => s.name)
        .join(', ')}`,
    );
  }
  return found;
};

describe('withJevCallSpan', () => {
  it('records the model and an OK status on success', async () => {
    const result = await withJevCallSpan(
      { feature: 'gap-triage', field: 'Social Security Number' },
      async () => ({ ok: true as const, model: 'jev-1' }),
    );

    expect(result.ok).toBe(true);
    const span = spanNamed('jev gap-triage');
    // The scope is what both exporters' customFilter matches on. If this
    // changes, spans stop reaching Braintrust without any other symptom.
    expect(span.instrumentationScope.name).toBe(TRACER_NAME);
    expect(span.attributes[ATTR.feature]).toBe('gap-triage');
    expect(span.attributes[ATTR.field]).toBe('Social Security Number');
    expect(span.attributes[ATTR.ok]).toBe(true);
    expect(span.attributes[ATTR.model]).toBe('jev-1');
    expect(span.attributes[ATTR.durationMs]).toBeTypeOf('number');
  });

  it('records the reason and an error status on failure', async () => {
    await withJevCallSpan({ feature: 'field-value' }, async () => ({
      ok: false as const,
      reason: 'rate limited',
    }));

    const span = spanNamed('jev field-value');
    expect(span.attributes[ATTR.ok]).toBe(false);
    expect(span.attributes[ATTR.reason]).toBe('rate limited');
    expect(span.attributes[ATTR.errorType]).toBe('jev_call_failed');
    expect(span.status.code).toBe(2); // SpanStatusCode.ERROR
  });

  it('puts the verdict on the active call span', async () => {
    await withJevCallSpan({ feature: 'summary-check' }, async () => {
      annotateVerdict('contradicts_record', 0.82);
      return { ok: true as const, model: 'jev-1' };
    });

    const span = spanNamed('jev summary-check');
    expect(span.attributes[ATTR.verdict]).toBe('contradicts_record');
    expect(span.attributes[ATTR.confidence]).toBe(0.82);
  });
});

describe('withJevBatchSpan', () => {
  it('counts attempted against annotated rows', async () => {
    const rows = await withJevBatchSpan(
      { feature: 'summary-check', enabled: true, rowsAttempted: 3 },
      async () => [{ jev: {} }, { jev: undefined }, { jev: {} }],
      (row) => row.jev !== undefined,
    );

    expect(rows).toHaveLength(3);
    const span = spanNamed('jev summary-check batch');
    expect(span.attributes[ATTR.enabled]).toBe(true);
    expect(span.attributes[ATTR.rowsAttempted]).toBe(3);
    // 2 of 3: the gap this span exists to make visible.
    expect(span.attributes[ATTR.rowsAnnotated]).toBe(2);
  });

  it('emits a span when the feature is off, with nothing attempted', async () => {
    await withJevBatchSpan(
      {
        feature: 'gap-triage',
        enabled: false,
        skipReason: 'flag_off',
        rowsAttempted: 7,
      },
      async () => [{ jev: undefined }],
      (row) => row.jev !== undefined,
    );

    const span = spanNamed('jev gap-triage batch');
    expect(span.attributes[ATTR.enabled]).toBe(false);
    expect(span.attributes[ATTR.reason]).toBe('flag_off');
    // Not 7: nothing was asked, so the denominator is zero, not the row count.
    expect(span.attributes[ATTR.rowsAttempted]).toBe(0);
  });
});
