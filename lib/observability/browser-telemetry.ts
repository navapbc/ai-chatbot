/**
 * OpenTelemetry instrumentation for the agent-browser subprocess boundary.
 *
 * Two outputs, deliberately, because they answer different questions:
 *
 * - **Spans** carry timing and attributes to whatever exporter is registered in
 *   `instrumentation.ts`. `@opentelemetry/api` is a no-op when no SDK is
 *   registered, so this is safe to call unconditionally.
 * - **Structured logs** go to stdout, which Cloud Run always collects. The
 *   exporter is gated on `BRAINTRUST_API_KEY`, which is not set in every
 *   environment; a subprocess that hangs must still be diagnosable there.
 *
 * Attribute names follow OTel semantic conventions where one exists
 * (`process.*`, `error.type`, `rpc.*`) so a future backend can group on them
 * without a translation layer.
 */

import { SpanStatusCode, trace, type Span } from '@opentelemetry/api';

const TRACER_NAME = 'labs-asp.agent-browser';

/** Attribute keys, named once so spans and logs cannot disagree. */
export const ATTR = {
  /** agent-browser subcommand, e.g. `snapshot` — low cardinality, safe to group by. */
  command: 'agent_browser.command',
  /** Full argv length; the args themselves may contain PII, so only the count. */
  argCount: 'agent_browser.arg_count',
  /** Daemon session key (`--session`). */
  session: 'agent_browser.session',
  /** Whether the call attached to a remote Kernel browser over CDP. */
  remote: 'agent_browser.remote',
  exitCode: 'process.exit.code',
  durationMs: 'agent_browser.duration_ms',
  timeoutMs: 'agent_browser.timeout_ms',
  outcome: 'agent_browser.outcome',
  errorType: 'error.type',
} as const;

export type CommandOutcome =
  | 'success'
  | 'command_error'
  | 'timeout'
  | 'aborted'
  | 'spawn_error';

/**
 * Emit one structured JSON line per event.
 *
 * Cloud Run parses JSON on stdout into `jsonPayload`, so these stay queryable
 * (`jsonPayload.event="agent_browser.command.start"`) without an exporter.
 * `severity` is the field Cloud Logging promotes to the entry's log level.
 */
function log(
  severity: 'INFO' | 'WARNING' | 'ERROR',
  event: string,
  fields: Record<string, unknown>,
): void {
  const line = JSON.stringify({ severity, event, ...fields });
  if (severity === 'ERROR' || severity === 'WARNING') console.error(line);
  else console.log(line);
}

export interface CommandTelemetry {
  /** Record the result and close the span. */
  end(result: { outcome: CommandOutcome; error?: string | null }): void;
}

/**
 * Record one completed step of the agent loop.
 *
 * The loop runs up to 500 steps inside a single HTTP request, so without a
 * per-step event a slow run is one long silence: you cannot tell which tool is
 * slow, or whether the model is still making progress at all.
 *
 * Tool *names* are recorded, never their arguments — argv and form values carry
 * applicant PII.
 */
export function logAgentStep(step: {
  index: number;
  toolNames: string[];
  finishReason?: string;
  inputTokens?: number;
  outputTokens?: number;
  durationMs?: number;
}): void {
  log('INFO', 'agent.step.finish', {
    step: step.index,
    tools: step.toolNames,
    finishReason: step.finishReason,
    inputTokens: step.inputTokens,
    outputTokens: step.outputTokens,
    durationMs: step.durationMs,
  });
}

/**
 * Wrap a Kernel SDK call in a span with start/end logs.
 *
 * Creating a browser is a multi-second remote call that previously logged only
 * on failure, so a slow or hanging create was indistinguishable from a slow
 * model response in the logs.
 */
export async function withKernelSpan<T>(
  operation: string,
  attributes: Record<string, string | number | boolean>,
  fn: () => Promise<T>,
): Promise<T> {
  const startedAt = Date.now();
  log('INFO', 'kernel.operation.start', { operation, ...attributes });

  return trace
    .getTracer(TRACER_NAME)
    .startActiveSpan(`kernel ${operation}`, { attributes }, async (span) => {
      try {
        const result = await fn();
        const durationMs = Date.now() - startedAt;
        span.setAttribute(ATTR.durationMs, durationMs);
        span.setStatus({ code: SpanStatusCode.OK });
        log('INFO', 'kernel.operation.end', {
          operation,
          durationMs,
          ...attributes,
        });
        return result;
      } catch (error: unknown) {
        const durationMs = Date.now() - startedAt;
        const message = error instanceof Error ? error.message : String(error);
        span.setAttribute(ATTR.durationMs, durationMs);
        span.setAttribute(ATTR.errorType, 'kernel_error');
        span.setStatus({ code: SpanStatusCode.ERROR, message });
        log('ERROR', 'kernel.operation.end', {
          operation,
          durationMs,
          error: message,
          ...attributes,
        });
        throw error;
      } finally {
        span.end();
      }
    });
}

/**
 * Open a span around one agent-browser invocation.
 *
 * Logs on **start** as well as end: a subprocess that never returns produces no
 * end event, so without the start line the window is silent and the hang is
 * invisible — which is exactly how a 52s stall reached production undiagnosed.
 */
export function startCommandTelemetry(
  argv: readonly string[],
  meta: {
    session: string;
    remote: boolean;
    timeoutMs: number;
  },
): CommandTelemetry {
  const command = argv[0] ?? 'unknown';
  const startedAt = Date.now();

  const attributes = {
    [ATTR.command]: command,
    [ATTR.argCount]: argv.length,
    [ATTR.session]: meta.session,
    [ATTR.remote]: meta.remote,
    [ATTR.timeoutMs]: meta.timeoutMs,
  };

  const span: Span = trace
    .getTracer(TRACER_NAME)
    .startSpan(`agent-browser ${command}`, { attributes });

  log('INFO', 'agent_browser.command.start', {
    command,
    session: meta.session,
    remote: meta.remote,
    timeoutMs: meta.timeoutMs,
  });

  return {
    end({ outcome, error }) {
      const durationMs = Date.now() - startedAt;
      span.setAttribute(ATTR.durationMs, durationMs);
      span.setAttribute(ATTR.outcome, outcome);

      if (outcome === 'success') {
        span.setStatus({ code: SpanStatusCode.OK });
      } else {
        span.setAttribute(ATTR.errorType, outcome);
        span.setStatus({
          code: SpanStatusCode.ERROR,
          message: error ?? outcome,
        });
      }
      span.end();

      log(
        outcome === 'success' ? 'INFO' : 'ERROR',
        'agent_browser.command.end',
        {
          command,
          session: meta.session,
          outcome,
          durationMs,
          ...(error ? { error } : {}),
        },
      );
    },
  };
}
