/**
 * OpenTelemetry instrumentation for Jev (TypeSafe System One) calls.
 *
 * Two outputs, for the same reason as lib/observability/browser-telemetry.ts:
 *
 * - **Spans** go to whatever exporter is registered (`instrumentation.ts` for
 *   the Next process, `agent/instrumentation.ts` for Eve). `@opentelemetry/api`
 *   is a no-op when no SDK is registered, so calling this is always safe.
 * - **Structured logs** go to stdout, which Cloud Run collects and `pnpm dev`
 *   prints. Both exporters are gated on `BRAINTRUST_API_KEY`, so the log line
 *   is the only signal in an environment without one.
 *
 * Both exporters run `filterAISpans: true`, which drops every span that is not
 * an AI-SDK span. Their `customFilter` has to name this tracer explicitly or
 * nothing here is exported — see the `JEV_TRACER` import in both files.
 *
 * What is deliberately NOT recorded: the participant record, the value the
 * agent entered, the candidate list, and the resolved value. Those carry
 * applicant PII. Verdicts, confidences and form-field labels are judgments and
 * labels, not applicant data, and the same reasoning that keeps the `network`
 * category out of lib/kernel/telemetry.ts applies here.
 */

import { SpanStatusCode, trace, type Span } from '@opentelemetry/api';
import type { JevFeature } from './client';

/** Scope name on every span this module creates; exporters filter on it. */
export const TRACER_NAME = 'labs-asp.jev';

/** Attribute keys, named once so spans and logs cannot disagree. */
export const ATTR = {
  /** `gap-triage` | `summary-check` | `field-value`. Low cardinality. */
  feature: 'jev.feature',
  /** False when the flag is off or no participant record was found. */
  enabled: 'jev.enabled',
  /** Whether the call returned answers. */
  ok: 'jev.ok',
  /** Failure text from askJev — rate limit, timeout, abort, bad key. */
  reason: 'jev.reason',
  /** Model id Jev reports back, so a model change is visible in the trace. */
  model: 'jev.model',
  durationMs: 'jev.duration_ms',
  /** Form-field label the judgment is about. Not a value. */
  field: 'jev.field',
  /** The chosen label, e.g. `present_in_record`. */
  verdict: 'jev.verdict',
  confidence: 'jev.confidence',
  /** Rows the batch tried to annotate, and how many came back annotated. */
  rowsAttempted: 'jev.rows_attempted',
  rowsAnnotated: 'jev.rows_annotated',
  errorType: 'error.type',
} as const;

/** Why a batch did not call Jev at all. */
export type JevSkipReason = 'flag_off' | 'no_participant';

function log(
  severity: 'INFO' | 'WARNING',
  event: string,
  fields: Record<string, unknown>,
): void {
  const line = JSON.stringify({ severity, event, ...fields });
  if (severity === 'WARNING') console.error(line);
  else console.log(line);
}

/**
 * Wrap one `askJev` request.
 *
 * Reports the outcome on the span rather than throwing: `askJev` never throws,
 * and a failure here is a missing annotation, not a failed form run. The span
 * status is still set to ERROR so a failed call is filterable.
 */
export async function withJevCallSpan<T extends { ok: boolean }>(
  meta: { feature: JevFeature; field?: string },
  fn: () => Promise<T>,
): Promise<T> {
  const startedAt = Date.now();

  return trace.getTracer(TRACER_NAME).startActiveSpan(
    `jev ${meta.feature}`,
    {
      attributes: {
        [ATTR.feature]: meta.feature,
        ...(meta.field ? { [ATTR.field]: meta.field } : {}),
      },
    },
    async (span: Span) => {
      const result = await fn();
      const durationMs = Date.now() - startedAt;

      span.setAttribute(ATTR.durationMs, durationMs);
      span.setAttribute(ATTR.ok, result.ok);

      if (result.ok) {
        const model = (result as { model?: string }).model;
        if (model) span.setAttribute(ATTR.model, model);
        span.setStatus({ code: SpanStatusCode.OK });
      } else {
        const reason = (result as { reason?: string }).reason ?? 'unknown';
        span.setAttribute(ATTR.reason, reason);
        span.setAttribute(ATTR.errorType, 'jev_call_failed');
        span.setStatus({ code: SpanStatusCode.ERROR, message: reason });
        // A failed call is the case this whole module exists for: without it,
        // an un-annotated row is indistinguishable from a disabled feature.
        log('WARNING', 'jev.call.failed', {
          feature: meta.feature,
          field: meta.field,
          reason,
          durationMs,
        });
      }

      span.end();
      return result;
    },
  );
}

/** Record the verdict a call produced, on the span that is active now. */
export function annotateVerdict(verdict: string, confidence: number): void {
  trace.getActiveSpan()?.setAttributes({
    [ATTR.verdict]: verdict,
    [ATTR.confidence]: confidence,
  });
}

/**
 * Wrap a whole tool call's worth of Jev requests.
 *
 * The per-call spans answer "did this request work"; this one answers "how
 * much of this card got annotated at all", which is the number that makes a
 * silently rate-limited feature visible. It also covers the disabled case,
 * where no request is made and there would otherwise be no span to find.
 */
export async function withJevBatchSpan<T>(
  meta: {
    feature: JevFeature;
    enabled: boolean;
    skipReason?: JevSkipReason;
    rowsAttempted: number;
  },
  fn: () => Promise<T[]>,
  /** Counts a returned row as annotated. */
  isAnnotated: (row: T) => boolean,
): Promise<T[]> {
  const startedAt = Date.now();

  return trace.getTracer(TRACER_NAME).startActiveSpan(
    `jev ${meta.feature} batch`,
    {
      attributes: {
        [ATTR.feature]: meta.feature,
        [ATTR.enabled]: meta.enabled,
        [ATTR.rowsAttempted]: meta.enabled ? meta.rowsAttempted : 0,
        ...(meta.skipReason ? { [ATTR.reason]: meta.skipReason } : {}),
      },
    },
    async (span: Span) => {
      const rows = await fn();
      const annotated = rows.filter(isAnnotated).length;
      const durationMs = Date.now() - startedAt;

      span.setAttribute(ATTR.rowsAnnotated, annotated);
      span.setAttribute(ATTR.durationMs, durationMs);
      span.setStatus({ code: SpanStatusCode.OK });
      span.end();

      log('INFO', 'jev.batch.finish', {
        feature: meta.feature,
        enabled: meta.enabled,
        skipReason: meta.skipReason,
        rowsAttempted: meta.enabled ? meta.rowsAttempted : 0,
        rowsAnnotated: annotated,
        durationMs,
      });

      return rows;
    },
  );
}
