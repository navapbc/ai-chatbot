import { registerOTel } from '@vercel/otel';
import { trace } from '@opentelemetry/api';
import { BraintrustExporter } from '@braintrust/otel';
import { TraceExporter } from '@google-cloud/opentelemetry-cloud-trace-exporter';
import {
  BatchSpanProcessor,
  type SpanProcessor,
} from '@opentelemetry/sdk-trace-base';

/**
 * Register OpenTelemetry exporters.
 *
 * Two backends, because they answer different questions and neither covers the
 * other:
 *
 * - **Braintrust** — eval-oriented: prompt/completion inspection, scoring,
 *   dataset regressions. Configured with `filterAISpans: true`, so it receives
 *   only AI spans; the `agent-browser` and `kernel` spans from
 *   `lib/observability/browser-telemetry.ts` are filtered out before export.
 * - **Cloud Trace** — operational: span waterfalls, latency distributions, and
 *   dependency timing for the browser/Kernel path. This is where the
 *   subprocess and SDK spans actually land, and it correlates with the Cloud
 *   Run request logs already being collected.
 *
 * Each is registered only when its prerequisite is present, so a missing key or
 * a non-GCP environment degrades to "no traces" rather than a boot failure.
 */
export function register() {
  const spanProcessors: SpanProcessor[] = [];
  const enabled: string[] = [];

  if (process.env.BRAINTRUST_API_KEY) {
    spanProcessors.push(
      new BatchSpanProcessor(new BraintrustExporter({ filterAISpans: true })),
    );
    enabled.push('braintrust');
  }

  // Gate on GOOGLE_CLOUD_PROJECT, which terraform/cloud_run.tf sets explicitly.
  // An earlier version gated on K_SERVICE — injected by the Cloud Run runtime
  // rather than the service spec, and evidently not visible to the
  // instrumentation hook, so the exporter was never registered and spans went
  // nowhere while Braintrust's initialized normally.
  const projectId = process.env.GOOGLE_CLOUD_PROJECT;
  if (projectId) {
    spanProcessors.push(
      new BatchSpanProcessor(new TraceExporter({ projectId })),
    );
    enabled.push('cloud-trace');
  }

  // Say which exporters came up. Silence here previously looked identical to a
  // working setup, which is what made the missing spans hard to attribute.
  console.log(
    JSON.stringify({
      severity: 'INFO',
      event: 'otel.register',
      exporters: enabled,
    }),
  );

  if (spanProcessors.length === 0) return;

  registerOTel({ serviceName: 'labs-asp-chat', spanProcessors });

  // Prove the provider registered here is the one this module can see. If the
  // bundler gives lib/observability a separate @opentelemetry/api instance,
  // that module's spans get an all-zero trace id while this one reports a real
  // one — the two logs together localize the split immediately.
  const probe = trace.getTracer('otel-self-check').startSpan('register-probe');
  const probeTraceId = probe.spanContext().traceId;
  probe.end();
  console.log(
    JSON.stringify({
      severity: 'INFO',
      event: 'otel.self_check',
      recording: probeTraceId !== '0'.repeat(32),
      traceId: probeTraceId,
    }),
  );
}
