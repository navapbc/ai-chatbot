import { registerOTel } from '@vercel/otel';
import {
  diag,
  DiagConsoleLogger,
  DiagLogLevel,
  trace,
} from '@opentelemetry/api';
import { BraintrustExporter } from '@braintrust/otel';
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-proto';
import { Compute } from 'google-auth-library';
import {
  BatchSpanProcessor,
  ConsoleSpanExporter,
  SimpleSpanProcessor,
  type SpanProcessor,
} from '@opentelemetry/sdk-trace-base';
import { registerTelemetry } from 'ai';
import { OpenTelemetry } from '@ai-sdk/otel';
import { TRACER_NAME as BROWSER_TRACER } from '@/lib/observability/browser-telemetry';
import { TRACER_NAME as JEV_TRACER } from '@/lib/jev/telemetry';

/**
 * Tracer scopes kept in addition to the AI spans.
 *
 * `filterAISpans: true` drops everything that is not an AI-SDK span, so a
 * module that creates its own tracer has to be named here or its spans are
 * silently discarded. Adding a tracer elsewhere in the codebase means adding
 * it to this set — in this file AND in agent/instrumentation.ts, which runs in
 * the separate Eve process.
 */
const KEEP_SCOPES = new Set([BROWSER_TRACER, JEV_TRACER]);

/**
 * Cloud Trace over OTLP. Replaces the deprecated cloud-trace-exporter, which
 * also capped spans at 32 attributes / 256-byte values — too small for GenAI
 * spans. `headers` is async because ADC tokens expire hourly.
 *
 * No x-goog-user-project header: it makes the API bill the request to that
 * project, which needs serviceusage.services.use — not in
 * roles/telemetry.tracesWriter, so every export 403'd. The gcp.project_id
 * resource attribute below already routes the spans.
 */
function cloudTraceProcessor(): SpanProcessor {
  // Compute, not GoogleAuth: ADC prefers GOOGLE_APPLICATION_CREDENTIALS (the
  // Vertex key file, no trace roles) over the runtime SA that terraform
  // grants telemetry.tracesWriter. Only the metadata server serves the SA.
  const auth = new Compute({
    scopes: ['https://www.googleapis.com/auth/cloud-platform'],
  });

  return new BatchSpanProcessor(
    new OTLPTraceExporter({
      url: 'https://telemetry.googleapis.com/v1/traces',
      async headers() {
        // google-auth-library@9 types this as Headers but returns a plain
        // object at runtime; Object.entries handles both.
        const authHeaders = await auth.getRequestHeaders();
        return Object.fromEntries(Object.entries(authHeaders));
      },
    }),
  );
}

export function register() {
  // BatchSpanProcessor swallows exporter errors, so a rejected export is
  // indistinguishable from no traffic. Set OTEL_DIAG=1 to surface them.
  if (process.env.OTEL_DIAG) {
    diag.setLogger(new DiagConsoleLogger(), DiagLogLevel.DEBUG);
  }

  const spanProcessors: SpanProcessor[] = [];
  const enabled: string[] = [];

  if (process.env.BRAINTRUST_API_KEY) {
    spanProcessors.push(
      new BatchSpanProcessor(
        new BraintrustExporter({
          // Keep the AI spans, plus the scopes named in KEEP_SCOPES.
          filterAISpans: true,
          customFilter: (span) =>
            KEEP_SCOPES.has(span.instrumentationScope?.name ?? '')
              ? true
              : undefined,
        }),
      ),
    );
    enabled.push('braintrust');
  }

  // Local development: print every span to the terminal instead of (or as
  // well as) shipping it. Unlike the two exporters above this needs no key and
  // no network, so it is the one way to see spans in an environment with
  // neither. SimpleSpanProcessor, not Batch: the point is to see the span as
  // soon as it ends. Noisy by design — leave it off unless you are reading it.
  if (process.env.OTEL_CONSOLE_SPANS) {
    spanProcessors.push(new SimpleSpanProcessor(new ConsoleSpanExporter()));
    enabled.push('console');
  }

  // Gate on GOOGLE_CLOUD_PROJECT (set in terraform), not K_SERVICE — the
  // runtime-injected vars are not visible to the instrumentation hook.
  const projectId = process.env.GOOGLE_CLOUD_PROJECT;
  if (projectId) {
    spanProcessors.push(cloudTraceProcessor());
    enabled.push('cloud-trace-otlp');
  }

  // Silence here is indistinguishable from a working setup, so say what came up.
  console.log(
    JSON.stringify({
      severity: 'INFO',
      event: 'otel.register',
      exporters: enabled,
    }),
  );

  if (spanProcessors.length === 0) return;

  // The Telemetry API rejects any payload whose resource lacks gcp.project_id
  // ("Resource is missing required attribute").
  registerOTel({
    serviceName: 'labs-asp-chat',
    spanProcessors,
    ...(projectId ? { attributes: { 'gcp.project_id': projectId } } : {}),
  });

  // AI SDK 7 emits no model spans until an integration is registered. Must
  // follow registerOTel — it binds the tracer from that provider.
  registerTelemetry(new OpenTelemetry());

  // All-zero trace id here means the bundler split @opentelemetry/api and this
  // provider is not the one route handlers see.
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
