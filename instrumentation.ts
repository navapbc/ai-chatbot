import { registerOTel } from '@vercel/otel';
import {
  diag,
  DiagConsoleLogger,
  DiagLogLevel,
  trace,
} from '@opentelemetry/api';
import { BraintrustExporter } from '@braintrust/otel';
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-proto';
import { GoogleAuth } from 'google-auth-library';
import {
  BatchSpanProcessor,
  type SpanProcessor,
} from '@opentelemetry/sdk-trace-base';
import { registerTelemetry } from 'ai';
import { OpenTelemetry } from '@ai-sdk/otel';

/**
 * Cloud Trace over OTLP. Replaces the deprecated cloud-trace-exporter, which
 * also capped spans at 32 attributes / 256-byte values — too small for GenAI
 * spans. `headers` is async because ADC tokens expire hourly.
 */
function cloudTraceProcessor(projectId: string): SpanProcessor {
  const auth = new GoogleAuth({
    scopes: 'https://www.googleapis.com/auth/cloud-platform',
  });

  return new BatchSpanProcessor(
    new OTLPTraceExporter({
      url: 'https://telemetry.googleapis.com/v1/traces',
      async headers() {
        const client = await auth.getClient();
        // google-auth-library@9 types this as Headers but returns a plain
        // object at runtime; Object.entries handles both.
        const authHeaders = await client.getRequestHeaders();
        return {
          ...Object.fromEntries(Object.entries(authHeaders)),
          'x-goog-user-project': projectId,
        };
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
      new BatchSpanProcessor(new BraintrustExporter({ filterAISpans: true })),
    );
    enabled.push('braintrust');
  }

  // Gate on GOOGLE_CLOUD_PROJECT (set in terraform), not K_SERVICE — the
  // runtime-injected vars are not visible to the instrumentation hook.
  const projectId = process.env.GOOGLE_CLOUD_PROJECT;
  if (projectId) {
    spanProcessors.push(cloudTraceProcessor(projectId));
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
