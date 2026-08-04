import { registerOTel } from '@vercel/otel';
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

  if (process.env.BRAINTRUST_API_KEY) {
    spanProcessors.push(
      new BatchSpanProcessor(new BraintrustExporter({ filterAISpans: true })),
    );
  }

  // K_SERVICE is set by Cloud Run. Off GCP there is no metadata server to
  // authenticate against, so skip rather than fail.
  if (process.env.K_SERVICE) {
    spanProcessors.push(
      new BatchSpanProcessor(
        new TraceExporter({ projectId: process.env.GOOGLE_CLOUD_PROJECT }),
      ),
    );
  }

  if (spanProcessors.length === 0) return;

  registerOTel({ serviceName: 'labs-asp-chat', spanProcessors });
}
