import { BraintrustExporter } from '@braintrust/otel';
import { registerOTel } from '@vercel/otel';
import { defineInstrumentation } from 'eve/instrumentation';
import { TRACER_NAME as BROWSER_TRACER } from '@/lib/observability/browser-telemetry';

/**
 * Exports Eve's agent spans to Braintrust.
 *
 * The agent loop runs in the Eve server process, not in Next, so the root
 * `instrumentation.ts` never sees these spans — this file is the only path
 * that gets them off the machine. It mirrors that file's exporter config so a
 * local trace looks like a labs-asp-preview one.
 *
 * Authoring this file also switches off Eve's zero-config local trace writer
 * (`.eve/traces/v1`), so `eve trace ls` stops recording new traces. Delete or
 * rename this file to get that back.
 */
export default defineInstrumentation({
  // recordInputs/recordOutputs are left at their default (true once this file
  // exists), so spans carry full message history and model outputs. That is
  // the point locally — it is what makes a trace debuggable — but it means any
  // Apricot participant data a session touches is sent to Braintrust. Set both
  // to false if that is ever not acceptable for the data in play.

  setup: ({ agentName }) => {
    // BraintrustExporter throws without a key. Terraform deliberately leaves
    // it unset on prod, so construct the exporter only when it is present —
    // otherwise `eve dev`/`eve start` would fail at startup wherever no key is
    // configured. With no key there is no Braintrust export and no local
    // trace file either, since authoring this file uninstalls that writer.
    if (!process.env.BRAINTRUST_API_KEY) return;

    registerOTel({
      serviceName: agentName,
      // No `spanProcessors` here, so @vercel/otel takes its "auto" branch and
      // adds a second, env-configured OTLP exporter whenever
      // OTEL_EXPORTER_OTLP_ENDPOINT is set — every span would ship twice, to
      // two different projects. Keep those vars out of .env.local.
      traceExporter: new BraintrustExporter({
        // Destination is BRAINTRUST_PARENT. Do not fall back to `agentName`:
        // Eve resolves it from the package name (`ai-chatbot`), which is not a
        // project anyone looks at. Unset, the exporter uses a project literally
        // named "default-otel-project".
        //
        // Keep the AI spans and the browser tracer's spans. Drop the rest.
        // agent/tools/browser.ts reaches lib/kernel/cli.ts, so the Kernel
        // browser spans are emitted in this process too, and filterAISpans
        // alone would discard them.
        filterAISpans: true,
        customFilter: (span) =>
          span.instrumentationScope?.name === BROWSER_TRACER ? true : undefined,
      }),
    });
  },
});
