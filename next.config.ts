import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  // cacheComponents disabled to allow runtime env vars in API routes
  // See: https://github.com/vercel/next.js/discussions/84894
  cacheComponents: false,
  // agent-browser is a native binary invoked as a subprocess, not an imported
  // module, so there is nothing for Next.js to bundle or externalize.
  //
  // The OpenTelemetry packages DO need to be external. `@opentelemetry/api`
  // keeps its global tracer provider in module scope, so if the bundler emits
  // one copy into instrumentation.js and another into the route chunks,
  // `register()` configures a provider that `trace.getTracer()` in
  // lib/observability never sees — spans are created against a no-op tracer
  // and silently vanish. Cloud Trace received Cloud Run's own request spans
  // but none of ours until these were externalized.
  serverExternalPackages: [
    '@opentelemetry/api',
    '@opentelemetry/sdk-trace-base',
    '@opentelemetry/exporter-trace-otlp-proto',
    '@braintrust/otel',
    '@vercel/otel',
    '@ai-sdk/otel',
  ],
  images: {
    remotePatterns: [
      {
        hostname: 'avatar.vercel.sh',
      },
    ],
  },
};

export default nextConfig;
