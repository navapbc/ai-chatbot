import type { NextConfig } from 'next';
import { withEve } from 'eve/next';

const nextConfig: NextConfig = {
  // cacheComponents disabled to allow runtime env vars in API routes
  // See: https://github.com/vercel/next.js/discussions/84894
  cacheComponents: false,
  experimental: {
    // `withEve()` rewrites /eve/v1/** to an ABSOLUTE 127.0.0.1 URL (the eve
    // child: an ephemeral port in dev, 4274 under `next start`), so Next treats
    // it as an external rewrite and proxies it through
    // `server/lib/router-utils/proxy-request.js`, which defaults
    // `proxyTimeout` to 30_000 and hands it to the bundled http-proxy as
    // `proxyReq.setTimeout(t, () => proxyReq.abort())`.
    //
    // That is an IDLE timeout on the upstream socket and it stays armed after
    // response headers are sent. Eve's NDJSON event stream sends no heartbeat
    // and goes completely silent while the model thinks or a subagent runs, so
    // any silence over 30s aborted the upstream mid-stream. Because
    // `headersSent` was already true, the downstream response was left OPEN
    // rather than ended: the /api/eve-chat reader saw neither an error nor EOF
    // and blocked until undici's own 300s body timeout, at which point the
    // reconnect loop in that route replayed the turn from its cursor.
    //
    // Measured on one run: eve finished the turn in 78s (a 55.7s model pause
    // after a 46KB snapshot was the only gap over 30s); the UI took 5.3min.
    // So a reconnect in any log predating this setting cannot be attributed to
    // a >300s silence — a 31s one produced the identical symptom. The reconnect
    // loop in app/(chat)/api/eve-chat/route.ts still earns its keep: this only
    // removes the proxy as the first thing to give up, and undici's 300s body
    // timeout on that route's own fetch is unchanged, so a subagent that stays
    // silent past 300s still resumes from the cursor as that comment describes.
    //
    // Sized to Cloud Run's request timeout (`chatbot_timeout`, 3600s in
    // terraform/variables.tf) — the proxy must never be the first thing to
    // give up. `0` does NOT disable the timeout: proxy-request.js evaluates
    // `proxyTimeout || 30000`, so a falsy value falls back to the 30s default.
    proxyTimeout: 3_600_000,
  },
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

export default withEve(nextConfig);
