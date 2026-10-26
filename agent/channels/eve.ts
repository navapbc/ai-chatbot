import { eveChannel } from "eve/channels/eve";
import {
  type AuthFn,
  isLoopbackRequest,
  localDev,
  placeholderAuth,
  vercelOidc,
} from "eve/channels/auth";
import { attributesFromHeaders } from "@/lib/ai/eve/loopback-attributes";

// Exposes adapter headers as session auth attributes (logic and rules in
// lib/ai/eve/loopback-attributes.ts):
//
//   x-eve-model        -> eveModel   dev/eval model override, loopback only.
//                                    Untrusted: only ever validated against
//                                    the allowlist in lib/ai/eve/model-map.ts.
//   x-eve-relay-url    -> relayUrl   the browser relay for this session
//   x-eve-relay-run    -> relayRunId the run's id (a UUID). Not a secret: the
//                                    browser tool fetches a short-lived agent
//                                    token with it at call time, because Eve
//                                    persists attributes. A token header is
//                                    never read.
//
// The relay pair is granted only on loopback AND with a matching
// `x-eve-internal-secret` (env EVE_INTERNAL_SECRET), because withEve() makes
// requests from the public origin look loopback. Returns null when nothing
// is granted, so the rest of the auth walk is unchanged.
const loopbackAttributeAuth: AuthFn<Request> = (request) => {
  const attributes = attributesFromHeaders(
    request.headers,
    isLoopbackRequest(request),
  );
  if (!attributes) return null;
  return {
    attributes,
    authenticator: "local-dev",
    principalId: "local-dev",
    principalType: "local-dev",
    subject: "local-dev",
  };
};

export default eveChannel({
  auth: [
    // Loopback-only adapter headers become session attributes (model override,
    // relay run reference). See comment above.
    loopbackAttributeAuth,
    // Lets the eve TUI and your Vercel deployments reach the deployed agent.
    vercelOidc(),
    // Open on localhost for `eve dev` and the REPL; ignored in production.
    localDev(),
    // This placeholder will not allow browser requests in production.
    // Replace it with your app's auth provider, like Auth.js or Clerk,
    // or use none() for a public demo.
    placeholderAuth(),
  ],
});
