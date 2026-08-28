import { eveChannel } from "eve/channels/eve";
import {
  type AuthFn,
  isLoopbackRequest,
  localDev,
  placeholderAuth,
  vercelOidc,
} from "eve/channels/auth";

// Dev/eval only: on a loopback request (same trust boundary as localDev —
// reuses eve's own isLoopbackRequest gate), read the adapter's `x-eve-model`
// header and expose it as an auth attribute the dynamic model resolver in
// agent/agent.ts reads. Returns null for anything non-loopback or
// header-less, so the existing auth walk (vercelOidc/localDev/placeholderAuth)
// is unchanged for all other traffic. The header value is untrusted input —
// downstream it is only ever validated against the Vertex model allowlist in
// lib/ai/eve/model-map.ts, never used as a credential here.
const modelAttributeAuth: AuthFn<Request> = (request) => {
  const model = request.headers.get("x-eve-model");
  if (!model || !isLoopbackRequest(request)) return null;
  return {
    attributes: { eveModel: model },
    authenticator: "local-dev",
    principalId: "local-dev",
    principalType: "local-dev",
    subject: "local-dev",
  };
};

export default eveChannel({
  auth: [
    // Dev/eval only: lets the model picker override the session model via
    // header on loopback requests. See comment above.
    modelAttributeAuth,
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
