import { eveChannel } from "eve/channels/eve";
import {
  type AuthFn,
  isLoopbackRequest,
  localDev,
  placeholderAuth,
  vercelOidc,
} from "eve/channels/auth";

// On a loopback request only (same trust boundary as localDev — reuses eve's
// own isLoopbackRequest gate), expose three adapter headers as session auth
// attributes. Returns null for anything non-loopback or with none of them
// present, so the existing auth walk (vercelOidc/localDev/placeholderAuth) is
// unchanged for all other traffic.
//
//   x-eve-model        -> eveModel   dev/eval model override, read by the
//                                    dynamic model resolver in agent/agent.ts.
//                                    Untrusted: only ever validated against
//                                    the allowlist in lib/ai/eve/model-map.ts.
//   x-eve-relay-url    -> relayUrl   the browser relay for this session
//   x-eve-relay-token  -> relayToken the run's bearer token for that relay.
//                                    A credential: read only by the browser
//                                    tool (lib/relay/eve-relay-browser.ts),
//                                    never put in message text or logged.
const loopbackAttributeAuth: AuthFn<Request> = (request) => {
  const attributes: Record<string, string> = {};
  const headers: Array<[header: string, attribute: string]> = [
    ["x-eve-model", "eveModel"],
    ["x-eve-relay-url", "relayUrl"],
    ["x-eve-relay-token", "relayToken"],
  ];
  for (const [header, attribute] of headers) {
    const value = request.headers.get(header);
    if (value) attributes[attribute] = value;
  }
  if (Object.keys(attributes).length === 0 || !isLoopbackRequest(request)) {
    return null;
  }
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
    // relay credentials). See comment above.
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
