const MIN_SECRET_LENGTH = 32;

export type RelayEnvCheck = { ok: true } | { ok: false; message: string };

/** Validates startup env. Messages are generic and never echo env values. */
export function checkRelayEnv(
  env: Record<string, string | undefined>,
): RelayEnvCheck {
  const secret = env.RELAY_TOKEN_SECRET;
  if (!secret || secret.length < MIN_SECRET_LENGTH) {
    return {
      ok: false,
      message: 'relay: RELAY_TOKEN_SECRET is missing or too short',
    };
  }

  for (const name of [
    'RELAY_ALLOWED_HOST_SUFFIXES',
    'RELAY_ALLOWED_HOSTS',
  ] as const) {
    const value = env[name];
    if (value?.trim() && value.split(',').some((entry) => !entry.trim())) {
      return { ok: false, message: `relay: ${name} has an empty entry` };
    }
  }

  // Known gaps when more than one relay instance shares a Redis bus
  // (ledgered; the supported deployment is a single instance without Redis):
  // - origin and user (token sub) binding are per hub, so a wrong-origin or
  //   other-user extension on another hub can briefly receive commands;
  // - a mismatched ext-up (origin or sub) can close the legitimate agent;
  // - "busy" is per hub;
  // - an agent on a hub that never saw the run is never closed;
  // - closed runs leave no tombstone, so a token can be replayed within TTL;
  // - ext-down/ext-up can reorder across publishers;
  // - pending sockets on another hub miss a remote closeRun.
  if (env.REDIS_URL && env.RELAY_MULTI_INSTANCE_ACKNOWLEDGED !== '1') {
    return {
      ok: false,
      message:
        'relay: REDIS_URL requires RELAY_MULTI_INSTANCE_ACKNOWLEDGED=1 (cross-instance origin binding is not enforced yet; see relay/main.ts)',
    };
  }
  return { ok: true };
}
