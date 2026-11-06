// The secret shared by Next and the Eve process (`EVE_INTERNAL_SECRET`). Free
// of `server-only` and Node imports so the Eve bundle can use it. One length
// rule lives here: a secret shorter than the minimum is treated as unset by
// every consumer.

export const MIN_INTERNAL_SECRET_LENGTH = 32;

/** The secret, or undefined when it is unset, empty or too short. */
export function internalSecret(
  raw: string | undefined = process.env.EVE_INTERNAL_SECRET,
): string | undefined {
  return typeof raw === 'string' && raw.length >= MIN_INTERNAL_SECRET_LENGTH
    ? raw
    : undefined;
}

/** True when a secret is set but unusable. Reports nothing about its value. */
export function internalSecretMisconfigured(
  raw: string | undefined = process.env.EVE_INTERNAL_SECRET,
): boolean {
  return Boolean(raw) && internalSecret(raw) === undefined;
}
