// Secret redaction shared by the CLI transport and the relay browser path.
// Pure and dependency-free so it can sit in front of telemetry.

export const REDACTED = '[redacted]';

/** Raw and URL-encoded forms of each secret, longest first. */
export function redactionTargets(secrets: readonly string[]): string[] {
  const forms = new Set<string>();
  for (const secret of secrets) {
    if (!secret) continue;
    forms.add(secret);
    forms.add(encodeURIComponent(secret));
  }
  return [...forms].sort((a, b) => b.length - a.length);
}

export function redactText(text: string, targets: readonly string[]): string {
  let out = text;
  for (const target of targets) out = out.split(target).join(REDACTED);
  return out;
}

/** Redacts every string inside a JSON-like value without changing its shape. */
export function redactDeep(
  value: unknown,
  targets: readonly string[],
): unknown {
  if (typeof value === 'string') return redactText(value, targets);
  if (Array.isArray(value)) return value.map((v) => redactDeep(v, targets));
  if (value !== null && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value).map(([k, v]) => [
        redactText(k, targets),
        redactDeep(v, targets),
      ]),
    );
  }
  return value;
}
