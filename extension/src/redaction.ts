import type { Redactor } from '../../lib/relay/redactor';
import type { FieldDescriptor } from '../../lib/relay/sensitive-fields';

// The bridge's fail-closed wrappers around the run's redactor. Pure (no
// chrome.*), so they are unit-tested with a stub redactor.

export const REDACTION_FAILED = 'response redaction failed';
const COMMAND_FAILED = 'command failed';

export type Describe = (
  backendNodeId: number,
) => Promise<FieldDescriptor | undefined>;

/**
 * The redacted result of a CDP command. Throws REDACTION_FAILED (sent as an
 * error reply) when the redactor throws, rejects, or answers `null` for a
 * result that was not null: a forwarded command always has an object
 * result, so `null` means the redactor could not make it safe. The raw
 * result is never returned.
 */
export async function redactReply(
  redactor: Redactor,
  method: string,
  raw: unknown,
  describe?: Describe,
): Promise<unknown> {
  let out: unknown;
  try {
    out = await redactor.redactResult(method, raw, describe);
  } catch {
    throw new Error(REDACTION_FAILED);
  }
  if (out === null && raw !== null) throw new Error(REDACTION_FAILED);
  return out;
}

/** The redacted event params, or `null` (drop the event) on any failure. */
export function redactEventParams(
  redactor: Redactor,
  method: string,
  params: unknown,
): unknown {
  try {
    return redactor.redactEvent(method, params);
  } catch {
    return null;
  }
}

/**
 * An error message with known values scrubbed (page text can reach a CDP
 * error message). Falls back to a generic message when it cannot be.
 */
export function redactErrorMessage(
  redactor: Redactor,
  message: string,
): string {
  try {
    const out = redactor.redactEvent('Bridge.error', { message });
    const scrubbed =
      typeof out === 'object' && out !== null
        ? (out as { message?: unknown }).message
        : undefined;
    return typeof scrubbed === 'string' ? scrubbed : COMMAND_FAILED;
  } catch {
    return COMMAND_FAILED;
  }
}
