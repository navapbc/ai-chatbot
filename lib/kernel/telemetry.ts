/**
 * Kernel Browser Telemetry → OpenTelemetry bridge.
 *
 * Kernel records what happens *inside* the browser VM — console errors, CDP
 * connects/disconnects, captcha outcomes, OOM kills — on a durable timeline
 * keyed by browser session. This module reads the slice of that timeline
 * matching one agent-browser command so the span layer can attach it to the
 * trace: a failed click then carries the page's own evidence.
 *
 * Capture is opt-in per category at browser creation (see
 * `getOrCreateBrowser`). `network` stays off: request/response headers and
 * bodies carry applicant PII.
 */

import Kernel from '@onkernel/sdk';
import type {
  TimelineCollector,
  TimelineEvent,
} from '@/lib/observability/browser-telemetry';

// Lazy: the SDK constructor throws without KERNEL_API_KEY, and this module's
// pure mapping half must stay importable (tests, environments without Kernel).
let client: Kernel | undefined;
function kernelClient(): Kernel {
  client ??= new Kernel();
  return client;
}

/**
 * Payload strings are page-controlled (console text, URLs); keep them short
 * enough for a span event and mark the cut.
 */
const MAX_ATTR_STRING = 500;

/** Payload fields can be arbitrary; a span event should stay skimmable. */
const MAX_ATTRS_PER_EVENT = 16;

/** Shape shared by every member of Kernel's telemetry event union. */
interface KernelEvent {
  category: string;
  type: string;
  /** Unix microseconds. */
  ts: number;
  data?: Record<string, unknown>;
  truncated?: boolean;
}

/**
 * Map one Kernel event to a span event: `kernel.<type>` named, stamped with
 * the event's own timestamp, carrying the primitive payload fields. Nested
 * objects (call stacks, header maps) are dropped — the durable timeline keeps
 * the full record, the trace needs the headline.
 *
 * Exported for tests.
 */
export function toTimelineEvent(
  event: KernelEvent,
  seq: number,
): TimelineEvent {
  const attributes: Record<string, string | number | boolean> = {
    'kernel.category': event.category,
    'kernel.seq': seq,
  };
  if (event.truncated) attributes['kernel.truncated'] = true;

  let count = 0;
  for (const [key, value] of Object.entries(event.data ?? {})) {
    if (count >= MAX_ATTRS_PER_EVENT) break;
    if (typeof value === 'string') {
      attributes[`kernel.${key}`] =
        value.length > MAX_ATTR_STRING
          ? `${value.slice(0, MAX_ATTR_STRING)}…`
          : value;
      count++;
    } else if (typeof value === 'number' || typeof value === 'boolean') {
      attributes[`kernel.${key}`] = value;
      count++;
    }
  }

  return {
    name: `kernel.${event.type}`,
    timestamp: new Date(event.ts / 1000),
    attributes,
  };
}

/**
 * Build a collector that reads a Kernel browser session's telemetry events
 * for a time window. The SDK paginates transparently; the caller caps volume.
 */
export function kernelTimelineCollector(
  kernelSessionId: string,
): TimelineCollector {
  return async ({ sinceIso, untilIso }) => {
    const events: TimelineEvent[] = [];
    const page = await kernelClient().browsers.telemetry.events(
      kernelSessionId,
      {
        since: sinceIso,
        until: untilIso,
      },
    );
    for await (const item of page) {
      events.push(toTimelineEvent(item.event as KernelEvent, item.seq));
      // Bound pagination; the span layer trims further for legibility.
      if (events.length >= 200) break;
    }
    return events;
  };
}
