/**
 * Bridge from Kernel Browser Telemetry to OpenTelemetry.
 *
 * Kernel records the events that occur in the browser VM: console errors,
 * CDP connects and disconnects, captcha results, and OOM kills. This module
 * reads the events that match one agent-browser command. The span layer
 * attaches them to the trace, so a failed click shows the browser's own
 * evidence.
 *
 * Capture is set for each category when the browser is created (see
 * `getOrCreateBrowser`). The `network` category stays off because request
 * data contains applicant PII.
 */

import Kernel from '@onkernel/sdk';
import type {
  TimelineCollector,
  TimelineEvent,
} from '@/lib/observability/browser-telemetry';

// Create the client on first use. The SDK constructor fails when
// KERNEL_API_KEY is not set, and tests import the pure functions here.
let client: Kernel | undefined;
function kernelClient(): Kernel {
  client ??= new Kernel();
  return client;
}

/** The page controls payload strings. Keep them short and mark the cut. */
const MAX_ATTR_STRING = 500;

/** Keep span events small so they are easy to read. */
const MAX_ATTRS_PER_EVENT = 16;

/** The fields that all members of Kernel's telemetry event union share. */
interface KernelEvent {
  category: string;
  type: string;
  /** Unix microseconds. */
  ts: number;
  data?: Record<string, unknown>;
  truncated?: boolean;
}

/**
 * Map one Kernel event to a span event named `kernel.<type>`, with the
 * event's own timestamp and its primitive payload fields. Nested objects are
 * dropped: Kernel's durable timeline keeps the full record.
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
 * Make a collector that reads one Kernel browser session's telemetry events
 * for a time window.
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
      // Stop after 200 events. The span layer applies a lower limit.
      if (events.length >= 200) break;
    }
    return events;
  };
}
