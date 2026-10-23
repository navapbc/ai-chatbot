import { createClient } from 'redis';

export interface Bus {
  publish(channel: string, message: string): Promise<void>;
  subscribe(
    channel: string,
    handler: (message: string) => void,
  ): Promise<() => Promise<void>>;
  close(): Promise<void>;
}

export type Direction = 'to-ext' | 'from-ext' | 'to-agent';

export const channelFor = (runId: string, direction: Direction): string =>
  `relay:${runId}:${direction}`;

export interface BusOptions {
  /**
   * Called with the error a subscriber threw. It never receives the message
   * payload, which can hold tokens. The default is a deliberate no-op: the
   * bus must not log on its own, and a bad subscriber must not break others.
   */
  onError?: (error: Error) => void;
}

export interface RedisBusOptions extends BusOptions {
  connectTimeoutMs?: number;
}

const CLOSED_MESSAGE = 'relay bus: closed';
const CONNECT_FAILED_MESSAGE = 'relay bus: could not connect to Redis';

function isolate(
  handler: (message: string) => void,
  onError: (error: Error) => void,
): (message: string) => void {
  return (message) => {
    try {
      handler(message);
    } catch (error) {
      try {
        onError(error instanceof Error ? error : new Error('handler failed'));
      } catch {
        // An onError that throws must not break delivery either.
      }
    }
  };
}

export function createLocalBus(opts: BusOptions = {}): Bus {
  const onError = opts.onError ?? (() => {});
  const handlers = new Map<string, Set<(message: string) => void>>();
  let closed = false;
  return {
    async publish(channel, message) {
      if (closed) throw new Error(CLOSED_MESSAGE);
      // Copy so a handler that unsubscribes mid-delivery cannot skip others.
      for (const handler of [...(handlers.get(channel) ?? [])]) {
        handler(message);
      }
    },
    async subscribe(channel, handler) {
      if (closed) throw new Error(CLOSED_MESSAGE);
      // Wrap so subscribing the same function twice yields two registrations.
      const entry = isolate(handler, onError);
      let set = handlers.get(channel);
      if (!set) {
        set = new Set();
        handlers.set(channel, set);
      }
      set.add(entry);
      return async () => {
        const current = handlers.get(channel);
        current?.delete(entry);
        if (current?.size === 0) handlers.delete(channel);
      };
    },
    async close() {
      closed = true;
      handlers.clear();
    },
  };
}

export async function createRedisBus(
  url: string,
  opts: RedisBusOptions = {},
): Promise<Bus> {
  const onError = opts.onError ?? (() => {});
  const timeoutMs = opts.connectTimeoutMs ?? 5000;
  const pub = createClient({ url });
  const sub = pub.duplicate();
  // Never let the URL (which may hold credentials) reach logs or errors.
  const ignore = () => {};
  pub.on('error', ignore);
  sub.on('error', ignore);

  let timer: ReturnType<typeof setTimeout> | undefined;
  const connecting = Promise.all([pub.connect(), sub.connect()]);
  connecting.catch(ignore); // a late rejection after a timeout is expected
  try {
    await Promise.race([
      connecting,
      new Promise<never>((_, reject) => {
        timer = setTimeout(() => reject(new Error('timeout')), timeoutMs);
      }),
    ]);
  } catch {
    for (const client of [pub, sub]) {
      try {
        client.destroy();
      } catch {
        // Already torn down.
      }
    }
    throw new Error(CONNECT_FAILED_MESSAGE);
  } finally {
    clearTimeout(timer);
  }

  let closed = false;
  return {
    async publish(channel, message) {
      if (closed) throw new Error(CLOSED_MESSAGE);
      await pub.publish(channel, message);
    },
    async subscribe(channel, handler) {
      if (closed) throw new Error(CLOSED_MESSAGE);
      const entry = isolate(handler, onError);
      await sub.subscribe(channel, entry);
      return async () => {
        if (closed) return;
        await sub.unsubscribe(channel, entry);
      };
    },
    async close() {
      if (closed) return;
      closed = true;
      await Promise.allSettled([pub.close(), sub.close()]);
    },
  };
}
