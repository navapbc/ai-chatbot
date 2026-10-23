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

export function createLocalBus(): Bus {
  const handlers = new Map<string, Set<(message: string) => void>>();
  return {
    async publish(channel, message) {
      // Copy so a handler that unsubscribes mid-delivery cannot skip others.
      for (const handler of [...(handlers.get(channel) ?? [])]) {
        handler(message);
      }
    },
    async subscribe(channel, handler) {
      // Wrap so subscribing the same function twice yields two registrations.
      const entry = (message: string) => handler(message);
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
      handlers.clear();
    },
  };
}

export async function createRedisBus(url: string): Promise<Bus> {
  const pub = createClient({ url });
  const sub = pub.duplicate();
  // Never let the URL (which may hold credentials) reach logs or errors.
  const onError = () => {};
  pub.on('error', onError);
  sub.on('error', onError);
  try {
    await pub.connect();
    await sub.connect();
  } catch {
    pub.destroy();
    sub.destroy();
    throw new Error('Failed to connect to the Redis bus');
  }
  return {
    async publish(channel, message) {
      await pub.publish(channel, message);
    },
    async subscribe(channel, handler) {
      const entry = (message: string) => handler(message);
      await sub.subscribe(channel, entry);
      return async () => {
        await sub.unsubscribe(channel, entry);
      };
    },
    async close() {
      await Promise.allSettled([pub.close(), sub.close()]);
    },
  };
}
