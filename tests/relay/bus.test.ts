import { afterEach, describe, expect, it } from 'vitest';
import {
  type Bus,
  channelFor,
  createLocalBus,
  createRedisBus,
} from '../../relay/bus';

const redisUrl = process.env.RELAY_TEST_REDIS_URL;
if (!redisUrl) {
  console.info('Skipping Redis bus tests: RELAY_TEST_REDIS_URL is not set');
}

const settle = () => new Promise((resolve) => setTimeout(resolve, 50));

describe('channelFor', () => {
  it('builds a distinct channel per run and direction', () => {
    expect(channelFor('r1', 'to-ext')).not.toBe(channelFor('r1', 'from-ext'));
    expect(channelFor('r1', 'to-ext')).not.toBe(channelFor('r2', 'to-ext'));
    expect(channelFor('r1', 'to-agent')).toContain('r1');
  });
});

interface Opts {
  onError?: (error: Error) => void;
}

function busSuite(
  name: string,
  make: (opts?: Opts) => Promise<Bus>,
  enabled = true,
) {
  describe.skipIf(!enabled)(name, () => {
    const buses: Bus[] = [];
    const open = async (opts?: Opts) => {
      const bus = await make(opts);
      buses.push(bus);
      return bus;
    };
    afterEach(async () => {
      await Promise.all(buses.splice(0).map((b) => b.close()));
    });

    it('delivers to every subscriber on the same channel', async () => {
      const bus = await open();
      const a: string[] = [];
      const b: string[] = [];
      await bus.subscribe('c1', (m) => a.push(m));
      await bus.subscribe('c1', (m) => b.push(m));
      await bus.publish('c1', 'hello');
      await settle();
      expect(a).toEqual(['hello']);
      expect(b).toEqual(['hello']);
    });

    it('does not deliver to other channels', async () => {
      const bus = await open();
      const got: string[] = [];
      await bus.subscribe('c1', (m) => got.push(m));
      await bus.publish('c2', 'nope');
      await settle();
      expect(got).toEqual([]);
    });

    it('stops delivering after unsubscribe, leaving other subscribers', async () => {
      const bus = await open();
      const a: string[] = [];
      const b: string[] = [];
      const unsubA = await bus.subscribe('c1', (m) => a.push(m));
      await bus.subscribe('c1', (m) => b.push(m));
      await unsubA();
      await bus.publish('c1', 'x');
      await settle();
      expect(a).toEqual([]);
      expect(b).toEqual(['x']);
    });

    it('isolates a throwing subscriber and reports only the error', async () => {
      const errors: unknown[] = [];
      const bus = await open({ onError: (e) => errors.push(e) });
      const got: string[] = [];
      await bus.subscribe('c1', () => {
        throw new Error('boom');
      });
      await bus.subscribe('c1', (m) => got.push(m));
      await bus.publish('c1', 'secret-payload');
      await settle();
      expect(got).toEqual(['secret-payload']);
      expect(errors).toHaveLength(1);
      expect((errors[0] as Error).message).toBe('boom');
      expect(JSON.stringify(errors)).not.toContain('secret-payload');
    });

    it('tolerates a throwing subscriber with the default onError', async () => {
      const bus = await open();
      const got: string[] = [];
      await bus.subscribe('c1', () => {
        throw new Error('boom');
      });
      await bus.subscribe('c1', (m) => got.push(m));
      await expect(bus.publish('c1', 'x')).resolves.toBeUndefined();
      await settle();
      expect(got).toEqual(['x']);
    });

    it('resolves when close is called twice', async () => {
      const bus = await open();
      await expect(bus.close()).resolves.toBeUndefined();
      await expect(bus.close()).resolves.toBeUndefined();
    });

    it('rejects publish and subscribe after close', async () => {
      const bus = await open();
      await bus.close();
      await expect(bus.publish('c1', 'x')).rejects.toThrow(
        new Error('relay bus: closed'),
      );
      await expect(bus.subscribe('c1', () => {})).rejects.toThrow(
        new Error('relay bus: closed'),
      );
    });

    it('tolerates calling unsubscribe twice', async () => {
      const bus = await open();
      const got: string[] = [];
      const other: string[] = [];
      const unsub = await bus.subscribe('c1', (m) => got.push(m));
      await bus.subscribe('c1', (m) => other.push(m));
      await unsub();
      await expect(unsub()).resolves.toBeUndefined();
      await bus.publish('c1', 'x');
      await settle();
      expect(got).toEqual([]);
      expect(other).toEqual(['x']);
    });

    it('delivers nothing after the last handler unsubscribes', async () => {
      const bus = await open();
      const got: string[] = [];
      const unsub = await bus.subscribe('c1', (m) => got.push(m));
      await unsub();
      await bus.publish('c1', 'x');
      await settle();
      expect(got).toEqual([]);
    });

    it('delivers across bus instances sharing a backend', async () => {
      if (name === 'local bus') return;
      const pub = await open();
      const sub = await open();
      const got: string[] = [];
      await sub.subscribe('c1', (m) => got.push(m));
      await pub.publish('c1', 'cross');
      await settle();
      expect(got).toEqual(['cross']);
    });
  });
}

busSuite('local bus', async (opts) => createLocalBus(opts));
busSuite(
  'redis bus',
  async (opts) => createRedisBus(redisUrl as string, opts),
  Boolean(redisUrl),
);

describe('createRedisBus with an unreachable server', () => {
  it('rejects quickly with a message that omits the address', async () => {
    const started = Date.now();
    const error = await createRedisBus('redis://127.0.0.1:1', {
      connectTimeoutMs: 300,
    }).then(
      () => null,
      (e: Error) => e,
    );
    expect(Date.now() - started).toBeLessThan(1500);
    expect(error?.message).toBe('relay bus: could not connect to Redis');
    expect(error?.message).not.toContain('127.0.0.1');
    expect(error?.message).not.toContain(':1');
  });
});
