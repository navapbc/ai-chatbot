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

function busSuite(name: string, make: () => Promise<Bus>, enabled = true) {
  describe.skipIf(!enabled)(name, () => {
    const buses: Bus[] = [];
    const open = async () => {
      const bus = await make();
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

busSuite('local bus', async () => createLocalBus());
busSuite(
  'redis bus',
  async () => createRedisBus(redisUrl as string),
  Boolean(redisUrl),
);
