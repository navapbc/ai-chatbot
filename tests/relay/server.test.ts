import type { Server } from 'node:http';
import type { AddressInfo } from 'node:net';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import WebSocket from 'ws';
import { mintRelayToken } from '../../lib/relay/token';
import { createLocalBus } from '../../relay/bus';
import { createRelayServer } from '../../relay/server';

const SECRET = 'test-secret-that-is-at-least-32-characters-long';
const ORIGIN = 'https://benefits.example.gov';
const EXT_ID = 'abcdefghijklmnopabcdefghijklmnop';
const EXT_ORIGIN = `chrome-extension://${EXT_ID}`;
const RUN = 'run-1';

let server: Server;
let port: number;
let logs: Array<Record<string, unknown>>;
let tokens: string[];
const sockets: WebSocket[] = [];

async function start(extra: Record<string, unknown> = {}) {
  logs = [];
  const bus = createLocalBus();
  server = createRelayServer({
    secret: SECRET,
    allowedHostSuffixes: ['.relay.example.test'],
    allowedHosts: ['127.0.0.1', 'localhost'],
    extensionId: EXT_ID,
    bus,
    log: (line) => logs.push(line),
    ...extra,
  });
  server.on('close', () => void bus.close());
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  port = (server.address() as AddressInfo).port;
}

const token = (
  role: 'extension' | 'agent',
  opts: { secret?: string; now?: number } = {},
) => {
  const t = mintRelayToken(
    { runId: RUN, role, origin: ORIGIN },
    opts.secret ?? SECRET,
    opts.now === undefined ? {} : { now: opts.now },
  );
  tokens.push(t);
  return t;
};

const url = (path: string, tok?: string) =>
  `ws://127.0.0.1:${port}${path}${tok === undefined ? '' : `?token=${tok}`}`;

function connect(
  target: string,
  headers: Record<string, string> = {},
): Promise<WebSocket> {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(target, { headers });
    sockets.push(ws);
    ws.once('open', () => resolve(ws));
    ws.once('error', reject);
  });
}

const rejected = (target: string, headers: Record<string, string> = {}) =>
  expect(connect(target, headers)).rejects.toBeDefined();

const nextMessage = (ws: WebSocket) =>
  new Promise<Record<string, unknown>>((resolve) =>
    ws.once('message', (d) => resolve(JSON.parse(d.toString()))),
  );

const closed = (ws: WebSocket) =>
  new Promise<{ code: number; reason: string }>((resolve) =>
    ws.once('close', (code, reason) =>
      resolve({ code, reason: reason.toString() }),
    ),
  );

async function pair() {
  const ext = await connect(url('/extension', token('extension')), {
    origin: EXT_ORIGIN,
  });
  ext.send(
    JSON.stringify({ type: 'register', runId: RUN, tabId: 7, origin: ORIGIN }),
  );
  const agent = await connect(url('/cdp', token('agent')));
  return { ext, agent };
}

beforeEach(() => {
  tokens = [];
});

afterEach(async () => {
  for (const ws of sockets.splice(0)) ws.terminate();
  await new Promise<void>((resolve) => server.close(() => resolve()));
});

describe('relay server', () => {
  it('answers /healthz without a token', async () => {
    await start();
    const res = await fetch(`http://127.0.0.1:${port}/healthz`);
    expect(res.status).toBe(200);
  });

  it('answers 421 for a bad Host on HTTP, including /healthz', async () => {
    await start();
    const res = await new Promise<number>((resolve, reject) => {
      import('node:http').then(({ request }) => {
        const req = request(
          {
            host: '127.0.0.1',
            port,
            path: '/healthz',
            headers: { host: 'evil.test' },
          },
          (r) => {
            r.resume();
            resolve(r.statusCode ?? 0);
          },
        );
        req.on('error', reject);
        req.end();
      });
    });
    expect(res).toBe(421);
  });

  it('accepts a Host that matches an allowed suffix', async () => {
    await start();
    const ws = await connect(url('/cdp', token('agent')), {
      host: 'relay-1.relay.example.test',
    });
    expect(ws.readyState).toBe(WebSocket.OPEN);
  });

  describe('rejects the upgrade', () => {
    beforeEach(() => start());

    it('without a token', async () => {
      await rejected(url('/cdp'));
    });
    it('with a token signed by another secret', async () => {
      await rejected(url('/cdp', token('agent', { secret: `${SECRET}-x` })));
    });
    it('with an expired token', async () => {
      await rejected(url('/cdp', token('agent', { now: 1000 })));
    });
    it('with the wrong role for /cdp', async () => {
      await rejected(url('/cdp', token('extension')));
    });
    it('with the wrong role for /extension', async () => {
      await rejected(url('/extension', token('agent')), { origin: EXT_ORIGIN });
    });
    it('for a web Origin on /cdp', async () => {
      await rejected(url('/cdp', token('agent')), { origin: ORIGIN });
    });
    it('for /extension without an Origin', async () => {
      await rejected(url('/extension', token('extension')));
    });
    it('for /extension with a web Origin', async () => {
      await rejected(url('/extension', token('extension')), { origin: ORIGIN });
    });
    it('for a wrong extension id', async () => {
      await rejected(url('/extension', token('extension')), {
        origin: 'chrome-extension://otherotherotherotherotherother12',
      });
    });
    it('for a bad Host', async () => {
      await rejected(url('/cdp', token('agent')), { host: 'evil.test' });
    });
    it('for an unknown path', async () => {
      await rejected(url('/other', token('agent')));
    });
  });

  it('accepts any chrome-extension Origin when no extension id is set', async () => {
    await start({ extensionId: undefined });
    const ws = await connect(url('/extension', token('extension')), {
      origin: 'chrome-extension://anything',
    });
    expect(ws.readyState).toBe(WebSocket.OPEN);
  });

  it('pairs an extension and an agent into one run', async () => {
    await start();
    const { ext, agent } = await pair();
    const eventAtAgent = nextMessage(agent);
    ext.send(
      JSON.stringify({
        type: 'event',
        sessionId: 's1',
        method: 'Page.loadEventFired',
        params: { ok: true },
      }),
    );
    expect(await eventAtAgent).toMatchObject({
      sessionId: 's1',
      method: 'Page.loadEventFired',
    });
  });

  it('closes a second agent with 1008 run busy', async () => {
    await start();
    await pair();
    const second = await connect(url('/cdp', token('agent')));
    expect(await closed(second)).toEqual({ code: 1008, reason: 'run busy' });
  });

  it('passes a 5 MB message from the extension to the agent intact', async () => {
    await start();
    const { ext, agent } = await pair();
    const blob = 'x'.repeat(5 * 1024 * 1024);
    const eventAtAgent = nextMessage(agent);
    ext.send(
      JSON.stringify({
        type: 'event',
        sessionId: 's1',
        method: 'Page.screencastFrame',
        params: { data: blob },
      }),
    );
    const received = await eventAtAgent;
    expect((received.params as { data: string }).data).toBe(blob);
  });

  it('ignores binary frames', async () => {
    await start();
    const { ext, agent } = await pair();
    const eventAtAgent = nextMessage(agent);
    ext.send(Buffer.from('{"type":"tabClosed"}'), { binary: true });
    ext.send(JSON.stringify({ type: 'event', method: 'Page.x', params: {} }));
    expect(await eventAtAgent).toMatchObject({ method: 'Page.x' });
  });

  it('logs each forwarded method once when recordMethods is on', async () => {
    await start({ recordMethods: true });
    const { ext, agent } = await pair();
    const askAndAnswer = async (id: number) => {
      const atExt = nextMessage(ext);
      const atAgent = nextMessage(agent);
      agent.send(
        JSON.stringify({
          id,
          sessionId: 's1',
          method: 'Runtime.evaluate',
          params: { expression: '1' },
        }),
      );
      const cmd = await atExt;
      ext.send(JSON.stringify({ type: 'reply', id: cmd.id, result: {} }));
      await atAgent;
    };
    await askAndAnswer(1);
    await askAndAnswer(2);
    const recorded = logs.filter((l) => l.event === 'method');
    expect(recorded).toEqual([{ event: 'method', method: 'Runtime.evaluate' }]);
  });

  it('does not emit method lines when recordMethods is off', async () => {
    await start();
    await pair();
    expect(logs.some((l) => l.event === 'method')).toBe(false);
  });

  it('never logs a token or the URL, for accepted and rejected connections', async () => {
    await start({ recordMethods: true });
    const { ext } = await pair();
    await rejected(url('/cdp', token('extension')));
    await rejected(url('/cdp', token('agent', { secret: `${SECRET}-x` })));
    await rejected(url('/cdp', token('agent')), { origin: ORIGIN });
    await rejected(url('/cdp', token('agent')), { host: 'evil.test' });
    await rejected(url('/nope', token('agent')));
    ext.send('not json');
    await connect(url('/cdp', token('agent'))).then((ws) => closed(ws));
    await new Promise((r) => setTimeout(r, 50));

    const output = JSON.stringify(logs);
    expect(logs.length).toBeGreaterThan(0);
    for (const t of tokens) {
      expect(output).not.toContain(t);
      expect(output).not.toContain(t.split('.')[0]);
    }
    expect(output).not.toContain('token=');
  });

  describe('token in the path', () => {
    it.each([
      (t: string) => `/cdp%3Ftoken=${t}`,
      (t: string) => `/cdp&token=${t}`,
      (t: string) => `/cdp;token=${t}`,
      (t: string) => `/cdp/${t}`,
      (t: string) => `/extension/${t}`,
    ])('is rejected as path and never logged (%#)', async (make) => {
      await start();
      const t = token('agent');
      await rejected(`ws://127.0.0.1:${port}${make(t)}`);
      await new Promise((r) => setTimeout(r, 20));
      const output = JSON.stringify(logs);
      expect(logs).toContainEqual({
        event: 'upgrade-rejected',
        path: 'other',
        reason: 'path',
      });
      for (let i = 0; i + 8 <= t.length; i += 1) {
        expect(output).not.toContain(t.slice(i, i + 8));
      }
    });
  });

  it('logs the reason for each kind of rejection', async () => {
    await start();
    const reasonOf = async (
      target: string,
      headers: Record<string, string> = {},
    ) => {
      logs.length = 0;
      await rejected(target, headers);
      await new Promise((r) => setTimeout(r, 20));
      return logs.find((l) => l.event === 'upgrade-rejected')?.reason;
    };
    expect(await reasonOf(url('/cdp'))).toBe('no-token');
    expect(await reasonOf(url('/cdp', 'garbage'))).toBe('token');
    expect(await reasonOf(url('/cdp', token('extension')))).toBe('role');
    expect(
      await reasonOf(url('/cdp', token('agent')), { host: 'evil.test' }),
    ).toBe('host');
    expect(
      await reasonOf(url('/cdp', token('agent')), { origin: ORIGIN }),
    ).toBe('origin');
    expect(await reasonOf(url('/nope', token('agent')))).toBe('path');
  });

  describe('Host allowlist', () => {
    const hostStatus = async (host: string | undefined) => {
      const { request } = await import('node:http');
      return new Promise<number>((resolve, reject) => {
        const req = request(
          {
            host: '127.0.0.1',
            port,
            path: '/healthz',
            headers: host === undefined ? {} : { host },
            setHost: false,
          },
          (r) => {
            r.resume();
            resolve(r.statusCode ?? 0);
          },
        );
        req.on('error', reject);
        req.end();
      });
    };

    it.each(['run.app', '.run.app'])(
      'matches suffix %s on a label boundary',
      async (suffix) => {
        await start({ allowedHostSuffixes: [suffix], allowedHosts: [] });
        for (const ok of ['x.run.app', 'run.app', 'X.Run.App:8080']) {
          expect(await hostStatus(ok)).toBe(200);
        }
        for (const bad of [
          'evilrun.app',
          'x.run.app.evil.com',
          'x.run.app@evil.com',
          'x.run.app:80@evil.com',
          'x.run.app evil',
          'x.run.app.',
        ]) {
          expect(await hostStatus(bad)).toBe(421);
        }
      },
    );

    it('rejects a missing Host', async () => {
      await start({ allowedHostSuffixes: ['run.app'], allowedHosts: [] });
      // Node answers 400 itself for HTTP/1.1 without Host; never 200.
      expect([400, 421]).toContain(await hostStatus(undefined));
    });
  });
});
