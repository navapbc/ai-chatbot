import { afterEach, describe, expect, it, vi } from 'vitest';
import { type Bus, createLocalBus } from '../../relay/bus';
import { type PeerSocket, RunHub } from '../../relay/hub';

const ORIGIN = 'https://benefits.example.gov';
const RUN = 'run-1';

class FakeSocket implements PeerSocket {
  sent: string[] = [];
  closed: { code?: number; reason?: string } | null = null;
  private messageListeners: Array<(data: string) => void> = [];
  private closeListeners: Array<() => void> = [];

  send(data: string) {
    if (this.closed) throw new Error('socket is closed');
    this.sent.push(data);
  }

  close(code?: number, reason?: string) {
    if (this.closed) return;
    this.closed = { code, reason };
    for (const cb of this.closeListeners) cb();
  }

  on(event: 'message', cb: (data: string) => void): void;
  on(event: 'close', cb: () => void): void;
  on(event: 'message' | 'close', cb: (data: string) => void) {
    if (event === 'message') this.messageListeners.push(cb);
    else this.closeListeners.push(cb as () => void);
  }

  receive(message: unknown) {
    const data =
      typeof message === 'string' ? message : JSON.stringify(message);
    for (const cb of this.messageListeners) cb(data);
  }

  messages(): Array<Record<string, unknown>> {
    return this.sent.map((s) => JSON.parse(s));
  }
}

const buses: Bus[] = [];
afterEach(async () => {
  for (const bus of buses.splice(0)) await bus.close();
});

function makeHub(
  opts: { bus?: Bus; commandTimeoutMs?: number; resumeWindowMs?: number } = {},
) {
  const bus = opts.bus ?? createLocalBus();
  if (!opts.bus) buses.push(bus);
  const logs: Array<Record<string, unknown>> = [];
  const hub = new RunHub({
    bus,
    commandTimeoutMs: opts.commandTimeoutMs,
    resumeWindowMs: opts.resumeWindowMs,
    log: (line) => logs.push(line),
  });
  return { hub, bus, logs };
}

// Answers whatever the hub sends the fake extension, after `onCommand` decides.
function answerExtension(
  ext: FakeSocket,
  onCommand: (cmd: Record<string, unknown>) => unknown,
) {
  let seen = 0;
  return () => {
    const msgs = ext.messages();
    for (const cmd of msgs.slice(seen)) {
      ext.receive({ type: 'reply', id: cmd.id, result: onCommand(cmd) });
    }
    seen = msgs.length;
  };
}

const lastReply = (agent: FakeSocket, id: number) =>
  agent.messages().find((m) => m.id === id);

describe('RunHub', () => {
  it('forwards a session-scoped command to the extension after the policy check', async () => {
    const { hub } = makeHub();
    const ext = new FakeSocket();
    const agent = new FakeSocket();
    hub.attachExtension(RUN, ORIGIN, ext);
    expect(hub.attachAgent(RUN, ORIGIN, agent)).toBe('ok');

    agent.receive({
      id: 1,
      sessionId: 'tab-7',
      method: 'Runtime.evaluate',
      params: { expression: '1' },
    });

    await vi.waitFor(() => expect(ext.sent).toHaveLength(1));
    const cmd = ext.messages()[0];
    expect(cmd).toMatchObject({
      type: 'cdp',
      sessionId: 'tab-7',
      method: 'Runtime.evaluate',
      params: { expression: '1' },
    });
    expect(cmd.id).toMatch(/^[^:]+:\d+$/);

    ext.receive({ type: 'reply', id: cmd.id, result: { value: 1 } });
    await vi.waitFor(() =>
      expect(lastReply(agent, 1)).toEqual({
        id: 1,
        sessionId: 'tab-7',
        result: { value: 1 },
      }),
    );
  });

  it('answers a denied command with a CDP error and never forwards it', async () => {
    const { hub } = makeHub();
    const ext = new FakeSocket();
    const agent = new FakeSocket();
    hub.attachExtension(RUN, ORIGIN, ext);
    hub.attachAgent(RUN, ORIGIN, agent);

    agent.receive({
      id: 2,
      sessionId: 'tab-7',
      method: 'Network.getAllCookies',
    });
    agent.receive({
      id: 3,
      sessionId: 'tab-7',
      method: 'Page.navigate',
      params: { url: 'https://evil.example.com/' },
    });
    agent.receive({
      id: 4,
      method: 'Target.createTarget',
      params: { url: 'https://evil.example.com/' },
    });

    await vi.waitFor(() => expect(agent.sent).toHaveLength(3));
    for (const id of [2, 3, 4]) {
      const reply = lastReply(agent, id);
      expect(reply?.result).toBeUndefined();
      expect(reply?.error).toMatchObject({ code: -32000 });
    }
    expect(lastReply(agent, 2)?.sessionId).toBe('tab-7');
    expect(ext.sent).toHaveLength(0);
  });

  it('refuses a session-scoped Target.createTarget even with an in-scope url', async () => {
    const { hub } = makeHub();
    const ext = new FakeSocket();
    const agent = new FakeSocket();
    hub.attachExtension(RUN, ORIGIN, ext);
    hub.attachAgent(RUN, ORIGIN, agent);

    agent.receive({
      id: 19,
      sessionId: 'tab-7',
      method: 'Target.createTarget',
      params: { url: `${ORIGIN}/next` },
    });
    await vi.waitFor(() =>
      expect(lastReply(agent, 19)?.error).toMatchObject({ code: -32000 }),
    );
    expect(ext.sent).toHaveLength(0);
  });

  it('returns only the targets the extension listed for Target.getTargets', async () => {
    const { hub } = makeHub();
    const ext = new FakeSocket();
    const agent = new FakeSocket();
    hub.attachExtension(RUN, ORIGIN, ext);
    hub.attachAgent(RUN, ORIGIN, agent);
    ext.receive({ type: 'register', runId: RUN, tabId: 7, origin: ORIGIN });

    const registered = {
      targetId: 'T7',
      type: 'page',
      title: 'Form',
      url: `${ORIGIN}/apply`,
      attached: false,
      canAccessOpener: false,
    };
    const answer = answerExtension(ext, (cmd) => {
      expect(cmd.type).toBe('listTargets');
      return { targetInfos: [registered, null, { targetId: 42 }] };
    });

    agent.receive({ id: 5, method: 'Target.getTargets' });
    await vi.waitFor(() => expect(ext.sent).toHaveLength(1));
    answer();
    await vi.waitFor(() =>
      expect(lastReply(agent, 5)).toEqual({
        id: 5,
        result: { targetInfos: [registered] },
      }),
    );
  });

  it('attaches only to a target the extension listed and emits attachedToTarget', async () => {
    const { hub } = makeHub();
    const ext = new FakeSocket();
    const agent = new FakeSocket();
    hub.attachExtension(RUN, ORIGIN, ext);
    hub.attachAgent(RUN, ORIGIN, agent);

    agent.receive({
      id: 6,
      method: 'Target.attachToTarget',
      params: { targetId: 'T7', flatten: true },
    });
    await vi.waitFor(() => expect(lastReply(agent, 6)?.error).toBeDefined());
    expect(ext.sent).toHaveLength(0);

    const targetInfo = { targetId: 'T7', type: 'page', url: `${ORIGIN}/` };
    const answer = answerExtension(ext, (cmd) =>
      cmd.type === 'listTargets'
        ? { targetInfos: [targetInfo] }
        : { sessionId: 'tab-7', targetInfo },
    );
    agent.receive({ id: 7, method: 'Target.getTargets' });
    await vi.waitFor(() => expect(ext.sent).toHaveLength(1));
    answer();
    await vi.waitFor(() => expect(lastReply(agent, 7)).toBeDefined());

    agent.receive({
      id: 8,
      method: 'Target.attachToTarget',
      params: { targetId: 'T7', flatten: true },
    });
    await vi.waitFor(() => expect(ext.sent).toHaveLength(2));
    expect(ext.messages()[1]).toMatchObject({ type: 'attach', targetId: 'T7' });
    answer();
    await vi.waitFor(() =>
      expect(
        agent.messages().find((m) => m.method === 'Target.attachedToTarget'),
      ).toEqual({
        method: 'Target.attachedToTarget',
        params: {
          sessionId: 'tab-7',
          targetInfo: { ...targetInfo, attached: true },
          waitingForDebugger: false,
        },
      }),
    );
    expect(lastReply(agent, 8)).toEqual({
      id: 8,
      result: { sessionId: 'tab-7' },
    });
  });

  it('answers unknown browser-level methods with an empty result', async () => {
    const { hub, logs } = makeHub();
    const ext = new FakeSocket();
    const agent = new FakeSocket();
    hub.attachExtension(RUN, ORIGIN, ext);
    hub.attachAgent(RUN, ORIGIN, agent);

    agent.receive({
      id: 9,
      method: 'Target.setDiscoverTargets',
      params: { discover: true },
    });
    await vi.waitFor(() =>
      expect(lastReply(agent, 9)).toEqual({ id: 9, result: {} }),
    );
    expect(ext.sent).toHaveLength(0);
    expect(logs).toContainEqual(
      expect.objectContaining({ method: 'Target.setDiscoverTargets' }),
    );
  });

  it("forwards the extension's events to the agent socket", async () => {
    const { hub } = makeHub();
    const ext = new FakeSocket();
    const agent = new FakeSocket();
    hub.attachExtension(RUN, ORIGIN, ext);
    hub.attachAgent(RUN, ORIGIN, agent);

    ext.receive({
      type: 'event',
      sessionId: 'tab-7',
      method: 'Page.loadEventFired',
      params: { timestamp: 1 },
    });
    await vi.waitFor(() =>
      expect(agent.messages()).toContainEqual({
        sessionId: 'tab-7',
        method: 'Page.loadEventFired',
        params: { timestamp: 1 },
      }),
    );
  });

  it("refuses a second agent socket for the same run with 'busy'", () => {
    const { hub } = makeHub();
    expect(hub.attachAgent(RUN, ORIGIN, new FakeSocket())).toBe('ok');
    expect(hub.attachAgent(RUN, ORIGIN, new FakeSocket())).toBe('busy');
    expect(hub.attachAgent('run-2', ORIGIN, new FakeSocket())).toBe('ok');
  });

  it('frees the agent slot when the agent socket closes', () => {
    const { hub } = makeHub();
    const first = new FakeSocket();
    hub.attachAgent(RUN, ORIGIN, first);
    first.close();
    expect(hub.attachAgent(RUN, ORIGIN, new FakeSocket())).toBe('ok');
  });

  it('rejects an unanswered command with "extension did not answer"', async () => {
    const { hub } = makeHub({ commandTimeoutMs: 50 });
    const ext = new FakeSocket();
    const agent = new FakeSocket();
    hub.attachExtension(RUN, ORIGIN, ext);
    hub.attachAgent(RUN, ORIGIN, agent);

    agent.receive({ id: 10, sessionId: 'tab-7', method: 'Page.enable' });
    await vi.waitFor(() =>
      expect(lastReply(agent, 10)?.error).toMatchObject({
        message: 'extension did not answer',
      }),
    );
  });

  it('rejects pending commands at once when the extension socket closes', async () => {
    const { hub } = makeHub({ commandTimeoutMs: 60_000 });
    const ext = new FakeSocket();
    const agent = new FakeSocket();
    hub.attachExtension(RUN, ORIGIN, ext);
    hub.attachAgent(RUN, ORIGIN, agent);

    agent.receive({ id: 11, sessionId: 'tab-7', method: 'Page.enable' });
    await vi.waitFor(() => expect(ext.sent).toHaveLength(1));
    ext.close();
    await vi.waitFor(() =>
      expect(lastReply(agent, 11)?.error).toMatchObject({
        message: 'tab or extension disconnected',
      }),
    );
    expect(agent.closed).toBeNull();
    hub.closeRun(RUN, 'test over');
  });

  it('routes an agent on one hub to an extension on another hub', async () => {
    const bus = createLocalBus();
    buses.push(bus);
    const a = makeHub({ bus });
    const b = makeHub({ bus });
    const ext = new FakeSocket();
    const agent = new FakeSocket();
    b.hub.attachExtension(RUN, ORIGIN, ext);
    a.hub.attachAgent(RUN, ORIGIN, agent);

    agent.receive({ id: 12, sessionId: 'tab-7', method: 'DOM.enable' });
    await vi.waitFor(() => expect(ext.sent).toHaveLength(1));
    const cmd = ext.messages()[0];
    ext.receive({ type: 'reply', id: cmd.id, result: { ok: true } });
    ext.receive({ type: 'event', method: 'Page.frameNavigated', params: {} });

    await vi.waitFor(() => {
      expect(lastReply(agent, 12)).toEqual({
        id: 12,
        sessionId: 'tab-7',
        result: { ok: true },
      });
      expect(agent.messages()).toContainEqual({
        method: 'Page.frameNavigated',
        params: {},
      });
    });
  });

  it('holds the run while the extension is away and resumes on reattach', async () => {
    const { hub } = makeHub({ resumeWindowMs: 60_000 });
    const ext = new FakeSocket();
    const agent = new FakeSocket();
    hub.attachExtension(RUN, ORIGIN, ext);
    hub.attachAgent(RUN, ORIGIN, agent);
    ext.close();

    agent.receive({ id: 13, sessionId: 'tab-7', method: 'Page.enable' });
    await new Promise((resolve) => setTimeout(resolve, 30));
    expect(agent.closed).toBeNull();
    expect(lastReply(agent, 13)).toBeUndefined();

    const returned = new FakeSocket();
    hub.attachExtension(RUN, ORIGIN, returned);
    await vi.waitFor(() => expect(returned.sent).toHaveLength(1));
    const cmd = returned.messages()[0];
    expect(cmd).toMatchObject({ type: 'cdp', method: 'Page.enable' });
    returned.receive({ type: 'reply', id: cmd.id, result: {} });
    await vi.waitFor(() =>
      expect(lastReply(agent, 13)).toEqual({
        id: 13,
        sessionId: 'tab-7',
        result: {},
      }),
    );
    expect(agent.closed).toBeNull();
  });

  it('closes the agent socket when the extension does not return in time', async () => {
    const { hub } = makeHub({ resumeWindowMs: 50 });
    const ext = new FakeSocket();
    const agent = new FakeSocket();
    hub.attachExtension(RUN, ORIGIN, ext);
    hub.attachAgent(RUN, ORIGIN, agent);
    ext.close();

    agent.receive({ id: 14, sessionId: 'tab-7', method: 'Page.enable' });
    await vi.waitFor(() =>
      expect(agent.closed?.reason).toBe('extension did not return'),
    );
  });

  it('makes an agent that connects inside the resume window wait too', async () => {
    const { hub } = makeHub({ resumeWindowMs: 60_000 });
    const ext = new FakeSocket();
    hub.attachExtension(RUN, ORIGIN, ext);
    const first = new FakeSocket();
    hub.attachAgent(RUN, ORIGIN, first);
    ext.close();
    first.close();

    const agent = new FakeSocket();
    expect(hub.attachAgent(RUN, ORIGIN, agent)).toBe('ok');
    agent.receive({ id: 20, sessionId: 'tab-7', method: 'Page.enable' });
    await new Promise((resolve) => setTimeout(resolve, 30));
    expect(lastReply(agent, 20)).toBeUndefined();

    const returned = new FakeSocket();
    hub.attachExtension(RUN, ORIGIN, returned);
    await vi.waitFor(() => expect(returned.sent).toHaveLength(1));
    returned.receive({
      type: 'reply',
      id: returned.messages()[0].id,
      result: {},
    });
    await vi.waitFor(() =>
      expect(lastReply(agent, 20)).toEqual({
        id: 20,
        sessionId: 'tab-7',
        result: {},
      }),
    );
  });

  it('closes the earlier extension socket when a new one attaches', async () => {
    const { hub } = makeHub();
    const first = new FakeSocket();
    const second = new FakeSocket();
    const agent = new FakeSocket();
    hub.attachExtension(RUN, ORIGIN, first);
    hub.attachAgent(RUN, ORIGIN, agent);
    hub.attachExtension(RUN, ORIGIN, second);
    expect(first.closed).not.toBeNull();

    agent.receive({ id: 15, sessionId: 'tab-7', method: 'Page.enable' });
    await vi.waitFor(() => expect(second.sent).toHaveLength(1));
    expect(first.sent).toHaveLength(0);
    expect(agent.closed).toBeNull();
  });

  it('closes an extension that registers for another run or origin', () => {
    const { hub } = makeHub();
    const wrongRun = new FakeSocket();
    hub.attachExtension(RUN, ORIGIN, wrongRun);
    wrongRun.receive({
      type: 'register',
      runId: 'other',
      tabId: 1,
      origin: ORIGIN,
    });
    expect(wrongRun.closed).not.toBeNull();

    const wrongOrigin = new FakeSocket();
    hub.attachExtension(RUN, ORIGIN, wrongOrigin);
    wrongOrigin.receive({
      type: 'register',
      runId: RUN,
      tabId: 1,
      origin: 'https://evil.example.com',
    });
    expect(wrongOrigin.closed).not.toBeNull();
    hub.closeRun(RUN, 'test over');
  });

  it('closes both sockets on closeRun and on tabClosed', async () => {
    const { hub } = makeHub();
    const ext = new FakeSocket();
    const agent = new FakeSocket();
    hub.attachExtension(RUN, ORIGIN, ext);
    hub.attachAgent(RUN, ORIGIN, agent);
    hub.closeRun(RUN, 'run finished');
    expect(ext.closed?.reason).toBe('run finished');
    expect(agent.closed?.reason).toBe('run finished');

    const ext2 = new FakeSocket();
    const agent2 = new FakeSocket();
    hub.attachExtension('run-2', ORIGIN, ext2);
    hub.attachAgent('run-2', ORIGIN, agent2);
    ext2.receive({ type: 'tabClosed' });
    await vi.waitFor(() => expect(agent2.closed?.reason).toBe('tab closed'));
    expect(ext2.closed).not.toBeNull();
  });

  it('ignores malformed messages from either socket without throwing', async () => {
    const { hub } = makeHub();
    const ext = new FakeSocket();
    const agent = new FakeSocket();
    hub.attachExtension(RUN, ORIGIN, ext);
    hub.attachAgent(RUN, ORIGIN, agent);

    for (const bad of ['not json', 'null', '42', '[]', '{"type":"reply"}']) {
      expect(() => ext.receive(bad)).not.toThrow();
    }
    for (const bad of ['not json', 'null', '"x"', '[]', '{"method":1}']) {
      expect(() => agent.receive(bad)).not.toThrow();
    }
    agent.receive({ id: 16, method: 7 });
    agent.receive({ id: 17, sessionId: 5, method: 'Page.enable' });
    await vi.waitFor(() => {
      expect(lastReply(agent, 16)?.error).toBeDefined();
      expect(lastReply(agent, 17)?.error).toBeDefined();
    });
    expect(ext.sent).toHaveLength(0);
    expect(ext.closed).toBeNull();
    expect(agent.closed).toBeNull();
  });

  it('never logs params, results or payloads', async () => {
    const { hub, logs } = makeHub();
    const ext = new FakeSocket();
    const agent = new FakeSocket();
    hub.attachExtension(RUN, ORIGIN, ext);
    hub.attachAgent(RUN, ORIGIN, agent);

    agent.receive({
      id: 18,
      sessionId: 'tab-7',
      method: 'Input.insertText',
      params: { text: 'SECRET-PARAM' },
    });
    await vi.waitFor(() => expect(ext.sent).toHaveLength(1));
    const cmd = ext.messages()[0];
    ext.receive({ type: 'reply', id: cmd.id, result: { v: 'SECRET-RESULT' } });
    ext.receive({
      type: 'event',
      method: 'X.y',
      params: { v: 'SECRET-EVENT' },
    });
    await vi.waitFor(() => expect(lastReply(agent, 18)).toBeDefined());

    const text = JSON.stringify(logs);
    expect(text).not.toContain('SECRET');
    expect(logs).toContainEqual(
      expect.objectContaining({
        runId: RUN,
        method: 'Input.insertText',
        durationMs: expect.any(Number),
      }),
    );
  });
});
