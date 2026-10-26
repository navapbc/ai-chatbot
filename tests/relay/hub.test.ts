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
  opts: {
    bus?: Bus;
    commandTimeoutMs?: number;
    resumeWindowMs?: number;
    registerTimeoutMs?: number;
    closedRunTtlMs?: number;
  } = {},
) {
  const bus = opts.bus ?? createLocalBus();
  if (!opts.bus) buses.push(bus);
  const logs: Array<Record<string, unknown>> = [];
  const hub = new RunHub({
    bus,
    commandTimeoutMs: opts.commandTimeoutMs,
    resumeWindowMs: opts.resumeWindowMs,
    registerTimeoutMs: opts.registerTimeoutMs,
    closedRunTtlMs: opts.closedRunTtlMs,
    log: (line) => logs.push(line),
  });
  return { hub, bus, logs };
}

// Attaches a fake extension and sends the `register` a real one sends first.
function attachExt(hub: RunHub, ext: FakeSocket, runId = RUN, origin = ORIGIN) {
  hub.attachExtension(runId, origin, ext);
  ext.receive({ type: 'register', runId, tabId: 7, origin });
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
    attachExt(hub, ext);
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
    const { hub, logs } = makeHub();
    const ext = new FakeSocket();
    const agent = new FakeSocket();
    attachExt(hub, ext);
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

    // The log names the method and a fixed reason code, never request text.
    const denials = logs.filter((l) => l.event === 'command-denied');
    expect(
      denials.map(({ method, level, reason }) => ({ method, level, reason })),
    ).toEqual([
      { method: 'Network.getAllCookies', level: 'session', reason: 'method' },
      { method: 'Page.navigate', level: 'session', reason: 'scope' },
      { method: 'Target.createTarget', level: 'browser', reason: 'scope' },
    ]);
    expect(JSON.stringify(denials)).not.toContain('evil.example.com');
  });

  it('refuses a session-scoped Target.createTarget even with an in-scope url', async () => {
    const { hub } = makeHub();
    const ext = new FakeSocket();
    const agent = new FakeSocket();
    attachExt(hub, ext);
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
    attachExt(hub, ext);
    hub.attachAgent(RUN, ORIGIN, agent);

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
    attachExt(hub, ext);
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
    attachExt(hub, ext);
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
    attachExt(hub, ext);
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
    attachExt(hub, ext);
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
    attachExt(hub, ext);
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
    attachExt(b.hub, ext);
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
    attachExt(hub, ext);
    hub.attachAgent(RUN, ORIGIN, agent);
    ext.close();

    agent.receive({ id: 13, sessionId: 'tab-7', method: 'Page.enable' });
    await new Promise((resolve) => setTimeout(resolve, 30));
    expect(agent.closed).toBeNull();
    expect(lastReply(agent, 13)).toBeUndefined();

    const returned = new FakeSocket();
    attachExt(hub, returned);
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
    attachExt(hub, ext);
    hub.attachAgent(RUN, ORIGIN, agent);
    ext.close();

    agent.receive({ id: 14, sessionId: 'tab-7', method: 'Page.enable' });
    await vi.waitFor(() =>
      expect(agent.closed?.reason).toBe('extension did not return'),
    );
  });

  it('treats an extension drop and re-attach inside the window as no close, then closes once the window lapses', async () => {
    const { hub } = makeHub({ resumeWindowMs: 200 });
    const ext = new FakeSocket();
    const agent = new FakeSocket();
    attachExt(hub, ext);
    hub.attachAgent(RUN, ORIGIN, agent);
    ext.close();

    const returned = new FakeSocket();
    attachExt(hub, returned);
    expect(returned.closed).toBeNull();
    agent.receive({ id: 21, sessionId: 'tab-7', method: 'Page.enable' });
    await vi.waitFor(() => expect(returned.sent).toHaveLength(1));
    const cmd = returned.messages()[0];
    returned.receive({ type: 'reply', id: cmd.id, result: {} });
    await vi.waitFor(() =>
      expect(lastReply(agent, 21)).toEqual({
        id: 21,
        sessionId: 'tab-7',
        result: {},
      }),
    );

    // Not tombstoned: the agent socket stays open and the run is still live
    // (a second agent is refused as busy, not as a closed run).
    await new Promise((resolve) => setTimeout(resolve, 250));
    expect(agent.closed).toBeNull();
    const other = new FakeSocket();
    expect(hub.attachAgent(RUN, ORIGIN, other)).toBe('busy');

    // Now the extension leaves for good: the window lapses and the run closes.
    returned.close();
    await vi.waitFor(() =>
      expect(agent.closed?.reason).toBe('extension did not return'),
    );
    const later = new FakeSocket();
    expect(hub.attachAgent(RUN, ORIGIN, later)).toBe('closed');
    expect(later.closed).toMatchObject({ code: 1008 });
  });

  it('makes an agent that connects inside the resume window wait too', async () => {
    const { hub } = makeHub({ resumeWindowMs: 60_000 });
    const ext = new FakeSocket();
    attachExt(hub, ext);
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
    attachExt(hub, returned);
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

  it('replaces the earlier extension socket only once the new one registers', async () => {
    const { hub } = makeHub();
    const first = new FakeSocket();
    const second = new FakeSocket();
    const agent = new FakeSocket();
    attachExt(hub, first);
    hub.attachAgent(RUN, ORIGIN, agent);
    hub.attachExtension(RUN, ORIGIN, second);
    expect(first.closed).toBeNull();

    second.receive({ type: 'register', runId: RUN, tabId: 8, origin: ORIGIN });
    expect(first.closed?.reason).toBe('replaced');

    agent.receive({ id: 15, sessionId: 'tab-7', method: 'Page.enable' });
    await vi.waitFor(() => expect(second.sent).toHaveLength(1));
    expect(first.sent).toHaveLength(0);
    expect(agent.closed).toBeNull();
  });

  it('keeps the first extension when a second one never registers', async () => {
    const { hub } = makeHub({ registerTimeoutMs: 50 });
    const first = new FakeSocket();
    const second = new FakeSocket();
    const agent = new FakeSocket();
    attachExt(hub, first);
    hub.attachAgent(RUN, ORIGIN, agent);
    hub.attachExtension(RUN, ORIGIN, second);
    // Anything but `register` is ignored before registration.
    second.receive({
      type: 'event',
      method: 'Page.loadEventFired',
      params: {},
    });
    second.receive({ type: 'tabClosed' });

    agent.receive({ id: 21, sessionId: 'tab-7', method: 'Page.enable' });
    await vi.waitFor(() => expect(first.sent).toHaveLength(1));
    first.receive({ type: 'reply', id: first.messages()[0].id, result: {} });
    await vi.waitFor(() => expect(lastReply(agent, 21)?.result).toEqual({}));
    first.receive({ type: 'event', method: 'Page.frameNavigated', params: {} });

    await vi.waitFor(() =>
      expect(second.closed?.reason).toBe('register timeout'),
    );
    expect(second.sent).toHaveLength(0);
    expect(first.closed).toBeNull();
    expect(agent.closed).toBeNull();
    expect(agent.messages().map((m) => m.method)).not.toContain(
      'Page.loadEventFired',
    );
  });

  it('closes a second extension that registers for another run or origin', async () => {
    const { hub } = makeHub();
    const first = new FakeSocket();
    const agent = new FakeSocket();
    attachExt(hub, first);
    hub.attachAgent(RUN, ORIGIN, agent);

    const wrongRun = new FakeSocket();
    hub.attachExtension(RUN, ORIGIN, wrongRun);
    wrongRun.receive({
      type: 'register',
      runId: 'other',
      tabId: 1,
      origin: ORIGIN,
    });
    expect(wrongRun.closed?.reason).toBe('registration mismatch');

    const wrongOrigin = new FakeSocket();
    hub.attachExtension(RUN, ORIGIN, wrongOrigin);
    wrongOrigin.receive({
      type: 'register',
      runId: RUN,
      tabId: 1,
      origin: 'https://evil.example.com',
    });
    expect(wrongOrigin.closed?.reason).toBe('registration mismatch');

    expect(first.closed).toBeNull();
    agent.receive({ id: 22, sessionId: 'tab-7', method: 'Page.enable' });
    await vi.waitFor(() => expect(first.sent).toHaveLength(1));
    expect(wrongRun.sent).toHaveLength(0);
    expect(wrongOrigin.sent).toHaveLength(0);
  });

  it('refuses an agent and an extension with different origins for one run', () => {
    const { hub } = makeHub();
    const other = 'https://other.example.gov';

    const agent = new FakeSocket();
    expect(hub.attachAgent(RUN, ORIGIN, agent)).toBe('ok');
    const ext = new FakeSocket();
    hub.attachExtension(RUN, other, ext);
    expect(ext.closed?.reason).toBe('origin mismatch');
    expect(agent.closed).toBeNull();

    const ext2 = new FakeSocket();
    attachExt(hub, ext2, 'run-2', other);
    const agent2 = new FakeSocket();
    expect(hub.attachAgent('run-2', ORIGIN, agent2)).toBe('origin-mismatch');
    expect(agent2.closed?.reason).toBe('origin mismatch');
    expect(ext2.closed).toBeNull();

    // A matching second extension is not affected by the refused one.
    const ext3 = new FakeSocket();
    attachExt(hub, ext3);
    expect(ext3.closed).toBeNull();
  });

  it('refuses an invalid, empty or non-normalized origin', () => {
    const { hub } = makeHub();
    for (const bad of [
      '',
      'not a url',
      'http://benefits.example.gov',
      'https://benefits.example.gov/',
      'HTTPS://BENEFITS.EXAMPLE.GOV',
      'https://benefits.example.gov/apply',
    ]) {
      const agent = new FakeSocket();
      expect(hub.attachAgent(RUN, bad, agent)).toBe('origin-mismatch');
      expect(agent.closed?.reason).toBe('invalid origin');
      const ext = new FakeSocket();
      hub.attachExtension(RUN, bad, ext);
      expect(ext.closed?.reason).toBe('invalid origin');
    }
    // Nothing was bound, so a valid origin still attaches.
    expect(hub.attachAgent(RUN, ORIGIN, new FakeSocket())).toBe('ok');
  });

  it('refuses an origin mismatch across two hubs', async () => {
    const bus = createLocalBus();
    buses.push(bus);
    const a = makeHub({ bus });
    const b = makeHub({ bus });
    const c = makeHub({ bus });
    const other = 'https://other.example.gov';

    // Hub A's agent is bound to ORIGIN; hub B never saw the run.
    const agent = new FakeSocket();
    a.hub.attachAgent(RUN, ORIGIN, agent);
    const wrongExt = new FakeSocket();
    attachExt(b.hub, wrongExt, RUN, other);
    await vi.waitFor(() =>
      expect(agent.closed?.reason).toBe('origin mismatch'),
    );

    // An extension for another origin on hub C does not displace hub A's.
    const ext = new FakeSocket();
    attachExt(a.hub, ext, 'run-2');
    const intruder = new FakeSocket();
    attachExt(c.hub, intruder, 'run-2', other);
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(ext.closed).toBeNull();

    a.hub.closeRun(RUN, 'test over');
    b.hub.closeRun(RUN, 'test over');
    a.hub.closeRun('run-2', 'test over');
    c.hub.closeRun('run-2', 'test over');
  });

  it('closes both sockets on closeRun and on tabClosed', async () => {
    const { hub } = makeHub();
    const ext = new FakeSocket();
    const agent = new FakeSocket();
    attachExt(hub, ext);
    hub.attachAgent(RUN, ORIGIN, agent);
    hub.closeRun(RUN, 'run finished');
    expect(ext.closed?.reason).toBe('run finished');
    expect(agent.closed?.reason).toBe('run finished');

    const ext2 = new FakeSocket();
    const agent2 = new FakeSocket();
    attachExt(hub, ext2, 'run-2');
    hub.attachAgent('run-2', ORIGIN, agent2);
    ext2.receive({ type: 'tabClosed' });
    await vi.waitFor(() => expect(agent2.closed?.reason).toBe('tab closed'));
    expect(ext2.closed).not.toBeNull();
  });

  it('closes the run at once when the registered extension sends stop', async () => {
    const { hub, logs } = makeHub();
    const ext = new FakeSocket();
    const agent = new FakeSocket();
    attachExt(hub, ext);
    hub.attachAgent(RUN, ORIGIN, agent);
    ext.receive({ type: 'stop', reason: 'left-origin' });
    await vi.waitFor(() =>
      expect(agent.closed).toEqual({ code: 1000, reason: 'run stopped' }),
    );
    expect(ext.closed).not.toBeNull();
    expect(logs).toContainEqual({
      event: 'run-stopped',
      runId: RUN,
      role: 'extension',
      reason: 'left-origin',
    });
    // Tombstoned: a re-dialing agent is refused with the same reason.
    const again = new FakeSocket();
    expect(hub.attachAgent(RUN, ORIGIN, again)).toBe('closed');
    expect(again.closed?.reason).toBe('run stopped');
  });

  it('ignores stop from an extension socket that has not registered', () => {
    const { hub } = makeHub();
    const ext = new FakeSocket();
    attachExt(hub, ext);
    const agent = new FakeSocket();
    hub.attachAgent(RUN, ORIGIN, agent);
    const pending = new FakeSocket();
    hub.attachExtension(RUN, ORIGIN, pending);
    pending.receive({ type: 'stop', reason: 'user' });
    expect(agent.closed).toBeNull();
    expect(ext.closed).toBeNull();
    expect(hub.attachAgent(RUN, ORIGIN, new FakeSocket())).toBe('busy');
    hub.closeRun(RUN, 'test over');
  });

  it('logs an unknown stop reason as user and never echoes it', async () => {
    const { hub, logs } = makeHub();
    const ext = new FakeSocket();
    const agent = new FakeSocket();
    attachExt(hub, ext);
    hub.attachAgent(RUN, ORIGIN, agent);
    ext.receive({ type: 'stop', reason: 'secret-token-value' });
    await vi.waitFor(() => expect(agent.closed?.reason).toBe('run stopped'));
    expect(JSON.stringify(logs)).not.toContain('secret-token-value');
    expect(logs).toContainEqual({
      event: 'run-stopped',
      runId: RUN,
      role: 'extension',
      reason: 'user',
    });
  });

  it('refuses new sockets for a closed run with the reason it closed', async () => {
    const { hub } = makeHub();
    const ext = new FakeSocket();
    attachExt(hub, ext);
    hub.attachAgent(RUN, ORIGIN, new FakeSocket());
    ext.receive({ type: 'tabClosed' });

    // A daemon that re-dials must fail at once, not wait for an extension.
    const agent = new FakeSocket();
    expect(hub.attachAgent(RUN, ORIGIN, agent)).toBe('closed');
    expect(agent.closed).toEqual({ code: 1008, reason: 'tab closed' });
    const ext2 = new FakeSocket();
    hub.attachExtension(RUN, ORIGIN, ext2);
    expect(ext2.closed).toEqual({ code: 1008, reason: 'tab closed' });

    // Other runs are unaffected.
    expect(hub.attachAgent('run-2', ORIGIN, new FakeSocket())).toBe('ok');
    hub.closeRun('run-2', 'test over');
  });

  it('forgets a closed run once its tokens have expired', () => {
    vi.useFakeTimers();
    try {
      const { hub } = makeHub({ closedRunTtlMs: 1000 });
      hub.attachAgent(RUN, ORIGIN, new FakeSocket());
      hub.closeRun(RUN, 'run finished');
      expect(hub.attachAgent(RUN, ORIGIN, new FakeSocket())).toBe('closed');
      vi.advanceTimersByTime(1001);
      expect(hub.attachAgent(RUN, ORIGIN, new FakeSocket())).toBe('ok');
      hub.closeRun(RUN, 'test over');
    } finally {
      vi.useRealTimers();
    }
  });

  it('lets an agent re-attach to a live run after its socket closed', () => {
    const { hub } = makeHub();
    attachExt(hub, new FakeSocket());
    const first = new FakeSocket();
    hub.attachAgent(RUN, ORIGIN, first);
    first.close();
    expect(hub.attachAgent(RUN, ORIGIN, new FakeSocket())).toBe('ok');
    hub.closeRun(RUN, 'test over');
  });

  it('ignores malformed messages from either socket without throwing', async () => {
    const { hub } = makeHub();
    const ext = new FakeSocket();
    const agent = new FakeSocket();
    attachExt(hub, ext);
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
    attachExt(hub, ext);
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

  describe('resumed (service-worker restart)', () => {
    // A restarted worker re-dials, registers, then sends `resumed` once.
    const restartExtension = (hub: RunHub, ext: FakeSocket) => {
      ext.close();
      const rebuilt = new FakeSocket();
      attachExt(hub, rebuilt);
      rebuilt.receive({ type: 'resumed' });
      return rebuilt;
    };

    it('closes the agent with "extension restarted" and keeps the run', async () => {
      const { hub, logs } = makeHub({ resumeWindowMs: 60_000 });
      const ext = new FakeSocket();
      const agent = new FakeSocket();
      attachExt(hub, ext);
      hub.attachAgent(RUN, ORIGIN, agent);

      const rebuilt = restartExtension(hub, ext);
      await vi.waitFor(() =>
        expect(agent.closed).toEqual({
          code: 1000,
          reason: 'extension restarted',
        }),
      );
      expect(rebuilt.closed).toBeNull();
      expect(logs).not.toContainEqual(
        expect.objectContaining({ event: 'run-closed' }),
      );

      // No tombstone: a new agent attaches and reaches the same extension.
      const next = new FakeSocket();
      expect(hub.attachAgent(RUN, ORIGIN, next)).toBe('ok');
      next.receive({ id: 30, sessionId: 'tab-7', method: 'Page.enable' });
      await vi.waitFor(() => expect(rebuilt.sent).toHaveLength(1));
      const cmd = rebuilt.messages()[0];
      expect(cmd).toMatchObject({ type: 'cdp', method: 'Page.enable' });
      rebuilt.receive({ type: 'reply', id: cmd.id, result: { ok: 1 } });
      await vi.waitFor(() =>
        expect(lastReply(next, 30)).toEqual({
          id: 30,
          sessionId: 'tab-7',
          result: { ok: 1 },
        }),
      );
      expect(next.closed).toBeNull();
      hub.closeRun(RUN, 'test over');
    });

    it('resets an agent attached through another hub', async () => {
      const bus = createLocalBus();
      buses.push(bus);
      const extHub = makeHub({ bus, resumeWindowMs: 60_000 }).hub;
      const agentHub = makeHub({ bus, resumeWindowMs: 60_000 }).hub;
      const ext = new FakeSocket();
      const agent = new FakeSocket();
      attachExt(extHub, ext);
      agentHub.attachAgent(RUN, ORIGIN, agent);

      restartExtension(extHub, ext);
      await vi.waitFor(() =>
        expect(agent.closed).toEqual({
          code: 1000,
          reason: 'extension restarted',
        }),
      );
      expect(agentHub.attachAgent(RUN, ORIGIN, new FakeSocket())).toBe('ok');
    });

    it("never flushes the old agent's queued command to the rebuilt extension", async () => {
      const { hub } = makeHub({ resumeWindowMs: 60_000 });
      const ext = new FakeSocket();
      const agent = new FakeSocket();
      attachExt(hub, ext);
      hub.attachAgent(RUN, ORIGIN, agent);
      ext.close();
      // Sent while the worker is down: held in the agent's queue.
      agent.receive({ id: 40, sessionId: 'tab-7', method: 'Page.enable' });
      await new Promise((resolve) => setTimeout(resolve, 20));

      const rebuilt = restartExtension(hub, ext);
      await vi.waitFor(() =>
        expect(agent.closed?.reason).toBe('extension restarted'),
      );
      await new Promise((resolve) => setTimeout(resolve, 30));
      expect(rebuilt.sent).toHaveLength(0);
      hub.closeRun(RUN, 'test over');
    });

    it('resets at once when resumed arrives after the extension was announced', async () => {
      const { hub, logs } = makeHub({ resumeWindowMs: 60_000 });
      const ext = new FakeSocket();
      const agent = new FakeSocket();
      attachExt(hub, ext);
      hub.attachAgent(RUN, ORIGIN, agent);
      ext.close();
      const rebuilt = new FakeSocket();
      attachExt(hub, rebuilt);
      await vi.waitFor(() =>
        expect(logs).toContainEqual(
          expect.objectContaining({ event: 'run-resumed' }),
        ),
      );
      expect(agent.closed).toBeNull();

      rebuilt.receive({ type: 'resumed' });
      await vi.waitFor(() =>
        expect(agent.closed).toEqual({
          code: 1000,
          reason: 'extension restarted',
        }),
      );
      expect(hub.attachAgent(RUN, ORIGIN, new FakeSocket())).toBe('ok');
      hub.closeRun(RUN, 'test over');
    });

    it('keeps a cross-hub run open past the resume window after a reset', async () => {
      const bus = createLocalBus();
      buses.push(bus);
      const extHub = makeHub({ bus, resumeWindowMs: 100 });
      const agentHub = makeHub({ bus, resumeWindowMs: 100 });
      const ext = new FakeSocket();
      const agent = new FakeSocket();
      attachExt(extHub.hub, ext);
      agentHub.hub.attachAgent(RUN, ORIGIN, agent);
      await new Promise((resolve) => setTimeout(resolve, 20));

      const rebuilt = restartExtension(extHub.hub, ext);
      await vi.waitFor(() =>
        expect(agent.closed?.reason).toBe('extension restarted'),
      );
      // Past the window: no hold may still be running on either hub.
      await new Promise((resolve) => setTimeout(resolve, 200));
      for (const { logs } of [extHub, agentHub]) {
        expect(logs).not.toContainEqual(
          expect.objectContaining({ event: 'run-closed' }),
        );
      }
      const next = new FakeSocket();
      expect(agentHub.hub.attachAgent(RUN, ORIGIN, next)).toBe('ok');
      next.receive({ id: 41, sessionId: 'tab-7', method: 'Page.enable' });
      await vi.waitFor(() => expect(rebuilt.sent).toHaveLength(1));
      const cmd = rebuilt.messages()[0];
      rebuilt.receive({ type: 'reply', id: cmd.id, result: {} });
      await vi.waitFor(() => expect(lastReply(next, 41)).toBeDefined());
      expect(next.closed).toBeNull();
    });

    it('does not reset when a socket that sent resumed early registers later', async () => {
      const { hub, logs } = makeHub({ resumeWindowMs: 60_000 });
      const ext = new FakeSocket();
      const agent = new FakeSocket();
      attachExt(hub, ext);
      hub.attachAgent(RUN, ORIGIN, agent);

      const pending = new FakeSocket();
      hub.attachExtension(RUN, ORIGIN, pending);
      pending.receive({ type: 'resumed' });
      pending.receive({
        type: 'register',
        runId: RUN,
        tabId: 7,
        origin: ORIGIN,
      });
      await new Promise((resolve) => setTimeout(resolve, 30));
      expect(agent.closed).toBeNull();
      expect(logs).not.toContainEqual(
        expect.objectContaining({ event: 'extension-restarted' }),
      );
      expect(logs).not.toContainEqual(
        expect.objectContaining({ event: 'agent-reset' }),
      );
      hub.closeRun(RUN, 'test over');
    });

    it('does nothing for resumed from a pending or unregistered socket', async () => {
      const { hub } = makeHub();
      const ext = new FakeSocket();
      const agent = new FakeSocket();
      attachExt(hub, ext);
      hub.attachAgent(RUN, ORIGIN, agent);

      const pending = new FakeSocket();
      hub.attachExtension(RUN, ORIGIN, pending);
      pending.receive({ type: 'resumed' });
      await new Promise((resolve) => setTimeout(resolve, 30));
      expect(agent.closed).toBeNull();
      expect(ext.closed).toBeNull();
      expect(pending.closed).toBeNull();
      expect(hub.attachAgent(RUN, ORIGIN, new FakeSocket())).toBe('busy');
      hub.closeRun(RUN, 'test over');
    });

    it('ignores a repeated or malformed resumed without closing a socket', async () => {
      const { hub } = makeHub({ resumeWindowMs: 60_000 });
      const ext = new FakeSocket();
      const agent = new FakeSocket();
      attachExt(hub, ext);
      hub.attachAgent(RUN, ORIGIN, agent);
      const rebuilt = restartExtension(hub, ext);
      await vi.waitFor(() => expect(agent.closed).not.toBeNull());

      const next = new FakeSocket();
      expect(hub.attachAgent(RUN, ORIGIN, next)).toBe('ok');
      for (const repeat of [
        { type: 'resumed' },
        { type: 'resumed', reason: 'again' },
        '{"type":"resumed"',
      ]) {
        expect(() => rebuilt.receive(repeat)).not.toThrow();
      }
      await new Promise((resolve) => setTimeout(resolve, 30));
      expect(next.closed).toBeNull();
      expect(rebuilt.closed).toBeNull();
      hub.closeRun(RUN, 'test over');
    });
  });
});
