import { randomUUID } from 'node:crypto';
import { checkCommand, normalizeOrigin } from '../lib/relay/policy';
import type { RelayToExtension } from '../lib/relay/protocol';
import { type Bus, channelFor, type Direction } from './bus';
import {
  type DistributiveOmit,
  type ExtensionCall,
  handleBrowserLevel,
  sanitizeTargetInfos,
} from './cdp-shim';

export type { DistributiveOmit, ExtensionCall } from './cdp-shim';

export interface PeerSocket {
  send(data: string): void;
  close(code?: number, reason?: string): void;
  on(event: 'message', cb: (data: string) => void): void;
  on(event: 'close', cb: () => void): void;
}

export interface RunHubOptions {
  bus: Bus;
  commandTimeoutMs?: number;
  resumeWindowMs?: number;
  log: (line: Record<string, unknown>) => void;
}

// Messages between hubs. Extension traffic travels as validated
// ExtensionToRelay shapes on `from-ext`; hub-only control messages travel on
// `to-ext` and `to-agent`, which no socket can publish to directly.
type ToExtFrame =
  | { t: 'cmd'; cmd: RelayToExtension }
  | { t: 'ext-attached'; by: string }
  | { t: 'close'; reason: string };
type ToAgentFrame =
  | { t: 'ext-up' }
  | { t: 'ext-down' }
  | { t: 'close'; reason: string };

type ExtCommand = DistributiveOmit<RelayToExtension, 'id'>;

interface Waiter {
  msg: ExtCommand;
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
}

interface Pending extends Waiter {
  timer: ReturnType<typeof setTimeout>;
}

interface AgentSide {
  socket: PeerSocket;
  origin: string;
  closed: boolean;
  extState: 'unknown' | 'up' | 'away';
  pending: Map<string, Pending>;
  queue: Waiter[];
  holdTimer?: ReturnType<typeof setTimeout>;
  seenTargets: Set<string>;
  ready: Promise<unknown>;
  unsubscribes: Array<Promise<() => Promise<void>>>;
}

interface ExtSide {
  socket: PeerSocket;
  origin: string;
  closed: boolean;
  unsubscribe?: Promise<() => Promise<void>>;
}

interface Run {
  agent?: AgentSide;
  ext?: ExtSide;
}

const DISCONNECTED = 'tab or extension disconnected';
const NO_ANSWER = 'extension did not answer';
const NOT_RETURNED = 'extension did not return';
const NORMAL_CLOSE = 1000;
const POLICY_CLOSE = 1008;

const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

function parseObject(raw: string): Record<string, unknown> | null {
  try {
    const value: unknown = JSON.parse(raw);
    return isObject(value) ? value : null;
  } catch {
    return null;
  }
}

const ignore = () => {};

/**
 * Pairs one agent socket with one extension socket per run and forwards CDP
 * between them under the relay policy. All traffic goes over the bus, so the
 * two sockets may sit on different hubs.
 */
export class RunHub {
  private readonly bus: Bus;
  private readonly commandTimeoutMs: number;
  private readonly resumeWindowMs: number;
  private readonly log: (line: Record<string, unknown>) => void;
  private readonly instanceId = randomUUID();
  private counter = 0;
  private readonly runs = new Map<string, Run>();

  constructor(opts: RunHubOptions) {
    this.bus = opts.bus;
    this.commandTimeoutMs = opts.commandTimeoutMs ?? 30_000;
    this.resumeWindowMs = opts.resumeWindowMs ?? 300_000;
    this.log = (line) => {
      try {
        opts.log(line);
      } catch {
        // A failing logger must not break forwarding.
      }
    };
  }

  attachExtension(runId: string, origin: string, socket: PeerSocket): void {
    const run = this.run(runId);
    const previous = run.ext;
    if (previous) {
      this.dropExtension(previous, NORMAL_CLOSE, 'replaced');
      run.ext = undefined;
      // Commands sent to the old socket will never be answered.
      this.publish(runId, 'to-agent', { t: 'ext-down' });
    }
    const side: ExtSide = { socket, origin, closed: false };
    run.ext = side;
    this.log({ event: 'extension-attached', runId, role: 'extension' });

    socket.on('message', (data) => {
      try {
        this.onExtensionMessage(runId, side, data);
      } catch {
        this.log({ event: 'malformed', runId, role: 'extension' });
      }
    });
    socket.on('close', () => {
      if (side.closed) return;
      this.dropExtension(side);
      if (run.ext === side) run.ext = undefined;
      this.forgetIfEmpty(runId, run);
      this.log({ event: 'extension-detached', runId, role: 'extension' });
      this.publish(runId, 'to-agent', { t: 'ext-down' });
    });

    side.unsubscribe = this.bus.subscribe(channelFor(runId, 'to-ext'), (raw) =>
      this.onToExt(runId, side, raw),
    );
    side.unsubscribe.then(
      () => {
        // Announce only once commands can reach this socket, and only if it
        // is still the run's extension.
        if (side.closed || this.runs.get(runId)?.ext !== side) return;
        this.publish(runId, 'to-ext', {
          t: 'ext-attached',
          by: this.instanceId,
        });
        this.publish(runId, 'to-agent', { t: 'ext-up' });
      },
      () => this.closeExtensionSocket(runId, side, 'relay bus unavailable'),
    );
  }

  attachAgent(
    runId: string,
    origin: string,
    socket: PeerSocket,
  ): 'ok' | 'busy' {
    const run = this.run(runId);
    if (run.agent) return 'busy';
    const agent: AgentSide = {
      socket,
      origin,
      closed: false,
      extState: run.ext ? 'up' : 'unknown',
      pending: new Map(),
      queue: [],
      seenTargets: new Set(),
      ready: Promise.resolve(),
      unsubscribes: [],
    };
    run.agent = agent;
    this.log({ event: 'agent-attached', runId, role: 'agent' });

    agent.unsubscribes = [
      this.bus.subscribe(channelFor(runId, 'from-ext'), (raw) =>
        this.onFromExt(agent, raw),
      ),
      this.bus.subscribe(channelFor(runId, 'to-agent'), (raw) =>
        this.onToAgent(runId, agent, raw),
      ),
    ];
    // Hold agent commands until replies can reach us.
    agent.ready = Promise.all(agent.unsubscribes);
    agent.ready.catch(() => {
      if (agent.closed) return;
      this.closeAgent(agent, 'relay bus unavailable');
      if (run.agent === agent) run.agent = undefined;
      this.forgetIfEmpty(runId, run);
    });

    socket.on('message', (data) => {
      agent.ready
        .then(() => this.onAgentMessage(runId, agent, data))
        .catch(ignore);
    });
    socket.on('close', () => {
      if (agent.closed) return;
      // Free the slot; the run stays up for the next agent connection.
      this.closeAgent(agent, 'agent disconnected', false);
      if (run.agent === agent) run.agent = undefined;
      this.forgetIfEmpty(runId, run);
      this.log({ event: 'agent-detached', runId, role: 'agent' });
    });
    return 'ok';
  }

  closeRun(runId: string, reason: string): void {
    this.closeLocal(runId, reason);
    this.publish(runId, 'to-agent', { t: 'close', reason });
    this.publish(runId, 'to-ext', { t: 'close', reason });
  }

  // --- extension side ------------------------------------------------------

  private onExtensionMessage(runId: string, side: ExtSide, data: string) {
    if (side.closed) return;
    const msg = parseObject(data);
    if (!msg) {
      this.log({ event: 'malformed', runId, role: 'extension' });
      return;
    }
    switch (msg.type) {
      case 'register': {
        const ok =
          msg.runId === runId &&
          typeof msg.tabId === 'number' &&
          typeof msg.origin === 'string' &&
          normalizeOrigin(msg.origin) === side.origin;
        if (!ok) {
          this.log({ event: 'register-refused', runId, role: 'extension' });
          this.closeExtensionSocket(runId, side, 'registration mismatch');
          return;
        }
        this.log({ event: 'registered', runId, role: 'extension' });
        return;
      }
      case 'reply': {
        if (typeof msg.id !== 'string') break;
        if (msg.error !== undefined && typeof msg.error !== 'string') break;
        this.publish(runId, 'from-ext', {
          type: 'reply',
          id: msg.id,
          result: msg.result,
          error: msg.error,
        });
        return;
      }
      case 'event': {
        if (typeof msg.method !== 'string') break;
        if (msg.sessionId !== undefined && typeof msg.sessionId !== 'string') {
          break;
        }
        this.publish(runId, 'from-ext', {
          type: 'event',
          sessionId: msg.sessionId,
          method: msg.method,
          params: msg.params,
        });
        return;
      }
      case 'tabClosed':
        this.log({ event: 'tab-closed', runId, role: 'extension' });
        this.closeRun(runId, 'tab closed');
        return;
    }
    this.log({ event: 'malformed', runId, role: 'extension' });
  }

  private onToExt(runId: string, side: ExtSide, raw: string) {
    const frame = parseObject(raw) as ToExtFrame | null;
    if (!frame || side.closed) return;
    if (frame.t === 'cmd') {
      if (!isObject(frame.cmd) || typeof frame.cmd.id !== 'string') return;
      this.sendTo(side.socket, frame.cmd);
    } else if (frame.t === 'ext-attached') {
      // A newer extension socket for this run attached on another hub.
      if (frame.by === this.instanceId) return;
      const run = this.runs.get(runId);
      this.dropExtension(side, NORMAL_CLOSE, 'replaced');
      if (run?.ext === side) run.ext = undefined;
      if (run) this.forgetIfEmpty(runId, run);
    } else if (frame.t === 'close' && typeof frame.reason === 'string') {
      this.closeLocal(runId, frame.reason);
    }
  }

  private closeExtensionSocket(runId: string, side: ExtSide, reason: string) {
    // Closing fires the socket's close handler, which pauses the run.
    try {
      side.socket.close(POLICY_CLOSE, reason);
    } catch {
      // Already gone; treat as closed below.
    }
    if (!side.closed) {
      this.dropExtension(side);
      const run = this.runs.get(runId);
      if (run?.ext === side) run.ext = undefined;
      this.publish(runId, 'to-agent', { t: 'ext-down' });
    }
  }

  /** Marks an extension side closed, unsubscribes and optionally closes it. */
  private dropExtension(side: ExtSide, code?: number, reason?: string) {
    side.closed = true;
    side.unsubscribe?.then((off) => off().catch(ignore), ignore);
    if (code !== undefined) {
      try {
        side.socket.close(code, reason);
      } catch {
        // Already closed.
      }
    }
  }

  // --- agent side ----------------------------------------------------------

  private async onAgentMessage(runId: string, agent: AgentSide, data: string) {
    if (agent.closed) return;
    const msg = parseObject(data);
    if (!msg || typeof msg.id !== 'number' || !Number.isSafeInteger(msg.id)) {
      this.log({ event: 'malformed', runId, role: 'agent' });
      return;
    }
    const id = msg.id;
    const sessionId =
      typeof msg.sessionId === 'string' ? msg.sessionId : undefined;
    const reply = (body: Record<string, unknown>) =>
      this.sendTo(
        agent.socket,
        { id, ...(sessionId !== undefined ? { sessionId } : {}), ...body },
        agent,
      );
    const fail = (message: string) =>
      reply({ error: { code: -32000, message } });

    const { method, params } = msg;
    if (
      typeof method !== 'string' ||
      (msg.sessionId !== undefined && sessionId === undefined) ||
      (params !== undefined && !isObject(params))
    ) {
      this.log({ event: 'malformed', runId, role: 'agent' });
      fail('invalid request');
      return;
    }

    const started = Date.now();
    const denied = (reason: string) => {
      this.log({ event: 'command-denied', runId, role: 'agent', method });
      fail(reason);
    };
    try {
      let result: unknown;
      if (sessionId !== undefined) {
        const check = checkCommand(
          { method, params },
          { origin: agent.origin },
        );
        if (!check.ok) return denied(check.reason);
        result = await this.extensionCall(
          runId,
          agent,
        )({
          type: 'cdp',
          sessionId,
          method,
          params,
        });
      } else {
        if (method === 'Target.createTarget') {
          const check = checkCommand(
            { method, params },
            { origin: agent.origin },
          );
          if (!check.ok) return denied(check.reason);
        }
        result = await handleBrowserLevel(
          method,
          params,
          this.extensionCall(runId, agent),
          {
            emit: (event) => this.sendTo(agent.socket, event, agent),
            onUnknown: (name) =>
              this.log({
                event: 'browser-method-stubbed',
                runId,
                role: 'agent',
                method: name,
              }),
          },
        );
      }
      reply({ result: result ?? {} });
      this.log({
        event: 'command',
        runId,
        role: 'agent',
        method,
        durationMs: Date.now() - started,
      });
    } catch (error) {
      fail(error instanceof Error ? error.message : 'command failed');
      this.log({
        event: 'command-failed',
        runId,
        role: 'agent',
        method,
        durationMs: Date.now() - started,
      });
    }
  }

  /** The run's ExtensionCall: refuses targets the extension never listed. */
  private extensionCall(runId: string, agent: AgentSide): ExtensionCall {
    return async (msg) => {
      if (
        (msg.type === 'attach' || msg.type === 'closeTarget') &&
        !agent.seenTargets.has(msg.targetId)
      ) {
        throw new Error('unknown target');
      }
      const result = await new Promise<unknown>((resolve, reject) => {
        const waiter: Waiter = { msg, resolve, reject };
        if (agent.closed) reject(new Error(DISCONNECTED));
        else if (agent.extState === 'away') agent.queue.push(waiter);
        else this.dispatch(runId, agent, waiter);
      });
      if (msg.type === 'listTargets') {
        for (const t of sanitizeTargetInfos(result)) {
          agent.seenTargets.add(t.targetId);
        }
      }
      return result;
    };
  }

  private dispatch(runId: string, agent: AgentSide, waiter: Waiter) {
    this.counter += 1;
    const id = `${this.instanceId}:${this.counter}`;
    const timer = setTimeout(() => {
      agent.pending.delete(id);
      waiter.reject(new Error(NO_ANSWER));
    }, this.commandTimeoutMs);
    agent.pending.set(id, { ...waiter, timer });
    const cmd = { ...waiter.msg, id } as RelayToExtension;
    this.bus
      .publish(channelFor(runId, 'to-ext'), JSON.stringify({ t: 'cmd', cmd }))
      .catch(() => {
        const pending = agent.pending.get(id);
        if (!pending) return;
        clearTimeout(pending.timer);
        agent.pending.delete(id);
        pending.reject(new Error('relay bus unavailable'));
      });
  }

  private onFromExt(agent: AgentSide, raw: string) {
    if (agent.closed) return;
    const msg = parseObject(raw);
    if (!msg) return;
    if (msg.type === 'reply' && typeof msg.id === 'string') {
      // Replies to other hubs' commands are not in our map.
      const pending = agent.pending.get(msg.id);
      if (!pending) return;
      clearTimeout(pending.timer);
      agent.pending.delete(msg.id);
      if (typeof msg.error === 'string') pending.reject(new Error(msg.error));
      else pending.resolve(msg.result);
    } else if (msg.type === 'event' && typeof msg.method === 'string') {
      this.sendTo(
        agent.socket,
        {
          ...(typeof msg.sessionId === 'string'
            ? { sessionId: msg.sessionId }
            : {}),
          method: msg.method,
          params: msg.params,
        },
        agent,
      );
    }
  }

  private onToAgent(runId: string, agent: AgentSide, raw: string) {
    const frame = parseObject(raw) as ToAgentFrame | null;
    if (!frame || agent.closed) return;
    if (frame.t === 'ext-down') {
      this.rejectPending(agent, DISCONNECTED);
      agent.extState = 'away';
      if (!agent.holdTimer) {
        this.log({ event: 'run-held', runId, role: 'agent' });
        agent.holdTimer = setTimeout(() => {
          agent.holdTimer = undefined;
          this.log({ event: 'run-expired', runId, role: 'agent' });
          this.closeRun(runId, NOT_RETURNED);
        }, this.resumeWindowMs);
      }
    } else if (frame.t === 'ext-up') {
      if (agent.holdTimer) {
        clearTimeout(agent.holdTimer);
        agent.holdTimer = undefined;
        this.log({ event: 'run-resumed', runId, role: 'agent' });
      }
      agent.extState = 'up';
      for (const waiter of agent.queue.splice(0)) {
        this.dispatch(runId, agent, waiter);
      }
    } else if (frame.t === 'close' && typeof frame.reason === 'string') {
      this.closeLocal(runId, frame.reason);
    }
  }

  private rejectPending(agent: AgentSide, reason: string) {
    for (const pending of agent.pending.values()) {
      clearTimeout(pending.timer);
      pending.reject(new Error(reason));
    }
    agent.pending.clear();
  }

  /** Fails every outstanding command and unsubscribes; optionally closes. */
  private closeAgent(agent: AgentSide, reason: string, closeSocket = true) {
    agent.closed = true;
    clearTimeout(agent.holdTimer);
    agent.holdTimer = undefined;
    this.rejectPending(agent, reason);
    for (const waiter of agent.queue.splice(0)) {
      waiter.reject(new Error(reason));
    }
    for (const unsubscribe of agent.unsubscribes) {
      unsubscribe.then((off) => off().catch(ignore), ignore);
    }
    if (closeSocket) {
      try {
        agent.socket.close(NORMAL_CLOSE, reason);
      } catch {
        // Already closed.
      }
    }
  }

  // --- shared --------------------------------------------------------------

  private closeLocal(runId: string, reason: string) {
    const run = this.runs.get(runId);
    if (!run) return;
    this.runs.delete(runId);
    if (run.agent && !run.agent.closed) this.closeAgent(run.agent, reason);
    if (run.ext && !run.ext.closed) {
      this.dropExtension(run.ext, NORMAL_CLOSE, reason);
    }
    this.log({ event: 'run-closed', runId });
  }

  private run(runId: string): Run {
    let run = this.runs.get(runId);
    if (!run) {
      run = {};
      this.runs.set(runId, run);
    }
    return run;
  }

  private forgetIfEmpty(runId: string, run: Run) {
    if (!run.agent && !run.ext && this.runs.get(runId) === run) {
      this.runs.delete(runId);
    }
  }

  private sendTo(socket: PeerSocket, message: unknown, agent?: AgentSide) {
    if (agent?.closed) return;
    try {
      socket.send(JSON.stringify(message));
    } catch {
      // The socket's close handler does the cleanup.
    }
  }

  private publish(
    runId: string,
    direction: Direction,
    message: ToExtFrame | ToAgentFrame | Record<string, unknown>,
  ) {
    this.bus
      .publish(channelFor(runId, direction), JSON.stringify(message))
      .catch(ignore);
  }
}
