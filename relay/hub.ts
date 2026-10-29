import { randomUUID } from 'node:crypto';
import {
  checkCommand,
  isMethodAllowed,
  normalizeOrigin,
} from '../lib/relay/policy';
import { type RelayToExtension, STOP_REASONS } from '../lib/relay/protocol';
import { RELAY_TOKEN_TTL_SECONDS } from '../lib/relay/token';
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
  /** Bytes queued for sending; a peer without it counts as under the cap. */
  readonly bufferedAmount?: number;
}

/** Outbound bytes a peer may have queued before its run is closed. */
export const MAX_BUFFERED_BYTES = 8 * 1024 * 1024;
const BACKPRESSURE_REASON = 'send buffer full';

export interface RunHubOptions {
  bus: Bus;
  commandTimeoutMs?: number;
  resumeWindowMs?: number;
  /** How long a new extension socket may take to send `register`. */
  registerTimeoutMs?: number;
  /** How long a closed run refuses new sockets. Defaults to the token TTL. */
  closedRunTtlMs?: number;
  log: (line: Record<string, unknown>) => void;
}

// Messages between hubs. Extension traffic travels as validated
// ExtensionToRelay shapes on `from-ext`; hub-only control messages travel on
// `to-ext` and `to-agent`, which no socket can publish to directly.
type ToExtFrame =
  | { t: 'cmd'; cmd: RelayToExtension }
  | { t: 'ext-attached'; by: string; origin: string; sub: string }
  | { t: 'close'; reason: string };
type ToAgentFrame =
  | { t: 'ext-up'; origin: string; sub: string }
  | { t: 'ext-down' }
  /** Drop the agent connection but keep the run (extension restarted). */
  | { t: 'agent-reset' }
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
  sub: string;
  closed: boolean;
  extState: 'unknown' | 'up' | 'away';
  pending: Map<string, Pending>;
  queue: Waiter[];
  seenTargets: Set<string>;
  ready: Promise<unknown>;
  unsubscribes: Array<Promise<() => Promise<void>>>;
}

interface ExtSide {
  socket: PeerSocket;
  origin: string;
  sub: string;
  closed: boolean;
  /** False until a matching `register`; a pending socket receives nothing. */
  registered: boolean;
  registerTimer?: ReturnType<typeof setTimeout>;
  unsubscribe?: Promise<() => Promise<void>>;
  /** True once `ext-up` was published for this socket. */
  announced: boolean;
  /** This socket already sent `resumed`; any later one is ignored. */
  resumed: boolean;
}

interface Run {
  /** Bound by the first attach; every later attach must match it. */
  origin: string;
  /** The user (token `sub`) bound by the first attach, like the origin. */
  sub: string;
  agent?: AgentSide;
  ext?: ExtSide;
  /** Extension sockets that have not sent a matching `register` yet. */
  pendingExts: Set<ExtSide>;
  /** Set while the extension is away; fires at the end of the resume window. */
  holdTimer?: ReturnType<typeof setTimeout>;
}

const DISCONNECTED = 'tab or extension disconnected';
const NO_ANSWER = 'extension did not answer';
const EXTENSION_RESTARTED = 'extension restarted';
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
  private readonly registerTimeoutMs: number;
  private readonly closedRunTtlMs: number;
  private readonly log: (line: Record<string, unknown>) => void;
  private readonly instanceId = randomUUID();
  private counter = 0;
  private readonly runs = new Map<string, Run>();
  /**
   * Runs closed on this hub, with the close reason and when to forget them.
   * A socket that re-dials a finished run (an agent-browser daemon after the
   * tab closed) is refused at once instead of waiting on a fresh empty run.
   * Local to this hub, like the rest of run state (single instance, R10).
   */
  private readonly closedRuns = new Map<
    string,
    { reason: string; until: number }
  >();

  constructor(opts: RunHubOptions) {
    this.bus = opts.bus;
    this.commandTimeoutMs = opts.commandTimeoutMs ?? 30_000;
    this.resumeWindowMs = opts.resumeWindowMs ?? 300_000;
    this.registerTimeoutMs = opts.registerTimeoutMs ?? 10_000;
    this.closedRunTtlMs = opts.closedRunTtlMs ?? RELAY_TOKEN_TTL_SECONDS * 1000;
    this.log = (line) => {
      try {
        opts.log(line);
      } catch {
        // A failing logger must not break forwarding.
      }
    };
  }

  attachExtension(
    runId: string,
    origin: string,
    sub: string,
    socket: PeerSocket,
  ): void {
    const closed = this.closedReason(runId);
    if (closed !== null) {
      this.log({ event: 'attach-refused', runId, role: 'extension' });
      this.refuse(socket, closed);
      return;
    }
    const refusal =
      this.originRefusal(runId, origin) ?? this.subRefusal(runId, sub);
    if (refusal) {
      this.log({ event: 'attach-refused', runId, role: 'extension' });
      this.refuse(socket, refusal);
      return;
    }
    const run = this.run(runId, origin, sub);
    // Held apart until it registers, so it cannot displace the current
    // extension or receive commands on the strength of its token alone.
    const side: ExtSide = {
      socket,
      origin,
      sub,
      closed: false,
      registered: false,
      announced: false,
      resumed: false,
    };
    run.pendingExts.add(side);
    side.registerTimer = setTimeout(() => {
      if (side.registered || side.closed) return;
      this.log({ event: 'register-timeout', runId, role: 'extension' });
      this.refusePending(runId, side, 'register timeout');
    }, this.registerTimeoutMs);
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
      if (side.registered) this.extensionGone(runId, side);
      else this.refusePending(runId, side);
    });
  }

  attachAgent(
    runId: string,
    origin: string,
    sub: string,
    socket: PeerSocket,
  ): 'ok' | 'busy' | 'origin-mismatch' | 'sub-mismatch' | 'closed' {
    const closed = this.closedReason(runId);
    if (closed !== null) {
      this.log({ event: 'attach-refused', runId, role: 'agent' });
      this.refuse(socket, closed);
      return 'closed';
    }
    const refusal = this.originRefusal(runId, origin);
    if (refusal) {
      this.log({ event: 'attach-refused', runId, role: 'agent' });
      this.refuse(socket, refusal);
      return 'origin-mismatch';
    }
    const subRefusal = this.subRefusal(runId, sub);
    if (subRefusal) {
      this.log({ event: 'attach-refused', runId, role: 'agent' });
      this.refuse(socket, subRefusal);
      return 'sub-mismatch';
    }
    const run = this.run(runId, origin, sub);
    if (run.agent) return 'busy';
    const agent: AgentSide = {
      socket,
      origin,
      sub,
      closed: false,
      // An agent that connects inside the resume window waits like the last.
      extState: run.ext ? 'up' : run.holdTimer ? 'away' : 'unknown',
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
        this.onFromExt(runId, agent, raw),
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
    if (msg.type === 'register') {
      const ok =
        msg.runId === runId &&
        typeof msg.tabId === 'number' &&
        typeof msg.origin === 'string' &&
        normalizeOrigin(msg.origin) === side.origin;
      if (!ok) {
        this.log({ event: 'register-refused', runId, role: 'extension' });
        if (side.registered) {
          this.closeExtensionSocket(runId, side, 'registration mismatch');
        } else {
          this.refusePending(runId, side, 'registration mismatch');
        }
        return;
      }
      this.log({ event: 'registered', runId, role: 'extension' });
      if (!side.registered) this.promote(runId, side);
      return;
    }
    if (!side.registered) {
      // Nothing but `register` is accepted before registration.
      this.log({ event: 'unregistered-message', runId, role: 'extension' });
      return;
    }
    switch (msg.type) {
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
      case 'resumed': {
        // Once per socket: a repeat could otherwise cut off the agent that
        // attached after the first one.
        if (side.resumed) {
          this.log({ event: 'resumed-repeated', runId, role: 'extension' });
          return;
        }
        side.resumed = true;
        this.log({ event: 'extension-restarted', runId, role: 'extension' });
        // Not announced yet: promote publishes the reset ahead of `ext-up`.
        // Already announced: `ext-up` may have flushed queued commands to
        // this socket; they fail there (`not attached`), and the Eve tool
        // rotates on that error.
        if (side.announced)
          this.publish(runId, 'to-agent', { t: 'agent-reset' });
        return;
      }
      case 'stop': {
        // Only a fixed reason is ever logged; anything else counts as 'user'.
        const reason =
          typeof msg.reason === 'string' && STOP_REASONS.has(msg.reason)
            ? msg.reason
            : 'user';
        this.log({ event: 'run-stopped', runId, role: 'extension', reason });
        this.closeRun(runId, 'run stopped');
        return;
      }
    }
    this.log({ event: 'malformed', runId, role: 'extension' });
  }

  private onToExt(runId: string, side: ExtSide, raw: string) {
    const frame = parseObject(raw) as ToExtFrame | null;
    if (!frame || side.closed) return;
    if (frame.t === 'cmd') {
      if (!isObject(frame.cmd) || typeof frame.cmd.id !== 'string') return;
      this.sendTo(runId, side.socket, frame.cmd);
    } else if (frame.t === 'ext-attached') {
      // A newer extension socket for this run attached on another hub.
      if (frame.by === this.instanceId) return;
      if (frame.origin !== side.origin) {
        this.log({ event: 'origin-mismatch', runId, role: 'extension' });
        return;
      }
      if (frame.sub !== side.sub) {
        this.log({ event: 'sub-mismatch', runId, role: 'extension' });
        return;
      }
      const run = this.runs.get(runId);
      this.dropExtension(side, NORMAL_CLOSE, 'replaced');
      if (run?.ext === side) run.ext = undefined;
      if (run) this.forgetIfEmpty(runId, run);
    } else if (frame.t === 'close' && typeof frame.reason === 'string') {
      this.closeLocal(runId, frame.reason);
    }
  }

  /** A pending extension sent a matching `register`: make it the run's. */
  private promote(runId: string, side: ExtSide) {
    const run = this.runs.get(runId);
    if (!run || side.closed || !run.pendingExts.has(side)) return;
    clearTimeout(side.registerTimer);
    side.registered = true;
    run.pendingExts.delete(side);
    const previous = run.ext;
    if (previous) {
      this.dropExtension(previous, NORMAL_CLOSE, 'replaced');
      // Commands sent to the old socket will never be answered.
      this.publish(runId, 'to-agent', { t: 'ext-down' });
    }
    run.ext = side;

    side.unsubscribe = this.bus.subscribe(channelFor(runId, 'to-ext'), (raw) =>
      this.onToExt(runId, side, raw),
    );
    side.unsubscribe.then(
      () => {
        // Announce only once commands can reach this socket, and only if it
        // is still the run's extension.
        const current = this.runs.get(runId);
        if (side.closed || current?.ext !== side) return;
        this.publish(runId, 'to-ext', {
          t: 'ext-attached',
          by: this.instanceId,
          origin: side.origin,
          sub: side.sub,
        });
        // A restarted extension's agent is reset before `ext-up`, so the
        // commands it queued while the worker was down are failed, never
        // flushed to an extension that holds none of its sessions.
        if (side.resumed) this.publish(runId, 'to-agent', { t: 'agent-reset' });
        this.publish(runId, 'to-agent', {
          t: 'ext-up',
          origin: side.origin,
          sub: side.sub,
        });
        side.announced = true;
        this.clearHold(runId, current);
      },
      () => this.closeExtensionSocket(runId, side, 'relay bus unavailable'),
    );
  }

  /** Closes an unregistered extension socket without touching the run's. */
  private refusePending(runId: string, side: ExtSide, reason?: string) {
    side.closed = true;
    clearTimeout(side.registerTimer);
    const run = this.runs.get(runId);
    run?.pendingExts.delete(side);
    if (reason !== undefined) this.refuse(side.socket, reason);
    if (run) this.forgetIfEmpty(runId, run);
  }

  private closeExtensionSocket(runId: string, side: ExtSide, reason: string) {
    // Closing fires the socket's close handler, which pauses the run.
    try {
      side.socket.close(POLICY_CLOSE, reason);
    } catch {
      // Already gone; treat as closed below.
    }
    if (!side.closed) this.extensionGone(runId, side);
  }

  /** The run's extension socket went away: hold the run for its window. */
  private extensionGone(runId: string, side: ExtSide) {
    this.dropExtension(side);
    this.log({ event: 'extension-detached', runId, role: 'extension' });
    const run = this.runs.get(runId);
    if (run?.ext !== side) return;
    run.ext = undefined;
    this.startHold(runId, run);
    this.publish(runId, 'to-agent', { t: 'ext-down' });
  }

  private startHold(runId: string, run: Run) {
    if (run.holdTimer) return;
    this.log({ event: 'run-held', runId });
    run.holdTimer = setTimeout(() => {
      run.holdTimer = undefined;
      if (this.runs.get(runId) !== run) return;
      this.log({ event: 'run-expired', runId });
      // Local only: the extension may already be back on another hub, which
      // tells its own agent hub with ext-up.
      this.closeLocal(runId, NOT_RETURNED);
    }, this.resumeWindowMs);
  }

  private clearHold(runId: string, run: Run) {
    if (!run.holdTimer) return;
    clearTimeout(run.holdTimer);
    run.holdTimer = undefined;
    this.log({ event: 'run-resumed', runId });
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
        runId,
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
    // Fixed vocabulary only: never request text. `level` tells a session
    // command from a browser-level one; `reason` a method that is not listed
    // from one refused for its url.
    const level = sessionId === undefined ? 'browser' : 'session';
    const denied = (reason: 'method' | 'scope', message: string) => {
      this.log({
        event: 'command-denied',
        runId,
        role: 'agent',
        method,
        level,
        reason,
      });
      fail(message);
    };
    try {
      let result: unknown;
      if (sessionId !== undefined) {
        // checkCommand exempts Target.createTarget for the browser-level shim;
        // on a session it must still pass the allowlist.
        if (!isMethodAllowed(method)) {
          return denied('method', `method not allowed: ${method}`);
        }
        const check = checkCommand(
          { method, params },
          { origin: agent.origin },
        );
        // The method passed the allowlist above, so a refusal is the scope.
        if (!check.ok) return denied('scope', check.reason);
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
          if (!check.ok) return denied('scope', check.reason);
        }
        result = await handleBrowserLevel(
          method,
          params,
          this.extensionCall(runId, agent),
          {
            emit: (event) => this.sendTo(runId, agent.socket, event, agent),
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
      // `durationMs` on this line feeds the relay's log-based metric on
      // command duration: do not rename the field or the event.
      this.log({
        event: 'command',
        runId,
        role: 'agent',
        method,
        level,
        durationMs: Date.now() - started,
      });
    } catch (error) {
      fail(error instanceof Error ? error.message : 'command failed');
      this.log({
        event: 'command-failed',
        runId,
        role: 'agent',
        method,
        level,
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

  private onFromExt(runId: string, agent: AgentSide, raw: string) {
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
        runId,
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
    const run = this.runs.get(runId);
    if (frame.t === 'ext-down') {
      this.rejectPending(agent, DISCONNECTED);
      agent.extState = 'away';
      if (run?.agent === agent) this.startHold(runId, run);
    } else if (frame.t === 'ext-up') {
      if (frame.origin !== agent.origin) {
        // An extension for another origin claimed this run on another hub.
        this.log({ event: 'origin-mismatch', runId, role: 'agent' });
        this.closeAgent(agent, 'origin mismatch');
        if (run?.agent === agent) {
          run.agent = undefined;
          this.forgetIfEmpty(runId, run);
        }
        return;
      }
      if (frame.sub !== agent.sub) {
        // Another user's extension claimed this run on another hub.
        this.log({ event: 'sub-mismatch', runId, role: 'agent' });
        this.closeAgent(agent, 'user mismatch');
        if (run?.agent === agent) {
          run.agent = undefined;
          this.forgetIfEmpty(runId, run);
        }
        return;
      }
      if (run?.agent === agent) this.clearHold(runId, run);
      agent.extState = 'up';
      for (const waiter of agent.queue.splice(0)) {
        this.dispatch(runId, agent, waiter);
      }
    } else if (frame.t === 'agent-reset') {
      // The extension lost its debugger session and every ref the agent
      // holds: end this connection, keep the run for the next one.
      this.closeAgent(agent, EXTENSION_RESTARTED);
      if (run?.agent === agent) {
        run.agent = undefined;
        // The extension is back (the reset follows its registration): a
        // hold left on this hub must not expire and close the run.
        this.clearHold(runId, run);
        this.forgetIfEmpty(runId, run);
      }
      this.log({ event: 'agent-reset', runId, role: 'agent' });
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
    this.rememberClosed(runId, reason);
    clearTimeout(run.holdTimer);
    run.holdTimer = undefined;
    if (run.agent && !run.agent.closed) this.closeAgent(run.agent, reason);
    if (run.ext && !run.ext.closed) {
      this.dropExtension(run.ext, NORMAL_CLOSE, reason);
    }
    for (const side of run.pendingExts) {
      side.closed = true;
      clearTimeout(side.registerTimer);
      this.refuse(side.socket, reason, NORMAL_CLOSE);
    }
    run.pendingExts.clear();
    this.log({ event: 'run-closed', runId });
  }

  private run(runId: string, origin: string, sub: string): Run {
    let run = this.runs.get(runId);
    if (!run) {
      run = { origin, sub, pendingExts: new Set() };
      this.runs.set(runId, run);
    }
    return run;
  }

  private rememberClosed(runId: string, reason: string) {
    const now = Date.now();
    for (const [id, entry] of this.closedRuns) {
      if (entry.until <= now) this.closedRuns.delete(id);
    }
    this.closedRuns.set(runId, { reason, until: now + this.closedRunTtlMs });
  }

  /** The reason a run closed, or null if it is not (or no longer) closed. */
  private closedReason(runId: string): string | null {
    const entry = this.closedRuns.get(runId);
    if (!entry) return null;
    if (entry.until > Date.now()) return entry.reason;
    this.closedRuns.delete(runId);
    return null;
  }

  /** Why an attach for this origin must be refused, or null if it may proceed. */
  private originRefusal(runId: string, origin: string): string | null {
    if (typeof origin !== 'string' || normalizeOrigin(origin) !== origin) {
      return 'invalid origin';
    }
    const run = this.runs.get(runId);
    if (run && run.origin !== origin) return 'origin mismatch';
    return null;
  }

  /**
   * Why an attach for this user must be refused, or null if it may proceed.
   * Reasons are fixed strings: the sub itself is never logged or sent.
   */
  private subRefusal(runId: string, sub: string): string | null {
    if (typeof sub !== 'string' || sub === '') return 'invalid user';
    const run = this.runs.get(runId);
    if (run && run.sub !== sub) return 'user mismatch';
    return null;
  }

  private refuse(socket: PeerSocket, reason: string, code = POLICY_CLOSE) {
    try {
      socket.close(code, reason);
    } catch {
      // Already closed.
    }
  }

  private forgetIfEmpty(runId: string, run: Run) {
    // A held run is kept until its window ends so a returning agent waits.
    if (
      !run.agent &&
      !run.ext &&
      run.pendingExts.size === 0 &&
      !run.holdTimer &&
      this.runs.get(runId) === run
    ) {
      this.runs.delete(runId);
    }
  }

  private sendTo(
    runId: string,
    socket: PeerSocket,
    message: unknown,
    agent?: AgentSide,
  ) {
    if (agent?.closed) return;
    const buffered = socket.bufferedAmount;
    if (typeof buffered === 'number' && buffered > MAX_BUFFERED_BYTES) {
      // The peer is not draining: drop the send and end the run, which both
      // sides see as a normal close with a fixed reason.
      if (this.runs.has(runId)) {
        this.log({
          event: 'backpressure-closed',
          severity: 'WARNING',
          runId,
        });
        this.closeRun(runId, BACKPRESSURE_REASON);
      }
      return;
    }
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
