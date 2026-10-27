import type { PendingInputRequest } from '@/lib/ai/eve/input-requests';
import type { RunProgress } from '@/lib/relay/progress';
import { createDbRunStore } from './db-run-store';

// Server-only through db-run-store (lib/db/queries.ts). Nothing under agent/
// (the Eve bundle) may import this module.

export interface EveRunRecord {
  eveSessionId: string;
  continuationToken: string;
  /** Count of stream events already forwarded; the next read's startIndex. */
  streamIndex: number;
  /** Exact https origin of the relay page; set only for extension runs. */
  relayOrigin: string | null;
  pendingRequests: PendingInputRequest[];
  progress: RunProgress | null;
}

/**
 * Thrown by `create` when a run already exists for (userId, chatId), or when
 * a relay run is created on a chatId that already has one (any user).
 */
export class EveRunExistsError extends Error {
  constructor(options?: { cause?: unknown }) {
    super('Eve run already exists for this user and chat', options);
    this.name = 'EveRunExistsError';
  }
}

/**
 * setPending, setProgress, advance and remove are no-ops when no run exists
 * for (userId, chatId); they never create a row.
 */
export interface EveRunStore {
  get(userId: string, chatId: string): Promise<EveRunRecord | undefined>;
  /** The relay run with this chatId; web rows never match. Token route only. */
  getByChatId(
    chatId: string,
  ): Promise<(EveRunRecord & { userId: string }) | undefined>;
  /**
   * Throws EveRunExistsError if a run already exists for (userId, chatId), or
   * if this is a relay run and any user already has a relay run on chatId;
   * any other failure propagates unchanged.
   */
  create(
    userId: string,
    chatId: string,
    init: {
      eveSessionId: string;
      continuationToken: string;
      relayOrigin?: string | null;
    },
  ): Promise<EveRunRecord>;
  /** Relay runs this user created at or after `since`. */
  countRecentRelayRuns(userId: string, since: Date): Promise<number>;
  /** Compare-and-set on continuationToken; true when the swap happened. */
  swapToken(
    userId: string,
    chatId: string,
    expected: string,
    next: string,
  ): Promise<boolean>;
  /**
   * Moves the cursor to streamIndex when that is greater than the stored one.
   * Never writes a token taken from the stream: a replayed turn boundary can
   * carry a token Eve already consumed. The only token write is `heal`: when
   * the stored token is still this turn's claim (its final swap failed), it
   * becomes the token Eve returned for this turn, whatever the index.
   * One atomic update.
   */
  advance(
    userId: string,
    chatId: string,
    update: { streamIndex: number; heal?: { claim: string; token: string } },
  ): Promise<void>;
  setPending(
    userId: string,
    chatId: string,
    requests: PendingInputRequest[],
  ): Promise<void>;
  setProgress(
    userId: string,
    chatId: string,
    progress: RunProgress,
  ): Promise<void>;
  remove(userId: string, chatId: string): Promise<void>;
}

const copy = (r: EveRunRecord): EveRunRecord => structuredClone(r);

/** Single-process store for tests and local use. Never awaits mid-update. */
export function createMemoryRunStore(): EveRunStore {
  const rows = new Map<
    string,
    EveRunRecord & { userId: string; chatId: string; createdAt: Date }
  >();
  const key = (userId: string, chatId: string) => `${userId}:${chatId}`;

  return {
    async get(userId, chatId) {
      const row = rows.get(key(userId, chatId));
      if (!row) return undefined;
      const { userId: _u, chatId: _c, createdAt: _t, ...record } = row;
      return copy(record);
    },
    async getByChatId(chatId) {
      const matches = [...rows.values()].filter(
        (r) => r.chatId === chatId && r.relayOrigin !== null,
      );
      if (matches.length !== 1) return undefined;
      const { chatId: _c, createdAt: _t, ...rest } = matches[0];
      return structuredClone(rest);
    },
    async create(userId, chatId, init) {
      const k = key(userId, chatId);
      if (rows.has(k)) throw new EveRunExistsError();
      if (
        init.relayOrigin != null &&
        [...rows.values()].some(
          (r) => r.chatId === chatId && r.relayOrigin !== null,
        )
      ) {
        throw new EveRunExistsError();
      }
      const record: EveRunRecord = {
        eveSessionId: init.eveSessionId,
        continuationToken: init.continuationToken,
        streamIndex: 0,
        relayOrigin: init.relayOrigin ?? null,
        pendingRequests: [],
        progress: null,
      };
      rows.set(k, { ...copy(record), userId, chatId, createdAt: new Date() });
      return record;
    },
    async countRecentRelayRuns(userId, since) {
      return [...rows.values()].filter(
        (r) =>
          r.userId === userId &&
          r.relayOrigin !== null &&
          r.createdAt >= since,
      ).length;
    },
    async swapToken(userId, chatId, expected, next) {
      const row = rows.get(key(userId, chatId));
      if (!row || row.continuationToken !== expected) return false;
      row.continuationToken = next;
      return true;
    },
    async advance(userId, chatId, update) {
      const row = rows.get(key(userId, chatId));
      if (!row) return;
      if (update.streamIndex > row.streamIndex) {
        row.streamIndex = update.streamIndex;
      }
      if (update.heal && row.continuationToken === update.heal.claim) {
        row.continuationToken = update.heal.token;
      }
    },
    async setPending(userId, chatId, requests) {
      const row = rows.get(key(userId, chatId));
      if (row) row.pendingRequests = structuredClone(requests);
    },
    async setProgress(userId, chatId, progress) {
      const row = rows.get(key(userId, chatId));
      if (row) row.progress = structuredClone(progress);
    },
    async remove(userId, chatId) {
      rows.delete(key(userId, chatId));
    },
  };
}

let store: EveRunStore | null = null;

/** The process-wide store; the Postgres-backed one unless a test swapped it. */
export function getRunStore(): EveRunStore {
  if (!store) store = createDbRunStore();
  return store;
}

export function setRunStoreForTests(next: EveRunStore | null): void {
  store = next;
}
