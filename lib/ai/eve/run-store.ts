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

/** Thrown by `create` when a run already exists for (userId, chatId). */
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
  /** Internal token route only. Undefined when two users hold the chatId. */
  getByChatId(
    chatId: string,
  ): Promise<(EveRunRecord & { userId: string }) | undefined>;
  /**
   * Throws EveRunExistsError if a run already exists for (userId, chatId);
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
  /** Compare-and-set on continuationToken; true when the swap happened. */
  swapToken(
    userId: string,
    chatId: string,
    expected: string,
    next: string,
  ): Promise<boolean>;
  /** Applies only when streamIndex is greater than the stored one. */
  advance(
    userId: string,
    chatId: string,
    update: { continuationToken: string; streamIndex: number },
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
  const rows = new Map<string, EveRunRecord & { userId: string; chatId: string }>();
  const key = (userId: string, chatId: string) => `${userId}:${chatId}`;

  return {
    async get(userId, chatId) {
      const row = rows.get(key(userId, chatId));
      if (!row) return undefined;
      const { userId: _u, chatId: _c, ...record } = row;
      return copy(record);
    },
    async getByChatId(chatId) {
      const matches = [...rows.values()].filter((r) => r.chatId === chatId);
      if (matches.length !== 1) return undefined;
      const { chatId: _c, ...rest } = matches[0];
      return structuredClone(rest);
    },
    async create(userId, chatId, init) {
      const k = key(userId, chatId);
      if (rows.has(k)) throw new EveRunExistsError();
      const record: EveRunRecord = {
        eveSessionId: init.eveSessionId,
        continuationToken: init.continuationToken,
        streamIndex: 0,
        relayOrigin: init.relayOrigin ?? null,
        pendingRequests: [],
        progress: null,
      };
      rows.set(k, { ...copy(record), userId, chatId });
      return record;
    },
    async swapToken(userId, chatId, expected, next) {
      const row = rows.get(key(userId, chatId));
      if (!row || row.continuationToken !== expected) return false;
      row.continuationToken = next;
      return true;
    },
    async advance(userId, chatId, update) {
      const row = rows.get(key(userId, chatId));
      if (!row || update.streamIndex <= row.streamIndex) return;
      row.streamIndex = update.streamIndex;
      row.continuationToken = update.continuationToken;
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
