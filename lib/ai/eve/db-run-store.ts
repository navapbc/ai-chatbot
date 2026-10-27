import 'server-only';

import {
  advanceEveRun,
  countRecentRelayRuns,
  createEveRun,
  deleteEveRun,
  getEveRun,
  getEveRunByChatId,
  setEveRunPending,
  setEveRunProgress,
  swapEveRunToken,
} from '@/lib/db/queries';
import type { EveRun } from '@/lib/db/schema';
import {
  EveRunExistsError,
  type EveRunRecord,
  type EveRunStore,
} from './run-store';

// (userId, chatId) constraint, and the one-relay-run-per-chatId partial index.
const UNIQUE_CONSTRAINTS = new Set([
  'EveRun_userId_chatId_unique',
  'EveRun_chatId_relay_unique',
]);

/** Postgres unique violation (23505) on one of the EveRun unique indexes. */
const isDuplicateRun = (error: unknown): boolean => {
  for (let e = error, depth = 0; e && depth < 3; depth++) {
    const { code, constraint_name } = e as {
      code?: unknown;
      constraint_name?: unknown;
    };
    if (
      code === '23505' &&
      typeof constraint_name === 'string' &&
      UNIQUE_CONSTRAINTS.has(constraint_name)
    ) {
      return true;
    }
    e = (e as { cause?: unknown }).cause;
  }
  return false;
};

const toRecord = (row: EveRun): EveRunRecord => ({
  eveSessionId: row.eveSessionId,
  continuationToken: row.continuationToken,
  streamIndex: row.streamIndex,
  relayOrigin: row.relayOrigin,
  pendingRequests: row.pendingRequests,
  progress: row.progress,
});

/** Postgres-backed run store over the EveRun table (lib/db/queries.ts). */
export function createDbRunStore(): EveRunStore {
  return {
    async get(userId, chatId) {
      const row = await getEveRun({ userId, chatId });
      return row ? toRecord(row) : undefined;
    },
    async getByChatId(chatId) {
      const row = await getEveRunByChatId({ chatId });
      return row ? { ...toRecord(row), userId: row.userId } : undefined;
    },
    async create(userId, chatId, init) {
      try {
        return toRecord(await createEveRun({ userId, chatId, ...init }));
      } catch (error) {
        if (isDuplicateRun(error)) throw new EveRunExistsError({ cause: error });
        throw error;
      }
    },
    countRecentRelayRuns: (userId, since) =>
      countRecentRelayRuns({ userId, since }),
    swapToken: (userId, chatId, expected, next) =>
      swapEveRunToken({ userId, chatId, expected, next }),
    advance: (userId, chatId, update) =>
      advanceEveRun({ userId, chatId, ...update }),
    setPending: (userId, chatId, pendingRequests) =>
      setEveRunPending({ userId, chatId, pendingRequests }),
    setProgress: (userId, chatId, progress) =>
      setEveRunProgress({ userId, chatId, progress }),
    remove: (userId, chatId) => deleteEveRun({ userId, chatId }),
  };
}
