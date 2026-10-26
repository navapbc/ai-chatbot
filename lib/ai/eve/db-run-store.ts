import 'server-only';

import {
  advanceEveRun,
  createEveRun,
  deleteEveRun,
  getEveRun,
  getEveRunByChatId,
  setEveRunPending,
  setEveRunProgress,
  swapEveRunToken,
} from '@/lib/db/queries';
import type { EveRun } from '@/lib/db/schema';
import type { EveRunRecord, EveRunStore } from './run-store';

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
      return toRecord(await createEveRun({ userId, chatId, ...init }));
    },
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
