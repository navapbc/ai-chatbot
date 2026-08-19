// In-memory per-(user, chat) Eve session continuity. SINGLE-PROCESS and lost
// on restart — SP-C replaces this with a Postgres-backed mapping. A restart
// mid-conversation simply starts a fresh Eve session on the next message.
export interface EveContinuity {
  eveSessionId: string;
  continuationToken: string;
}

const store = new Map<string, EveContinuity>();
const key = (userId: string, chatId: string) => `${userId}:${chatId}`;

export function getContinuity(userId: string, chatId: string): EveContinuity | undefined {
  return store.get(key(userId, chatId));
}
export function setContinuity(userId: string, chatId: string, value: EveContinuity): void {
  store.set(key(userId, chatId), value);
}
export function clearContinuity(userId: string, chatId: string): void {
  store.delete(key(userId, chatId));
}
