// In-memory per-(user, chat) Eve session continuity. SINGLE-PROCESS and lost
// on restart — SP-C replaces this with a Postgres-backed mapping. A restart
// mid-conversation simply starts a fresh Eve session on the next message.
export interface EveContinuity {
  eveSessionId: string;
  continuationToken: string;
  /**
   * Absolute count of stream events already forwarded to the client, i.e. the
   * `startIndex` the next read should resume from.
   *
   * Eve's session stream is durable and replays from event 0 when `startIndex`
   * is omitted (verified against a running server). Without this cursor a
   * follow-up turn re-reads the previous turn's events, hits its
   * `session.waiting`, and stops before reaching any of the new turn's — so the
   * agent's work after a gap-analysis reply never reaches the UI.
   */
  streamIndex: number;
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
