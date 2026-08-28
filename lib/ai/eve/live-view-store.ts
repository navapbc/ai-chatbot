// Live-view URLs for Kernel browsers driven by the Eve agent, keyed per
// (user, chat).
//
// Why this exists: on the Eve transport the browser is created inside the
// `eve dev` process (lib/kernel/eve-browser.ts), so its in-memory session map is
// unreachable from any Next.js route — different OS process. The URL instead
// rides out on the browser tool's result, through Eve's event stream, and the
// /api/eve-chat reader records it here. /api/kernel-browser then serves it to
// the browser artifact, which needs no knowledge of which transport produced it.
//
// SINGLE-PROCESS and lost on restart, matching `session-continuity.ts`. Losing
// an entry only blanks the live-view panel until the agent's next browser
// command re-reports the same URL; it never affects the automation itself.
const store = new Map<string, string>();
const key = (userId: string, chatId: string) => `${userId}:${chatId}`;

export function getLiveViewUrl(
  userId: string,
  chatId: string,
): string | undefined {
  return store.get(key(userId, chatId));
}

export function setLiveViewUrl(
  userId: string,
  chatId: string,
  liveViewUrl: string,
): void {
  store.set(key(userId, chatId), liveViewUrl);
}

export function clearLiveViewUrl(userId: string, chatId: string): void {
  store.delete(key(userId, chatId));
}
