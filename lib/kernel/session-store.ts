/**
 * Pure, dependency-free helpers for the browser session store.
 *
 * Kept separate from `lib/kernel/browser.ts` (which imports `@onkernel/sdk`) so
 * this logic can be unit-tested in the browser-mode vitest environment without
 * bundling that server-only dependency.
 */

export interface SessionStatus {
  exists: boolean;
  standby: boolean;
  liveViewUrl: string | null;
  startedAt: number;
  lastActivityAt: number;
  /** Epoch ms (server clock) at the moment this status was produced. */
  now: number;
}

/** Minimal session shape the pure helpers need (subset of BrowserSession). */
export interface SessionLike {
  liveViewUrl: string;
  startedAt: number;
  lastActivityAt: number;
  standby: boolean;
}

/** Cache key for the in-memory session→browser map. */
export function cacheKey(userId: string, sessionId: string): string {
  return `${userId}:${sessionId}`;
}

/**
 * Name of the agent-browser daemon session backing a browser session.
 *
 * The CLI keys its daemon (and therefore the live CDP connection and the
 * `@eN` ref map) by `--session`, so every caller driving the same browser must
 * derive the same name. Reuses `cacheKey` so the daemon and the in-memory cache
 * cannot drift apart.
 */
export function cliSessionName(userId: string, sessionId: string): string {
  return cacheKey(userId, sessionId);
}

/**
 * Decide whether a profile is usable after a `profiles.create` attempt, given
 * the error status (if any).
 * - no error      → created, usable
 * - 409 conflict  → already exists, usable (create is idempotent for us)
 * - anything else → not usable; caller should fall back to no profile
 *
 * Kept pure so the resilience policy can be unit-tested without the SDK.
 */
export function isProfileUsable(errorStatus: number | undefined): boolean {
  if (errorStatus === undefined) return true;
  return errorStatus === 409;
}

/**
 * Stable, Kernel-valid profile name for a session. Kernel requires 1-255 chars
 * of letters, numbers, dots, underscores, or hyphens — sanitize the sessionId
 * (which is `${chatId}-${userId}`) accordingly.
 */
export function profileNameFor(sessionId: string): string {
  return `sess-${sessionId.replace(/[^a-zA-Z0-9._-]/g, '-')}`.slice(0, 255);
}

/**
 * Shape the lifecycle snapshot the client polls. `liveViewUrl` is intentionally
 * null while standby so we never hand back a URL pointing at a paused browser.
 */
export function buildSessionStatus(
  session: SessionLike | undefined,
  now: number,
): SessionStatus {
  if (!session) {
    return {
      exists: false,
      standby: false,
      liveViewUrl: null,
      startedAt: 0,
      lastActivityAt: 0,
      now,
    };
  }
  return {
    exists: true,
    standby: session.standby,
    liveViewUrl: session.standby ? null : session.liveViewUrl,
    startedAt: session.startedAt,
    lastActivityAt: session.lastActivityAt,
    now,
  };
}
