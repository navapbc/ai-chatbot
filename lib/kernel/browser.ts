import Kernel from '@onkernel/sdk';
import { BrowserManager } from 'agent-browser/dist/browser.js';
import { KERNEL_TIMEOUT_SECONDS } from './session-config';
import {
  buildSessionStatus,
  cacheKey,
  isProfileUsable,
  profileNameFor,
  type SessionStatus,
} from './session-store';

const kernel = new Kernel();

// =============================================================================
// Types
// =============================================================================

export interface BrowserSession {
  kernelSessionId: string;
  liveViewUrl: string;
  cdpWsUrl: string;
  userId: string;
  browserManager: BrowserManager;
  replayId?: string;
  /** Kernel profile name backing this session, so state survives standby. */
  profileName: string;
  /** Epoch ms when the session was first created (drives the hard cap). */
  startedAt: number;
  /** Epoch ms of the last agent or user action (drives the idle timer). */
  lastActivityAt: number;
  /**
   * True while the CDP connection is intentionally disconnected so Kernel can
   * drop the browser to standby (no cost, state preserved). The Kernel browser
   * still exists and can be reconnected.
   */
  standby: boolean;
}

export type { SessionStatus };

// =============================================================================
// In-memory session cache
//
// Single source of truth for session→browser mapping within this process.
// Kernel.sh is the ultimate source of truth for browser lifecycle/timeout.
// No Redis needed — this is a single Cloud Run instance talking to Kernel.
// =============================================================================

const sessions = new Map<string, BrowserSession>();
const pendingCreations = new Map<string, Promise<BrowserSession>>();

/**
 * Ensure a Kernel profile exists so it can be loaded into a browser session.
 *
 * Kernel requires profiles to be created *beforehand* — passing the name of a
 * non-existent profile to `browsers.create` fails with `400 profile not found`.
 * Creating is idempotent from our side: a name that already exists returns a
 * 409 conflict, which we treat as success.
 *
 * Returns true if the profile is ready to use, false if it could not be
 * ensured (caller should then create a browser without a profile rather than
 * fail outright).
 */
async function ensureProfile(profileName: string): Promise<boolean> {
  try {
    await kernel.profiles.create({ name: profileName });
    return true;
  } catch (err: unknown) {
    const status = (err as { status?: number }).status;
    // 409 (already exists) is success for us; any other status is a real
    // failure — log it and let the caller fall back to a profile-less browser.
    if (isProfileUsable(status)) return true;
    console.error('[Kernel] Failed to ensure profile:', err);
    return false;
  }
}

// =============================================================================
// Core operations
// =============================================================================

/**
 * Get or create a browser session for a user's chat.
 *
 * Uses in-memory cache to dedup. If a create is already in-flight for this
 * session, awaits it instead of creating a duplicate. A per-session Kernel
 * profile backs the browser so its state survives standby/reconnect.
 */
export async function getOrCreateBrowser(
  sessionId: string,
  userId: string,
  options?: { isMobile?: boolean },
): Promise<BrowserSession> {
  if (!userId) {
    throw new Error(
      '[Kernel] userId is required for browser session isolation',
    );
  }

  const key = cacheKey(userId, sessionId);

  // 1. Check in-memory cache. A cached session counts as agent activity.
  const cached = sessions.get(key);
  if (cached) {
    cached.lastActivityAt = Date.now();
    return cached;
  }

  // 2. If a create is already in-flight, await it
  const pending = pendingCreations.get(key);
  if (pending) {
    return pending;
  }

  // 3. Create new browser via Kernel SDK
  const createPromise = (async () => {
    try {
      const viewport = options?.isMobile
        ? { width: 1024, height: 768 }
        : { width: 1280, height: 800 };
      const profileName = profileNameFor(sessionId);

      // A persistent profile lets standby → reconnect restore the exact browser
      // state. The profile must exist before it's referenced, so create it
      // first. If that fails for any reason, fall back to a profile-less
      // browser so session creation never breaks — standby reconnect just
      // starts fresh instead of restoring state.
      const hasProfile = await ensureProfile(profileName);

      const browser = (await kernel.browsers.create({
        viewport,
        timeout_seconds: KERNEL_TIMEOUT_SECONDS,
        kiosk_mode: false,
        stealth: true,
        ...(hasProfile
          ? { profile: { name: profileName, save_changes: true } }
          : {}),
      })) as {
        session_id: string;
        cdp_ws_url: string;
        browser_live_view_url: string;
      };

      const manager = new BrowserManager();
      await manager.launch({
        id: 'launch',
        action: 'launch',
        cdpUrl: browser.cdp_ws_url,
      });

      // Start session replay recording
      let replayId: string | undefined;
      try {
        const replay = await kernel.browsers.replays.start(browser.session_id);
        replayId = replay.replay_id;
      } catch (err) {
        console.error('[Kernel] Failed to start replay recording:', err);
      }

      const now = Date.now();
      const session: BrowserSession = {
        kernelSessionId: browser.session_id,
        liveViewUrl: browser.browser_live_view_url,
        cdpWsUrl: browser.cdp_ws_url,
        userId,
        browserManager: manager,
        replayId,
        profileName,
        startedAt: now,
        lastActivityAt: now,
        standby: false,
      };

      sessions.set(key, session);

      return session;
    } finally {
      pendingCreations.delete(key);
    }
  })();

  pendingCreations.set(key, createPromise);
  return createPromise;
}

/**
 * Get an existing browser session from cache.
 * Also awaits any in-flight creation so callers can poll for a browser
 * that another code path (e.g. the tool) is currently creating.
 * Returns null if no session exists and none is being created.
 */
export async function getBrowser(
  sessionId: string,
  userId: string,
): Promise<BrowserSession | null> {
  if (!userId) {
    throw new Error('[Kernel] userId is required for browser session access');
  }

  const key = cacheKey(userId, sessionId);

  const cached = sessions.get(key);
  if (cached) return cached;

  // Await in-flight creation from another code path (e.g. tool execution)
  const pending = pendingCreations.get(key);
  if (pending) {
    return pending;
  }

  return null;
}

/**
 * Record an agent or user action against a session, resetting its idle timer.
 * No-op if the session isn't cached (e.g. already torn down).
 */
export function touchActivity(sessionId: string, userId: string): boolean {
  const session = sessions.get(cacheKey(userId, sessionId));
  if (!session) return false;
  session.lastActivityAt = Date.now();
  return true;
}

/** Lifecycle snapshot for the client's idle/cap timers. */
export function getSessionStatus(
  sessionId: string,
  userId: string,
): SessionStatus {
  const session = sessions.get(cacheKey(userId, sessionId));
  return buildSessionStatus(session, Date.now());
}

/**
 * Move a session to standby: drop the CDP connection so Kernel snapshots the
 * browser and scales to zero (no cost), while the Kernel browser and its
 * profile state are preserved for reconnect. Does NOT delete the browser.
 */
export async function standbyBrowser(
  sessionId: string,
  userId: string,
): Promise<boolean> {
  const session = sessions.get(cacheKey(userId, sessionId));
  if (!session || session.standby) return false;

  session.standby = true;

  // Closing the BrowserManager disconnects Playwright/CDP. Kernel treats an
  // idle CDP connection as the trigger to drop the unikernel into standby.
  try {
    await session.browserManager.close();
  } catch (err) {
    console.error('[Kernel] Failed to close BrowserManager for standby:', err);
  }

  return true;
}

/**
 * Reconnect a standby session: re-establish CDP to the same Kernel browser,
 * waking it from standby with its state intact. If the underlying Kernel
 * browser is gone (reaped past its timeout), recreate it from the persistent
 * profile so state is still restored.
 */
export async function reconnectBrowser(
  sessionId: string,
  userId: string,
  options?: { isMobile?: boolean },
): Promise<BrowserSession> {
  const key = cacheKey(userId, sessionId);
  const session = sessions.get(key);

  // No cached session (or its Kernel browser may be gone) → recreate from
  // profile. getOrCreateBrowser reuses the same profile name, restoring state.
  if (!session) {
    sessions.delete(key);
    return getOrCreateBrowser(sessionId, userId, options);
  }

  // Try to wake the existing browser by reconnecting CDP.
  try {
    const manager = new BrowserManager();
    await manager.launch({
      id: 'launch',
      action: 'launch',
      cdpUrl: session.cdpWsUrl,
    });

    session.browserManager = manager;
    session.standby = false;
    session.lastActivityAt = Date.now();
    return session;
  } catch (err) {
    // Kernel browser likely reaped — recreate from the persistent profile.
    console.error(
      '[Kernel] CDP reconnect failed, recreating from profile:',
      err,
    );
    sessions.delete(key);
    return getOrCreateBrowser(sessionId, userId, options);
  }
}

/**
 * Delete a browser session.
 * Removes from cache, then tells Kernel to destroy the browser instance.
 */
export async function deleteBrowser(
  sessionId: string,
  userId: string,
): Promise<void> {
  const key = cacheKey(userId, sessionId);
  const session = sessions.get(key);

  if (!session) return;

  // Remove from cache first
  sessions.delete(key);

  // Stop replay recording and log the view URL
  if (session.replayId) {
    try {
      await kernel.browsers.replays.stop(session.replayId, {
        id: session.kernelSessionId,
      });
      await kernel.browsers.replays.list(session.kernelSessionId);
    } catch (err) {
      console.error('[Kernel] Failed to stop/list replays:', err);
    }
  }

  // Close BrowserManager (disconnects Playwright from CDP)
  try {
    await session.browserManager.close();
  } catch (err) {
    console.error('[Kernel] Failed to close BrowserManager:', err);
  }

  // Delete from Kernel
  try {
    await kernel.browsers.deleteByID(session.kernelSessionId);
  } catch (err: unknown) {
    const error = err as { status?: number };
    if (error.status !== 404) {
      console.error('[Kernel] Failed to delete browser:', err);
    }
  }
}
