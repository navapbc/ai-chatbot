import Kernel from '@onkernel/sdk';
import { BrowserManager } from 'agent-browser/dist/browser.js';

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
}

// =============================================================================
// In-memory session cache
//
// Single source of truth for session→browser mapping within this process.
// Kernel.sh is the ultimate source of truth for browser lifecycle/timeout.
// No Redis needed — this is a single Cloud Run instance talking to Kernel.
// =============================================================================

const sessions = new Map<string, BrowserSession>();
const stoppedSessions = new Map<string, Omit<BrowserSession, 'browserManager'>>();
const pendingCreations = new Map<string, Promise<BrowserSession>>();

function cacheKey(userId: string, sessionId: string): string {
  return `${userId}:${sessionId}`;
}

// =============================================================================
// Core operations
// =============================================================================

/**
 * Get or create a browser session for a user's chat.
 *
 * Uses in-memory cache to dedup. If a create is already in-flight for this
 * session, awaits it instead of creating a duplicate. Kernel handles all
 * timeout/lifecycle logic.
 */
export async function getOrCreateBrowser(
  sessionId: string,
  userId: string,
  options?: { isMobile?: boolean },
): Promise<BrowserSession> {
  if (!userId) {
    throw new Error('[Kernel] userId is required for browser session isolation');
  }

  const key = cacheKey(userId, sessionId);

  // 1. Check in-memory cache
  const cached = sessions.get(key);
  if (cached) {
    console.log(
      `[Kernel] Reusing browser ${cached.kernelSessionId} for session ${sessionId}`,
    );
    return cached;
  }

  // 2. Check if there's a stopped session to resume
  const stopped = stoppedSessions.get(key);
  if (stopped) {
    const resumed = await resumeBrowser(sessionId, userId);
    if (resumed) return resumed;
  }

  // 3. If a create is already in-flight, await it
  const pending = pendingCreations.get(key);
  if (pending) {
    console.log(`[Kernel] Awaiting in-flight create for session ${sessionId}`);
    return pending;
  }

  // 4. Create new browser via Kernel SDK
  const createPromise = (async () => {
    try {
      const viewport = options?.isMobile
        ? { width: 1024, height: 768 }
        : { width: 1280, height: 800 };

      const browser = (await kernel.browsers.create({
        viewport,
        timeout_seconds: 600,
        kiosk_mode: true,
        stealth: true,
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

      const session: BrowserSession = {
        kernelSessionId: browser.session_id,
        liveViewUrl: browser.browser_live_view_url,
        cdpWsUrl: browser.cdp_ws_url,
        userId,
        browserManager: manager,
      };

      sessions.set(key, session);
      console.log(
        `[Kernel] Created browser ${browser.session_id} for session ${sessionId}`,
      );

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
    console.log(
      `[Kernel] getBrowser awaiting in-flight create for session ${sessionId}`,
    );
    return pending;
  }

  return null;
}

/**
 * Stop a browser session.
 * Closes the BrowserManager (CDP connection) so the agent can no longer control
 * the browser. The Kernel browser instance is NOT deleted — it stays alive for
 * user interaction via the live view. Session info is preserved so it can be
 * resumed later via resumeBrowser().
 */
export async function stopBrowser(
  sessionId: string,
  userId: string,
): Promise<void> {
  const key = cacheKey(userId, sessionId);
  const session = sessions.get(key);

  if (!session) return;

  // Move session info to stopped cache for later resume
  stoppedSessions.set(key, {
    kernelSessionId: session.kernelSessionId,
    liveViewUrl: session.liveViewUrl,
    cdpWsUrl: session.cdpWsUrl,
    userId: session.userId,
  });

  // Remove from active sessions so the agent can no longer access it
  sessions.delete(key);

  // Close BrowserManager (disconnects Playwright from CDP)
  try {
    await session.browserManager.close();
    console.log(`[Kernel] Stopped browser ${session.kernelSessionId} — agent disconnected, browser still alive on Kernel`);
  } catch (err) {
    console.error('[Kernel] Failed to close BrowserManager:', err);
  }
}

/**
 * Resume a stopped browser session.
 * Reconnects a new BrowserManager to the existing Kernel browser via CDP,
 * restoring agent control. Returns null if no stopped session exists.
 */
export async function resumeBrowser(
  sessionId: string,
  userId: string,
): Promise<BrowserSession | null> {
  const key = cacheKey(userId, sessionId);
  const stopped = stoppedSessions.get(key);

  if (!stopped) return null;

  try {
    const manager = new BrowserManager();
    await manager.launch({
      id: 'launch',
      action: 'launch',
      cdpUrl: stopped.cdpWsUrl,
    });

    const session: BrowserSession = {
      ...stopped,
      browserManager: manager,
    };

    // Move back to active sessions
    sessions.set(key, session);
    stoppedSessions.delete(key);
    console.log(`[Kernel] Resumed browser ${stopped.kernelSessionId} — agent CDP reconnected`);

    return session;
  } catch (err) {
    console.error('[Kernel] Failed to resume browser:', err);
    // Clean up the stopped entry if the browser is no longer reachable
    stoppedSessions.delete(key);
    return null;
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

  if (!session) {
    // Also clean up any stopped session entry
    stoppedSessions.delete(key);
    return;
  }

  // Remove from both caches
  sessions.delete(key);
  stoppedSessions.delete(key);

  // Close BrowserManager (disconnects Playwright from CDP)
  try {
    await session.browserManager.close();
  } catch (err) {
    console.error('[Kernel] Failed to close BrowserManager:', err);
  }

  // Delete from Kernel
  try {
    await kernel.browsers.deleteByID(session.kernelSessionId);
    console.log(`[Kernel] Deleted browser ${session.kernelSessionId}`);
  } catch (err: unknown) {
    const error = err as { status?: number };
    if (error.status === 404) {
      console.log(
        `[Kernel] Browser ${session.kernelSessionId} already deleted (404)`,
      );
    } else {
      console.error('[Kernel] Failed to delete browser:', err);
    }
  }
}

