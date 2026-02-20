import Kernel from '@onkernel/sdk';
import { BrowserManager } from 'agent-browser/dist/browser.js';
import {
  saveBrowserSession,
  getBrowserSession,
  deleteBrowserSession,
} from '@/lib/db/queries';

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
const pendingCreations = new Map<string, Promise<BrowserSession>>();

function cacheKey(userId: string, sessionId: string): string {
  return `${userId}:${sessionId}`;
}

// =============================================================================
// DB reconnection helper
// =============================================================================

/**
 * Try to reconnect to a browser session stored in the DB.
 * Returns null if no record exists or the Kernel session is dead.
 */
async function tryReconnectFromDB(
  sessionId: string,
  userId: string,
): Promise<BrowserSession | null> {
  const record = await getBrowserSession({ sessionId });
  if (!record) return null;

  try {
    const kernelSession = (await kernel.browsers.retrieve(
      record.kernelSessionId,
    )) as {
      session_id: string;
      cdp_ws_url: string;
      browser_live_view_url: string;
      deleted_at?: string | null;
    };

    // Session expired or deleted
    if (kernelSession.deleted_at) {
      await deleteBrowserSession({ sessionId });
      return null;
    }

    // Reconnect BrowserManager via CDP
    const manager = new BrowserManager();
    await manager.launch({
      id: 'reconnect',
      action: 'launch',
      cdpUrl: kernelSession.cdp_ws_url,
    });

    const session: BrowserSession = {
      kernelSessionId: kernelSession.session_id,
      liveViewUrl: kernelSession.browser_live_view_url,
      cdpWsUrl: kernelSession.cdp_ws_url,
      userId,
      browserManager: manager,
    };

    // Update cache and DB with fresh URLs
    const key = cacheKey(userId, sessionId);
    sessions.set(key, session);

    await saveBrowserSession({
      sessionId,
      userId,
      chatId: record.chatId,
      kernelSessionId: session.kernelSessionId,
      liveViewUrl: session.liveViewUrl,
      cdpWsUrl: session.cdpWsUrl,
    });

    console.log(
      `[Kernel] Reconnected to browser ${session.kernelSessionId} from DB for session ${sessionId}`,
    );
    return session;
  } catch (err: unknown) {
    const error = err as { status?: number };
    if (error.status === 404) {
      console.log(
        `[Kernel] DB record for session ${sessionId} points to deleted Kernel session, cleaning up`,
      );
    } else {
      console.error('[Kernel] Failed to reconnect from DB:', err);
    }
    await deleteBrowserSession({ sessionId });
    return null;
  }
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

  // 2. If a create is already in-flight, await it
  const pending = pendingCreations.get(key);
  if (pending) {
    console.log(`[Kernel] Awaiting in-flight create for session ${sessionId}`);
    return pending;
  }

  // 3. Try to reconnect from DB (survives Cloud Run restarts)
  const reconnected = await tryReconnectFromDB(sessionId, userId);
  if (reconnected) return reconnected;

  // 4. Create new browser via Kernel SDK
  const createPromise = (async () => {
    try {
      const viewport = options?.isMobile
        ? { width: 1024, height: 768 }
        : { width: 1280, height: 800 };

      const browser = (await kernel.browsers.create({
        viewport,
        timeout_seconds: 3600,
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

      // Persist to DB for reconnection after Cloud Run restarts
      const chatId = sessionId.replace(`-${userId}`, '');
      await saveBrowserSession({
        sessionId,
        userId,
        chatId,
        kernelSessionId: session.kernelSessionId,
        liveViewUrl: session.liveViewUrl,
        cdpWsUrl: session.cdpWsUrl,
      });

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

  // Fall back to DB (survives Cloud Run restarts)
  return tryReconnectFromDB(sessionId, userId);
}

/**
 * Stop in-flight browser operations without destroying the Kernel browser.
 *
 * Closes the current BrowserManager (severing the CDP connection, which kills
 * all in-flight Playwright commands), then reconnects a fresh BrowserManager
 * to the same cdp_ws_url so future tool calls still work.
 *
 * The Kernel browser itself stays alive — the live-view iframe keeps it alive.
 */
export async function stopBrowserOperations(
  sessionId: string,
  userId: string,
): Promise<void> {
  const key = cacheKey(userId, sessionId);
  const session = sessions.get(key);
  if (!session) return;

  const { cdpWsUrl } = session;

  // Close current BrowserManager — kills all in-flight Playwright actions
  try {
    await session.browserManager.close();
  } catch (err) {
    console.error('[Kernel] Failed to close BrowserManager during stop:', err);
  }

  // Reconnect fresh BrowserManager to same browser via CDP
  try {
    const newManager = new BrowserManager();
    await newManager.launch({ id: 'relaunch', action: 'launch', cdpUrl: cdpWsUrl });
    session.browserManager = newManager;
  } catch (err) {
    // Reconnect failed — remove from cache so next tool call creates fresh
    console.error('[Kernel] Failed to reconnect BrowserManager:', err);
    sessions.delete(key);
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

  // Always clean up DB record (may exist from a previous instance)
  await deleteBrowserSession({ sessionId });

  if (!session) return;

  // Remove from cache
  sessions.delete(key);

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

/**
 * Disconnect CDP without destroying the Kernel browser.
 *
 * Closes the BrowserManager (severs CDP — stops billing), removes from
 * in-memory cache, but keeps the DB record so we can reconnect later.
 */
export async function disconnectBrowser(
  sessionId: string,
  userId: string,
): Promise<void> {
  const key = cacheKey(userId, sessionId);
  const session = sessions.get(key);
  if (!session) return;

  // Close BrowserManager (severs CDP connection)
  try {
    await session.browserManager.close();
  } catch (err) {
    console.error('[Kernel] Failed to close BrowserManager during disconnect:', err);
  }

  // Remove from in-memory cache — DB record stays for reconnection
  sessions.delete(key);
  console.log(`[Kernel] Disconnected CDP for session ${sessionId}`);
}

/**
 * Reconnect to a browser session.
 *
 * Checks in-memory cache first, then falls back to DB lookup + Kernel retrieve.
 */
export async function reconnectBrowser(
  sessionId: string,
  userId: string,
): Promise<BrowserSession | null> {
  const key = cacheKey(userId, sessionId);

  // Already in memory
  const cached = sessions.get(key);
  if (cached) return cached;

  return tryReconnectFromDB(sessionId, userId);
}

