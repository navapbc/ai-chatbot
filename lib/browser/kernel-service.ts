import Kernel from '@onkernel/sdk';
import { Redis } from '@upstash/redis';

export interface BrowserSession {
  sessionId: string;
  cdpWsUrl: string;
  liveViewUrl: string;
  chatId: string;
  createdAt: string;
}

const REDIS_SESSION_PREFIX = 'browser-session:';
const SESSION_TTL_SECONDS = 3600; // 1 hour TTL

// Use existing Upstash Redis client
const redis = new Redis({
  url: process.env.UPSTASH_REDIS_REST_URL!,
  token: process.env.UPSTASH_REDIS_REST_TOKEN!,
});

export async function createBrowserSession(chatId: string): Promise<BrowserSession> {
  const kernel = new Kernel();

  // Create browser with stealth mode and kiosk mode enabled
  // headless: false is required to get browser_live_view_url
  const browser = await kernel.browsers.create({
    stealth: true,
    kiosk_mode: true,
    headless: false,
    timeout_seconds: 3600, // 1 hour timeout
  });

  if (!browser.browser_live_view_url) {
    throw new Error('Failed to get browser live view URL from Kernel');
  }

  const session: BrowserSession = {
    sessionId: browser.session_id,
    cdpWsUrl: browser.cdp_ws_url,
    liveViewUrl: browser.browser_live_view_url,
    chatId,
    createdAt: new Date().toISOString(),
  };

  await redis.setex(
    `${REDIS_SESSION_PREFIX}${chatId}`,
    SESSION_TTL_SECONDS,
    JSON.stringify(session)
  );

  return session;
}

export async function deleteBrowserSession(chatId: string): Promise<void> {
  const session = await getBrowserSession(chatId);
  if (session) {
    try {
      const kernel = new Kernel();
      await kernel.browsers.deleteByID(session.sessionId);
    } catch (error) {
      console.error('Failed to delete Kernel browser:', error);
    }
    await redis.del(`${REDIS_SESSION_PREFIX}${chatId}`);
  }
}

export async function getBrowserSession(chatId: string): Promise<BrowserSession | null> {
  const data = await redis.get<string>(`${REDIS_SESSION_PREFIX}${chatId}`);
  if (!data) return null;

  // Handle both string and object responses from Redis
  if (typeof data === 'string') {
    return JSON.parse(data);
  }
  return data as unknown as BrowserSession;
}

export async function getOrCreateBrowserSession(chatId: string): Promise<BrowserSession> {
  let session = await getBrowserSession(chatId);
  if (!session) {
    session = await createBrowserSession(chatId);
  }
  return session;
}

// Refresh TTL when session is actively used
export async function refreshSessionTTL(chatId: string): Promise<void> {
  await redis.expire(`${REDIS_SESSION_PREFIX}${chatId}`, SESSION_TTL_SECONDS);
}
