import { auth } from '@/app/(auth)/auth';
import {
  getOrCreateBrowser,
  deleteBrowser,
  refreshSession,
} from '@/lib/kernel/browser';

// Prevent any CDN, proxy, or browser from caching API responses.
// Cross-user response caching is the most likely cause of users briefly
// seeing another user's Kernel browser session.
const NO_CACHE_HEADERS = {
  'Cache-Control': 'no-store, no-cache, must-revalidate, proxy-revalidate',
  'Pragma': 'no-cache',
  'Expires': '0',
} as const;

function json(data: unknown, init?: ResponseInit) {
  return Response.json(data, {
    ...init,
    headers: { ...NO_CACHE_HEADERS, ...init?.headers },
  });
}

export async function POST(request: Request) {
  const session = await auth();
  if (!session?.user) {
    return json({ error: 'Unauthorized' }, { status: 401 });
  }

  const userId = session.user.id;

  try {
    const { action, sessionId, isMobile } = await request.json();

    if (!sessionId) {
      return json({ error: 'sessionId is required' }, { status: 400 });
    }

    // Validate session ownership: sessionId must end with `-{userId}`
    if (!sessionId.endsWith(`-${userId}`)) {
      return json(
        { error: 'Forbidden: session does not belong to user' },
        { status: 403 },
      );
    }

    if (action === 'create') {
      const browser = await getOrCreateBrowser(sessionId, userId, { isMobile });
      return json({
        liveViewUrl: browser.liveViewUrl,
        kernelSessionId: browser.kernelSessionId,
        // Echo back ownership info so the client can verify the response
        // belongs to the correct session (prevents stale/cached cross-user responses)
        ownerSessionId: sessionId,
        ownerUserId: userId,
      });
    }

    if (action === 'delete') {
      await deleteBrowser(sessionId, userId);
      return json({ success: true, ownerSessionId: sessionId });
    }

    if (action === 'heartbeat') {
      const browser = await refreshSession(sessionId, userId);
      if (!browser) {
        return json(
          { error: 'Session expired or not found', ownerSessionId: sessionId },
          { status: 404 },
        );
      }
      return json({
        success: true,
        liveViewUrl: browser.liveViewUrl,
        // Echo back for client-side verification
        ownerSessionId: sessionId,
        ownerUserId: userId,
      });
    }

    return json({ error: 'Invalid action' }, { status: 400 });
  } catch (error) {
    console.error('Kernel browser API error:', error);
    return json(
      {
        error:
          error instanceof Error ? error.message : 'Failed to manage browser',
      },
      { status: 500 },
    );
  }
}
