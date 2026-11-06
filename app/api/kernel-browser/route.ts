import { auth } from '@/app/(auth)/auth';
import {
  getOrCreateBrowser,
  deleteBrowser,
  getBrowser,
  touchActivity,
  getSessionStatus,
  standbyBrowser,
  reconnectBrowser,
} from '@/lib/kernel/browser';
import { getLiveViewUrl } from '@/lib/ai/eve/live-view-store';

export async function POST(request: Request) {
  const session = await auth();
  if (!session?.user) {
    return Response.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const userId = session.user.id;

  try {
    const { action, sessionId, isMobile, useEve } = await request.json();

    if (!sessionId) {
      return Response.json({ error: 'sessionId is required' }, { status: 400 });
    }

    // Validate session ownership: sessionId must end with `-{userId}`
    if (!sessionId.endsWith(`-${userId}`)) {
      return Response.json(
        { error: 'Forbidden: session does not belong to user' },
        { status: 403 },
      );
    }

    if (action === 'get') {
      const browser = await getBrowser(sessionId, userId);
      if (!browser) {
        // No legacy browser: on the Eve transport the browser belongs to the
        // `eve dev` process, so the only handle this process has is the URL the
        // browser tool reported over the chat stream. `sessionId` is
        // `${chatId}-${userId}` (already ownership-checked above), so trimming
        // the suffix recovers the chatId that store is keyed by.
        const chatId = sessionId.slice(0, -(userId.length + 1));
        const eveLiveViewUrl = getLiveViewUrl(userId, chatId);
        // Deliberately no `sessionId` in this response: the Kernel session id
        // is not known on this side, and the artifact only needs the URL.
        return Response.json({ liveViewUrl: eveLiveViewUrl ?? null });
      }
      return Response.json({
        liveViewUrl: browser.liveViewUrl,
        sessionId: browser.kernelSessionId,
      });
    }

    if (action === 'create') {
      // On the Eve transport the agent owns the browser's lifecycle, inside the
      // `eve dev` process. Creating one HERE would start a second, real Kernel
      // browser (billed, replay-recorded) that the agent is not driving and the
      // user would be watching by mistake — so never create on this path, just
      // report whatever the agent has reported to us.
      if (useEve) {
        const chatId = sessionId.slice(0, -(userId.length + 1));
        return Response.json({
          liveViewUrl: getLiveViewUrl(userId, chatId) ?? null,
        });
      }
      const browser = await getOrCreateBrowser(sessionId, userId, {
        isMobile,
      });
      return Response.json({
        liveViewUrl: browser.liveViewUrl,
        sessionId: browser.kernelSessionId,
      });
    }

    if (action === 'delete') {
      await deleteBrowser(sessionId, userId);
      return Response.json({ success: true });
    }

    if (action === 'heartbeat') {
      const browser = await getBrowser(sessionId, userId);
      if (!browser) {
        return Response.json(
          { error: 'Session expired or not found' },
          { status: 404 },
        );
      }
      return Response.json({
        success: true,
        liveViewUrl: browser.liveViewUrl,
      });
    }

    // Record a user action (click/type in takeover), resetting the idle timer.
    if (action === 'activity') {
      const touched = touchActivity(sessionId, userId);
      return Response.json({ success: touched });
    }

    // Lifecycle snapshot the client uses to drive its idle/cap timers.
    if (action === 'status') {
      return Response.json(getSessionStatus(sessionId, userId));
    }

    // Idle expiry: drop CDP so Kernel moves the browser to standby (no cost,
    // state preserved). The session stays reconnectable.
    if (action === 'standby') {
      const ok = await standbyBrowser(sessionId, userId);
      return Response.json({ success: ok });
    }

    // Wake a standby session, restoring its state.
    if (action === 'reconnect') {
      const browser = await reconnectBrowser(sessionId, userId, { isMobile });
      return Response.json({
        liveViewUrl: browser.liveViewUrl,
        sessionId: browser.kernelSessionId,
      });
    }

    return Response.json({ error: 'Invalid action' }, { status: 400 });
  } catch (error) {
    console.error('Kernel browser API error:', error);
    return Response.json(
      {
        error:
          error instanceof Error ? error.message : 'Failed to manage browser',
      },
      { status: 500 },
    );
  }
}
