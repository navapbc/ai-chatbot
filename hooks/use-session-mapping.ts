'use client';

import { useEffect, useRef } from 'react';
import { usePostHog } from 'posthog-js/react';

/**
 * Reports the current PostHog session id to the server so it can be correlated
 * with the chat (and, server-side, with the Kernel browser session/replay) in
 * the SessionMapping table.
 *
 * The PostHog session id is only knowable in the browser, so the client is the
 * only place that can supply it. We wait until `ready` is true — i.e. the chat
 * row exists in the DB — because the server route enforces a chat-ownership
 * check before inserting (the chat is created lazily on the first message).
 *
 * Re-reports whenever PostHog rotates the session id (its 30-min idle / 24-h
 * cap), keyed so we POST at most once per (chatId, sessionId) pair.
 */
export function useSessionMapping({
  chatId,
  ready,
}: {
  chatId: string;
  ready: boolean;
}) {
  const posthog = usePostHog();
  const reportedKey = useRef<string | null>(null);

  useEffect(() => {
    if (!ready || !posthog) return;

    const posthogSessionId = posthog.get_session_id?.();
    if (!posthogSessionId) return;

    const key = `${chatId}:${posthogSessionId}`;
    if (reportedKey.current === key) return;
    reportedKey.current = key;

    fetch('/api/session-mapping', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ chatId, posthogSessionId }),
    }).catch((error) => {
      // Best-effort telemetry — don't surface to the user, but allow a retry.
      console.error('Failed to report session mapping:', error);
      reportedKey.current = null;
    });
  }, [chatId, ready, posthog]);
}
