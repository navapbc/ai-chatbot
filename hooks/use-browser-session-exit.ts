'use client';

import { useState, useCallback } from 'react';
import { useArtifact } from './use-artifact';

/**
 * Hook to manage browser session exit warnings
 * Returns methods to check if navigation should be intercepted
 * and to handle the confirmation flow
 */
export function useBrowserSessionExit() {
  const { artifact, metadata } = useArtifact();
  const [showExitWarning, setShowExitWarning] = useState(false);
  const [pendingAction, setPendingAction] = useState<(() => void) | null>(null);

  /**
   * Check if there's an active browser session that requires exit warning
   */
  const hasActiveBrowserSession = useCallback(() => {
    return artifact.kind === 'browser' && metadata?.isConnected === true;
  }, [artifact.kind, metadata?.isConnected]);

  /**
   * Intercept navigation and show warning if needed
   * @param action - The navigation action to perform after confirmation
   * @returns boolean - true if navigation should proceed immediately, false if intercepted
   */
  const interceptNavigation = useCallback(
    (action: () => void) => {
      if (hasActiveBrowserSession()) {
        setPendingAction(() => action);
        setShowExitWarning(true);
        return false;
      }
      // No active session, proceed immediately
      action();
      return true;
    },
    [hasActiveBrowserSession],
  );

  /**
   * Handle user confirming they want to leave the session.
   *
   * Leaving is an explicit teardown: tell the server to delete the browser,
   * which stops the replay recording and archives the video to storage before
   * the Kernel session is destroyed. Without this, navigating away would just
   * abandon the session — it would idle into standby (delayed) or be reaped by
   * Kernel's timeout with no video written. Fire-and-forget via sendBeacon so
   * it survives the navigation that runs immediately after.
   */
  const handleConfirmLeave = useCallback(() => {
    setShowExitWarning(false);

    const sessionId = metadata?.sessionId;
    if (sessionId) {
      try {
        const payload = JSON.stringify({ action: 'delete', sessionId });
        navigator.sendBeacon(
          '/api/kernel-browser',
          new Blob([payload], { type: 'application/json' }),
        );
      } catch {
        // Best-effort — don't block leaving on cleanup.
      }
    }

    if (pendingAction) {
      pendingAction();
      setPendingAction(null);
    }
  }, [pendingAction, metadata?.sessionId]);

  /**
   * Handle user canceling the exit
   */
  const handleCancelLeave = useCallback(() => {
    setShowExitWarning(false);
    setPendingAction(null);
  }, []);

  return {
    showExitWarning,
    setShowExitWarning,
    hasActiveBrowserSession: hasActiveBrowserSession(),
    interceptNavigation,
    handleConfirmLeave,
    handleCancelLeave,
  };
}
