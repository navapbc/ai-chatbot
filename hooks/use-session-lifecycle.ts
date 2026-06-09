'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import {
  ACTIVITY_POLL_MS,
  CAP_WARNING_BEFORE_MS,
  HARD_CAP_MS,
  IDLE_COUNTDOWN_MS,
  IDLE_DISCONNECT_AFTER_MS,
  IDLE_WARNING_AFTER_MS,
} from '@/lib/kernel/session-config';

/**
 * Why the warning modal is showing.
 * - `idle`: inactivity reached IDLE_WARNING_AFTER_MS; counts down to standby.
 * - `cap`:  approaching the hard session cap; counts down to a full end.
 */
export type WarningReason = 'idle' | 'cap';

/** Action the controller should take this tick. */
export type LifecycleAction =
  | { kind: 'none' }
  | { kind: 'warn'; reason: WarningReason; countdownSeconds: number }
  | { kind: 'standby' }
  | { kind: 'hard-end' };

/**
 * Pure idle + hard-cap policy. Given the clock and the session's start/activity
 * timestamps, decide what should happen. Hard cap takes precedence over idle.
 * Extracted so the timing rules can be unit-tested without React/timers.
 */
export function evaluateLifecycle(
  now: number,
  startedAt: number,
  lastActivityAt: number,
): LifecycleAction {
  const idleFor = now - lastActivityAt;
  const aliveFor = now - startedAt;

  const capRemaining = HARD_CAP_MS - aliveFor;
  if (capRemaining <= 0) return { kind: 'hard-end' };
  if (capRemaining <= CAP_WARNING_BEFORE_MS) {
    return {
      kind: 'warn',
      reason: 'cap',
      countdownSeconds: Math.ceil(capRemaining / 1000),
    };
  }

  if (idleFor >= IDLE_DISCONNECT_AFTER_MS) return { kind: 'standby' };
  if (idleFor >= IDLE_WARNING_AFTER_MS) {
    return {
      kind: 'warn',
      reason: 'idle',
      countdownSeconds: Math.ceil((IDLE_DISCONNECT_AFTER_MS - idleFor) / 1000),
    };
  }

  return { kind: 'none' };
}

export interface SessionLifecycle {
  /** Whether the warning modal should be visible. */
  warningOpen: boolean;
  /** Why the warning is showing (drives copy + the expiry action). */
  warningReason: WarningReason | null;
  /** Seconds left in the active countdown (for the modal timer). */
  countdownSeconds: number;
  /** True once the session has been moved to standby on idle expiry. */
  isStandby: boolean;
  /** Report a user action (e.g. entering takeover); resets the idle timer. */
  recordUserActivity: () => void;
  /** "Continue session" — dismiss the warning and reset idle activity. */
  continueSession: () => void;
  /** Reconnect from standby (or a hard-ended session). */
  reconnect: () => void;
}

interface Options {
  sessionId: string;
  /** True once the live view is connected; the timers only run while connected. */
  isConnected: boolean;
  isMobile: boolean;
  /** Called when idle expiry moves the session to standby. */
  onStandby: () => void;
  /** Called when the hard cap fully ends the session. */
  onHardEnd: () => void;
  /** Called when the user reconnects; receives the fresh live view URL. */
  onReconnected: (liveViewUrl: string) => void;
}

async function postAction(
  sessionId: string,
  action: string,
  isMobile: boolean,
) {
  return fetch('/api/kernel-browser', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action, sessionId, isMobile }),
  });
}

/**
 * Single session-lifecycle controller for the embedded browser.
 *
 * Owns the idle timer (last user OR agent action → warning → countdown →
 * standby) and the hard-cap timer (session start → warning → end), and exposes
 * the state the warning modal needs. Agent activity is tracked server-side
 * (each browser tool call bumps `lastActivityAt`); user activity is reported
 * via `recordUserActivity`. The client polls server status so agent actions —
 * which happen without any client event — also reset the idle countdown.
 */
export function useSessionLifecycle({
  sessionId,
  isConnected,
  isMobile,
  onStandby,
  onHardEnd,
  onReconnected,
}: Options): SessionLifecycle {
  const [warningOpen, setWarningOpen] = useState(false);
  const [warningReason, setWarningReason] = useState<WarningReason | null>(
    null,
  );
  const [countdownSeconds, setCountdownSeconds] = useState(0);
  const [isStandby, setIsStandby] = useState(false);

  // Authoritative timestamps (epoch ms, server-aligned). Refs so the ticking
  // interval reads fresh values without re-subscribing every second.
  const lastActivityAtRef = useRef<number>(Date.now());
  const startedAtRef = useRef<number>(Date.now());
  const warningReasonRef = useRef<WarningReason | null>(null);
  warningReasonRef.current = warningReason;

  // Keep callbacks/flags in refs to avoid re-arming the timer loop.
  const isMobileRef = useRef(isMobile);
  isMobileRef.current = isMobile;
  const onStandbyRef = useRef(onStandby);
  onStandbyRef.current = onStandby;
  const onHardEndRef = useRef(onHardEnd);
  onHardEndRef.current = onHardEnd;

  const enterStandby = useCallback(() => {
    setIsStandby(true);
    setWarningOpen(false);
    setWarningReason(null);
    postAction(sessionId, 'standby', isMobileRef.current).catch((err) =>
      console.error('[lifecycle] standby failed:', err),
    );
    onStandbyRef.current();
  }, [sessionId]);

  const recordUserActivity = useCallback(() => {
    lastActivityAtRef.current = Date.now();
    // Best-effort server sync so agent-side status reflects the user action.
    postAction(sessionId, 'activity', isMobileRef.current).catch(() => {});
  }, [sessionId]);

  const continueSession = useCallback(() => {
    setWarningOpen(false);
    setWarningReason(null);
    recordUserActivity();
  }, [recordUserActivity]);

  const reconnect = useCallback(async () => {
    try {
      const res = await postAction(sessionId, 'reconnect', isMobileRef.current);
      const data = await res.json();
      if (data.liveViewUrl) {
        const now = Date.now();
        startedAtRef.current = now;
        lastActivityAtRef.current = now;
        setIsStandby(false);
        setWarningOpen(false);
        setWarningReason(null);
        onReconnected(data.liveViewUrl);
      }
    } catch (err) {
      console.error('[lifecycle] reconnect failed:', err);
    }
  }, [sessionId, onReconnected]);

  // Poll server status so agent activity (which produces no client event) and
  // the authoritative session start time keep the local timers honest.
  useEffect(() => {
    if (!isConnected) return;
    let cancelled = false;

    const sync = async () => {
      try {
        const res = await postAction(sessionId, 'status', isMobileRef.current);
        const data = await res.json();
        if (cancelled || !data?.exists) return;
        // Translate server clock to local clock via the `now` it reported.
        const skew = Date.now() - data.now;
        lastActivityAtRef.current = Math.max(
          lastActivityAtRef.current,
          data.lastActivityAt + skew,
        );
        startedAtRef.current = data.startedAt + skew;
      } catch {
        // Network hiccup — local timers keep running off last known values.
      }
    };

    sync();
    const id = setInterval(sync, ACTIVITY_POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [isConnected, sessionId]);

  // The one-second tick that evaluates idle + cap policy.
  useEffect(() => {
    if (!isConnected || isStandby) return;

    const tick = () => {
      const action = evaluateLifecycle(
        Date.now(),
        startedAtRef.current,
        lastActivityAtRef.current,
      );

      switch (action.kind) {
        case 'hard-end':
          setWarningOpen(false);
          onHardEndRef.current();
          return;
        case 'standby':
          enterStandby();
          return;
        case 'warn':
          setWarningReason(action.reason);
          setWarningOpen(true);
          setCountdownSeconds(action.countdownSeconds);
          return;
        case 'none':
          // Within the normal activity window — ensure the warning is hidden.
          if (warningReasonRef.current) {
            setWarningOpen(false);
            setWarningReason(null);
          }
          return;
      }
    };

    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, [isConnected, isStandby, enterStandby]);

  return {
    warningOpen,
    warningReason,
    countdownSeconds,
    isStandby,
    recordUserActivity,
    continueSession,
    reconnect,
  };
}

// Re-export timing so consumers (and tests) can reference the same constants.
export {
  IDLE_COUNTDOWN_MS,
  IDLE_WARNING_AFTER_MS,
  HARD_CAP_MS,
  CAP_WARNING_BEFORE_MS,
};
