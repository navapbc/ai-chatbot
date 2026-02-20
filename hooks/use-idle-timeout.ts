import { useEffect, useRef, useCallback } from 'react';

interface UseIdleTimeoutOptions {
  /** Idle threshold in ms before triggering */
  timeoutMs: number;
  /** Whether idle detection is active */
  enabled: boolean;
  /** Called when idle threshold is reached */
  onIdle: () => void;
}

const ACTIVITY_EVENTS: Array<keyof DocumentEventMap> = [
  'mousemove',
  'mousedown',
  'keydown',
  'scroll',
  'touchstart',
];

const THROTTLE_MS = 1000;

export function useIdleTimeout({
  timeoutMs,
  enabled,
  onIdle,
}: UseIdleTimeoutOptions) {
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastActivityRef = useRef<number>(0);
  const onIdleRef = useRef(onIdle);
  onIdleRef.current = onIdle;

  const clearTimer = useCallback(() => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const startTimer = useCallback(() => {
    clearTimer();
    timerRef.current = setTimeout(() => {
      onIdleRef.current();
    }, timeoutMs);
  }, [timeoutMs, clearTimer]);

  const resetIdleTimer = useCallback(() => {
    if (!enabled) return;
    startTimer();
  }, [enabled, startTimer]);

  useEffect(() => {
    if (!enabled) {
      clearTimer();
      return;
    }

    // Start the timer immediately when enabled
    startTimer();

    const handleActivity = () => {
      const now = Date.now();
      if (now - lastActivityRef.current < THROTTLE_MS) return;
      lastActivityRef.current = now;
      startTimer();
    };

    for (const event of ACTIVITY_EVENTS) {
      document.addEventListener(event, handleActivity, { passive: true });
    }

    return () => {
      clearTimer();
      for (const event of ACTIVITY_EVENTS) {
        document.removeEventListener(event, handleActivity);
      }
    };
  }, [enabled, startTimer, clearTimer]);

  return { resetIdleTimer };
}
