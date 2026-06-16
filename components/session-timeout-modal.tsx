'use client';

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatTime(totalSeconds: number): string {
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, '0')}`;
}

// ---------------------------------------------------------------------------
// SessionTimeoutModal
// ---------------------------------------------------------------------------

interface SessionTimeoutModalProps {
  /** Whether the dialog is visible. */
  open: boolean;
  /** Called when the dialog open state changes. */
  onOpenChange: (open: boolean) => void;
  /**
   * Live countdown **in seconds**, owned by the caller (the session-lifecycle
   * controller). The modal renders this value; it does not run its own timer,
   * so the displayed time stays in sync with the authoritative idle/cap clock.
   */
  countdownSeconds: number;
  /**
   * Why the session is ending — controls copy only. `idle` (default) follows
   * inactivity; `cap` is the hard session-length limit.
   */
  reason?: 'idle' | 'cap';
  /** Called when the user clicks "End session". */
  onEndSession: () => void;
  /** Called when the user clicks "Continue session". */
  onContinueSession: () => void;
}

export function SessionTimeoutModal({
  open,
  onOpenChange,
  countdownSeconds,
  reason = 'idle',
  onEndSession,
  onContinueSession,
}: SessionTimeoutModalProps) {
  const remaining = Math.max(0, countdownSeconds);

  const handleEndSession = () => {
    onOpenChange(false);
    onEndSession();
  };

  const handleContinueSession = () => {
    onOpenChange(false);
    onContinueSession();
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        showCloseButton={false}
        className="max-w-[480px] bg-card rounded-[6px] border border-border p-0 gap-0"
      >
        <DialogHeader className="px-6 pt-6 pb-0">
          <DialogTitle className="font-source-serif font-semibold leading-[28px] text-card-foreground text-left break-words">
            Your session is ending soon
          </DialogTitle>
        </DialogHeader>

        <p
          className="font-source-serif text-[64px] font-light leading-[64px] text-card-foreground text-left px-6 pt-6 pb-6"
          aria-live="polite"
          aria-atomic="true"
        >
          {formatTime(remaining)}
        </p>

        <div className="h-px bg-border mx-6" />

        <DialogDescription className="px-6 pt-4 pb-2 font-inter text-[14px] font-normal leading-[22px] text-foreground text-left">
          {reason === 'cap' ? (
            <>
              You&apos;ve reached the maximum session length. Select{' '}
              <strong className="font-bold text-foreground">
                Continue session
              </strong>{' '}
              to start a fresh session and keep working.
            </>
          ) : (
            <>
              To keep the system running smoothly, sessions end after
              inactivity. Select{' '}
              <strong className="font-bold text-foreground">
                Continue session
              </strong>{' '}
              to keep working.
            </>
          )}
        </DialogDescription>

        <DialogFooter className="px-6 pb-6 pt-3 flex-row justify-end gap-3">
          <Button
            variant="outline"
            onClick={handleEndSession}
            className="border border-border bg-card text-card-foreground text-[14px] font-medium leading-[24px] px-5 py-2 rounded-[6px] hover:bg-secondary/80 transition-colors"
          >
            End session
          </Button>
          <Button
            onClick={handleContinueSession}
            className="bg-primary text-primary-foreground text-[14px] font-medium leading-[24px] px-5 py-2 rounded-[6px] hover:bg-primary/90 transition-colors border-0"
          >
            Continue session
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
