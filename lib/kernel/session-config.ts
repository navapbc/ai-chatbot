/**
 * Browser session lifecycle timing.
 *
 * Single source of truth for the idle-timeout + hard-cap policy so product can
 * tune durations without touching the controller logic. All values are in
 * milliseconds; the `*_SECONDS` exports are derived for the few places that need
 * seconds (e.g. Kernel's `timeout_seconds`).
 *
 * Policy (see idle-timeout spec):
 *   IDLE:  last user/agent action --IDLE_WARNING_AFTER_MS--> warning modal
 *          --IDLE_COUNTDOWN_MS countdown--> disconnect to standby (no cost,
 *          state preserved). Reconnect from UI.
 *   CAP:   session start --HARD_CAP_MS--> hard end. A CAP_WARNING_BEFORE_MS
 *          warning precedes it.
 */

const MINUTE_MS = 60_000;

// --- Idle policy ----------------------------------------------------------

/** Inactivity before the idle warning modal appears. */
export const IDLE_WARNING_AFTER_MS = 12 * MINUTE_MS;

/** Countdown shown in the warning modal before disconnecting to standby. */
export const IDLE_COUNTDOWN_MS = 3 * MINUTE_MS;

/** Total inactivity before disconnect (warning + countdown). */
export const IDLE_DISCONNECT_AFTER_MS =
  IDLE_WARNING_AFTER_MS + IDLE_COUNTDOWN_MS;

// --- Hard cap -------------------------------------------------------------

/** Maximum lifetime of a session regardless of activity. */
export const HARD_CAP_MS = 60 * MINUTE_MS;

/** Warning shown before the hard cap ends the session. */
export const CAP_WARNING_BEFORE_MS = 5 * MINUTE_MS;

// --- Kernel server-side backstop -----------------------------------------

/**
 * Inactivity timeout passed to Kernel's `browsers.create`. Kernel counts CDP +
 * live-view connections as activity, so while the tab is open this never fires;
 * it only reaps orphaned sessions (closed tab, crashed client). We set it
 * comfortably above the hard cap so our own controller, not Kernel, governs the
 * happy path, while standby sessions (CDP intentionally disconnected) survive
 * long enough to be reconnected within the cap.
 */
export const KERNEL_TIMEOUT_SECONDS = Math.ceil(
  (HARD_CAP_MS + 10 * MINUTE_MS) / 1000,
);

// --- Client polling -------------------------------------------------------

/** How often the client reports/polls session liveness. */
export const ACTIVITY_POLL_MS = 30_000;
