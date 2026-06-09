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

// =============================================================================
// TEMPORARY SHORT TIMINGS FOR TESTING
// TODO: restore production timings (12-min warning, 3-min countdown, 60-min
// cap, 5-min cap warning) before merging. See PRODUCTION values below.
// =============================================================================

// --- Idle policy ----------------------------------------------------------

/** Inactivity before the idle warning modal appears. */
export const IDLE_WARNING_AFTER_MS = 1 * MINUTE_MS; // PROD: 12 * MINUTE_MS

/** Countdown shown in the warning modal before disconnecting to standby. */
export const IDLE_COUNTDOWN_MS = 1 * MINUTE_MS; // PROD: 3 * MINUTE_MS

/** Total inactivity before disconnect (warning + countdown). */
export const IDLE_DISCONNECT_AFTER_MS =
  IDLE_WARNING_AFTER_MS + IDLE_COUNTDOWN_MS;

// --- Hard cap -------------------------------------------------------------

/** Maximum lifetime of a session regardless of activity. */
export const HARD_CAP_MS = 10 * MINUTE_MS; // PROD: 60 * MINUTE_MS

/** Warning shown before the hard cap ends the session. */
export const CAP_WARNING_BEFORE_MS = 2 * MINUTE_MS; // PROD: 5 * MINUTE_MS

// --- Kernel server-side backstop -----------------------------------------

/**
 * Inactivity timeout passed to Kernel's `browsers.create`. This is the hard
 * cost safety net: if our own standby/cleanup ever fails to disconnect a
 * session, Kernel reaps it after this much network inactivity so billing
 * stops no matter what.
 *
 * Trade-off: a session that has gone to standby is, by definition, network-idle
 * — so Kernel's timer is running against it. This value therefore also bounds
 * how long after standby a user can reconnect before Kernel reaps the browser
 * (after which reconnect recreates from the persistent profile, restoring
 * state). Keep it short for cost safety; reconnect-from-profile covers the rest.
 * Kernel min is 10s, max 72h.
 */
export const KERNEL_TIMEOUT_SECONDS = 90; // PROD: consider 5–10 min

// --- Client polling -------------------------------------------------------

/** How often the client reports/polls session liveness. */
export const ACTIVITY_POLL_MS = 30_000;
