export type RelayToExtension =
  | { id: string; type: 'listTargets' }
  | { id: string; type: 'attach'; targetId: string }
  | { id: string; type: 'createTarget'; url?: string }
  | { id: string; type: 'closeTarget'; targetId: string }
  | {
      id: string;
      type: 'cdp';
      sessionId: string;
      method: string;
      params?: unknown;
    };

/** Why the extension ended a run. A fixed set; the relay logs only these. */
export type StopReason =
  | 'user'
  | 'left-origin'
  | 'debugger-canceled'
  | 'token-expired'
  | 'replaced';

export const STOP_REASONS: ReadonlySet<string> = new Set<StopReason>([
  'user',
  'left-origin',
  'debugger-canceled',
  'token-expired',
  'replaced',
]);

export type ExtensionToRelay =
  | { type: 'register'; runId: string; tabId: number; origin: string }
  | { type: 'reply'; id: string; result?: unknown; error?: string }
  | { type: 'event'; sessionId?: string; method: string; params: unknown }
  | { type: 'tabClosed' }
  | { type: 'stop'; reason?: StopReason }
  /**
   * Sent once, right after `register`, by a run the extension rebuilt from
   * storage after its service worker restarted. Its debugger attachment and
   * child sessions are gone, so the relay drops the agent connection.
   */
  | { type: 'resumed' };
