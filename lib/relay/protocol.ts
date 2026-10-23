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

export type ExtensionToRelay =
  | { type: 'register'; runId: string; tabId: number; origin: string }
  | { type: 'reply'; id: string; result?: unknown; error?: string }
  | { type: 'event'; sessionId?: string; method: string; params: unknown }
  | { type: 'tabClosed' };
