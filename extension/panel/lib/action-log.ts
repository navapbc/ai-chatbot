// A session-local log of what the agent reported doing. The agent sends at
// most 5 recent actions per progress update, so successive updates overlap:
// only the entries not already at the tail of the log are appended. Text only,
// never persisted. The entries are untrusted agent text: callers render them
// as plain text.

export const MAX_ACTION_CHARS = 120;
export const MAX_LOG_ENTRIES = 200;

export type ActionLog = {
  add(actions: string[]): void;
  entries(): string[];
};

export const createActionLog = (max = MAX_LOG_ENTRIES): ActionLog => {
  let log: string[] = [];

  return {
    add(actions) {
      const window: string[] = [];
      for (const raw of actions) {
        if (typeof raw !== 'string') continue;
        const text = raw.trim().slice(0, MAX_ACTION_CHARS).trim();
        if (text) window.push(text);
      }
      // The longest suffix of the log that is a prefix of the new window.
      let overlap = Math.min(log.length, window.length);
      for (; overlap > 0; overlap -= 1) {
        let same = true;
        for (let i = 0; i < overlap; i += 1) {
          if (log[log.length - overlap + i] !== window[i]) {
            same = false;
            break;
          }
        }
        if (same) break;
      }
      const fresh = window.slice(overlap);
      if (fresh.length === 0) return;
      log = log.concat(fresh);
      if (log.length > max) log = log.slice(log.length - max);
    },
    entries() {
      return log.slice();
    },
  };
};
