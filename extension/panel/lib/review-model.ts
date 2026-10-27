import type { RunProgress } from '../../../lib/relay/progress';
import { cleanSection, cut } from './progress-view';

// Pure view model for the Review panel. Names and counts come from the
// agent's progress, so they go through the same cleaning as the Running
// view (cut by code points, counts clamped). Never throws.

export type LeftItem = {
  name: string;
  reason: 'needs_you' | 'incomplete';
  /** "2 of 5" when the total is known, else "2". */
  count: string;
};

export type ReviewModel = { left: LeftItem[]; hasProgress: boolean };

const MAX_SECTIONS = 40;
const MAX_ACTIONS = 50;
const MAX_ACTION = 120;

export const buildReviewModel = (progress: RunProgress | null): ReviewModel => {
  const list: unknown =
    progress && typeof progress === 'object' && !Array.isArray(progress)
      ? progress.sections
      : null;
  if (!Array.isArray(list) || list.length === 0) {
    return { left: [], hasProgress: false };
  }
  const left: LeftItem[] = [];
  for (const raw of list.slice(0, MAX_SECTIONS)) {
    const s = cleanSection(raw);
    const needsYou = s.status === 'needs_you';
    const incomplete = s.total !== undefined && s.filled < s.total;
    if (!needsYou && !incomplete) continue;
    left.push({
      name: s.name,
      reason: needsYou ? 'needs_you' : 'incomplete',
      count:
        s.total === undefined ? `${s.filled}` : `${s.filled} of ${s.total}`,
    });
  }
  return { left, hasProgress: true };
};

/** The session's action log for display: last 50, each cut to 120. */
export const reviewActions = (actions: unknown): string[] =>
  Array.isArray(actions)
    ? actions
        .filter((a): a is string => typeof a === 'string')
        .slice(-MAX_ACTIONS)
        .map((a) => cut(a, MAX_ACTION))
    : [];

/**
 * A run ended through "I submitted it" goes straight to Confirmed, so the
 * running screen shows nothing for it instead of flashing "Run ended".
 */
export const isConfirming = (state: { endedReason?: string } | null): boolean =>
  state?.endedReason === 'submitted';
