import type { RunProgress, SectionStatus } from '../../../lib/relay/progress';

// Pure view model for the Running screen's progress. Everything here is
// untrusted text from the agent: it is trimmed and capped, never interpreted.
// Escaping is the renderer's job (React text nodes). Never throws.

export type SectionView = {
  name: string;
  status: SectionStatus;
  filled: number;
  total?: number;
  /** True for the active section only. */
  open: boolean;
  /** "3 of 8" when the total is known, else "3". */
  count: string;
  /** The last actions; only on the active section (the log is global). */
  recent?: string[];
};

export type ProgressViewModel = {
  /** `value` (0..100) only for a determinate bar (every total known). */
  bar: { filled: number; total?: number; value?: number; label: string };
  /** "Section X of Y"; absent when there are no sections. */
  sectionLabel?: string;
  sections: SectionView[];
  /** Flat list of recent steps, used only when there are no sections. */
  steps: string[];
};

const MAX_SECTIONS = 40;
const MAX_NAME = 80;
const MAX_ACTION = 120;
const MAX_STEPS = 8;
const MAX_RECENT = 5;
const STATUSES: readonly SectionStatus[] = [
  'waiting',
  'filling',
  'done',
  'needs_you',
];

// Capped so the sum of 40 sections stays an exact, readable integer.
const MAX_COUNT = 1_000_000;

const toCount = (v: unknown): number =>
  typeof v === 'number' && Number.isFinite(v)
    ? Math.min(MAX_COUNT, Math.max(0, Math.floor(v)))
    : 0;

// Cut by code points so an emoji is never split into a lone surrogate.
export const cut = (text: string, max: number): string =>
  Array.from(text).slice(0, max).join('');

const toName = (v: unknown): string => {
  const text = typeof v === 'string' ? cut(v.trim(), MAX_NAME).trim() : '';
  return text === '' ? 'Untitled section' : text;
};

const toActions = (v: unknown): string[] =>
  Array.isArray(v)
    ? v
        .filter((a): a is string => typeof a === 'string')
        .map((a) => cut(a, MAX_ACTION))
    : [];

export type Clean = {
  name: string;
  status: SectionStatus;
  filled: number;
  total?: number;
};

export const cleanSection = (raw: unknown): Clean => {
  const s = (typeof raw === 'object' && raw !== null ? raw : {}) as Record<
    string,
    unknown
  >;
  const status = STATUSES.includes(s.status as SectionStatus)
    ? (s.status as SectionStatus)
    : 'waiting';
  const hasTotal = typeof s.total === 'number' && Number.isFinite(s.total);
  const total = hasTotal ? toCount(s.total) : undefined;
  const filled = toCount(s.filled);
  return {
    name: toName(s.name),
    status,
    filled: total === undefined ? filled : Math.min(filled, total),
    ...(total === undefined ? {} : { total }),
  };
};

const activeIndex = (sections: Clean[], current: unknown): number => {
  if (sections.length === 0) return -1;
  if (typeof current === 'string') {
    const wanted = toName(current);
    const named = sections.findIndex((s) => s.name === wanted);
    if (named >= 0) return named;
  }
  for (const status of ['filling', 'needs_you'] as const) {
    const at = sections.findIndex((s) => s.status === status);
    if (at >= 0) return at;
  }
  const open = sections.findIndex((s) => s.status !== 'done');
  return open >= 0 ? open : sections.length - 1;
};

export const buildProgressView = (
  progress: RunProgress | null,
  actions: string[],
): ProgressViewModel => {
  const log = toActions(actions);
  const raw =
    progress && typeof progress === 'object' && !Array.isArray(progress)
      ? progress
      : null;
  if (!raw) {
    return {
      bar: { filled: 0, label: 'Starting...' },
      sections: [],
      steps: log.slice(-MAX_STEPS),
    };
  }
  const list: unknown[] = Array.isArray(raw.sections) ? raw.sections : [];
  const clean = list.slice(0, MAX_SECTIONS).map(cleanSection);
  const filled = clean.reduce((sum, s) => sum + s.filled, 0);
  const known = clean.length > 0 && clean.every((s) => s.total !== undefined);
  const total = known
    ? clean.reduce((sum, s) => sum + (s.total ?? 0), 0)
    : undefined;

  const at = activeIndex(clean, raw.current);
  // A finished section stays collapsed even when `current` still names it.
  const activeOpen = at >= 0 && clean[at].status !== 'done';

  const sections: SectionView[] = clean.map((s, i) => {
    const open = i === at && activeOpen;
    return {
      ...s,
      open,
      count:
        s.total === undefined ? `${s.filled}` : `${s.filled} of ${s.total}`,
      ...(open ? { recent: log.slice(-MAX_RECENT) } : {}),
    };
  });

  const ratio = total !== undefined && total > 0 ? (filled / total) * 100 : 0;
  const value =
    Number.isFinite(ratio) && ratio >= 0 && ratio <= 100 ? ratio : 0;

  return {
    bar: {
      filled,
      ...(total === undefined ? {} : { total, value }),
      label:
        total === undefined
          ? `${filled} fields so far`
          : `${filled} of ${total} fields`,
    },
    ...(clean.length > 0
      ? { sectionLabel: `Section ${at + 1} of ${clean.length}` }
      : {}),
    sections,
    steps: clean.length === 0 ? log.slice(-MAX_STEPS) : [],
  };
};
