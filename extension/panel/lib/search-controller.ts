import type { ParticipantSummary } from '../../../lib/data/participant-source';
import type { AppApi } from '../../src/app-api';
import { type ErrorMessage, messageFor } from './copy';

// Debounced, abortable participant search, free of React so it can be tested
// with fake timers. Only the latest started request may produce a state.

export type SearchState =
  | { status: 'idle' }
  | { status: 'loading'; query: string }
  | { status: 'results'; query: string; results: ParticipantSummary[] }
  | { status: 'error'; query: string; error: ErrorMessage; code?: string };

export const MAX_QUERY_LENGTH = 100;

type Options = {
  debounceMs?: number;
  setTimeout?: typeof setTimeout;
  clearTimeout?: typeof clearTimeout;
};

const codeOf = (error: unknown): string | undefined => {
  if (typeof error !== 'object' || error === null) return undefined;
  const code = (error as { code?: unknown }).code;
  return typeof code === 'string' ? code : undefined;
};

export const createSearchController = (
  api: Pick<AppApi, 'searchParticipants'>,
  onState: (state: SearchState) => void,
  opts: Options = {},
) => {
  const debounceMs = opts.debounceMs ?? 250;
  const schedule = opts.setTimeout ?? setTimeout;
  const unschedule = opts.clearTimeout ?? clearTimeout;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let controller: AbortController | undefined;
  let seq = 0;
  let disposed = false;

  const cancel = () => {
    if (timer !== undefined) unschedule(timer);
    timer = undefined;
    controller?.abort();
    controller = undefined;
    // Invalidates any response still on its way.
    seq += 1;
  };

  const emit = (state: SearchState) => {
    if (!disposed) onState(state);
  };

  const run = (query: string) => {
    timer = undefined;
    controller?.abort();
    const mine = new AbortController();
    controller = mine;
    seq += 1;
    const id = seq;
    emit({ status: 'loading', query });
    api.searchParticipants(query, mine.signal).then(
      (results) => {
        if (id !== seq || mine.signal.aborted) return;
        emit({ status: 'results', query, results });
      },
      (error: unknown) => {
        if (id !== seq || mine.signal.aborted) return;
        emit({
          status: 'error',
          query,
          error: messageFor(error),
          code: codeOf(error),
        });
      },
    );
  };

  return {
    setQuery(raw: string) {
      if (disposed) return;
      cancel();
      const query = raw.trim().slice(0, MAX_QUERY_LENGTH);
      if (!query) {
        emit({ status: 'idle' });
        return;
      }
      timer = schedule(() => run(query), debounceMs);
    },
    dispose() {
      disposed = true;
      cancel();
    },
  };
};

/**
 * Mount/unmount lifecycle for an effect: every mount builds a fresh
 * controller, every unmount disposes it for good. Safe under React
 * StrictMode's simulated unmount and remount.
 */
export const createControllerLifecycle = <C extends { dispose(): void }>(
  factory: () => C,
) => {
  let current: C | undefined;
  return {
    mount(): C {
      current?.dispose();
      current = factory();
      return current;
    },
    unmount() {
      current?.dispose();
      current = undefined;
    },
  };
};
