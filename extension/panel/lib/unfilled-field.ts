import type { UnfilledFieldResult } from './worker-api';

// The state machine behind Review's "Show me the unfilled field" button. It
// keeps the last result (booleans, a count or a code: nothing from the page)
// and a busy flag that is set before the first await, so a double press makes
// one worker call.

export type UnfilledView = {
  busy: boolean;
  result: UnfilledFieldResult | null;
};

export type UnfilledDeps = {
  runId: string;
  showUnfilledField: (runId: string) => Promise<UnfilledFieldResult>;
};

export type UnfilledField = {
  view: () => UnfilledView;
  subscribe: (listener: (view: UnfilledView) => void) => () => void;
  press: () => Promise<void>;
  /** Ignores any call still in flight. */
  dispose: () => void;
};

export const createUnfilledField = (deps: UnfilledDeps): UnfilledField => {
  let busy = false;
  let result: UnfilledFieldResult | null = null;
  let disposed = false;
  const listeners = new Set<(view: UnfilledView) => void>();

  const view = (): UnfilledView => ({ busy, result });
  const emit = () => {
    if (disposed) return;
    const next = view();
    for (const listener of listeners) listener(next);
  };

  const press = async () => {
    if (disposed || busy) return;
    busy = true;
    result = null;
    emit();
    let next: UnfilledFieldResult = { ok: false, code: 'worker_error' };
    try {
      next = await deps.showUnfilledField(deps.runId);
    } catch {
      next = { ok: false, code: 'worker_error' };
    } finally {
      busy = false;
    }
    result = next;
    emit();
  };

  return {
    view,
    subscribe: (listener) => {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
    press,
    dispose: () => {
      disposed = true;
      listeners.clear();
    },
  };
};
