import type { TypeSensitiveCode, TypeSensitiveResult } from './worker-api';

// The state machine behind the sensitive question's masked input. It holds
// the typed value for as long as the card is mounted and nowhere else: not in
// the panel reducer, a Draft, an answer payload, storage, a log or an error
// string. The only thing that leaves is the value handed to typeSensitive,
// trimmed at both ends. A refusal keeps the value (the caseworker may focus
// the right field and retry); success and dispose clear it.

export type FillView = {
  value: string;
  busy: boolean;
  /** The fixed-copy code of the last refusal, never the worker's text. */
  code: TypeSensitiveCode | null;
};

export type FillDeps = {
  runId: string;
  typeSensitive: (runId: string, value: string) => Promise<TypeSensitiveResult>;
  /** Called after the value was typed and cleared; carries nothing. */
  onFilled: () => void;
};

export type SensitiveFill = {
  view: () => FillView;
  subscribe: (listener: (view: FillView) => void) => () => void;
  setValue: (value: string) => void;
  fill: () => Promise<void>;
  /** Clears the value and ignores any call still in flight. */
  dispose: () => void;
};

export const createSensitiveFill = (deps: FillDeps): SensitiveFill => {
  let value = '';
  let busy = false;
  let code: TypeSensitiveCode | null = null;
  let disposed = false;
  const listeners = new Set<(view: FillView) => void>();

  const view = (): FillView => ({ value, busy, code });
  const emit = () => {
    if (disposed) return;
    const next = view();
    for (const listener of listeners) listener(next);
  };

  const setValue = (next: string) => {
    if (disposed || busy) return;
    value = next;
    code = null;
    emit();
  };

  const fill = async () => {
    // `busy` is set before the first await: a second press finds it set.
    if (disposed || busy) return;
    const trimmed = value.trim();
    if (trimmed === '') return;
    busy = true;
    code = null;
    emit();
    let result: TypeSensitiveResult = { ok: false, code: 'worker_error' };
    try {
      result = await deps.typeSensitive(deps.runId, trimmed);
    } catch {
      result = { ok: false, code: 'worker_error' };
    } finally {
      busy = false;
      if (result.ok) value = '';
    }
    if (disposed) return;
    if (!result.ok) {
      code = result.code;
      emit();
      return;
    }
    emit();
    deps.onFilled();
  };

  const dispose = () => {
    disposed = true;
    value = '';
    listeners.clear();
  };

  return {
    view,
    subscribe: (listener) => {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
    setValue,
    fill,
    dispose,
  };
};
