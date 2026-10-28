import { Eye, EyeOff } from 'lucide-react';
import { type ChangeEvent, useEffect, useId, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { NEEDS_YOU_COPY, TYPE_SENSITIVE_COPY } from '../lib/copy';
import {
  createSensitiveFill,
  type FillView,
  type SensitiveFill as FillController,
} from '../lib/sensitive-fill';
import { createWorkerApi, type WorkerApi } from '../lib/worker-api';

// The masked input on a sensitive question. The typed value lives in the
// controller (lib/sensitive-fill.ts) and this component's view of it: it is
// never a prop, never in a Draft or the reducer, never logged or stored, and
// it is cleared on success and on unmount. The input has no name and no id,
// and the browser is told not to remember or check it.

type Props = {
  runId: string;
  /** The value was typed in the page and cleared; the answer is `provided`. */
  onFilled: () => void;
  /** The caseworker will type it in the page: the answer is `provided`. */
  onTypeMyself: () => void;
  /** Disables every control (the card is sending or done). */
  disabled?: boolean;
  /** "I typed it in the page" is the current answer. */
  typedMyself?: boolean;
  /** Skip is the current answer. */
  skipped?: boolean;
  /** The worker call; defaults to the real worker client. */
  typeSensitive?: WorkerApi['typeSensitive'];
};

const defaultTypeSensitive: WorkerApi['typeSensitive'] = (runId, value) =>
  createWorkerApi().typeSensitive(runId, value);

const EMPTY: FillView = { value: '', busy: false, code: null };

export const SensitiveFill = ({
  runId,
  onFilled,
  onTypeMyself,
  disabled = false,
  typedMyself = false,
  skipped = false,
  typeSensitive = defaultTypeSensitive,
}: Props) => {
  const helpId = useId();
  const [view, setView] = useState<FillView>(EMPTY);
  const [shown, setShown] = useState(false);
  const controller = useRef<FillController | null>(null);
  const latest = useRef({ onFilled, typeSensitive });
  latest.current = { onFilled, typeSensitive };

  useEffect(() => {
    const next = createSensitiveFill({
      runId,
      typeSensitive: (id, value) => latest.current.typeSensitive(id, value),
      onFilled: () => latest.current.onFilled(),
    });
    controller.current = next;
    setView(next.view());
    const unsubscribe = next.subscribe(setView);
    return () => {
      unsubscribe();
      next.dispose();
      controller.current = null;
    };
  }, [runId]);

  // Another answer was chosen: the typed value is not needed any more.
  useEffect(() => {
    if (typedMyself || skipped) controller.current?.clear();
  }, [typedMyself, skipped]);

  const handleChange = (event: ChangeEvent<HTMLInputElement>) => {
    controller.current?.setValue(event.target.value);
  };
  const handleFill = () => {
    void controller.current?.fill();
  };
  const handleToggleShown = () => {
    setShown((current) => !current);
  };

  const empty = view.value.trim() === '';

  return (
    <div className="flex flex-col gap-2">
      <p id={helpId} className="text-sm">
        {NEEDS_YOU_COPY.fillInstruction}
      </p>
      <div className="flex items-center gap-2">
        <Input
          type={shown ? 'text' : 'password'}
          autoComplete="off"
          spellCheck={false}
          aria-label={NEEDS_YOU_COPY.fillInputLabel}
          aria-describedby={helpId}
          readOnly={view.busy}
          disabled={disabled}
          value={view.value}
          onChange={handleChange}
        />
        <Button
          type="button"
          variant="outline"
          size="icon"
          aria-label={NEEDS_YOU_COPY.showTyped}
          aria-pressed={shown}
          disabled={disabled}
          onClick={handleToggleShown}
        >
          {shown ? (
            <EyeOff aria-hidden="true" className="size-4" />
          ) : (
            <Eye aria-hidden="true" className="size-4" />
          )}
        </Button>
      </div>
      {view.code ? (
        <p role="alert" className="text-sm">
          {TYPE_SENSITIVE_COPY[view.code]}
        </p>
      ) : null}
      <div className="flex flex-wrap gap-2">
        <Button
          type="button"
          aria-busy={view.busy}
          disabled={disabled || empty || view.busy}
          onClick={handleFill}
        >
          {NEEDS_YOU_COPY.fillIt}
        </Button>
        <Button
          type="button"
          variant={typedMyself ? 'default' : 'outline'}
          aria-pressed={typedMyself}
          disabled={disabled}
          onClick={onTypeMyself}
        >
          {NEEDS_YOU_COPY.typeMyself}
        </Button>
      </div>
    </div>
  );
};
