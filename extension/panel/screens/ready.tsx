import { useId, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import { modelLabel } from '../lib/model-label';
import { DEBUGGER_BAR_COPY, type ErrorMessage } from '../lib/copy';
import {
  buildStartRequest,
  type ReadyModel,
  type StartRequest,
} from '../lib/start-request';
import type { Action, State } from '../lib/state';

/**
 * Wired by the shell to the run starter. A rejection becomes `startFailed`,
 * which shows its fixed message here and enables Start again.
 */
export type OnStart = (args: {
  tabId: number;
  siteName: string;
  request: StartRequest;
}) => Promise<unknown> | undefined;

type Props = {
  state: State;
  dispatch: (action: Action) => void;
  onStart: OnStart;
};

const STATEMENTS = [
  'The agent works only on this site.',
  'It asks you for anything that is missing.',
  'It stops before Submit — you submit the form yourself.',
] as const;

const INVALID: ErrorMessage = {
  title: 'We could not start this run',
  detail: 'Go back and check what you chose or entered, then try again.',
  retryable: false,
};

// Names, details and the site are rendered as React text children only. A
// native select is used for the model: it is keyboard- and screen-reader-ready
// with no portal, and works inside a narrow extension page.
export const ReadyScreen = ({ state, dispatch, onStart }: Props) => {
  const modelLabelId = useId();
  const pending = useRef(false);
  const [invalid, setInvalid] = useState(false);
  const { target, participant, config } = state;
  if (!target || !participant || !config) return null;

  const modelId = state.modelId ?? config.defaultModel;
  const starting = state.screen === 'starting';
  const error = invalid ? INVALID : state.error;

  const handleModelChange = (value: string) => {
    dispatch({ type: 'chooseModel', modelId: value });
  };

  const handleBack = () => dispatch({ type: 'goBack' });

  const handleStart = () => {
    if (pending.current || starting) return;
    let request: StartRequest;
    try {
      request = buildStartRequest({
        siteName: target.siteName,
        origin: target.origin,
        participant,
        models: config.models,
        defaultModel: config.defaultModel,
        modelId,
      } satisfies ReadyModel);
    } catch {
      setInvalid(true);
      return;
    }
    setInvalid(false);
    pending.current = true;
    dispatch({ type: 'startRequested' });
    Promise.resolve()
      .then(() =>
        onStart({ tabId: target.tabId, siteName: target.siteName, request }),
      )
      .catch((failure: unknown) => {
        pending.current = false;
        dispatch({ type: 'startFailed', error: failure });
      });
  };

  return (
    <section className="flex flex-col gap-4" aria-label="Ready">
      <div className="flex flex-col gap-1">
        <h2 className="text-lg font-semibold">Ready to fill</h2>
        <p className="truncate text-sm text-muted-foreground">
          {participant.kind === 'record'
            ? participant.name
            : 'Details you entered'}
          {' · '}
          {target.siteName}
        </p>
      </div>
      <div className="flex flex-col gap-2">
        <label htmlFor={modelLabelId} className="text-sm font-medium">
          Model
        </label>
        <select
          id={modelLabelId}
          value={modelId}
          disabled={starting || config.models.length === 1}
          onChange={(event) => handleModelChange(event.target.value)}
          className="h-11 w-full rounded-lg border border-input bg-background px-3 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50"
        >
          {config.models.map((m) => (
            <option key={m.id} value={m.id}>
              {modelLabel(m.id)}
              {m.id === config.defaultModel ? ' (default)' : ''}
            </option>
          ))}
        </select>
        <p className="text-sm text-muted-foreground">
          Faster models cost less; slower ones handle harder forms.
        </p>
      </div>
      <div className="flex flex-col gap-2 rounded-xl border border-border p-3">
        <h3 className="text-sm font-semibold">The agent will</h3>
        <ul className="flex flex-col gap-1 text-sm">
          {STATEMENTS.map((line) => (
            <li key={line}>{line}</li>
          ))}
        </ul>
      </div>
      <p className="text-sm text-muted-foreground">{DEBUGGER_BAR_COPY}</p>
      {error ? (
        <div role="alert" className="flex flex-col gap-1 text-sm">
          <p className="font-semibold">{error.title}</p>
          <p>{error.detail}</p>
        </div>
      ) : null}
      <div className="flex flex-col gap-2">
        <Button type="button" disabled={starting} onClick={handleStart}>
          Start filling
        </Button>
        <Button
          type="button"
          variant="outline"
          disabled={starting}
          onClick={handleBack}
        >
          Back
        </Button>
      </div>
    </section>
  );
};
