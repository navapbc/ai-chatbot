import { useId, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
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
      <h2 className="text-base font-semibold">Ready</h2>
      <dl className="flex flex-col gap-2 text-sm">
        <div>
          <dt className="text-muted-foreground">Site</dt>
          <dd className="font-medium">{target.siteName}</dd>
          <dd className="text-muted-foreground">{target.origin}</dd>
        </div>
        <div>
          <dt className="text-muted-foreground">For</dt>
          <dd className="font-medium">
            {participant.kind === 'record'
              ? participant.name
              : 'Details you entered'}
          </dd>
        </div>
        <div className="flex flex-col gap-1">
          <dt className="text-muted-foreground">
            {config.models.length === 1 ? (
              'Model'
            ) : (
              <label htmlFor={modelLabelId}>Model</label>
            )}
          </dt>
          <dd>
            {config.models.length === 1 ? (
              <span className="font-medium">{config.models[0].id}</span>
            ) : (
              <select
                id={modelLabelId}
                value={modelId}
                disabled={starting}
                onChange={(event) => handleModelChange(event.target.value)}
                className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50"
              >
                {config.models.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.id}
                    {m.id === config.defaultModel ? ' (default)' : ''}
                  </option>
                ))}
              </select>
            )}
          </dd>
        </div>
      </dl>
      <ul className="flex list-disc flex-col gap-1 pl-5 text-sm">
        {STATEMENTS.map((line) => (
          <li key={line}>{line}</li>
        ))}
      </ul>
      <p className="text-sm text-muted-foreground">{DEBUGGER_BAR_COPY}</p>
      {error ? (
        <div role="alert" className="flex flex-col gap-1 text-sm">
          <p className="font-semibold">{error.title}</p>
          <p>{error.detail}</p>
        </div>
      ) : null}
      <div className="flex flex-col gap-2">
        <Button type="button" disabled={starting} onClick={handleStart}>
          Start
        </Button>
        <Button
          type="button"
          variant="ghost"
          disabled={starting}
          onClick={handleBack}
        >
          Back
        </Button>
      </div>
    </section>
  );
};
