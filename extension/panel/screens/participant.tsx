import { type KeyboardEvent, useEffect, useId, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { ParticipantTabs } from './participant-tabs';
import type { AppApi } from '../../src/app-api';
import {
  createControllerLifecycle,
  createSearchController,
  type SearchState,
} from '../lib/search-controller';

type Props = {
  api: Pick<AppApi, 'searchParticipants'>;
  siteName: string;
  onChoose: (participant: {
    kind: 'record';
    recordId: string;
    name: string;
  }) => void;
  onDetails: () => void;
  onBack: () => void;
  onSignedOut: () => void;
};

const announce = (state: SearchState): string => {
  if (state.status === 'loading') return 'Searching';
  if (state.status !== 'results') return '';
  if (state.results.length === 0) return 'No participants found';
  return state.results.length === 1
    ? '1 result'
    : `${state.results.length} results`;
};

// Every server-provided string below is rendered as a React text child, never
// as markup.
export const ParticipantScreen = ({
  api,
  siteName,
  onChoose,
  onDetails,
  onBack,
  onSignedOut,
}: Props) => {
  const [value, setValue] = useState('');
  const [state, setState] = useState<SearchState>({ status: 'idle' });
  const [chosen, setChosen] = useState<{
    recordId: string;
    name: string;
  } | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const items = useRef<(HTMLButtonElement | null)[]>([]);
  const labelId = useId();

  const controller = useRef<ReturnType<typeof createSearchController> | null>(
    null,
  );
  // Built in the effect so a StrictMode unmount/remount gets a fresh one.
  useEffect(() => {
    const lifecycle = createControllerLifecycle(() =>
      createSearchController(api, setState),
    );
    controller.current = lifecycle.mount();
    return () => {
      lifecycle.unmount();
      controller.current = null;
    };
  }, [api]);
  useEffect(() => {
    input.current?.focus();
  }, []);

  const handleChange = (next: string) => {
    setValue(next);
    setChosen(null);
    controller.current?.setQuery(next);
  };

  const handleRetry = () => controller.current?.setQuery(value);

  const handleInputKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key !== 'ArrowDown') return;
    event.preventDefault();
    items.current[0]?.focus();
  };

  const handleItemKeyDown = (
    event: KeyboardEvent<HTMLButtonElement>,
    index: number,
  ) => {
    if (event.key === 'ArrowDown') {
      event.preventDefault();
      items.current[index + 1]?.focus();
      return;
    }
    if (event.key === 'ArrowUp') {
      event.preventDefault();
      if (index === 0) input.current?.focus();
      else items.current[index - 1]?.focus();
    }
  };

  const error = state.status === 'error' ? state : null;

  const handleContinue = () => {
    if (!chosen) return;
    onChoose({ kind: 'record', recordId: chosen.recordId, name: chosen.name });
  };

  return (
    <section className="flex flex-col gap-4" aria-label="Participant">
      <div className="flex flex-col gap-1">
        <h2 className="text-lg font-semibold">Participant</h2>
        <p className="truncate text-sm text-muted-foreground">{siteName}</p>
      </div>
      <ParticipantTabs
        active="system"
        onSystem={() => {}}
        onDetails={() => onDetails()}
      />
      <div className="flex flex-col gap-2">
        <label htmlFor={labelId} className="sr-only">
          Search participants
        </label>
        <Input
          id={labelId}
          ref={input}
          type="search"
          aria-label="Search participants"
          placeholder="Search by name or record ID"
          autoComplete="off"
          value={value}
          onChange={(event) => handleChange(event.target.value)}
          onKeyDown={handleInputKeyDown}
        />
      </div>
      <p role="status" aria-live="polite" className="text-sm">
        {announce(state)}
      </p>
      {state.status === 'results' && state.results.length > 0 ? (
        <ul className="flex flex-col gap-2" aria-label="Participants">
          {state.results.map((p, index) => (
            <li key={p.recordId}>
              <button
                type="button"
                aria-pressed={chosen?.recordId === p.recordId}
                ref={(el) => {
                  items.current[index] = el;
                }}
                className={`flex w-full items-center justify-between gap-2 rounded-xl border p-3 text-left text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring ${
                  chosen?.recordId === p.recordId
                    ? 'border-primary/40 bg-accent'
                    : 'border-border bg-secondary'
                }`}
                onClick={() => setChosen(p)}
                onKeyDown={(event) => handleItemKeyDown(event, index)}
              >
                <span className="flex min-w-0 flex-col">
                  <span className="truncate font-semibold">{p.name}</span>
                  <span className="truncate text-muted-foreground">
                    {p.county}
                  </span>
                </span>
                <span className="shrink-0 rounded-full bg-secondary px-2 py-0.5 text-xs text-muted-foreground">
                  {p.recordId}
                </span>
              </button>
            </li>
          ))}
        </ul>
      ) : null}
      {error ? (
        <div role="alert" className="flex flex-col gap-2 text-sm">
          <p className="font-semibold">{error.error.title}</p>
          <p>{error.error.detail}</p>
          {error.code === 'signed_out' ? (
            <Button type="button" onClick={() => onSignedOut()}>
              Sign in again
            </Button>
          ) : null}
          {error.code !== 'signed_out' && error.error.retryable ? (
            <Button type="button" onClick={handleRetry}>
              Try again
            </Button>
          ) : null}
        </div>
      ) : null}
      <div className="flex flex-col gap-2">
        <Button type="button" disabled={!chosen} onClick={handleContinue}>
          Continue
        </Button>
        <Button type="button" variant="ghost" onClick={() => onBack()}>
          Back
        </Button>
      </div>
    </section>
  );
};
