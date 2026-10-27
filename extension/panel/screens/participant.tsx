import {
  type KeyboardEvent,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
} from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import type { AppApi } from '../../src/app-api';
import {
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
  const input = useRef<HTMLInputElement>(null);
  const items = useRef<(HTMLButtonElement | null)[]>([]);
  const labelId = useId();

  const controller = useMemo(
    () => createSearchController(api, setState),
    [api],
  );
  useEffect(() => () => controller.dispose(), [controller]);
  useEffect(() => {
    input.current?.focus();
  }, []);

  const handleChange = (next: string) => {
    setValue(next);
    controller.setQuery(next);
  };

  const handleRetry = () => controller.setQuery(value);

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

  return (
    <section className="flex flex-col gap-4" aria-label="Participant">
      <p className="text-sm text-muted-foreground">Site: {siteName}</p>
      <h2 className="text-base font-semibold">Who is this for?</h2>
      <div className="flex flex-col gap-2">
        <label htmlFor={labelId} className="text-sm font-medium">
          Search participants
        </label>
        <Input
          id={labelId}
          ref={input}
          type="search"
          aria-label="Search participants"
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
        <ul className="flex flex-col gap-1" aria-label="Participants">
          {state.results.map((p, index) => (
            <li key={p.recordId}>
              <button
                type="button"
                ref={(el) => {
                  items.current[index] = el;
                }}
                className="flex w-full flex-col rounded-md border border-input px-3 py-2 text-left text-sm hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                onClick={() =>
                  onChoose({
                    kind: 'record',
                    recordId: p.recordId,
                    name: p.name,
                  })
                }
                onKeyDown={(event) => handleItemKeyDown(event, index)}
              >
                <span className="font-medium">{p.name}</span>
                <span className="text-muted-foreground">
                  {p.county}, {p.recordId}
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
        <Button type="button" variant="outline" onClick={() => onDetails()}>
          Enter details instead
        </Button>
        <Button type="button" variant="ghost" onClick={() => onBack()}>
          Back
        </Button>
      </div>
    </section>
  );
};
