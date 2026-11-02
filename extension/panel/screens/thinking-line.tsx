import { useEffect, useState } from 'react';

// Shown while the agent works. With a real step from the agent it shows that
// step; before the first one it cycles fixed phrases so the screen does not
// sit still. Each change fades in (see .thinking-fade in panel.css). The step
// text is untrusted agent text, rendered as a React text node only.
const PHRASES = [
  'Opening the form...',
  'Reading the page...',
  'Finding the fields...',
  'Matching them to the record...',
  'Working through the form...',
] as const;

const PHRASE_MS = 3500;

type Props = {
  /** The agent's latest step, when it has reported one. */
  latest?: string;
};

export const ThinkingLine = ({ latest }: Props) => {
  const [index, setIndex] = useState(0);

  useEffect(() => {
    if (latest) return;
    const timer = setInterval(
      () => setIndex((i) => Math.min(i + 1, PHRASES.length - 1)),
      PHRASE_MS,
    );
    return () => clearInterval(timer);
  }, [latest]);

  const text = latest || PHRASES[index];
  return (
    <p
      key={text}
      aria-live="off"
      className="thinking-fade break-words text-sm text-muted-foreground"
    >
      {text}
    </p>
  );
};
