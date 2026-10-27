import { Check } from 'lucide-react';
import { Progress } from '@/components/ui/progress';
import type { ProgressViewModel, SectionView } from '../lib/progress-view';

type Props = {
  model: ProgressViewModel;
};

// Every string below the fixed copy is untrusted agent text: rendered as a
// React text node only, never as HTML or a URL.

const cardClass = 'rounded-xl border border-accent bg-background p-3';

const SectionCard = ({ section }: { section: SectionView }) => {
  if (section.status === 'done') {
    return (
      <li className={`${cardClass} flex items-center gap-2 text-sm`}>
        <Check aria-hidden="true" className="size-4 shrink-0" />
        <span className="min-w-0 flex-1 break-words">{section.name}</span>
        <span className="shrink-0 text-muted-foreground">
          {section.count}
          <span className="sr-only"> fields, done</span>
        </span>
      </li>
    );
  }
  if (section.open) {
    return (
      <li
        aria-current="step"
        className={`${cardClass} flex flex-col gap-2 text-sm`}
      >
        <div className="flex items-center gap-2">
          <span className="min-w-0 flex-1 break-words font-medium">
            {section.name}
          </span>
          {section.status === 'needs_you' ? (
            <span className="shrink-0 font-medium">Needs you</span>
          ) : null}
          <span className="shrink-0 text-muted-foreground">
            {section.count}
          </span>
        </div>
        {section.recent && section.recent.length > 0 ? (
          <ul className="flex flex-col gap-1">
            {section.recent.map((action, i) => (
              <li
                // The log has no ids and may repeat a line; order is stable.
                key={`${i}-${action}`}
                className="break-words font-ibm-plex-mono text-xs text-muted-foreground"
              >
                {action}
              </li>
            ))}
          </ul>
        ) : null}
      </li>
    );
  }
  return (
    <li
      className={`${cardClass} flex items-center gap-2 text-sm text-muted-foreground`}
    >
      <span className="min-w-0 flex-1 break-words">{section.name}</span>
      {section.status === 'needs_you' ? (
        <span className="shrink-0 font-medium text-foreground">Needs you</span>
      ) : null}
      <span className="shrink-0">{section.count}</span>
    </li>
  );
};

/** The progress bar, section cards and (without sections) a flat step list. */
export const ProgressView = ({ model }: Props) => {
  const { bar, sections, steps, sectionLabel } = model;
  const known = bar.total !== undefined;
  const value =
    known && bar.total !== undefined && bar.total > 0
      ? Math.min(100, (bar.filled / bar.total) * 100)
      : 0;
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-col gap-1">
        <Progress
          aria-label="Fields filled"
          value={known ? value : undefined}
          max={100}
          className={known ? 'h-2' : 'h-2 animate-pulse'}
        />
        <div className="flex justify-between gap-2 text-xs text-muted-foreground">
          <span>{bar.label}</span>
          {sectionLabel ? <span>{sectionLabel}</span> : null}
        </div>
      </div>
      {sections.length > 0 ? (
        <ul aria-label="Form sections" className="flex flex-col gap-2">
          {sections.map((section, i) => (
            // Section names can repeat; the list is rebuilt whole each poll.
            <SectionCard key={`${i}-${section.name}`} section={section} />
          ))}
        </ul>
      ) : null}
      {sections.length === 0 && steps.length > 0 ? (
        <ul aria-label="Recent steps" className="flex flex-col gap-1">
          {steps.map((step, i) => (
            <li
              key={`${i}-${step}`}
              className="break-words font-ibm-plex-mono text-xs text-muted-foreground"
            >
              {step}
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
};
