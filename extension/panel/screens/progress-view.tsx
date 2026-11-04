import { Check, CircleAlert, Minus } from 'lucide-react';
import { Progress } from '@/components/ui/progress';
import type { FieldProgress } from '../../../lib/relay/progress';
import type { ProgressViewModel, SectionView } from '../lib/progress-view';

type Props = {
  model: ProgressViewModel;
};

// Every string below the fixed copy is untrusted agent text: rendered as a
// React text node only, never as HTML or a URL.

const cardClass = 'rounded-xl border border-accent bg-background p-3';

const FIELD_TEXT: Record<FieldProgress['state'], string> = {
  waiting: 'Waiting',
  filled: 'Filled',
  needs_you: 'Needs you',
  skipped: 'Skipped',
};

/** Field labels with a state each; labels only, never values. */
const FieldList = ({ fields }: { fields: FieldProgress[] }) => {
  if (fields.length === 0) return null;
  return (
    <ul aria-label="Fields" className="flex flex-col gap-1">
      {fields.map((field, i) => (
        <li
          // Labels can repeat within a section; order is stable.
          key={`${i}-${field.label}`}
          className="flex items-center gap-2 text-xs"
        >
          {field.state === 'filled' ? (
            <Check aria-hidden="true" className="size-3.5 shrink-0" />
          ) : field.state === 'needs_you' ? (
            <CircleAlert aria-hidden="true" className="size-3.5 shrink-0" />
          ) : field.state === 'skipped' ? (
            <Minus aria-hidden="true" className="size-3.5 shrink-0" />
          ) : (
            <span aria-hidden="true" className="size-3.5 shrink-0" />
          )}
          <span
            className={`min-w-0 flex-1 break-words ${
              field.state === 'waiting' ? 'text-muted-foreground' : ''
            }`}
          >
            {field.label}
          </span>
          <span className="shrink-0 text-muted-foreground">
            {FIELD_TEXT[field.state]}
          </span>
        </li>
      ))}
    </ul>
  );
};

/** A collapsed card with its field list; the active section opens itself. */
const FieldsDisclosure = ({ section }: { section: SectionView }) => {
  if (section.fields.length === 0) return null;
  return (
    <details open={section.open} className="text-xs">
      <summary className="cursor-pointer text-muted-foreground">
        Fields ({section.fields.length})
      </summary>
      <div className="pt-1">
        <FieldList fields={section.fields} />
      </div>
    </details>
  );
};

const SectionCard = ({ section }: { section: SectionView }) => {
  if (section.status === 'done') {
    return (
      <li className={`${cardClass} flex flex-col gap-2 text-sm`}>
        <div className="flex items-center gap-2">
          <Check aria-hidden="true" className="size-4 shrink-0" />
          <span className="min-w-0 flex-1 break-words">{section.name}</span>
          <span className="shrink-0 text-muted-foreground">
            {section.count}
            <span className="sr-only"> fields, done</span>
          </span>
        </div>
        <FieldsDisclosure section={section} />
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
        <FieldsDisclosure section={section} />
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
      className={`${cardClass} flex flex-col gap-2 text-sm text-muted-foreground`}
    >
      <div className="flex items-center gap-2">
        <span className="min-w-0 flex-1 break-words">{section.name}</span>
        {section.status === 'needs_you' ? (
          <span className="shrink-0 font-medium text-foreground">
            Needs you
          </span>
        ) : null}
        <span className="shrink-0">{section.count}</span>
      </div>
      <FieldsDisclosure section={section} />
    </li>
  );
};

/** The progress bar, section cards and (without sections) a flat step list. */
export const ProgressView = ({ model }: Props) => {
  const { bar, sections, steps, sectionLabel } = model;
  const determinate = bar.value !== undefined && Number.isFinite(bar.value);
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-col gap-1">
        <Progress
          aria-label="Fields filled"
          value={determinate ? bar.value : undefined}
          max={100}
          className={determinate ? 'h-2' : 'h-2 motion-safe:animate-pulse'}
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
