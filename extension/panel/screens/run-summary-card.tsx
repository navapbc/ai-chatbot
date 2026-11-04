import type { RunSummary } from '../lib/run-summary';

type Props = { summary: RunSummary };

// The client the run is for, so the caseworker can compare it with the page.
// Values come from what the caseworker chose or typed in this panel; they are
// rendered as React text nodes only.
export const RunSummaryCard = ({ summary }: Props) => (
  <section
    aria-label="Client"
    className="flex flex-col gap-1 rounded-xl border border-accent bg-background p-3 text-sm"
  >
    <p className="break-words font-medium">{summary.name}</p>
    {summary.rows.length > 0 ? (
      <dl className="flex flex-col gap-0.5 text-xs">
        {summary.rows.map((row) => (
          <div key={row.label} className="flex gap-2">
            <dt className="shrink-0 text-muted-foreground">{row.label}</dt>
            <dd className="min-w-0 break-words">{row.value}</dd>
          </div>
        ))}
      </dl>
    ) : null}
  </section>
);
