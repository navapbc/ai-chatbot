type Props = {
  active: 'system' | 'details';
  onSystem: () => void;
  onDetails: () => void;
};

const tabClass = (active: boolean) =>
  `flex-1 rounded-md px-2.5 py-1.5 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring ${
    active
      ? 'bg-accent text-foreground'
      : 'text-foreground/80 hover:text-foreground'
  }`;

// The switch shown on both ways of identifying the participant.
export const ParticipantTabs = ({ active, onSystem, onDetails }: Props) => (
  <div
    role="tablist"
    aria-label="How to identify the participant"
    className="flex gap-1 rounded-lg border border-border p-1"
  >
    <button
      type="button"
      role="tab"
      aria-selected={active === 'system'}
      className={tabClass(active === 'system')}
      onClick={() => active !== 'system' && onSystem()}
    >
      From the system
    </button>
    <button
      type="button"
      role="tab"
      aria-selected={active === 'details'}
      className={tabClass(active === 'details')}
      onClick={() => active !== 'details' && onDetails()}
    >
      Enter details
    </button>
  </div>
);
