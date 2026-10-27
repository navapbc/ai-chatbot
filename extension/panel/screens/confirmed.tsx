import { Button } from '@/components/ui/button';

type Props = {
  siteName: string;
  onStartAnother: () => void;
  onHome: () => void;
};

// Honest wording: this records what the caseworker said, not that the site
// accepted the form. No confirmation number is read or shown.
export const ConfirmedScreen = ({
  siteName,
  onStartAnother,
  onHome,
}: Props) => (
  <section className="flex flex-col gap-4" aria-label="Submitted">
    <h2 className="text-base font-semibold">Submitted</h2>
    <p role="status" aria-live="polite" className="text-sm">
      You said you submitted the form on {siteName}.
    </p>
    <div className="flex flex-col gap-2">
      <Button type="button" onClick={onStartAnother}>
        Start another form for this person
      </Button>
      <Button type="button" variant="outline" onClick={onHome}>
        Back to Home
      </Button>
    </div>
  </section>
);
