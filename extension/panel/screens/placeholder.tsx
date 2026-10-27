import { Button } from '@/components/ui/button';

type Props = {
  name: string;
  onHome: () => void;
};

/** Stub for a screen a later task builds; it only leads back to Home. */
export const Placeholder = ({ name, onHome }: Props) => {
  const handleHome = () => onHome();
  return (
    <section className="flex flex-col gap-4" aria-label={name}>
      <h2 className="text-base font-semibold">{name}</h2>
      <Button type="button" onClick={handleHome}>
        Back to Home
      </Button>
    </section>
  );
};
