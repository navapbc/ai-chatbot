import { useEffect, useId, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  clearSavedDetails,
  DETAIL_FIELDS,
  loadSavedDetails,
  pickDetails,
  SensitiveDetailError,
  saveDetails,
} from '../lib/saved-details';

type Props = {
  onSubmit: (details: Record<string, string>, save: boolean) => void;
  onBack: () => void;
};

const INPUT_TYPES: Record<string, 'tel' | 'email' | 'text'> = {
  phone: 'tel',
  email: 'email',
};

const SAVE_REFUSED = 'These details could not be saved.';

// Values live in component state until submit; nothing here logs or stores
// them except the opt-in saveDetails call.
export const DetailsScreen = ({ onSubmit, onBack }: Props) => {
  const [values, setValues] = useState<Record<string, string>>({});
  const [saved, setSaved] = useState<Record<string, string> | null>(null);
  const [save, setSave] = useState(false);
  const [notice, setNotice] = useState('');
  const prefix = useId();

  useEffect(() => {
    let current = true;
    loadSavedDetails().then((record) => current && setSaved(record));
    return () => {
      current = false;
    };
  }, []);

  const handleChange = (key: string, value: string) =>
    setValues((previous) => ({ ...previous, [key]: value }));

  const handleReuse = () => {
    if (saved) setValues({ ...saved });
  };

  const handleClearSaved = async () => {
    await clearSavedDetails();
    setSaved(null);
    setNotice('Saved details cleared.');
  };

  const handleSubmit = async (event: { preventDefault: () => void }) => {
    event.preventDefault();
    const details = pickDetails(values);
    if (Object.keys(details).length === 0) return;
    if (save) {
      try {
        await saveDetails(details);
      } catch (error) {
        if (error instanceof SensitiveDetailError) {
          setNotice(SAVE_REFUSED);
          return;
        }
        // Storage failed: continue without saving.
      }
    }
    onSubmit(details, save);
  };

  const handleBack = () => onBack();

  const isEmpty = Object.keys(pickDetails(values)).length === 0;

  return (
    <section className="flex flex-col gap-4" aria-label="Details">
      <h2 className="text-base font-semibold">Enter details</h2>
      {saved ? (
        <div className="flex flex-col gap-2">
          <Button type="button" variant="outline" onClick={handleReuse}>
            Reuse saved details
          </Button>
          <Button type="button" variant="ghost" onClick={handleClearSaved}>
            Clear saved details
          </Button>
        </div>
      ) : null}
      <form className="flex flex-col gap-3" onSubmit={handleSubmit}>
        {DETAIL_FIELDS.map(({ key, label }) => (
          <div key={key} className="flex flex-col gap-1">
            <Label htmlFor={`${prefix}-${key}`}>{label}</Label>
            <Input
              id={`${prefix}-${key}`}
              type={INPUT_TYPES[key] ?? 'text'}
              autoComplete="off"
              maxLength={500}
              value={values[key] ?? ''}
              onChange={(event) => handleChange(key, event.target.value)}
            />
          </div>
        ))}
        <div className="flex items-center gap-2">
          <Checkbox
            id={`${prefix}-save`}
            checked={save}
            onCheckedChange={(checked) => setSave(checked === true)}
          />
          <Label htmlFor={`${prefix}-save`}>Save these for 8 hours</Label>
        </div>
        <p className="text-sm text-muted-foreground">
          Saved details stay on this computer for 8 hours, then are deleted.
          Nothing is saved unless you check the box.
        </p>
        <p role="status" aria-live="polite" className="text-sm">
          {notice}
        </p>
        <Button type="submit" disabled={isEmpty}>
          Continue
        </Button>
        <Button type="button" variant="ghost" onClick={handleBack}>
          Back
        </Button>
      </form>
    </section>
  );
};
