import { PARTICIPANTS, type Participant } from '../../../lib/data/participants';
import { DETAIL_FIELDS } from './saved-details';

// Quick-start people for testing without the Participant table. They come
// from the checked-in fixtures and go through the same entered-details path as
// typed values, so only the fixed non-sensitive fields leave the panel (no
// record id, no calworks id).

export type SampleParticipant = {
  recordId: string;
  name: string;
  details: Record<string, string>;
};

const ALLOWED = new Set<string>(DETAIL_FIELDS.map((f) => f.key));

const nameOf = (p: Participant): string =>
  [p.participant.name.first, p.participant.name.middle, p.participant.name.last]
    .filter(Boolean)
    .join(' ');

export const detailsFromParticipant = (
  p: Participant,
): Record<string, string> => {
  const { phones, email } = p.contact_information;
  const home = p.address.residential;
  const candidates: Record<string, string | null | undefined> = {
    first_name: p.participant.name.first,
    middle_name: p.participant.name.middle,
    last_name: p.participant.name.last,
    date_of_birth: p.participant.date_of_birth,
    phone: phones.cell ?? phones.home ?? phones.main,
    email,
    street: home.street,
    unit: home.unit,
    city: home.city,
    state: home.state,
    county: home.county,
    zip: home.zip,
  };
  const details: Record<string, string> = {};
  for (const [key, value] of Object.entries(candidates)) {
    if (ALLOWED.has(key) && typeof value === 'string' && value.trim() !== '') {
      details[key] = value;
    }
  }
  return details;
};

export const SAMPLE_PARTICIPANTS: SampleParticipant[] = PARTICIPANTS.map(
  (p) => ({
    recordId: p.record_id,
    name: nameOf(p),
    details: detailsFromParticipant(p),
  }),
);
