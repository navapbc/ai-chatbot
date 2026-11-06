import type { Participant } from './state';

// What the Running screen shows about the client, so the caseworker can
// compare the page with it while the agent works. It is built in the panel
// from what the caseworker already chose or typed, only from the fixed
// non-sensitive detail keys (see DETAIL_FIELDS), and is never stored, logged
// or sent anywhere. A panel reopened mid-run has no participant, so it gets
// no summary.

export type RunSummary = {
  name: string;
  rows: { label: string; value: string }[];
};

const MAX_VALUE = 120;

const clean = (v: unknown): string =>
  typeof v === 'string'
    ? Array.from(v.trim()).slice(0, MAX_VALUE).join('')
    : '';

const join = (...parts: string[]): string => parts.filter(Boolean).join(' ');

export const buildRunSummary = (
  participant: Participant | undefined,
): RunSummary | undefined => {
  if (!participant) return undefined;
  if (participant.kind === 'record') {
    const name = clean(participant.name);
    const id = clean(participant.recordId);
    if (!name && !id) return undefined;
    return {
      name: name || 'Client',
      rows: id ? [{ label: 'Record ID', value: id }] : [],
    };
  }
  const d = participant.details;
  const name = join(
    clean(d.first_name),
    clean(d.middle_name),
    clean(d.last_name),
  );
  const address = [
    join(clean(d.street), clean(d.unit)),
    join(clean(d.city), clean(d.state), clean(d.zip)),
  ]
    .filter(Boolean)
    .join(', ');
  const rows = [
    { label: 'Date of birth', value: clean(d.date_of_birth) },
    { label: 'Phone', value: clean(d.phone) },
    { label: 'Email', value: clean(d.email) },
    { label: 'Address', value: address },
  ].filter((r) => r.value !== '');
  if (!name && rows.length === 0) return undefined;
  return { name: name || 'Client', rows };
};
