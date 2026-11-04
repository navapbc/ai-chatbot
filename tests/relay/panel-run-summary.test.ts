import { describe, expect, it } from 'vitest';
import { buildRunSummary } from '../../extension/panel/lib/run-summary';

describe('buildRunSummary', () => {
  it('is absent without a participant (a reopened panel)', () => {
    expect(buildRunSummary(undefined)).toBeUndefined();
  });

  it('shows the name and record id for a system record', () => {
    expect(
      buildRunSummary({
        kind: 'record',
        recordId: '339619',
        name: 'Celeste Thomas',
      }),
    ).toEqual({
      name: 'Celeste Thomas',
      rows: [{ label: 'Record ID', value: '339619' }],
    });
  });

  it('builds name, date of birth, contact and address from entered details', () => {
    const s = buildRunSummary({
      kind: 'details',
      details: {
        first_name: 'Ana',
        last_name: 'Torres',
        date_of_birth: '04/17/1991',
        street: '1 Main St',
        unit: 'Apt 2',
        city: 'Fresno',
        state: 'CA',
        zip: '93701',
      },
    });
    expect(s?.name).toBe('Ana Torres');
    expect(s?.rows).toEqual([
      { label: 'Date of birth', value: '04/17/1991' },
      { label: 'Address', value: '1 Main St Apt 2, Fresno CA 93701' },
    ]);
  });

  it('ignores any key outside the fixed non-sensitive set', () => {
    const s = buildRunSummary({
      kind: 'details',
      details: { first_name: 'Ana', ssn: '123-45-6789', calworks_id: '9' },
    });
    expect(JSON.stringify(s)).not.toMatch(/123-45|calworks|ssn/i);
  });

  it('returns nothing for empty details', () => {
    expect(buildRunSummary({ kind: 'details', details: {} })).toBeUndefined();
  });
});
