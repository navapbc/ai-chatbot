import { describe, expect, it } from 'vitest';
import dataset from '@/evals/datasets/sensitive-fields.json';
import {
  type FieldDescriptor,
  isSensitiveField,
  MASK,
  SENSITIVE_CATEGORIES,
} from '@/lib/relay/sensitive-fields';

interface Row {
  id: string;
  field: FieldDescriptor;
  sensitive: boolean;
  why: string;
}

describe('sensitive field dataset', () => {
  it('has at least 40 rows with both classes', () => {
    const rows = dataset as Row[];
    expect(rows.length).toBeGreaterThanOrEqual(40);
    expect(rows.some((r) => r.sensitive)).toBe(true);
    expect(rows.some((r) => !r.sensitive)).toBe(true);
  });

  for (const row of dataset as Row[]) {
    it(`${row.id}: ${row.why}`, () => {
      expect(isSensitiveField(row.field)).toBe(row.sensitive);
    });
  }
});

describe('isSensitiveField', () => {
  it('exports the mask and the category list', () => {
    expect(MASK).toBe('[masked]');
    expect(SENSITIVE_CATEGORIES.length).toBeGreaterThan(5);
  });

  it('password type is sensitive whatever the label', () => {
    for (const label of [undefined, '', 'Code', 'Nickname', 'City']) {
      expect(isSensitiveField({ inputType: 'password', label })).toBe(true);
    }
  });

  it('autocomplete cc-number current-password one-time-code is sensitive', () => {
    for (const autocomplete of [
      'cc-number',
      'cc-csc',
      'cc-exp-month',
      'cc-exp-year',
      'current-password',
      'new-password',
      'one-time-code',
      'billing cc-number',
    ]) {
      expect(isSensitiveField({ label: 'Field', autocomplete })).toBe(true);
    }
  });

  it('an ordinary text field is not sensitive', () => {
    expect(
      isSensitiveField({ inputType: 'text', label: 'City', name: 'city' }),
    ).toBe(false);
  });

  it('a missing descriptor is not sensitive but password type alone is', () => {
    expect(isSensitiveField({})).toBe(false);
    expect(isSensitiveField({ inputType: 'password' })).toBe(true);
  });

  it('fails toward sensitive when signals conflict', () => {
    expect(
      isSensitiveField({ label: 'Favorite color', name: 'user_ssn' }),
    ).toBe(true);
    expect(
      isSensitiveField({ label: 'Social Security Number', name: 'color' }),
    ).toBe(true);
    expect(
      isSensitiveField({ label: 'City', placeholder: 'Enter your PIN' }),
    ).toBe(true);
    expect(
      isSensitiveField({ label: 'City', autocomplete: 'one-time-code' }),
    ).toBe(true);
  });

  it('splits camelCase and snake_case names', () => {
    expect(isSensitiveField({ name: 'socialSecurityNumber' })).toBe(true);
    expect(isSensitiveField({ name: 'routing_number' })).toBe(true);
    expect(isSensitiveField({ name: 'socialWorkerName' })).toBe(false);
  });

  it.each([
    'socialsecuritynumber',
    'socialsecurity',
    'ssno',
    's s n',
    's-s-n',
    'CIN',
    'cin',
    'Client CIN',
    'client index number',
    'clientindexnumber',
  ])('treats the key %s as sensitive', (key) => {
    expect(isSensitiveField({ name: key, label: key })).toBe(true);
  });

  it.each(['cinema', 'decision', 'date of birth', 'Date of birth', 'dob'])(
    'keeps %s non-sensitive',
    (key) => {
      expect(isSensitiveField({ name: key, label: key })).toBe(false);
    },
  );
});
