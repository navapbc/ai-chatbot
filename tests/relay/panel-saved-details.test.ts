import { isSensitiveField } from '../../lib/relay/sensitive-fields';
import { beforeEach, describe, expect, it } from 'vitest';
import {
  clearSavedDetails,
  DETAIL_FIELDS,
  ensureSweepAlarm,
  loadSavedDetails,
  pickDetails,
  SensitiveDetailError,
  saveDetails,
  startSavedDetailsSweep,
  sweepSavedDetails,
} from '../../extension/panel/lib/saved-details';

const KEY = 'savedDetails';
const HOUR = 3_600_000;
const NOW = 1_800_000_000_000;

let store: Record<string, unknown>;
let failing = false;
let writes = 0;
let alarms: { name: string; info: unknown }[];
let alarmFails = false;

beforeEach(() => {
  store = {};
  failing = false;
  writes = 0;
  alarms = [];
  alarmFails = false;
  (globalThis as unknown as { chrome: unknown }).chrome = {
    storage: {
      local: {
        get: async (key: string) => {
          if (failing) throw new Error('boom');
          return key in store ? { [key]: store[key] } : {};
        },
        set: async (items: Record<string, unknown>) => {
          if (failing) throw new Error('boom');
          writes += 1;
          Object.assign(store, JSON.parse(JSON.stringify(items)));
        },
        remove: async (key: string) => {
          if (failing) throw new Error('boom');
          writes += 1;
          delete store[key];
        },
      },
    },
    alarms: {
      create: async (name: string, info: unknown) => {
        if (alarmFails) throw new Error('boom');
        alarms = alarms.filter((a) => a.name !== name);
        alarms.push({ name, info });
      },
    },
  };
});

const seed = (record: unknown) => {
  store[KEY] = record;
};

describe('DETAIL_FIELDS', () => {
  it('is the fixed ordered list', () => {
    expect(DETAIL_FIELDS.map((f) => f.key)).toEqual([
      'first_name',
      'middle_name',
      'last_name',
      'date_of_birth',
      'phone',
      'email',
      'street',
      'unit',
      'city',
      'state',
      'county',
      'zip',
    ]);
    expect(DETAIL_FIELDS[0]).toEqual({
      key: 'first_name',
      label: 'First name',
    });
    expect(DETAIL_FIELDS[11].label).toBe('ZIP code');
  });
});

describe('DETAIL_FIELDS and the classifier', () => {
  it('flags none of the fixed keys', () => {
    for (const { key } of DETAIL_FIELDS) {
      expect(isSensitiveField({ name: key, label: key })).toBe(false);
    }
  });
});

describe('loadSavedDetails', () => {
  it('returns null when nothing is stored', async () => {
    expect(await loadSavedDetails(NOW)).toBeNull();
  });

  it('keeps a record at exactly 8 hours and drops one 1 ms older', async () => {
    seed({ savedAt: NOW - 8 * HOUR, values: { city: 'Fresno' } });
    expect(await loadSavedDetails(NOW)).toEqual({ city: 'Fresno' });
    seed({ savedAt: NOW - 8 * HOUR - 1, values: { city: 'Fresno' } });
    expect(await loadSavedDetails(NOW)).toBeNull();
    expect(KEY in store).toBe(false);
  });

  it('drops a record dated more than 5 minutes in the future', async () => {
    seed({ savedAt: NOW + 5 * 60_000, values: { city: 'a' } });
    expect(await loadSavedDetails(NOW)).toEqual({ city: 'a' });
    seed({ savedAt: NOW + 5 * 60_000 + 1, values: { city: 'a' } });
    expect(await loadSavedDetails(NOW)).toBeNull();
    expect(KEY in store).toBe(false);
  });

  it.each([
    ['ssn', { ssn: '1' }],
    ['calworks id', { 'calworks id': '1' }],
    ['bank_account', { bank_account: '1' }],
    ['unknown key', { nickname: 'x' }],
    ['non-string value', { city: 5 }],
    ['socialsecuritynumber', { socialsecuritynumber: '1' }],
    ['s s n', { 's s n': '1' }],
    ['bank-account', { 'bank-account': '1' }],
    ['CIN', { CIN: '1' }],
    ['long value', { city: 'x'.repeat(501) }],
  ])('drops and deletes a record with %s', async (_n, values) => {
    seed({ savedAt: NOW, values });
    expect(await loadSavedDetails(NOW)).toBeNull();
    expect(KEY in store).toBe(false);
  });

  it('drops a record with more than 12 keys', async () => {
    const values: Record<string, string> = {};
    for (const f of DETAIL_FIELDS) values[f.key] = 'x';
    values.extra = 'x';
    seed({ savedAt: NOW, values });
    expect(await loadSavedDetails(NOW)).toBeNull();
    expect(KEY in store).toBe(false);
  });

  it.each([
    ['extra top-level key', { savedAt: NOW, values: {}, more: 1 }],
    ['missing values', { savedAt: NOW }],
    ['NaN time', { savedAt: Number.NaN, values: {} }],
    ['Infinity time', { savedAt: Number.POSITIVE_INFINITY, values: {} }],
    ['-Infinity time', { savedAt: Number.NEGATIVE_INFINITY, values: {} }],
    ['array values', { savedAt: NOW, values: ['a'] }],
    ['string', 'junk'],
    ['array', []],
    ['number', 3],
    ['null', null],
  ])('drops and deletes %s', async (_n, record) => {
    seed(record);
    expect(await loadSavedDetails(NOW)).toBeNull();
    expect(KEY in store).toBe(false);
  });

  it('returns null without throwing when storage throws', async () => {
    failing = true;
    await expect(loadSavedDetails(NOW)).resolves.toBeNull();
  });
});

describe('saveDetails', () => {
  it('stores only savedAt and values, trimmed, in a plain record', async () => {
    await saveDetails({ first_name: '  Ann ', city: '  ' }, NOW);
    expect(store[KEY]).toEqual({ savedAt: NOW, values: { first_name: 'Ann' } });
    expect(Object.keys(store)).toEqual([KEY]);
    expect(Object.keys(store[KEY] as object).sort()).toEqual([
      'savedAt',
      'values',
    ]);
  });

  it.each([
    ['sensitive key', { first_name: 'Ann', ssn: '1' }],
    ['calworks key', { 'calworks id': '1' }],
    ['unknown key', { first_name: 'Ann', nickname: 'x' }],
  ])('refuses a %s and writes nothing', async (_n, values) => {
    store[KEY] = { savedAt: 1, values: { city: 'old' } };
    const before = JSON.stringify(store);
    await expect(saveDetails(values, NOW)).rejects.toBeInstanceOf(
      SensitiveDetailError,
    );
    expect(JSON.stringify(store)).toBe(before);
    expect(writes).toBe(0);
  });

  it('does not put a refused value in the error', async () => {
    const error = await saveDetails({ ssn: '123-45-6789' }, NOW).catch(
      (e) => e,
    );
    expect(String(error.message)).not.toContain('123');
  });

  it('writes nothing and clears an old record when all values are empty', async () => {
    store[KEY] = { savedAt: NOW, values: { city: 'old' } };
    await saveDetails({ city: ' ', zip: '' }, NOW);
    expect(KEY in store).toBe(false);
    writes = 0;
    await saveDetails({}, NOW);
    expect(Object.keys(store)).toEqual([]);
  });

  it('caps each value at 500 characters', async () => {
    await saveDetails({ street: 'x'.repeat(900) }, NOW);
    expect(
      (store[KEY] as { values: { street: string } }).values.street,
    ).toHaveLength(500);
  });

  it('replaces an older record', async () => {
    await saveDetails({ city: 'A' }, NOW);
    await saveDetails({ zip: '9' }, NOW + 1);
    expect(store[KEY]).toEqual({ savedAt: NOW + 1, values: { zip: '9' } });
  });
});

describe('clear and sweep', () => {
  it('clearSavedDetails removes the record', async () => {
    seed({ savedAt: NOW, values: { city: 'a' } });
    await clearSavedDetails();
    expect(KEY in store).toBe(false);
  });

  it('sweep deletes an expired record and keeps a fresh one', async () => {
    seed({ savedAt: NOW - 9 * HOUR, values: { city: 'a' } });
    await sweepSavedDetails(NOW);
    expect(KEY in store).toBe(false);
    seed({ savedAt: NOW - HOUR, values: { city: 'a' } });
    await sweepSavedDetails(NOW);
    expect(KEY in store).toBe(true);
  });

  it('sweep deletes an invalid record and never throws', async () => {
    seed({ savedAt: NOW, values: { ssn: '1' } });
    await sweepSavedDetails(NOW);
    expect(KEY in store).toBe(false);
    failing = true;
    await expect(sweepSavedDetails(NOW)).resolves.toBeUndefined();
  });
});

describe('pickDetails', () => {
  it('keeps trimmed non-empty known keys in field order', () => {
    const picked = pickDetails({
      zip: ' 93721 ',
      ssn: '1',
      first_name: 'Ann',
      city: '  ',
    });
    expect(picked).toEqual({ first_name: 'Ann', zip: '93721' });
    expect(Object.keys(picked)).toEqual(['first_name', 'zip']);
  });
});

describe('startSavedDetailsSweep', () => {
  it('creates the 30 minute alarm, idempotently', async () => {
    await startSavedDetailsSweep();
    await startSavedDetailsSweep();
    await ensureSweepAlarm();
    expect(alarms).toEqual([
      { name: 'saved-details-sweep', info: { periodInMinutes: 30 } },
    ]);
  });

  it('swallows a failing alarm create', async () => {
    alarmFails = true;
    await expect(startSavedDetailsSweep()).resolves.toBeUndefined();
  });
});
