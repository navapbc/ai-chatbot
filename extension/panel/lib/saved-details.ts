import { isSensitiveField } from '../../../lib/relay/sensitive-fields';
import { SAVED_DETAILS_SWEEP_ALARM } from '../../src/panel-behavior';

// Saved details: non-sensitive entered-details values kept for 8 hours in
// chrome.storage.local so a caseworker can reuse them. The stored value is
// untrusted on read; an invalid or expired record is deleted, never returned.
// A key outside DETAIL_FIELDS, or one the sensitive-field classifier flags, is
// refused before storage rather than stripped. No value is ever logged.

export const DETAIL_FIELDS = [
  { key: 'first_name', label: 'First name' },
  { key: 'middle_name', label: 'Middle name' },
  { key: 'last_name', label: 'Last name' },
  { key: 'date_of_birth', label: 'Date of birth' },
  { key: 'phone', label: 'Phone' },
  { key: 'email', label: 'Email' },
  { key: 'street', label: 'Street address' },
  { key: 'unit', label: 'Apartment or unit' },
  { key: 'city', label: 'City' },
  { key: 'state', label: 'State' },
  { key: 'county', label: 'County' },
  { key: 'zip', label: 'ZIP code' },
] as const;

export class SensitiveDetailError extends Error {
  constructor() {
    super('These details cannot be saved.');
    this.name = 'SensitiveDetailError';
  }
}

const KEY = 'savedDetails';
const MAX_AGE_MS = 8 * 3600 * 1000;
const MAX_FUTURE_MS = 5 * 60 * 1000;
const MAX_VALUE_LENGTH = 500;
const ALLOWED = new Set<string>(DETAIL_FIELDS.map((f) => f.key));

const isAllowedKey = (key: string): boolean =>
  ALLOWED.has(key) && !isSensitiveField({ name: key, label: key });

const isPlainObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const parseRecord = (
  stored: unknown,
  now: number,
): Record<string, string> | null => {
  if (!isPlainObject(stored)) return null;
  const keys = Object.keys(stored);
  if (keys.length !== 2 || !('savedAt' in stored) || !('values' in stored)) {
    return null;
  }
  const { savedAt, values } = stored;
  if (typeof savedAt !== 'number' || !Number.isFinite(savedAt)) return null;
  if (now - savedAt > MAX_AGE_MS || savedAt - now > MAX_FUTURE_MS) return null;
  if (!isPlainObject(values)) return null;
  const entries = Object.entries(values);
  if (entries.length > DETAIL_FIELDS.length) return null;
  const out: Record<string, string> = {};
  for (const [key, value] of entries) {
    if (!isAllowedKey(key)) return null;
    if (typeof value !== 'string' || value.length > MAX_VALUE_LENGTH) {
      return null;
    }
    out[key] = value;
  }
  return out;
};

/** The valid saved record's values, or null (an invalid one is deleted). */
export const loadSavedDetails = async (
  now: number = Date.now(),
): Promise<Record<string, string> | null> => {
  try {
    const stored = (await chrome.storage.local.get(KEY))[KEY];
    if (stored === undefined) return null;
    const values = parseRecord(stored, now);
    if (values) return values;
    await chrome.storage.local.remove(KEY);
    return null;
  } catch {
    return null;
  }
};

export const clearSavedDetails = async (): Promise<void> => {
  try {
    await chrome.storage.local.remove(KEY);
  } catch {
    // Value-free: nothing to report.
  }
};

/** Deletes an expired or invalid record. Never throws. */
export const sweepSavedDetails = async (
  now: number = Date.now(),
): Promise<void> => {
  await loadSavedDetails(now);
};

/** Throws SensitiveDetailError, writing nothing, for any refused key. */
export const saveDetails = async (
  values: Record<string, string>,
  now: number = Date.now(),
): Promise<void> => {
  if (!Object.keys(values).every(isAllowedKey))
    throw new SensitiveDetailError();
  const kept: Record<string, string> = {};
  for (const { key } of DETAIL_FIELDS) {
    const value = values[key];
    if (typeof value !== 'string') continue;
    const trimmed = value.trim().slice(0, MAX_VALUE_LENGTH);
    if (trimmed) kept[key] = trimmed;
  }
  if (Object.keys(kept).length === 0) {
    await chrome.storage.local.remove(KEY);
    return;
  }
  await chrome.storage.local.set({ [KEY]: { savedAt: now, values: kept } });
};

/** The flat object sent as participant.details: known keys, trimmed, ordered. */
export const pickDetails = (
  values: Record<string, string>,
): Record<string, string> => {
  const out: Record<string, string> = {};
  for (const { key } of DETAIL_FIELDS) {
    const value = values[key];
    if (typeof value !== 'string') continue;
    const trimmed = value.trim().slice(0, MAX_VALUE_LENGTH);
    if (trimmed) out[key] = trimmed;
  }
  return out;
};

/**
 * On panel start: sweep once, create the periodic alarm (idempotent: creating
 * an alarm with an existing name replaces it) and sweep on each firing. Errors
 * are swallowed value-free.
 */
export const startSavedDetailsSweep = async (): Promise<void> => {
  await sweepSavedDetails();
  try {
    chrome.alarms.onAlarm?.addListener((alarm) => {
      if (alarm.name === SAVED_DETAILS_SWEEP_ALARM) void sweepSavedDetails();
    });
    await chrome.alarms.create(SAVED_DETAILS_SWEEP_ALARM, {
      periodInMinutes: 30,
    });
  } catch {
    // Value-free: the read-time check still enforces the 8 hours.
  }
};
