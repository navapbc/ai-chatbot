// The first turn of an extension (relay) run. The client names the page and
// the participant; the server builds the opening message, so the full record
// only ever exists inside the server.
//
// Pure: no `server-only`, no database. The route resolves a record id through
// the participant source and passes the record in.

import {
  buildApplicationPrompt,
  type Participant,
} from '@/lib/data/participants';
import { stripSensitiveFields } from '@/lib/data/sensitive-record';
import { normalizeOrigin } from '@/lib/relay/policy';
import { isSensitiveField } from '@/lib/relay/sensitive-fields';

/** Caseworker-entered details (P4a-f): a flat object of strings. */
export const MAX_DETAIL_KEYS = 40;
export const MAX_DETAIL_CHARS = 500;
const MAX_RECORD_ID_CHARS = 128;
const DEFAULT_MAX_RUNS_PER_HOUR = 20;

export type RelayParticipantInput =
  | { recordId: string }
  | { details: Record<string, string> };

export type RelayStart =
  | {
      ok: true;
      origin: string;
      participant: RelayParticipantInput;
      model?: string;
    }
  | { ok: false; reason: string };

// Keys an object literal or JSON.parse treats specially: `__proto__` would be
// dropped when copied, so it is refused, never silently stripped.
const RESERVED_KEYS = new Set(['__proto__', 'constructor', 'prototype']);

const isPlainObject = (v: unknown): v is Record<string, unknown> =>
  typeof v === 'object' && v !== null && !Array.isArray(v);

type DetailsCheck =
  | { ok: true; details: Record<string, string> }
  | { ok: false; reason: string };

const parseDetails = (raw: unknown): DetailsCheck => {
  if (!isPlainObject(raw)) return { ok: false, reason: 'details_invalid' };
  const keys = Object.keys(raw);
  // Counted before anything is iterated: the input is client-controlled.
  if (keys.length > MAX_DETAIL_KEYS) {
    return { ok: false, reason: 'details_too_large' };
  }
  if (keys.length === 0) return { ok: false, reason: 'details_invalid' };
  const details: Record<string, string> = {};
  for (const key of keys) {
    const value = raw[key];
    if (
      key.trim() === '' ||
      RESERVED_KEYS.has(key) ||
      typeof value !== 'string'
    ) {
      return { ok: false, reason: 'details_invalid' };
    }
    if (key.length > MAX_DETAIL_CHARS || value.length > MAX_DETAIL_CHARS) {
      return { ok: false, reason: 'details_too_large' };
    }
    // Refused, never stripped: a value the caseworker typed under a sensitive
    // label must not be silently dropped or forwarded.
    if (isSensitiveField({ name: key, label: key })) {
      return { ok: false, reason: 'sensitive_detail' };
    }
    details[key] = value;
  }
  return { ok: true, details };
};

/**
 * Validates the `relay` field of a first turn. Returns only the fields it
 * checked, never the caller's object. The model is type-checked here; the
 * allowlist is the route's (it reads the environment).
 */
export function parseRelayStart(raw: unknown): RelayStart {
  if (!isPlainObject(raw)) return { ok: false, reason: 'invalid_relay' };
  const { origin, participant, model } = raw;
  if (typeof origin !== 'string' || normalizeOrigin(origin) !== origin) {
    return { ok: false, reason: 'invalid_origin' };
  }
  if (model !== undefined && typeof model !== 'string') {
    return { ok: false, reason: 'model_not_allowed' };
  }
  if (!isPlainObject(participant)) {
    return { ok: false, reason: 'invalid_participant' };
  }
  const hasRecord = participant.recordId !== undefined;
  const hasDetails = participant.details !== undefined;
  if (hasRecord === hasDetails) {
    return { ok: false, reason: 'invalid_participant' };
  }
  let parsed: RelayParticipantInput;
  if (hasRecord) {
    const recordId = participant.recordId;
    if (
      typeof recordId !== 'string' ||
      recordId.trim() === '' ||
      recordId.length > MAX_RECORD_ID_CHARS
    ) {
      return { ok: false, reason: 'invalid_participant' };
    }
    parsed = { recordId };
  } else {
    const details = parseDetails(participant.details);
    if (!details.ok) return details;
    parsed = { details: details.details };
  }
  return {
    ok: true,
    origin,
    participant: parsed,
    ...(model !== undefined ? { model } : {}),
  };
}

const isParticipantRecord = (
  p: Participant | Record<string, string>,
): p is Participant => isPlainObject((p as Participant).participant);

/**
 * The opening message of a relay run, before the extension-run marker is
 * added: `buildApplicationPrompt` for the origin, with the record (or the
 * entered details) passed through `stripSensitiveFields`. Details use the same
 * prefix and JSON format as a record.
 */
export function buildRelayOpeningMessage(args: {
  origin: string;
  participant: Participant | Record<string, string>;
}): string {
  const { origin, participant } = args;
  // Details are not a Participant, but buildApplicationPrompt only serializes
  // what it is given; going through it keeps the prefix and format identical.
  const stripped = isParticipantRecord(participant)
    ? stripSensitiveFields(participant)
    : (stripSensitiveFields(participant) as unknown as Participant);
  return buildApplicationPrompt({ participant: stripped, target: origin });
}

type Env = Record<string, string | undefined>;

/** P4a-b: relay runs a user may create per rolling hour. Invalid -> 20. */
export function maxRelayRunsPerHour(env: Env = process.env): number {
  const raw = env.RELAY_MAX_RUNS_PER_HOUR?.trim() ?? '';
  return /^[1-9]\d{0,5}$/.test(raw) ? Number(raw) : DEFAULT_MAX_RUNS_PER_HOUR;
}
