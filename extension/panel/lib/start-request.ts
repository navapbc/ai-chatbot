import { normalizeOrigin } from '../../../lib/relay/policy';
import { DETAIL_FIELDS } from './saved-details';

// The body the server's `parseRelayStart` accepts, built from what the panel
// holds. The site name and the participant's name are display-only and never
// enter the request. Errors carry a fixed message and none of the input.

export type ReadyModel = {
  siteName: string;
  origin: string;
  participant:
    | { kind: 'record'; recordId: string; name: string }
    | { kind: 'details'; details: Record<string, string> };
  models: { id: string }[];
  defaultModel: string;
  modelId: string;
};

export type StartRequest = {
  origin: string;
  participant: { recordId: string } | { details: Record<string, string> };
  model?: string;
};

export class StartRequestError extends Error {
  constructor() {
    super('The start request is not valid.');
    this.name = 'StartRequestError';
  }
}

const MAX_RECORD_ID = 128;
const MAX_VALUE = 500;
const FIELD_KEYS: readonly string[] = DETAIL_FIELDS.map((f) => f.key);

const pickDetails = (details: Record<string, string>) => {
  const picked: Record<string, string> = {};
  for (const key of FIELD_KEYS) {
    if (!Object.hasOwn(details, key)) continue;
    const raw = details[key];
    if (typeof raw !== 'string') continue;
    const value = raw.trim();
    if (value === '') continue;
    if (value.length > MAX_VALUE) throw new StartRequestError();
    picked[key] = value;
  }
  if (Object.keys(picked).length === 0) throw new StartRequestError();
  return picked;
};

export const buildStartRequest = (model: ReadyModel): StartRequest => {
  const { origin, participant } = model;
  if (typeof origin !== 'string' || normalizeOrigin(origin) !== origin) {
    throw new StartRequestError();
  }

  let who: StartRequest['participant'];
  if (participant.kind === 'record') {
    const { recordId } = participant;
    if (
      typeof recordId !== 'string' ||
      recordId.trim() === '' ||
      recordId.length > MAX_RECORD_ID
    ) {
      throw new StartRequestError();
    }
    who = { recordId };
  } else {
    who = { details: pickDetails(participant.details) };
  }

  if (model.modelId === model.defaultModel) return { origin, participant: who };
  if (!model.models.some((m) => m.id === model.modelId)) {
    throw new StartRequestError();
  }
  return { origin, participant: who, model: model.modelId };
};
