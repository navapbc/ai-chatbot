import { describe, it, expect, afterEach } from 'vitest';
import {
  extractParticipant,
  extractParticipantFromText,
} from '@/lib/jev/participant';
import { PARTICIPANTS, buildApplicationPrompt } from '@/lib/data/participants';
import { isJevEnabled } from '@/lib/jev/client';

const PARTICIPANT = PARTICIPANTS[0];

describe('extractParticipantFromText', () => {
  it('round-trips a record written by buildApplicationPrompt', () => {
    const text = buildApplicationPrompt({
      participant: PARTICIPANT,
      target: 'a WIC application',
    });
    expect(extractParticipantFromText(text)?.record_id).toBe(
      PARTICIPANT.record_id,
    );
  });

  it('skips JSON that is not a participant', () => {
    const text = `{"foo":"bar"} then ${JSON.stringify(PARTICIPANT)}`;
    expect(extractParticipantFromText(text)?.record_id).toBe(
      PARTICIPANT.record_id,
    );
  });

  it('is not confused by braces inside string values', () => {
    const withBrace = {
      ...PARTICIPANT,
      family_profile: { linked: true, notes: 'note with a } brace' },
    };
    expect(
      extractParticipantFromText(JSON.stringify(withBrace))?.record_id,
    ).toBe(PARTICIPANT.record_id);
  });

  it('returns null when there is no record', () => {
    expect(extractParticipantFromText('just a question')).toBeNull();
    expect(extractParticipantFromText('{"incomplete":')).toBeNull();
  });
});

describe('extractParticipant', () => {
  const opening = buildApplicationPrompt({
    participant: PARTICIPANT,
    target: 'a WIC application',
  });

  it('finds the record in an earlier turn, not only the latest', () => {
    // A session carries across forms; the record is only in the first message.
    const messages = [
      { role: 'user', content: opening },
      { role: 'assistant', content: 'Done.' },
      { role: 'user', content: 'now do IHSS for the same person' },
    ];
    expect(extractParticipant(messages)?.record_id).toBe(PARTICIPANT.record_id);
  });

  it('reads array-style message content', () => {
    const messages = [
      { role: 'user', content: [{ type: 'text', text: opening }] },
    ];
    expect(extractParticipant(messages)?.record_id).toBe(PARTICIPANT.record_id);
  });

  it('ignores non-user messages', () => {
    expect(
      extractParticipant([{ role: 'assistant', content: opening }]),
    ).toBeNull();
  });

  it('returns null for no messages', () => {
    expect(extractParticipant(undefined)).toBeNull();
    expect(extractParticipant([])).toBeNull();
  });
});

describe('isJevEnabled', () => {
  const saved = { ...process.env };
  afterEach(() => {
    process.env.JEV_FEATURES = saved.JEV_FEATURES;
    process.env.TYPESAFE_API_KEY = saved.TYPESAFE_API_KEY;
  });

  it('is off when JEV_FEATURES is unset', () => {
    process.env.TYPESAFE_API_KEY = 'test-key';
    process.env.JEV_FEATURES = undefined;
    expect(isJevEnabled('gap-triage')).toBe(false);
  });

  it('enables only the named features', () => {
    process.env.TYPESAFE_API_KEY = 'test-key';
    process.env.JEV_FEATURES = 'gap-triage, field-value';
    expect(isJevEnabled('gap-triage')).toBe(true);
    expect(isJevEnabled('field-value')).toBe(true);
    expect(isJevEnabled('summary-check')).toBe(false);
  });

  it('supports "all"', () => {
    process.env.TYPESAFE_API_KEY = 'test-key';
    process.env.JEV_FEATURES = 'all';
    expect(isJevEnabled('summary-check')).toBe(true);
  });

  it('ignores unknown feature names', () => {
    process.env.TYPESAFE_API_KEY = 'test-key';
    process.env.JEV_FEATURES = 'not-a-feature';
    expect(isJevEnabled('gap-triage')).toBe(false);
  });

  // A missing key must read as "off", not as an error: an unset secret in a
  // deploy should degrade to pre-Jev behavior, not fail a form run.
  it('is off when the API key is missing, even with the flag on', () => {
    process.env.JEV_FEATURES = 'all';
    process.env.TYPESAFE_API_KEY = '';
    expect(isJevEnabled('gap-triage')).toBe(false);
  });
});

describe('currentParticipant (Eve)', () => {
  it('returns null outside an eve context instead of throwing', async () => {
    // defineState throws when read outside eve-managed code, and a subagent
    // inherits no state. Both must read as "no record", which every Jev
    // feature treats the same as the flag being off.
    const { currentParticipant } = await import('@/agent/lib/jev');
    expect(currentParticipant()).toBeNull();
  });
});
