import { describe, it, expect, beforeEach, vi } from 'vitest';
import { PARTICIPANTS, buildApplicationPrompt } from '@/lib/data/participants';

// `defineState` needs a live eve runtime context, so the durable slot is
// swapped for an in-memory one. This exercises the hook's own logic — extract
// from the inbound message, first record wins — without booting a session.
// It does NOT prove eve invokes the hook or persists the state; only a real
// session does that.
// Keyed by state name: the hook now fills TWO slots (the participant record
// and the caseworker's turns). A single shared slot let the second
// `defineState` clobber the first.
const slots = new Map<string, unknown>();
vi.mock('eve/context', () => ({
  defineState: (name: string, init: () => unknown) => {
    if (!slots.has(name)) slots.set(name, init());
    return {
      get: () => slots.get(name),
      update: (fn: (c: unknown) => unknown) => {
        slots.set(name, fn(slots.get(name)));
      },
    };
  },
}));

const PARTICIPANT_SLOT = 'labs-asp.participant';
const MESSAGES_SLOT = 'labs-asp.caseworker-messages';
const participantSlot = () => slots.get(PARTICIPANT_SLOT);
const messagesSlot = () =>
  ((slots.get(MESSAGES_SLOT) ?? []) as { text: string }[]).map((t) => t.text);
vi.mock('eve/hooks', () => ({
  defineHook: (definition: unknown) => definition,
}));

const PARTICIPANT = PARTICIPANTS[0];
const OTHER = PARTICIPANTS[1];
const opening = (p: typeof PARTICIPANT) =>
  buildApplicationPrompt({ participant: p, target: 'a WIC application' });

const loadHook = async () => {
  const mod = await import('@/agent/hooks/participant');
  return (mod.default as { events: Record<string, (e: unknown) => void> })
    .events['message.received'];
};

let turnCounter = 0;
const received = (message: string, turnId?: string) => ({
  data: {
    message,
    sequence: 0,
    turnId: turnId ?? `t${++turnCounter}`,
  },
  type: 'message.received' as const,
});

describe('participant capture hook', () => {
  beforeEach(() => {
    slots.set(PARTICIPANT_SLOT, null);
    slots.set(MESSAGES_SLOT, []);
    turnCounter = 0;
  });

  it('captures the record from an inbound message', async () => {
    const onMessage = await loadHook();
    onMessage(received(opening(PARTICIPANT)));
    expect((participantSlot() as { record_id: string })?.record_id).toBe(
      PARTICIPANT.record_id,
    );
  });

  it('keeps the first record when later messages arrive', async () => {
    const onMessage = await loadHook();
    onMessage(received(opening(PARTICIPANT)));
    onMessage(received('now do IHSS for the same person'));
    onMessage(received(opening(OTHER)));
    expect((participantSlot() as { record_id: string })?.record_id).toBe(
      PARTICIPANT.record_id,
    );
  });

  it('leaves state empty when no record is present', async () => {
    const onMessage = await loadHook();
    onMessage(received('can you help me with a form?'));
    expect(participantSlot()).toBeNull();
  });

  it('tolerates a message with no text', async () => {
    const onMessage = await loadHook();
    expect(() =>
      onMessage({
        data: { sequence: 0, turnId: 't1' },
        type: 'message.received',
      }),
    ).not.toThrow();
    expect(participantSlot()).toBeNull();
  });

  // The caseworker's turns are what let Jev tell an inference drawn from the
  // task apart from an invention (lib/jev/questions.ts).
  it('records every caseworker turn, with the record JSON stripped', async () => {
    const onMessage = await loadHook();
    onMessage(received(opening(PARTICIPANT)));
    onMessage(received('SSN is 123456789'));
    const said = messagesSlot();
    expect(said).toHaveLength(2);
    expect(said[0]).toContain('WIC');
    expect(said[0]).not.toContain(PARTICIPANT.record_id);
    expect(said[1]).toBe('SSN is 123456789');
  });

  it('keeps recording turns after the record has been captured', async () => {
    // Regression: the record capture returns early once a record is held. If
    // the message were recorded after that return, every later turn — i.e.
    // exactly the gap-analysis answers — would be lost.
    const onMessage = await loadHook();
    onMessage(received(opening(PARTICIPANT)));
    onMessage(received('first answer'));
    onMessage(received('second answer'));
    expect(messagesSlot().slice(-2)).toEqual(['first answer', 'second answer']);
  });

  it('does not duplicate a replayed delivery of the same turn', async () => {
    // eve's docs do not promise `message.received` fires exactly once, the
    // workflow replays completed steps, and /api/eve-chat re-opens the stream
    // on a body timeout. A duplicate would repeat in every Jev request.
    const onMessage = await loadHook();
    onMessage(received('an answer', 'turn_7'));
    onMessage(received('an answer', 'turn_7'));
    expect(messagesSlot()).toEqual(['an answer']);
  });

  it('records nothing for a message with no text', async () => {
    const onMessage = await loadHook();
    onMessage({
      data: { sequence: 0, turnId: 't1' },
      type: 'message.received',
    });
    expect(messagesSlot()).toEqual([]);
  });
});
