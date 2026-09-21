import { describe, it, expect, beforeEach, vi } from 'vitest';
import { PARTICIPANTS, buildApplicationPrompt } from '@/lib/data/participants';

// `defineState` needs a live eve runtime context, so the durable slot is
// swapped for an in-memory one. This exercises the hook's own logic — extract
// from the inbound message, first record wins — without booting a session.
// It does NOT prove eve invokes the hook or persists the state; only a real
// session does that.
const slot: { value: unknown } = { value: null };
vi.mock('eve/context', () => ({
  defineState: (_name: string, init: () => unknown) => {
    slot.value = init();
    return {
      get: () => slot.value,
      update: (fn: (c: unknown) => unknown) => {
        slot.value = fn(slot.value);
      },
    };
  },
}));
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

const received = (message: string) => ({
  data: { message, sequence: 0, turnId: 't1' },
  type: 'message.received' as const,
});

describe('participant capture hook', () => {
  beforeEach(() => {
    slot.value = null;
  });

  it('captures the record from an inbound message', async () => {
    const onMessage = await loadHook();
    onMessage(received(opening(PARTICIPANT)));
    expect((slot.value as { record_id: string })?.record_id).toBe(
      PARTICIPANT.record_id,
    );
  });

  it('keeps the first record when later messages arrive', async () => {
    const onMessage = await loadHook();
    onMessage(received(opening(PARTICIPANT)));
    onMessage(received('now do IHSS for the same person'));
    onMessage(received(opening(OTHER)));
    expect((slot.value as { record_id: string })?.record_id).toBe(
      PARTICIPANT.record_id,
    );
  });

  it('leaves state empty when no record is present', async () => {
    const onMessage = await loadHook();
    onMessage(received('can you help me with a form?'));
    expect(slot.value).toBeNull();
  });

  it('tolerates a message with no text', async () => {
    const onMessage = await loadHook();
    expect(() =>
      onMessage({
        data: { sequence: 0, turnId: 't1' },
        type: 'message.received',
      }),
    ).not.toThrow();
    expect(slot.value).toBeNull();
  });
});
