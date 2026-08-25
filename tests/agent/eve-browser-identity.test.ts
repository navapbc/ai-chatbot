import { describe, it, expect, vi } from 'vitest';

// `lib/kernel/eve-browser` constructs a Kernel client at module load, which
// throws when KERNEL_API_KEY is unset (as it is in CI). browserIdentity is pure
// and touches none of it, so stub the SDK rather than requiring a real key.
vi.mock('@onkernel/sdk', () => ({
  default: class {
    profiles = { create: vi.fn() };
    browsers = { create: vi.fn() };
  },
}));

import { browserIdentity } from '@/lib/kernel/eve-browser';
import { cliSessionName } from '@/lib/kernel/session-store';

type Principal = {
  attributes: Record<string, string | readonly string[]>;
  authenticator: string;
  principalId: string;
  principalType: string;
  subject?: string;
};

const principal = (over: Partial<Principal> = {}): Principal => ({
  attributes: {},
  authenticator: 'test',
  principalId: 'principal-1',
  principalType: 'user',
  subject: 'subject-1',
  ...over,
});

// Minimal stand-in for eve's ToolContext: browserIdentity reads only
// `session.id` and `session.auth`.
const ctx = (auth: {
  current: Principal | null;
  initiator: Principal | null;
}) =>
  ({ session: { id: 'wrun_session_1', auth } }) as unknown as Parameters<
    typeof browserIdentity
  >[0];

describe('browserIdentity', () => {
  it('passes the eve session id through unchanged', () => {
    const { sessionId } = browserIdentity(
      ctx({ current: principal(), initiator: principal() }),
    );
    expect(sessionId).toBe('wrun_session_1');
  });

  it('prefers the initiator over the current caller', () => {
    const { userId } = browserIdentity(
      ctx({
        current: principal({ subject: 'follow-up-caller' }),
        initiator: principal({ subject: 'session-creator' }),
      }),
    );
    expect(userId).toBe('session-creator');
  });

  // The regression this file exists for: keying on `auth.current` renamed the
  // agent-browser daemon on the first follow-up turn, which orphaned the live
  // browser and every `@eN` ref mid-form ("Unknown ref: e22").
  it('stays stable across turns when only the current caller changes', () => {
    const initiator = principal({ subject: 'session-creator' });

    // Turn 0: the creator's own request carries a principal.
    const turn0 = browserIdentity(ctx({ current: initiator, initiator }));
    // Turn 1: a follow-up arrives with no principal on `current`.
    const turn1 = browserIdentity(ctx({ current: null, initiator }));
    // Turn 2: a different caller follows up on the same session.
    const turn2 = browserIdentity(
      ctx({ current: principal({ subject: 'someone-else' }), initiator }),
    );

    expect(turn1).toEqual(turn0);
    expect(turn2).toEqual(turn0);

    // What actually matters: one daemon name, so refs survive the turn boundary.
    const names = [turn0, turn1, turn2].map((i) =>
      cliSessionName(i.userId, i.sessionId),
    );
    expect(new Set(names).size).toBe(1);
  });

  it('falls back to principalId when the principal carries no subject', () => {
    const { userId } = browserIdentity(
      ctx({
        current: null,
        initiator: principal({ subject: undefined, principalId: 'pid-9' }),
      }),
    );
    expect(userId).toBe('pid-9');
  });

  it('reads both fields off one principal, never mixing the two', () => {
    // initiator wins, so current's principalId must not leak in when the
    // initiator has no subject.
    const { userId } = browserIdentity(
      ctx({
        current: principal({ subject: 'cur-sub', principalId: 'cur-pid' }),
        initiator: principal({ subject: undefined, principalId: 'init-pid' }),
      }),
    );
    expect(userId).toBe('init-pid');
  });

  it('uses the current caller when there is no initiator', () => {
    const { userId } = browserIdentity(
      ctx({ current: principal({ subject: 'only-caller' }), initiator: null }),
    );
    expect(userId).toBe('only-caller');
  });

  it('falls back to a constant when there is no auth at all (standalone eve dev)', () => {
    const { userId } = browserIdentity(ctx({ current: null, initiator: null }));
    expect(userId).toBe('eve-local');
  });
});
