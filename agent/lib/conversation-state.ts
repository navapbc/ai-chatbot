// Durable per-session slot holding the caseworker's own turns for the Eve agent.
//
// Sibling of `participant-state.ts` and it exists for the same structural
// reason: Eve tools are module-level `defineTool` default exports with no
// construction step, and `ToolContext` exposes no message history, so nothing
// a tool can reach knows what the caseworker said. The legacy route just
// passes `extractCaseworkerMessages(messages)` into the tool factory; here a
// hook has to put it in state first.
//
// Why Jev needs it: `summaryFieldQuestions` used to see only the participant
// record, so it could not distinguish an inference drawn from the caseworker's
// instructions from an invention. On run wrun_01M37CZX that produced two false
// `unsupported` verdicts out of six.
//
// `defineState` is durable and survives compaction — which matters, because
// compaction is exactly what would otherwise drop the opening message.

import { defineState } from 'eve/context';
import { CASEWORKER_CONTEXT_CHAR_BUDGET } from '@/lib/jev/participant';

/**
 * One recorded turn. The key exists because appending is NOT idempotent and
 * eve's docs do not promise `message.received` fires exactly once — the
 * workflow replays completed steps (docs/concepts/security-model.md), and
 * /api/eve-chat re-opens the stream on a body timeout. A replayed delivery
 * would otherwise duplicate a turn into every Jev request and eat the budget.
 */
export interface RecordedTurn {
  /** `${turnId}#${sequence}` — stable across a replay of the same delivery. */
  key: string;
  text: string;
}

export const caseworkerMessagesState = defineState<RecordedTurn[]>(
  'labs-asp.caseworker-messages',
  () => [],
);

/** Stable key for one inbound message delivery. */
export const turnKey = (turnId: unknown, sequence: unknown): string =>
  `${String(turnId ?? '?')}#${String(sequence ?? 0)}`;

/**
 * Append a turn, dropping the oldest when the budget is exceeded.
 *
 * These messages ride on EVERY summary row's Jev request (~25 per card), so an
 * unbounded transcript multiplies cost by the row count. Trimming from the
 * front keeps the recent turns, which is where a gap analysis' answers live —
 * the thing `caseworker_supplied` has to be checked against.
 *
 * Exported as a pure function so the trimming is testable without eve.
 */
export const appendWithinBudget = (
  existing: readonly RecordedTurn[],
  incoming: RecordedTurn,
  charBudget: number = CASEWORKER_CONTEXT_CHAR_BUDGET,
): RecordedTurn[] => {
  if (!incoming.text) return [...existing];
  // Replayed delivery — already recorded, nothing to do.
  if (existing.some((t) => t.key === incoming.key)) return [...existing];

  const next = [...existing, incoming];
  let total = next.reduce((n, t) => n + t.text.length, 0);
  // Always keep at least the newest turn, even if it alone exceeds the budget.
  while (next.length > 1 && total > charBudget) {
    total -= (next.shift() as RecordedTurn).text.length;
  }
  return next;
};
