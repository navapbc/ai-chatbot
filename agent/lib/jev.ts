// Eve-side helpers for reading the participant out of session state.
//
// `defineState` throws when called outside eve-managed code, and a subagent
// does not inherit the root agent's state (agent/subagents/* carry their own
// copies of everything for this reason). Both cases read as "no record",
// which every Jev feature already treats the same as being switched off — so
// a subagent simply gets un-annotated rows rather than an error.

import type { Participant } from '@/lib/data/participants';
import { participantState } from './participant-state';

export const currentParticipant = (): Participant | null => {
  try {
    return participantState.get();
  } catch {
    return null;
  }
};
