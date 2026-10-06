// Captures the participant record AND the caseworker's own turns from inbound
// messages into session state.
//
// The record only ever exists as JSON inside the caseworker's opening message
// (`buildApplicationPrompt` in lib/data/participants.ts). Hooks are the right
// place to lift it out: they run on code's schedule rather than the model's,
// so nothing depends on the agent choosing to call a tool first.
//
// Hooks are observe-only with respect to model context — this writes session
// state, which is a side effect, not an injected message.

import { defineHook } from 'eve/hooks';
import {
  extractParticipantFromText,
  stripParticipantRecord,
} from '@/lib/jev/participant';
import {
  appendWithinBudget,
  caseworkerMessagesState,
  turnKey,
} from '../lib/conversation-state';
import { participantState } from '../lib/participant-state';

export default defineHook({
  events: {
    'message.received'(event) {
      const message = event.data.message ?? '';

      // Every caseworker turn is kept, with the record JSON stripped out —
      // the record already travels to Jev as `participant_record`, and a
      // second copy inside the message text would let it be double-counted as
      // corroboration. Recorded BEFORE the early return below, or every turn
      // after the first would be dropped.
      const said = stripParticipantRecord(message);
      if (said) {
        const key = turnKey(event.data.turnId, event.data.sequence);
        caseworkerMessagesState.update((prev) =>
          appendWithinBudget(prev, { key, text: said }),
        );
      }

      // First record wins. A session carries across forms (WIC → IHSS →
      // BenefitsCal) and only the opening message has the record; later
      // messages must not clear it.
      if (participantState.get()) return;
      const found = extractParticipantFromText(message);
      if (found) participantState.update(() => found);
    },
  },
});
