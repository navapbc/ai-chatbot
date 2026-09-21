// Captures the participant record from inbound messages into session state.
//
// The record only ever exists as JSON inside the caseworker's opening message
// (`buildApplicationPrompt` in lib/data/participants.ts). Hooks are the right
// place to lift it out: they run on code's schedule rather than the model's,
// so nothing depends on the agent choosing to call a tool first.
//
// Hooks are observe-only with respect to model context — this writes session
// state, which is a side effect, not an injected message.

import { defineHook } from 'eve/hooks';
import { extractParticipantFromText } from '@/lib/jev/participant';
import { participantState } from '../lib/participant-state';

export default defineHook({
  events: {
    'message.received'(event) {
      // First record wins. A session carries across forms (WIC → IHSS →
      // BenefitsCal) and only the opening message has the record; later
      // messages must not clear it.
      if (participantState.get()) return;
      const found = extractParticipantFromText(event.data.message ?? '');
      if (found) participantState.update(() => found);
    },
  },
});
