import { defineHook } from 'eve/hooks';
import { Preparation } from '../../../lib/gsn/contract';
import { requestState } from '../lib/request';
export default defineHook({
  events: {
    'message.received'(event) {
      if (requestState.get()) return;
      const value = Preparation.parse(JSON.parse(event.data.message ?? ''));
      requestState.update(() => value);
    },
  },
});
