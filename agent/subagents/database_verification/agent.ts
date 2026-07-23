import { defineAgent } from 'eve';

export default defineAgent({
  description:
    'Retrieve a participant\'s Apricot records and resolve every field_NNNN to its confirmed label before any value is trusted. Returns source-tagged, verified data. Delegate here before reasoning about participant data.',
  model: 'anthropic/claude-sonnet-4.6',
});
