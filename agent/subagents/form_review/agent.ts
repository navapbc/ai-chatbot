import { defineAgent } from 'eve';

export default defineAgent({
  description:
    'Walk the application\'s review/summary screen at the end of filling and produce the structured, source-tagged formSummary field list for the caseworker to review before submission.',
  model: 'anthropic/claude-sonnet-4.6',
});
