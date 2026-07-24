import { defineAgent } from 'eve';

export default defineAgent({
  description:
    'Research a benefits program\'s application up front and enumerate ALL fields it will require across every page, so gap analysis is complete before form-filling starts. Returns a field checklist.',
  model: 'anthropic/claude-sonnet-4.6',
});
