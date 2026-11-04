import { defineTool } from 'eve/tools';
import { runProgressSchema } from '@/lib/relay/progress';

// Status for the extension panel. The tool has no side effects and echoes
// nothing back: the eve-chat route reads the call's input off the event stream
// (lib/ai/eve/stream-adapter.ts), sanitizes it, and stores it. Labels, counts
// and statuses only — never field values.
export default defineTool({
  description:
    'Report form-filling progress to the caseworker panel. Send section names, statuses (waiting, filling, done, needs_you), filled/total counts, an optional list of field labels per section (each with a state of waiting, filled, needs_you, or skipped), an optional current section, and up to 5 short recent actions. Names, labels and counts only: never include field values or applicant data.',
  inputSchema: runProgressSchema,
  async execute() {
    return { ok: true };
  },
});
