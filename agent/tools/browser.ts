import { defineTool } from 'eve/tools';
import { z } from 'zod';

// Example tool ONLY. Production browser automation runs against Kernel.sh via
// lib/kernel/browser.ts + lib/ai/tools/browser.ts, and re-architecting it for
// Eve's durable execution is migration sub-project 3 (see docs/eve-spike-findings.md
// "Browser session sketch"). Eve tools run in the app runtime (not the sandbox),
// so a real port would call Kernel.sh here, re-resolving the session by its
// stable id each call. This stub only shows the command shape.
export default defineTool({
  description:
    'Send a structured browser command (navigate, snapshot, click, fill, type, select, check, evaluate, wait). Snapshot before interacting.',
  inputSchema: z.object({
    action: z.enum([
      'navigate',
      'snapshot',
      'click',
      'fill',
      'type',
      'select',
      'check',
      'evaluate',
      'press',
      'wait',
      'inputvalue',
      'back',
      'reload',
    ]),
    url: z.string().optional(),
    selector: z.string().optional(),
    value: z.string().optional(),
    text: z.string().optional(),
  }),
  async execute(input) {
    // Demonstrative stub: production dispatches to Kernel.sh.
    return { ok: true, action: input.action };
  },
});
