import { defineTool } from 'eve/tools';
import { z } from 'zod';
import { runBrowserCommand } from '@/lib/kernel/eve-browser';

export default defineTool({
  description: `Execute browser automation commands on a remote Kernel browser.

Send a structured command with an "action" field and action-specific parameters. See the browser-automation skill for snapshot discipline, selector strategy, and workflow rules. Snapshot before interacting; re-snapshot after every DOM change. NEVER navigate away from the target application domain and NEVER click the final submit button.

Actions: navigate, snapshot (optional selector / interactive), click, fill, type (clear?), select (values[]), getbylabel (subaction), press (key), hover, check, uncheck, scrollintoview, wait (selector or timeout), waitforloadstate (state), gettext, inputvalue, url, title, scroll (direction/amount), screenshot, back, forward, evaluate (script — reading only), tab_list/tab_switch/tab_new/tab_close, dialog (response), frame/mainframe.`,
  inputSchema: z.object({
    action: z.string(),
    selector: z.string().optional(),
    value: z.string().optional(),
    text: z.string().optional(),
    url: z.string().optional(),
    key: z.string().optional(),
    label: z.string().optional(),
    subaction: z.string().optional(),
    script: z.string().optional(),
    values: z.array(z.string()).optional(),
    timeout: z.number().optional(),
    amount: z.number().optional(),
    delay: z.number().optional(),
    interactive: z.boolean().optional(),
    clear: z.boolean().optional(),
    direction: z.string().optional(),
    state: z.string().optional(),
    index: z.number().optional(),
    response: z.string().optional(),
    promptText: z.string().optional(),
  }),
  async execute(params, ctx) {
    try {
      const response = await runBrowserCommand(ctx, params);
      if (response.success) {
        const output =
          typeof response.data === 'string'
            ? response.data
            : JSON.stringify(response.data);
        return { success: true, output, error: null };
      }
      return { success: false, output: null, error: response.error ?? 'command failed' };
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : String(error);
      return { success: false, output: null, error: message };
    }
  },
});
