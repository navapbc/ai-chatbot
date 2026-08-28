import { defineTool } from 'eve/tools';
import { z } from 'zod';
import { runBrowserCommand } from '@/lib/kernel/eve-browser';

export default defineTool({
  description: `Execute an agent-browser command on a remote Kernel browser.

Pass the command as an argv array, exactly as the agent-browser CLI takes it — the first element is the command, the rest are its arguments and flags. Values are passed through as-is, so never quote or shell-escape them. See the browser-automation skill for snapshot discipline, selector strategy, and workflow rules. Snapshot before interacting; re-snapshot after every DOM change. NEVER navigate away from the target application domain and NEVER click the final submit button.

Common commands:
- ["open", "<url>"] - Navigate to URL (waits for load; no separate wait needed)
- ["snapshot"] - Full accessibility tree (ALWAYS do this first)
- ["snapshot", "-s", "form"] - Scoped snapshot (reduces noise)
- ["snapshot", "-i"] - Interactive elements only, with refs
- ["click", "@e1"] - Click element by ref
- ["fill", "@e1", "text"] - Clear field and fill (use for plain text fields)
- ["type", "@e1", "text"] - Real keystrokes, no clear (use for masked fields: SSN, date, phone, state, zip)
- ["select", "@e1", "option"] - Select native dropdown option (repeat the value argument for multi-select)
- ["find", "label", "Field Name", "fill", "val"] - Act on a field by its accessible label
- ["find", "role", "button", "click", "--name", "Submit"] - Act on an element by ARIA role
- ["press", "Enter"] - Press key (Tab, Escape, ArrowDown, Control+a, …)
- ["hover", "@e1"] / ["check", "@e1"] / ["uncheck", "@e1"]
- ["scrollintoview", "@e1"] - Scroll element into view
- ["wait", "@e1"] - Wait for element; ["wait", "2000"] - wait ms; ["wait", "--load", "networkidle"]
- ["get", "text", "@e1"] / ["get", "value", "@e1"] / ["get", "url"] / ["get", "title"]
- ["scroll", "down", "500"] - Scroll down 500px
- ["screenshot"] - Take screenshot
- ["back"] / ["forward"] - Browser navigation (AVOID during form filling — may wipe state)
- ["eval", "document.title"] - Run JavaScript (ONLY for reading simple values — NEVER to find/click elements)
- ["tab"] / ["tab", "t2"] / ["tab", "new"] / ["tab", "close"] - Tab management (ids look like t1, t2)
- ["dialog", "accept"] / ["dialog", "dismiss"] - Handle browser dialogs
- ["frame", "#iframe"] / ["frame", "main"] - Switch between frames

NEVER navigate away from the target application domain. Do NOT click social media links, share buttons, or external links.`,
  inputSchema: z
    .object({
      command: z
        .array(z.string())
        .min(1)
        .describe(
          'agent-browser CLI argv, e.g. ["click", "@e1"] or ["fill", "@e1", "John"]. One argument per array element; do not quote or escape values.',
        ),
    })
    .describe('An agent-browser CLI command as an argv array'),
  async execute({ command }, ctx) {
    try {
      const response = await runBrowserCommand(ctx, command);
      // `liveViewUrl` is carried for the chat UI's live browser panel, not for
      // the model: `eve dev` is a separate process from the Next app, so the
      // tool result is how the URL crosses over. The Next stream reader caches
      // it (lib/ai/eve/live-view-store.ts) and /api/kernel-browser serves it.
      if (response.success) {
        const output =
          typeof response.data === 'string'
            ? response.data
            : JSON.stringify(response.data);
        return {
          success: true,
          output,
          error: null,
          liveViewUrl: response.liveViewUrl,
        };
      }
      return {
        success: false,
        output: null,
        error: response.error ?? 'command failed',
        liveViewUrl: response.liveViewUrl,
      };
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : String(error);
      return {
        success: false,
        output: null,
        error: message,
        liveViewUrl: null,
      };
    }
  },
});
