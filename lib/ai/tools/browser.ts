import { tool } from 'ai';
import { z } from 'zod';
import { getOrCreateBrowser } from '@/lib/kernel/browser';
import { runCommand } from '@/lib/kernel/cli';
import { kernelTimelineCollector } from '@/lib/kernel/telemetry';
import { cliSessionName } from '@/lib/kernel/session-store';

const COMMAND_TIMEOUT_MS = 120_000; // 2 minutes

/**
 * Input contract for the browser tool: agent-browser's own CLI argv.
 *
 * Exported so the eval harness drives the agent through the identical schema
 * rather than a hand-copied duplicate that can drift.
 */
export const browserInputSchema = z
  .object({
    command: z
      .array(z.string())
      .min(1)
      .describe(
        'agent-browser CLI argv, e.g. ["click", "@e1"] or ["fill", "@e1", "John"]. One argument per array element; do not quote or escape values.',
      ),
  })
  .describe('An agent-browser CLI command as an argv array');

/**
 * Per-session mutex to serialize browser commands.
 *
 * The agent-browser daemon serializes work per `--session` internally, but the
 * AI SDK can fire parallel tool calls, and interleaving them would scramble the
 * `@eN` ref map: a snapshot's refs must still describe the page when the
 * following click runs. Queueing per session keeps each read→act pair adjacent.
 */
const sessionQueues = new Map<string, Promise<unknown>>();

function withSessionQueue<T>(
  sessionId: string,
  fn: () => Promise<T>,
): Promise<T> {
  const prev = sessionQueues.get(sessionId) ?? Promise.resolve();
  const next = prev.then(fn, fn); // always advance the queue even on error
  sessionQueues.set(
    sessionId,
    next.then(
      () => {},
      () => {},
    ),
  ); // swallow to prevent unhandled rejection on queue chain
  return next;
}

/**
 * Creates a browser automation tool for a specific session.
 *
 * The model emits agent-browser's own CLI argv (`["click", "@e1"]`) rather than
 * a bespoke action vocabulary this repo would have to keep mapping. New
 * agent-browser commands therefore work the moment the package is upgraded, and
 * the CLI's own errors reach the model unaltered.
 *
 * The daemon is keyed by `--session` and holds the CDP connection between
 * calls, so `@eN` refs from a snapshot stay valid for the commands that follow.
 *
 * @param sessionId - The chat/session ID for browser isolation
 * @param userId - The user ID for ownership validation and security
 *
 * @see https://agent-browser.dev
 */
export const createBrowserTool = (sessionId: string, userId: string) =>
  tool({
    description: `Execute an agent-browser command on a remote Kernel browser.

Pass the command as an argv array, exactly as the agent-browser CLI takes it — the first element is the command, the rest are its arguments and flags. Values are passed through as-is, so never quote or shell-escape them. See the Browser Automation skill for snapshot discipline, selector strategy, and workflow rules.

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
- ["press", "Enter"] - Press key (Tab, Escape, ArrowDown, …)
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
    inputSchema: browserInputSchema,
    execute: async (
      { command }: { command: string[] },
      { abortSignal }: { abortSignal?: AbortSignal },
    ) => {
      return withSessionQueue(sessionId, async () => {
        try {
          // Ensure we have a Kernel browser instance (creates one if needed)
          const session = await getOrCreateBrowser(sessionId, userId);

          const response = await runCommand(command, {
            session: cliSessionName(userId, sessionId),
            cdpUrl: session.cdpWsUrl,
            timeoutMs: COMMAND_TIMEOUT_MS,
            signal: abortSignal,
            collectTimeline: kernelTimelineCollector(session.kernelSessionId),
          });

          if (response.success) {
            const output =
              typeof response.data === 'string'
                ? response.data
                : JSON.stringify(response.data);
            return { success: true, output, error: null };
          }

          console.error('[browser-tool] Command error:', response.error);
          return { success: false, output: null, error: response.error };
        } catch (error: unknown) {
          const message =
            error instanceof Error ? error.message : String(error);

          if (abortSignal?.aborted || message.includes('stopped by user')) {
            return {
              success: false,
              output: null,
              error: 'Browser command stopped by user',
            };
          }

          console.error('[browser-tool] Error:', message);
          return {
            success: false,
            output: null,
            error: message,
          };
        }
      });
    },
  });
