import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { tool, type ToolExecutionOptions } from 'ai';
import { z } from 'zod';
import { getOrCreateBrowser } from '@/lib/kernel/browser';

const execFileAsync = promisify(execFile);

const COMMAND_TIMEOUT_MS = 120_000; // 2 minutes
const AGENT_BROWSER_BIN = 'agent-browser';

const sessionQueues = new Map<string, Promise<unknown>>();
const connectedSessions = new Set<string>();

function withSessionQueue<T>(sessionId: string, fn: () => Promise<T>): Promise<T> {
  const prev = sessionQueues.get(sessionId) ?? Promise.resolve();
  const next = prev.then(fn, fn);
  sessionQueues.set(sessionId, next.then(() => {}, () => {}));
  return next;
}

/** Stable agent-browser `--session` name from our kernel session id. */
function toAgentSession(kernelSessionId: string): string {
  return `kernel-${kernelSessionId}`;
}

/**
 * Translate our structured `{ action, ... }` object into agent-browser CLI
 * argv. Returns an array suitable for `agent-browser batch --json` stdin
 * (a single command as `[cmd, ...args]`).
 */
function toCliCommand(params: Record<string, unknown>): string[] {
  const p = params as Record<string, string | number | boolean | string[] | undefined>;
  const action = String(p.action);

  switch (action) {
    case 'navigate':
      return ['open', String(p.url)];

    case 'snapshot': {
      const args = ['snapshot'];
      if (p.interactive) args.push('-i');
      if (p.selector) args.push('-s', String(p.selector));
      return args;
    }

    case 'click':
      return ['click', String(p.selector)];

    case 'fill':
      return ['fill', String(p.selector), String(p.value ?? '')];

    case 'type': {
      const args = ['type', String(p.selector), String(p.text ?? '')];
      if (p.clear) args.push('--clear');
      return args;
    }

    case 'select': {
      const values = Array.isArray(p.values) ? p.values : [String(p.value ?? '')];
      return ['select', String(p.selector), ...values.map(String)];
    }

    case 'press':
      return ['press', String(p.key)];

    case 'hover':
      return ['hover', String(p.selector)];

    case 'check':
      return ['check', String(p.selector)];

    case 'uncheck':
      return ['uncheck', String(p.selector)];

    case 'scrollintoview':
      return ['scrollintoview', String(p.selector)];

    case 'wait': {
      if (p.state) return ['wait', '--load', String(p.state)];
      if (p.selector) return ['wait', String(p.selector)];
      return ['wait', String(p.timeout ?? 1000)];
    }

    case 'waitforloadstate':
      return ['wait', '--load', String(p.state ?? 'networkidle')];

    case 'gettext':
      return ['get', 'text', String(p.selector)];

    case 'inputvalue':
      return ['get', 'value', String(p.selector)];

    case 'url':
      return ['get', 'url'];

    case 'title':
      return ['get', 'title'];

    case 'scroll': {
      const dir = String(p.direction ?? 'down');
      const amount = p.amount != null ? String(p.amount) : '300';
      return ['scroll', dir, amount];
    }

    case 'screenshot':
      return ['screenshot'];

    case 'back':
      return ['back'];

    case 'forward':
      return ['forward'];

    case 'evaluate':
      return ['eval', String(p.script ?? '')];

    case 'tab_list':
      return ['tab'];

    case 'tab_switch':
      return ['tab', String(p.index ?? 0)];

    case 'tab_new':
      return p.url ? ['tab', 'new', String(p.url)] : ['tab', 'new'];

    case 'tab_close':
      return p.index != null ? ['tab', 'close', String(p.index)] : ['tab', 'close'];

    case 'dialog': {
      const response = String(p.response ?? 'accept');
      if (response === 'accept' && p.promptText) {
        return ['dialog', 'accept', String(p.promptText)];
      }
      return ['dialog', response];
    }

    case 'frame':
      return ['frame', String(p.selector)];

    case 'mainframe':
      return ['frame', 'main'];

    case 'getbylabel': {
      const sub = String(p.subaction ?? 'click');
      const args = ['find', 'label', String(p.label), sub];
      if (p.value != null) args.push(String(p.value));
      return args;
    }

    default:
      throw new Error(`Unknown browser action: ${action}`);
  }
}

interface BatchResponse {
  success: boolean;
  command?: string[];
  result?: unknown;
  error?: string | null;
}

/**
 * Creates a browser automation tool for a specific session.
 * Spawns the agent-browser Rust CLI in `batch --json` mode and pipes a
 * single command via stdin. Commands target our Kernel-managed browser
 * via `--cdp <cdp_ws_url>` (stateless per-command; no daemon session state).
 *
 * @param sessionId - The chat/session ID for browser isolation
 * @param userId - The user ID for ownership validation and security
 *
 * @see https://www.kernel.sh/docs/integrations/agent-browser
 */
export const createBrowserTool = (sessionId: string, userId: string) =>
  tool({
    description: `Execute browser automation commands on a remote Kernel browser.

Send structured JSON commands with an "action" field and action-specific parameters. See the Browser Automation skill for snapshot discipline, selector strategy, and workflow rules.

Commands:
- { action: "navigate", url: "<url>" } - Navigate to URL
- { action: "snapshot" } - Full accessibility tree (ALWAYS do this first)
- { action: "snapshot", selector: "form" } - Scoped snapshot (reduces noise)
- { action: "snapshot", interactive: true } - Interactive elements only with refs
- { action: "click", selector: "@e1" } - Click element by ref
- { action: "fill", selector: "@e1", value: "text" } - Clear field and fill (programmatic — use for plain text fields)
- { action: "type", selector: "@e1", text: "text", clear: true } - Simulate real keystrokes (use for masked fields: SSN, date, phone, state, zip)
- { action: "select", selector: "@e1", values: ["option"] } - Select native dropdown option
- { action: "getbylabel", label: "Field Name", subaction: "fill", value: "val" } - Fill by accessible label
- { action: "press", key: "Enter" } - Press key (Tab, Escape, ArrowDown, etc.)
- { action: "hover", selector: "@e1" } - Hover over element
- { action: "check", selector: "@e1" } - Toggle checkbox on
- { action: "uncheck", selector: "@e1" } - Toggle checkbox off
- { action: "scrollintoview", selector: "@e1" } - Scroll element into view
- { action: "wait", selector: "@e1" } - Wait for element
- { action: "wait", timeout: 2000 } - Wait milliseconds
- { action: "waitforloadstate", state: "networkidle" } - Wait for network to settle
- { action: "gettext", selector: "@e1" } - Get element text content
- { action: "inputvalue", selector: "@e1" } - Get input field value
- { action: "url" } - Get current URL
- { action: "title" } - Get page title
- { action: "scroll", direction: "down", amount: 500 } - Scroll down 500px
- { action: "screenshot" } - Take screenshot
- { action: "back" } / { action: "forward" } - Browser navigation (AVOID during form filling — may wipe state)
- { action: "evaluate", script: "document.title" } - Run JavaScript (ONLY for reading simple values — NEVER to find/click elements)
- { action: "tab_list" } / { action: "tab_switch", index: N } / { action: "tab_new" } / { action: "tab_close" } - Tab management
- { action: "dialog", response: "accept" } / { action: "dialog", response: "dismiss" } - Handle browser dialogs
- { action: "frame", selector: "#iframe" } / { action: "mainframe" } - Switch between frames

NEVER navigate away from the target application domain. Do NOT click social media links, share buttons, or external links.`,
    inputSchema: z
      .object({
        action: z.string().describe('The command action (e.g. "navigate", "click", "snapshot", "fill")'),
        selector: z.string().optional().describe('Element selector: ref (@e1), CSS (#id), or label'),
        value: z.string().optional().describe('Value for fill action'),
        text: z.string().optional().describe('Text for type action'),
        url: z.string().optional().describe('URL for navigate action'),
        key: z.string().optional().describe('Key for press action (e.g. "Enter", "Tab")'),
        label: z.string().optional().describe('Label text for getbylabel action'),
        subaction: z.string().optional().describe('Sub-action for getbylabel ("click", "fill", "check")'),
        script: z.string().optional().describe('JavaScript for evaluate action'),
        values: z.array(z.string()).optional().describe('Option values for select action — must be an array'),
        timeout: z.number().optional().describe('Timeout in ms for wait action — must be a number'),
        amount: z.number().optional().describe('Scroll amount in px — must be a number'),
        delay: z.number().optional().describe('Delay between keystrokes in ms — must be a number'),
        interactive: z.boolean().optional().describe('Show only interactive elements in snapshot — must be boolean'),
        clear: z.boolean().optional().describe('Clear field before typing — must be boolean'),
        direction: z.string().optional().describe('Scroll direction: "up" or "down"'),
        state: z.string().optional().describe('Load state for waitforloadstate (e.g. "networkidle")'),
        index: z.number().optional().describe('Tab index for tab_switch/tab_close'),
        response: z.string().optional().describe('Dialog response: "accept" or "dismiss"'),
        promptText: z.string().optional().describe('Text to enter in prompt dialog'),
      })
      .describe('Structured command object with action and action-specific parameters'),
    execute: async (
      params: Record<string, unknown>,
      { abortSignal }: ToolExecutionOptions,
    ) => {
      return withSessionQueue(sessionId, async () => {
        try {
          const session = await getOrCreateBrowser(sessionId, userId);
          const agentSession = toAgentSession(session.kernelSessionId);
          const argv = toCliCommand(params);
          const batchInput = JSON.stringify([argv]);

          console.log('[browser-tool] Session:', sessionId);
          console.log('[browser-tool] Executing:', params.action, JSON.stringify(params));

          // First call per kernel browser: `agent-browser connect <ws>` to
          // attach the agent-browser session to the Kernel CDP endpoint.
          // Subsequent calls target the same session by name.
          if (!connectedSessions.has(agentSession)) {
            const { stdout: connectOut, stderr: connectErr } = await execFileAsync(
              AGENT_BROWSER_BIN,
              ['--session', agentSession, 'connect', session.cdpWsUrl],
              { timeout: COMMAND_TIMEOUT_MS },
            );
            if (connectErr) console.log('[browser-tool] connect stderr:', connectErr);
            if (connectOut) console.log('[browser-tool] connect stdout:', connectOut);
            connectedSessions.add(agentSession);
            console.log('[browser-tool] Connected agent-browser session:', agentSession);
          }

          const child = execFileAsync(
            AGENT_BROWSER_BIN,
            ['--session', agentSession, 'batch', '--json'],
            {
              input: batchInput,
              timeout: COMMAND_TIMEOUT_MS,
              maxBuffer: 32 * 1024 * 1024,
            } as Parameters<typeof execFileAsync>[2] & { input: string },
          );

          if (abortSignal) {
            abortSignal.addEventListener('abort', () => {
              child.child.kill('SIGTERM');
            });
          }

          const { stdout } = await child;

          const parsed = JSON.parse(stdout) as BatchResponse | BatchResponse[];
          const response: BatchResponse = Array.isArray(parsed) ? parsed[0] : parsed;

          if (response.success) {
            const output =
              typeof response.result === 'string'
                ? response.result
                : JSON.stringify(response.result);
            console.log('[browser-tool] Success. Output length:', output?.length);
            return { success: true, output, error: null };
          }

          console.error('[browser-tool] Command error:', response.error);
          return { success: false, output: null, error: response.error ?? 'unknown error' };
        } catch (error: unknown) {
          const err = error as { message?: string; stderr?: string; stdout?: string; code?: number };
          const message = err.message ?? String(error);
          const stderr = err.stderr?.toString?.() ?? '';
          const stdout = err.stdout?.toString?.() ?? '';

          if (abortSignal?.aborted || message.includes('stopped by user') || message.includes('SIGTERM')) {
            console.log('[browser-tool] Command aborted by user');
            return {
              success: false,
              output: null,
              error: 'Browser command stopped by user',
            };
          }

          console.error('[browser-tool] Error:', message);
          if (stderr) console.error('[browser-tool] stderr:', stderr);
          if (stdout) console.error('[browser-tool] stdout:', stdout);

          const detail = stderr.trim() || stdout.trim() || message;
          return {
            success: false,
            output: null,
            error: detail,
          };
        }
      });
    },
  });
