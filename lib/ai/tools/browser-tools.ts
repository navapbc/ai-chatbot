import { tool } from 'ai';
import { z } from 'zod';
import { execa } from 'execa';
import type { Session } from 'next-auth';
import type { BrowserSession } from '@/lib/browser/kernel-service';
import type { UIMessageStreamWriter } from 'ai';
import type { ChatMessage } from '@/lib/types';

// agent-browser CLI - uses npx which finds the binary in node_modules
const AGENT_BROWSER_CMD = 'npx';
const AGENT_BROWSER_ARGS = ['agent-browser'];

interface BrowserToolProps {
  session: Session;
  browserSession: BrowserSession;
  dataStream: UIMessageStreamWriter<ChatMessage>;
}

// Stream a browser action notification to the client
function streamBrowserAction(
  dataStream: UIMessageStreamWriter<ChatMessage>,
  action: string,
  details?: Record<string, any>
) {
  dataStream.write({
    type: 'data-browserAction',
    data: {
      type: 'browser-action',
      action,
      timestamp: Date.now(),
      ...details,
    },
    transient: true,
  });
}

async function runAgentBrowser(
  cdpWsUrl: string,
  args: string[],
  dataStream?: UIMessageStreamWriter<ChatMessage>,
  actionDescription?: string
): Promise<any> {
  const command = args[0];
  const startTime = Date.now();

  // Stream action start if dataStream provided
  if (dataStream && actionDescription) {
    streamBrowserAction(dataStream, actionDescription, { status: 'running', command });
  }

  try {
    console.log('[agent-browser] Running:', command, args.slice(1).join(' '));
    const result = await execa(AGENT_BROWSER_CMD, [
      ...AGENT_BROWSER_ARGS,
      '--cdp', cdpWsUrl,
      ...args,
      '--json'
    ]);

    const duration = Date.now() - startTime;

    // Only log truncated output for snapshot command (too verbose)
    if (command === 'snapshot') {
      console.log('[agent-browser] snapshot completed, length:', result.stdout.length, 'in', duration, 'ms');
    } else {
      console.log('[agent-browser] result:', result.stdout.substring(0, 200), 'in', duration, 'ms');
    }

    const parsed = JSON.parse(result.stdout);

    // Stream action completion
    if (dataStream && actionDescription) {
      streamBrowserAction(dataStream, actionDescription, {
        status: 'complete',
        command,
        duration,
        success: true,
      });
    }

    return parsed;
  } catch (error: any) {
    const duration = Date.now() - startTime;
    console.error('[agent-browser] Error:', error.message);
    console.error('[agent-browser] stdout:', error.stdout);
    console.error('[agent-browser] stderr:', error.stderr);

    // Stream action error
    if (dataStream && actionDescription) {
      streamBrowserAction(dataStream, actionDescription, {
        status: 'error',
        command,
        duration,
        error: error.message,
      });
    }

    // Try to parse error output as JSON
    if (error.stdout) {
      try {
        return JSON.parse(error.stdout);
      } catch {
        // Fall through
      }
    }
    return {
      success: false,
      error: error.stderr || error.message || 'agent-browser command failed',
    };
  }
}

export const browserNavigate = ({ browserSession, dataStream }: BrowserToolProps) =>
  tool({
    description: 'Navigate to a URL in the browser. Use this to open websites.',
    inputSchema: z.object({
      url: z.string().describe('The URL to navigate to'),
    }),
    execute: async ({ url }) => {
      return await runAgentBrowser(browserSession.cdpWsUrl, ['open', url], dataStream, `Opening ${url}`);
    },
  });

export const browserSnapshot = ({ browserSession, dataStream }: BrowserToolProps) =>
  tool({
    description: 'Get accessibility tree snapshot of the current page with element refs (@e1, @e2, etc.). Use this to understand page structure and find elements to interact with.',
    inputSchema: z.object({}),
    execute: async () => {
      return await runAgentBrowser(browserSession.cdpWsUrl, ['snapshot', '-i'], dataStream, 'Reading page');
    },
  });

export const browserClick = ({ browserSession, dataStream }: BrowserToolProps) =>
  tool({
    description: 'Click an element by its ref (e.g., @e1) from a previous snapshot.',
    inputSchema: z.object({
      ref: z.string().describe('Element ref from snapshot (e.g., @e1)'),
    }),
    execute: async ({ ref }) => {
      return await runAgentBrowser(browserSession.cdpWsUrl, ['click', ref], dataStream, `Clicking ${ref}`);
    },
  });

export const browserType = ({ browserSession, dataStream }: BrowserToolProps) =>
  tool({
    description: 'Type text into a focused element or at cursor position.',
    inputSchema: z.object({
      text: z.string().describe('Text to type'),
    }),
    execute: async ({ text }) => {
      const preview = text.length > 20 ? text.slice(0, 20) + '...' : text;
      return await runAgentBrowser(browserSession.cdpWsUrl, ['type', 'body', text], dataStream, `Typing "${preview}"`);
    },
  });

export const browserFill = ({ browserSession, dataStream }: BrowserToolProps) =>
  tool({
    description: 'Clear an input field and fill it with new text. Use element ref from snapshot.',
    inputSchema: z.object({
      ref: z.string().describe('Element ref from snapshot (e.g., @e1)'),
      text: z.string().describe('Text to fill'),
    }),
    execute: async ({ ref, text }) => {
      const preview = text.length > 15 ? text.slice(0, 15) + '...' : text;
      return await runAgentBrowser(browserSession.cdpWsUrl, ['fill', ref, text], dataStream, `Filling ${ref} with "${preview}"`);
    },
  });

export const browserScroll = ({ browserSession, dataStream }: BrowserToolProps) =>
  tool({
    description: 'Scroll the page in a direction.',
    inputSchema: z.object({
      direction: z.enum(['up', 'down', 'left', 'right']).describe('Scroll direction'),
      pixels: z.number().optional().describe('Number of pixels to scroll (default: 300)'),
    }),
    execute: async ({ direction, pixels }) => {
      const args = pixels ? [direction, String(pixels)] : [direction];
      return await runAgentBrowser(browserSession.cdpWsUrl, ['scroll', ...args], dataStream, `Scrolling ${direction}`);
    },
  });

export const browserPress = ({ browserSession, dataStream }: BrowserToolProps) =>
  tool({
    description: 'Press a key or key combination (e.g., Enter, Tab, Control+a, Escape).',
    inputSchema: z.object({
      key: z.string().describe('Key to press (e.g., Enter, Tab, Escape, Control+a)'),
    }),
    execute: async ({ key }) => {
      return await runAgentBrowser(browserSession.cdpWsUrl, ['press', key], dataStream, `Pressing ${key}`);
    },
  });

export const browserSelect = ({ browserSession, dataStream }: BrowserToolProps) =>
  tool({
    description: 'Select an option from a dropdown/select element.',
    inputSchema: z.object({
      ref: z.string().describe('Element ref of the select element'),
      value: z.string().describe('Option value to select'),
    }),
    execute: async ({ ref, value }) => {
      return await runAgentBrowser(browserSession.cdpWsUrl, ['select', ref, value], dataStream, `Selecting "${value}"`);
    },
  });

export const browserCheck = ({ browserSession, dataStream }: BrowserToolProps) =>
  tool({
    description: 'Check a checkbox element.',
    inputSchema: z.object({
      ref: z.string().describe('Element ref of the checkbox'),
    }),
    execute: async ({ ref }) => {
      return await runAgentBrowser(browserSession.cdpWsUrl, ['check', ref], dataStream, `Checking ${ref}`);
    },
  });

export const browserUncheck = ({ browserSession, dataStream }: BrowserToolProps) =>
  tool({
    description: 'Uncheck a checkbox element.',
    inputSchema: z.object({
      ref: z.string().describe('Element ref of the checkbox'),
    }),
    execute: async ({ ref }) => {
      return await runAgentBrowser(browserSession.cdpWsUrl, ['uncheck', ref], dataStream, `Unchecking ${ref}`);
    },
  });

export const browserHover = ({ browserSession, dataStream }: BrowserToolProps) =>
  tool({
    description: 'Hover over an element to reveal hidden content or trigger hover states.',
    inputSchema: z.object({
      ref: z.string().describe('Element ref from snapshot'),
    }),
    execute: async ({ ref }) => {
      return await runAgentBrowser(browserSession.cdpWsUrl, ['hover', ref], dataStream, `Hovering ${ref}`);
    },
  });

export const browserWait = ({ browserSession, dataStream }: BrowserToolProps) =>
  tool({
    description: 'Wait for an element to appear or for a specified time.',
    inputSchema: z.object({
      target: z.string().describe('Element ref to wait for, or milliseconds to wait (e.g., "@e1" or "2000")'),
    }),
    execute: async ({ target }) => {
      return await runAgentBrowser(browserSession.cdpWsUrl, ['wait', target], dataStream, `Waiting for ${target}`);
    },
  });

export const browserBack = ({ browserSession, dataStream }: BrowserToolProps) =>
  tool({
    description: 'Navigate back to the previous page.',
    inputSchema: z.object({}),
    execute: async () => {
      return await runAgentBrowser(browserSession.cdpWsUrl, ['back'], dataStream, 'Going back');
    },
  });

export const browserForward = ({ browserSession, dataStream }: BrowserToolProps) =>
  tool({
    description: 'Navigate forward to the next page.',
    inputSchema: z.object({}),
    execute: async () => {
      return await runAgentBrowser(browserSession.cdpWsUrl, ['forward'], dataStream, 'Going forward');
    },
  });

export const browserReload = ({ browserSession, dataStream }: BrowserToolProps) =>
  tool({
    description: 'Reload the current page.',
    inputSchema: z.object({}),
    execute: async () => {
      return await runAgentBrowser(browserSession.cdpWsUrl, ['reload'], dataStream, 'Reloading page');
    },
  });

export const browserGetText = ({ browserSession, dataStream }: BrowserToolProps) =>
  tool({
    description: 'Get the text content of an element.',
    inputSchema: z.object({
      ref: z.string().describe('Element ref from snapshot'),
    }),
    execute: async ({ ref }) => {
      return await runAgentBrowser(browserSession.cdpWsUrl, ['get', 'text', ref], dataStream, `Reading text from ${ref}`);
    },
  });

export const browserGetUrl = ({ browserSession, dataStream }: BrowserToolProps) =>
  tool({
    description: 'Get the current page URL.',
    inputSchema: z.object({}),
    execute: async () => {
      return await runAgentBrowser(browserSession.cdpWsUrl, ['get', 'url'], dataStream, 'Getting URL');
    },
  });

export const browserGetTitle = ({ browserSession, dataStream }: BrowserToolProps) =>
  tool({
    description: 'Get the current page title.',
    inputSchema: z.object({}),
    execute: async () => {
      return await runAgentBrowser(browserSession.cdpWsUrl, ['get', 'title'], dataStream, 'Getting title');
    },
  });

export const browserScreenshot = ({ browserSession, dataStream }: BrowserToolProps) =>
  tool({
    description: 'Take a screenshot of the current page. Returns base64-encoded image.',
    inputSchema: z.object({
      fullPage: z.boolean().optional().describe('Capture full page including scrollable area'),
    }),
    execute: async ({ fullPage }) => {
      const args = fullPage ? ['screenshot', '-f'] : ['screenshot'];
      return await runAgentBrowser(browserSession.cdpWsUrl, args, dataStream, 'Taking screenshot');
    },
  });

// Export all browser tools as a function that takes props and returns tool map
export function createBrowserTools(props: BrowserToolProps) {
  return {
    browserNavigate: browserNavigate(props),
    browserSnapshot: browserSnapshot(props),
    browserClick: browserClick(props),
    browserType: browserType(props),
    browserFill: browserFill(props),
    browserScroll: browserScroll(props),
    browserPress: browserPress(props),
    browserSelect: browserSelect(props),
    browserCheck: browserCheck(props),
    browserUncheck: browserUncheck(props),
    browserHover: browserHover(props),
    browserWait: browserWait(props),
    browserBack: browserBack(props),
    browserForward: browserForward(props),
    browserReload: browserReload(props),
    browserGetText: browserGetText(props),
    browserGetUrl: browserGetUrl(props),
    browserGetTitle: browserGetTitle(props),
    browserScreenshot: browserScreenshot(props),
  };
}
