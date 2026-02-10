import { BrowserManager } from 'agent-browser/dist/browser.js';
import { executeCommand as abExecuteCommand } from 'agent-browser/dist/actions.js';
import type { Command, Response } from 'agent-browser/dist/types.js';
import { randomUUID } from 'crypto';
import { getCdpUrl } from './browser';

// =============================================================================
// BrowserManager pool — one instance per (userId, sessionId) pair
// =============================================================================

interface PoolEntry {
  browser: BrowserManager;
  cdpUrl: string;
}

const pool = new Map<string, PoolEntry>();

function poolKey(userId: string, sessionId: string): string {
  return `${userId}:${sessionId}`;
}

/**
 * Get or create a BrowserManager connected via CDP for the given session.
 *
 * If the CDP URL has changed (browser was recreated), the old connection is
 * closed and a new one established.
 */
export async function getOrCreateBrowserClient(
  sessionId: string,
  userId: string,
): Promise<BrowserManager> {
  const key = poolKey(userId, sessionId);
  const cdpUrl = await getCdpUrl(sessionId, userId);

  if (!cdpUrl) {
    throw new Error(
      `[AgentBrowserClient] No CDP URL for session ${sessionId}`,
    );
  }

  const existing = pool.get(key);

  // If we already have a connection to the same CDP URL, reuse it
  if (existing && existing.cdpUrl === cdpUrl && existing.browser.isLaunched()) {
    return existing.browser;
  }

  // CDP URL changed or browser disconnected — close old and reconnect
  if (existing) {
    console.log(
      `[AgentBrowserClient] CDP URL changed for ${sessionId}, reconnecting`,
    );
    try {
      await existing.browser.close();
    } catch {
      // Ignore close errors on stale connections
    }
    pool.delete(key);
  }

  const browser = new BrowserManager();
  await browser.launch({ action: 'launch', id: randomUUID(), cdpUrl });
  pool.set(key, { browser, cdpUrl });

  console.log(`[AgentBrowserClient] Connected to ${sessionId} via CDP`);
  return browser;
}

/**
 * Close and remove a BrowserManager from the pool.
 * Called when a browser session is deleted.
 */
export async function closeBrowserClient(
  sessionId: string,
  userId: string,
): Promise<void> {
  const key = poolKey(userId, sessionId);
  const entry = pool.get(key);
  if (!entry) return;

  pool.delete(key);
  try {
    await entry.browser.close();
    console.log(`[AgentBrowserClient] Disconnected from ${sessionId}`);
  } catch {
    // Ignore — browser may already be gone
  }
}

// =============================================================================
// CLI string → Command object parser
//
// Converts the CLI-style strings that the AI agent sends (e.g. "open https://example.com",
// "click @e1", "fill @e3 \"text\"") into the JSON Command objects that
// executeCommand() expects.
// =============================================================================

/**
 * Execute a CLI-style command string against a BrowserManager.
 *
 * Handles parsing the CLI string, dispatching to executeCommand(), and
 * formatting the response as a string suitable for the AI agent.
 */
export async function executeBrowserCommand(
  browser: BrowserManager,
  commandStr: string,
): Promise<{ output: string; success: boolean }> {
  const command = parseCliCommand(commandStr);
  const response: Response = await abExecuteCommand(command, browser);

  if (response.success) {
    const data = response.data;
    if (data === null || data === undefined) {
      return { output: 'Command completed successfully', success: true };
    }
    if (typeof data === 'string') {
      return { output: data, success: true };
    }
    return { output: JSON.stringify(data, null, 2), success: true };
  }

  return { output: response.error, success: false };
}

/**
 * Parse a CLI-style command string into a Command object.
 *
 * Maps the CLI syntax used by the AI agent to the JSON Command interface
 * expected by executeCommand().
 */
export function parseCliCommand(commandStr: string): Command {
  const tokens = tokenize(commandStr);
  if (tokens.length === 0) {
    throw new Error('Empty command');
  }

  const id = randomUUID();
  const verb = tokens[0].toLowerCase();

  switch (verb) {
    // Navigation
    case 'open':
    case 'navigate':
      return { action: 'navigate', url: tokens[1] || '', id } as Command;

    // Clicks
    case 'click':
      return { action: 'click', selector: tokens[1] || '', id } as Command;
    case 'dblclick':
    case 'doubleclick':
      return { action: 'dblclick', selector: tokens[1] || '', id } as Command;

    // Input
    case 'fill':
      return {
        action: 'fill',
        selector: tokens[1] || '',
        value: tokens[2] || '',
        id,
      } as Command;
    case 'type':
      return {
        action: 'type',
        selector: tokens[1] || '',
        text: tokens[2] || '',
        id,
      } as Command;

    // Checkbox
    case 'check':
      return { action: 'check', selector: tokens[1] || '', id } as Command;
    case 'uncheck':
      return { action: 'uncheck', selector: tokens[1] || '', id } as Command;

    // Select
    case 'select':
      return {
        action: 'select',
        selector: tokens[1] || '',
        values: tokens[2] || '',
        id,
      } as Command;

    // Keyboard
    case 'press':
      return { action: 'press', key: tokens[1] || '', id } as Command;

    // Hover
    case 'hover':
      return { action: 'hover', selector: tokens[1] || '', id } as Command;

    // Scroll
    case 'scroll':
      return parseScrollCommand(tokens, id);
    case 'scrollintoview':
      return {
        action: 'scrollintoview',
        selector: tokens[1] || '',
        id,
      } as Command;

    // Snapshot — supports flags: -i (interactive), -s "selector", -c (compact), -d N (maxDepth)
    case 'snapshot':
      return parseSnapshotCommand(tokens, id);

    // Screenshot
    case 'screenshot':
      return parseScreenshotCommand(tokens, id);

    // Evaluate
    case 'eval':
    case 'evaluate':
      return {
        action: 'evaluate',
        script: tokens.slice(1).join(' '),
        id,
      } as Command;

    // Navigation actions
    case 'back':
      return { action: 'back', id } as Command;
    case 'forward':
      return { action: 'forward', id } as Command;
    case 'reload':
      return { action: 'reload', id } as Command;

    // Wait — supports multiple forms
    case 'wait':
      return parseWaitCommand(tokens, id);

    // Get — get text/url/title/value
    case 'get':
      return parseGetCommand(tokens, id);

    // Find — find label/role/text/placeholder
    case 'find':
      return parseFindCommand(tokens, id);

    // Focus
    case 'focus':
      return { action: 'focus', selector: tokens[1] || '', id } as Command;

    // Upload
    case 'upload':
      return {
        action: 'upload',
        selector: tokens[1] || '',
        files: tokens[2] || '',
        id,
      } as Command;

    // Content
    case 'content':
      return { action: 'content', id } as Command;

    default:
      // Try treating the whole thing as a single-word action
      return { action: verb, id } as Command;
  }
}

// =============================================================================
// Sub-parsers for complex commands
// =============================================================================

function parseSnapshotCommand(tokens: string[], id: string): Command {
  const opts: Record<string, unknown> = {
    action: 'snapshot',
    id,
  };

  for (let i = 1; i < tokens.length; i++) {
    const t = tokens[i];
    if (t === '-i') {
      opts.interactive = true;
    } else if (t === '-c') {
      opts.compact = true;
    } else if (t === '-s' && i + 1 < tokens.length) {
      opts.selector = tokens[++i];
    } else if (t === '-d' && i + 1 < tokens.length) {
      opts.maxDepth = parseInt(tokens[++i], 10);
    }
  }

  return opts as unknown as Command;
}

function parseScreenshotCommand(tokens: string[], id: string): Command {
  const opts: Record<string, unknown> = {
    action: 'screenshot',
    id,
  };

  for (let i = 1; i < tokens.length; i++) {
    const t = tokens[i];
    if (t === '--full') {
      opts.fullPage = true;
    } else if (t === '--selector' && i + 1 < tokens.length) {
      opts.selector = tokens[++i];
    } else if (t === '--format' && i + 1 < tokens.length) {
      opts.format = tokens[++i];
    } else if (!t.startsWith('-')) {
      // Positional arg = path
      opts.path = t;
    }
  }

  return opts as unknown as Command;
}

function parseScrollCommand(tokens: string[], id: string): Command {
  // "scroll down 500", "scroll up", "scroll @e1 down 300"
  const direction = tokens.find((t) =>
    ['up', 'down', 'left', 'right'].includes(t.toLowerCase()),
  );
  const amount = tokens.find((t) => /^\d+$/.test(t));
  const selector = tokens.find(
    (t) =>
      t.startsWith('@') ||
      (t !== 'scroll' &&
        !['up', 'down', 'left', 'right'].includes(t.toLowerCase()) &&
        !/^\d+$/.test(t)),
  );

  return {
    action: 'scroll',
    id,
    ...(direction && { direction: direction.toLowerCase() }),
    ...(amount && { amount: parseInt(amount, 10) }),
    ...(selector && { selector }),
  } as unknown as Command;
}

function parseWaitCommand(tokens: string[], id: string): Command {
  // "wait 2000" — timeout
  // "wait --load networkidle" — waitforloadstate
  // "wait --text "Welcome"" — wait for text
  // "wait --url "https://..."" — waitforurl
  // "wait @e1" — wait for selector

  for (let i = 1; i < tokens.length; i++) {
    if (tokens[i] === '--load' && i + 1 < tokens.length) {
      return {
        action: 'waitforloadstate',
        state: tokens[i + 1],
        id,
      } as unknown as Command;
    }
    if (tokens[i] === '--text' && i + 1 < tokens.length) {
      return {
        action: 'wait',
        selector: `text=${tokens[i + 1]}`,
        id,
      } as Command;
    }
    if (tokens[i] === '--url' && i + 1 < tokens.length) {
      return {
        action: 'waitforurl',
        url: tokens[i + 1],
        id,
      } as unknown as Command;
    }
  }

  // "wait 2000" — just a timeout
  if (tokens.length === 2 && /^\d+$/.test(tokens[1])) {
    return {
      action: 'wait',
      timeout: parseInt(tokens[1], 10),
      id,
    } as Command;
  }

  // "wait @e1" — wait for selector
  if (tokens.length >= 2) {
    return { action: 'wait', selector: tokens[1], id } as Command;
  }

  return { action: 'wait', id } as Command;
}

function parseGetCommand(tokens: string[], id: string): Command {
  // "get text @e1", "get url", "get title", "get value @e1"
  const sub = tokens[1]?.toLowerCase();
  switch (sub) {
    case 'text':
      return {
        action: 'gettext',
        selector: tokens[2] || '',
        id,
      } as Command;
    case 'url':
      return { action: 'url', id } as Command;
    case 'title':
      return { action: 'title', id } as Command;
    case 'value':
      return {
        action: 'inputvalue',
        selector: tokens[2] || '',
        id,
      } as Command;
    case 'attribute':
      return {
        action: 'getattribute',
        selector: tokens[2] || '',
        attribute: tokens[3] || '',
        id,
      } as Command;
    default:
      return { action: 'gettext', selector: tokens[1] || '', id } as Command;
  }
}

function parseFindCommand(tokens: string[], id: string): Command {
  // "find label "Email" fill "value""
  // "find role button click --name "Submit""
  // "find text "Sign In" click"
  // "find placeholder "Search" fill "query""
  const strategy = tokens[1]?.toLowerCase();

  switch (strategy) {
    case 'label': {
      const label = tokens[2] || '';
      const subaction = tokens[3]?.toLowerCase() || 'click';
      const value = tokens[4];
      return {
        action: 'getbylabel',
        label,
        subaction,
        ...(value !== undefined && { value }),
        id,
      } as unknown as Command;
    }
    case 'role': {
      const role = tokens[2] || '';
      // Find subaction and --name flag
      let subaction = 'click';
      let name: string | undefined;
      for (let i = 3; i < tokens.length; i++) {
        if (tokens[i] === '--name' && i + 1 < tokens.length) {
          name = tokens[++i];
        } else if (
          ['click', 'fill', 'check', 'hover'].includes(
            tokens[i].toLowerCase(),
          )
        ) {
          subaction = tokens[i].toLowerCase();
        }
      }
      return {
        action: 'getbyrole',
        role,
        subaction,
        ...(name !== undefined && { name }),
        id,
      } as unknown as Command;
    }
    case 'text': {
      const text = tokens[2] || '';
      const subaction = tokens[3]?.toLowerCase() || 'click';
      return {
        action: 'getbytext',
        text,
        subaction,
        id,
      } as unknown as Command;
    }
    case 'placeholder': {
      const placeholder = tokens[2] || '';
      const subaction = tokens[3]?.toLowerCase() || 'click';
      const value = tokens[4];
      return {
        action: 'getbyplaceholder',
        placeholder,
        subaction,
        ...(value !== undefined && { value }),
        id,
      } as unknown as Command;
    }
    default:
      throw new Error(`Unknown find strategy: ${strategy}`);
  }
}

// =============================================================================
// Tokenizer
// =============================================================================

function tokenize(command: string): string[] {
  const args: string[] = [];
  let current = '';
  let inQuote: string | null = null;

  for (let i = 0; i < command.length; i++) {
    const ch = command[i];

    if (inQuote) {
      if (ch === '\\' && i + 1 < command.length) {
        current += command[++i];
      } else if (ch === inQuote) {
        inQuote = null;
      } else {
        current += ch;
      }
    } else if (ch === '"' || ch === "'") {
      inQuote = ch;
    } else if (ch === ' ' || ch === '\t') {
      if (current) {
        args.push(current);
        current = '';
      }
    } else {
      current += ch;
    }
  }
  if (current) args.push(current);
  return args;
}
