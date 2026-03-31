import { execFile } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { platform, arch } from 'node:os';

/**
 * Shorten a session key so the daemon's Unix socket path stays under 103 bytes.
 * Uses a 16-char hex hash — unique enough for concurrent sessions.
 */
function shortSessionKey(key: string): string {
  if (key.length <= 32) return key;
  return createHash('sha256').update(key).digest('hex').slice(0, 16);
}

// =============================================================================
// Binary resolution
// =============================================================================

function getBinaryPath(): string {
  const os = platform();
  const cpu = arch();

  const osKey = os === 'win32' ? 'win32' : os === 'darwin' ? 'darwin' : 'linux';
  const archKey = cpu === 'arm64' || cpu === 'aarch64' ? 'arm64' : 'x64';
  const ext = os === 'win32' ? '.exe' : '';
  const binaryName = `agent-browser-${osKey}-${archKey}${ext}`;

  // Search for the native binary in node_modules.
  // We can't use require.resolve() because Next.js serverExternalPackages
  // rewrites it to a virtual path. Instead, walk up from cwd looking for
  // the pnpm store or a direct node_modules install.
  const searchRoots = [
    process.cwd(),                          // e.g. /app/client
    join(process.cwd(), '..'),              // parent dir (monorepo)
  ];

  for (const root of searchRoots) {
    // Direct install (npm/yarn)
    const directPath = join(root, 'node_modules', 'agent-browser', 'bin', binaryName);
    if (existsSync(directPath)) return directPath;

    // pnpm store — glob for the versioned directory
    const pnpmBase = join(root, 'node_modules', '.pnpm');
    if (existsSync(pnpmBase)) {
      try {
        const entries = readdirSync(pnpmBase).filter(e => e.startsWith('agent-browser@'));
        for (const entry of entries) {
          const pnpmPath = join(pnpmBase, entry, 'node_modules', 'agent-browser', 'bin', binaryName);
          if (existsSync(pnpmPath)) return pnpmPath;
        }
      } catch {
        // readdirSync may fail on permissions — continue searching
      }
    }
  }

  throw new Error(
    `agent-browser binary "${binaryName}" not found in node_modules. Run "pnpm install" to ensure the package is installed.`,
  );
}

let cachedBinaryPath: string | null = null;

function resolveBinary(): string {
  if (!cachedBinaryPath) {
    cachedBinaryPath = getBinaryPath();
  }
  return cachedBinaryPath;
}

// =============================================================================
// Params → CLI args (passthrough with minimal structure mapping)
// =============================================================================

/**
 * Convert the flat tool input object into one or more CLI arg arrays.
 *
 * The tool schema uses the same command names as the agent-browser CLI.
 * This function just arranges the params into positional args and flags
 * that the CLI expects.
 *
 * Returns an array of arg arrays — usually one, but `type` with `clear: true`
 * produces two (fill "" first, then type).
 */
function toArgs(params: Record<string, unknown>): string[][] {
  const action = String(params.action ?? '');
  const selector = params.selector as string | undefined;
  const value = params.value as string | undefined;
  const text = params.text as string | undefined;

  switch (action) {
    // --- Navigation ---
    case 'open':
      return [['open', params.url as string ?? '']];

    case 'back':
    case 'forward':
    case 'reload':
    case 'close':
      return [[action]];

    // --- Snapshot ---
    case 'snapshot': {
      const args = ['snapshot'];
      if (params.interactive) args.push('-i');
      if (selector) args.push('-s', selector);
      return [args];
    }

    // --- Interactions ---
    case 'click':
      return [['click', selector ?? '']];

    case 'dblclick':
      return [['dblclick', selector ?? '']];

    case 'fill':
      return [['fill', selector ?? '', value ?? '']];

    case 'type': {
      if (params.clear && selector) {
        // CLI `type` has no --clear flag. Select all + delete before typing.
        // Can't use `fill ""` because it bypasses JS input masks.
        return [
          ['click', selector],
          ['press', 'Control+a'],
          ['press', 'Backspace'],
          ['type', selector, text ?? ''],
        ];
      }
      return [['type', selector ?? '', text ?? '']];
    }

    case 'press':
      return [['press', params.key as string ?? '']];

    case 'select':
      return [['select', selector ?? '', ...((params.values as string[]) ?? [])]];

    case 'hover':
      return [['hover', selector ?? '']];

    case 'focus':
      return [['focus', selector ?? '']];

    case 'check':
      return [['check', selector ?? '']];

    case 'uncheck':
      return [['uncheck', selector ?? '']];

    case 'scroll':
      return [['scroll', (params.direction as string) ?? 'down', String(params.amount ?? 300)]];

    case 'scrollintoview':
      return [['scrollintoview', selector ?? '']];

    case 'drag':
      return [['drag', selector ?? '', (params.target as string) ?? '']];

    case 'upload':
      return [['upload', selector ?? '', ...((params.files as string[]) ?? [])]];

    // --- Wait ---
    case 'wait':
      if (selector) return [['wait', selector]];
      if (params.timeout !== undefined) return [['wait', String(params.timeout)]];
      if (params.text) return [['wait', '--text', params.text as string]];
      if (params.url) return [['wait', '--url', params.url as string]];
      if (params.state) return [['wait', '--load', params.state as string]];
      if (params.fn) return [['wait', '--fn', params.fn as string]];
      return [['wait', '1000']];

    // --- Get info ---
    case 'get text':
      return [['get', 'text', selector ?? '']];

    case 'get value':
      return [['get', 'value', selector ?? '']];

    case 'get url':
      return [['get', 'url']];

    case 'get title':
      return [['get', 'title']];

    case 'get html':
      return [['get', 'html', selector ?? '']];

    case 'get attr':
      return [['get', 'attr', selector ?? '', (params.attr as string) ?? '']];

    case 'get count':
      return [['get', 'count', selector ?? '']];

    case 'get box':
      return [['get', 'box', selector ?? '']];

    case 'get styles':
      return [['get', 'styles', selector ?? '']];

    // --- Screenshot / eval ---
    case 'screenshot':
      return [['screenshot']];

    case 'eval':
      return [['eval', (params.script as string) ?? '']];

    // --- Tabs ---
    case 'tab':
      return [['tab']];

    case 'tab new':
      return [['tab', 'new']];

    case 'tab close':
      return [['tab', 'close']];

    case 'tab switch':
      return [['tab', String(params.index ?? 0)]];

    // --- Dialog ---
    case 'dialog':
      if (params.promptText) return [['dialog', (params.response as string) ?? 'accept', params.promptText as string]];
      return [['dialog', (params.response as string) ?? 'accept']];

    // --- Frames ---
    case 'frame':
      return [['frame', selector ?? '']];

    case 'frame main':
      return [['frame', 'main']];

    // --- Find (semantic locators) ---
    case 'find label': {
      const args = ['find', 'label', (params.label as string) ?? ''];
      if (params.subaction) args.push(params.subaction as string);
      if (value) args.push(value);
      return [args];
    }

    case 'find role': {
      const args = ['find', 'role', (params.role as string) ?? ''];
      if (params.subaction) args.push(params.subaction as string);
      if (params.name) args.push('--name', params.name as string);
      return [args];
    }

    case 'find text':
      return [['find', 'text', text ?? '', (params.subaction as string) ?? 'click']];

    // --- Default passthrough ---
    default: {
      // Unknown action: pass through as-is for forward compatibility.
      // Split compound commands like "get text" into ["get", "text"].
      const parts = action.split(' ');
      if (selector) parts.push(selector);
      if (value) parts.push(value);
      return [parts];
    }
  }
}

// =============================================================================
// CLI execution
// =============================================================================

interface CLIResult {
  success: boolean;
  data?: string | Record<string, unknown>;
  error?: string;
}

/**
 * Spawn the agent-browser CLI and return parsed JSON output.
 */
function spawnCLI(
  args: string[],
  abortSignal?: AbortSignal,
): Promise<CLIResult> {
  return new Promise((resolve, reject) => {
    const binary = resolveBinary();

    // Use node for .js wrapper, direct exec for native binary
    const isJsWrapper = binary.endsWith('.js');
    const cmd = isJsWrapper ? process.execPath : binary;
    const cmdArgs = isJsWrapper ? [binary, ...args] : args;

    const child = execFile(
      cmd,
      cmdArgs,
      {
        encoding: 'utf-8',
        maxBuffer: 10 * 1024 * 1024, // 10MB for large snapshots
        timeout: 0, // Handled by caller's Promise.race
        env: {
          ...process.env,
          AGENT_BROWSER_IDLE_TIMEOUT_MS: process.env.AGENT_BROWSER_IDLE_TIMEOUT_MS ?? '300000',
        },
      },
      (error, stdout, stderr) => {
        if (abortSignal?.aborted) {
          reject(new Error('Browser command stopped by user'));
          return;
        }

        if (stderr) {
          console.error('[browser-cli] stderr:', stderr.slice(0, 500));
        }

        if (error) {
          // Try to parse JSON error from stdout
          if (stdout) {
            try {
              const parsed = JSON.parse(stdout);
              resolve({
                success: false,
                error: parsed.error ?? parsed.message ?? error.message,
              });
              return;
            } catch {
              // Fall through
            }
          }
          resolve({ success: false, error: error.message });
          return;
        }

        try {
          const parsed = JSON.parse(stdout);
          resolve({
            success: parsed.success ?? true,
            data: parsed.data ?? parsed.output ?? stdout,
            error: parsed.error,
          });
        } catch {
          // Non-JSON output — return as-is
          resolve({ success: true, data: stdout.trim() });
        }
      },
    );

    if (abortSignal) {
      const onAbort = () => child.kill('SIGTERM');
      abortSignal.addEventListener('abort', onAbort, { once: true });
      child.on('exit', () => abortSignal.removeEventListener('abort', onAbort));
    }
  });
}

// =============================================================================
// Public API
// =============================================================================

/**
 * Establish the agent-browser daemon connection to a CDP endpoint.
 * Call once per session before executing commands.
 */
export async function connectSession(
  sessionKey: string,
  cdpWsUrl: string,
): Promise<void> {
  const shortKey = shortSessionKey(sessionKey);
  console.log(`[browser-cli] Connecting session "${shortKey}" to CDP`);
  const result = await spawnCLI([
    '--session', shortKey,
    '--cdp', cdpWsUrl,
    'get', 'url', '--json',
  ]);

  if (!result.success) {
    throw new Error(`Failed to connect agent-browser to CDP: ${result.error}`);
  }

  console.log(`[browser-cli] Session "${shortKey}" connected`);
}

/**
 * Execute a browser command via the agent-browser CLI.
 *
 * Params use the same action names as the CLI (open, snapshot, click, fill, etc.).
 * The function arranges them into positional args and spawns the binary.
 */
export async function executeCLICommand(
  sessionKey: string,
  cdpWsUrl: string,
  params: Record<string, unknown>,
  abortSignal?: AbortSignal,
): Promise<{ success: boolean; output: string | null; error: string | null }> {
  const argSets = toArgs(params);

  let lastResult: CLIResult = { success: true };

  for (const args of argSets) {
    console.log('[browser-cli] Executing:', args.join(' '));

    lastResult = await spawnCLI(
      ['--session', shortSessionKey(sessionKey), '--cdp', cdpWsUrl, ...args, '--json'],
      abortSignal,
    );

    if (!lastResult.success) break;
  }

  const output = lastResult.data != null
    ? (typeof lastResult.data === 'string' ? lastResult.data : JSON.stringify(lastResult.data))
    : null;

  return {
    success: lastResult.success ?? false,
    output,
    error: lastResult.error ?? null,
  };
}

/**
 * Close the agent-browser session (disconnect from CDP, stop daemon for this session).
 */
export async function closeSession(sessionKey: string, cdpWsUrl: string): Promise<void> {
  try {
    const shortKey = shortSessionKey(sessionKey);
    await spawnCLI(['--session', shortKey, '--cdp', cdpWsUrl, 'close', '--json']);
    console.log(`[browser-cli] Session "${shortKey}" closed`);
  } catch (err) {
    console.error(`[browser-cli] Error closing session "${sessionKey}":`, err);
  }
}
