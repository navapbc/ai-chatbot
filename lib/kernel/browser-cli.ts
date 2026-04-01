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
// Params → CLI args
// =============================================================================

/**
 * Convert tool params to CLI args.
 *
 * Most commands follow: `<action> [selector] [value]`.
 * Compound actions like "get text" or "tab new" are split on spaces.
 * Snapshot and wait have special flag syntax.
 * Everything else is a direct passthrough — no command-specific logic.
 *
 * Returns an array of arg arrays. Usually one, but `type` with `clear: true`
 * produces two: fill "" to clear, then type to enter text char-by-char.
 */
function toArgs(params: Record<string, unknown>): string[][] {
  const action = String(params.action ?? '');
  const args = action.split(' ');

  // --- type with clear: daemon supports clear but CLI doesn't expose it.
  // Clear with fill "" (programmatic — fine for empty field), then type
  // char-by-char to trigger JS input masks.
  if (action === 'type' && params.clear && params.selector) {
    return [
      ['fill', String(params.selector), ''],
      ['type', String(params.selector), String(params.text ?? '')],
    ];
  }

  // --- snapshot: uses short flags ---
  if (action === 'snapshot') {
    if (params.interactive) args.push('-i');
    if (params.selector) args.push('-s', String(params.selector));
    return [args];
  }

  // --- wait: polymorphic (selector, ms, or --flag) ---
  if (action === 'wait') {
    if (params.selector) return [[...args, String(params.selector)]];
    if (params.timeout !== undefined) return [[...args, String(params.timeout)]];
    if (params.text) return [[...args, '--text', String(params.text)]];
    if (params.url) return [[...args, '--url', String(params.url)]];
    if (params.state) return [[...args, '--load', String(params.state)]];
    if (params.fn) return [[...args, '--fn', String(params.fn)]];
    return [args];
  }

  // --- scroll: direction before selector ---
  if (action === 'scroll') {
    if (params.direction) args.push(String(params.direction));
    if (params.amount !== undefined) args.push(String(params.amount));
    return [args];
  }

  // --- Everything else: selector, then value-like positional, then extras ---
  if (params.selector) args.push(String(params.selector));

  // First value positional (only one applies per command)
  const val = params.value ?? params.text ?? params.url ?? params.key
    ?? params.script ?? params.label ?? params.response;
  if (val !== undefined) args.push(String(val));

  // Extras: subaction, array values, index, promptText, --name flag
  if (params.subaction) args.push(String(params.subaction));
  if (params.subaction && params.value) args.push(String(params.value));
  if (Array.isArray(params.values)) params.values.forEach(v => args.push(String(v)));
  if (params.index !== undefined) args.push(String(params.index));
  if (params.promptText) args.push(String(params.promptText));
  if (params.name) args.push('--name', String(params.name));

  return [args];
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
