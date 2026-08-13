/**
 * agent-browser CLI transport.
 *
 * agent-browser is a native binary, not a library — it is driven as a
 * subprocess and answers with a JSON envelope on stdout. This module owns that
 * boundary: argv construction (pure, unit-tested) and process execution.
 *
 * Commands are passed through as the CLI's own argv, so the model, the prompts
 * and this code all speak one vocabulary. There is deliberately no action-name
 * translation layer here: a mapping table would silently rot every time
 * agent-browser adds or renames a command.
 *
 * @see https://agent-browser.dev
 */

import { execFile } from 'node:child_process';
import {
  startCommandTelemetry,
  type TimelineCollector,
} from '@/lib/observability/browser-telemetry';

/** The JSON envelope every `--json` invocation prints on stdout. */
export interface CliResponse {
  success: boolean;
  data?: unknown;
  error?: string | null;
}

export interface CliOptions {
  /** Kernel CDP endpoint to drive. Omit to use the configured provider. */
  cdpUrl?: string;
  /** Isolates daemon state; one per browser session. */
  session: string;
  timeoutMs?: number;
  signal?: AbortSignal;
  /**
   * When set, the browser-internal events that occurred during this command
   * (from Kernel Browser Telemetry) are attached to the trace as a child span.
   */
  collectTimeline?: TimelineCollector;
}

/** Resolved from PATH — the Docker image symlinks the native binary there. */
const CLI_BIN = process.env.AGENT_BROWSER_BIN ?? 'agent-browser';

export const DEFAULT_TIMEOUT_MS = 120_000;

/**
 * Build the argv for one CLI invocation.
 *
 * `connect <cdpUrl>` is expressed as the global `--cdp` flag so a single
 * invocation both attaches to the Kernel browser and runs the command; the
 * daemon reuses that connection on later calls. We attach by CDP URL rather
 * than `-p kernel` because this app creates and owns its Kernel browsers
 * through the SDK (profiles, replay recording, standby), and letting the CLI
 * create its own would strand those.
 *
 * Exported for tests: argv is the whole contract with the binary.
 */
export function buildArgs(
  command: readonly string[],
  options: Pick<CliOptions, 'cdpUrl' | 'session'>,
): string[] {
  if (command.length === 0) {
    throw new Error('[agent-browser] command must not be empty');
  }
  return [
    '--session',
    options.session,
    ...(options.cdpUrl ? ['--cdp', options.cdpUrl] : []),
    '--json',
    ...command,
  ];
}

/**
 * Environment for one invocation.
 *
 * `AGENT_BROWSER_PROVIDER` is stripped whenever we attach by CDP: the CLI
 * rejects `--cdp` combined with a provider ("Cannot use --cdp and -p/--provider
 * together"), and an inherited value would break every browser command at
 * runtime with no local reproduction. Kernel is reached through the CDP URL of
 * a browser this app already created, so a provider is never wanted here.
 *
 * Exported for tests.
 */
export function cliEnv(
  options: Pick<CliOptions, 'cdpUrl'>,
  base: NodeJS.ProcessEnv = process.env,
): NodeJS.ProcessEnv {
  if (!options.cdpUrl) return base;
  const { AGENT_BROWSER_PROVIDER: _dropped, ...rest } = base;
  return rest;
}

/**
 * Parse the CLI's stdout into an envelope.
 *
 * A non-zero exit is still a structured failure when stdout carries JSON (the
 * binary reports e.g. "element not found" that way), so the caller gets the
 * real message instead of "exited with code 1". Non-JSON output means the
 * binary itself failed — surface stderr, which is where that lands.
 *
 * Exported for tests.
 */
export function parseResponse(stdout: string, stderr: string): CliResponse {
  const text = stdout.trim();
  if (text) {
    try {
      const parsed = JSON.parse(text) as CliResponse;
      // `batch` answers with an array; keep it whole under `data`.
      if (Array.isArray(parsed)) return { success: true, data: parsed };
      if (
        typeof parsed === 'object' &&
        parsed !== null &&
        'success' in parsed
      ) {
        return parsed;
      }
      return { success: true, data: parsed };
    } catch {
      // Fall through — not JSON, so treat stderr as the failure.
    }
  }
  return {
    success: false,
    data: null,
    error: stderr.trim() || text || 'agent-browser produced no output',
  };
}

/**
 * Run one agent-browser command against a Kernel browser.
 *
 * Never rejects on a browser-level failure — those come back as
 * `{ success: false, error }` so tool call sites can hand the message to the
 * model. Rejects only if the process cannot be run at all.
 */
export async function runCommand(
  command: readonly string[],
  options: CliOptions,
): Promise<CliResponse> {
  const args = buildArgs(command, options);
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const env = cliEnv(options);

  const telemetry = startCommandTelemetry(command, {
    session: options.session,
    remote: Boolean(options.cdpUrl),
    timeoutMs,
    collectTimeline: options.collectTimeline,
  });

  return new Promise<CliResponse>((resolve, reject) => {
    execFile(
      CLI_BIN,
      args,
      {
        timeout: timeoutMs,
        signal: options.signal,
        maxBuffer: 32 * 1024 * 1024, // snapshots of large pages
        encoding: 'utf8',
        env,
      },
      (error, stdout, stderr) => {
        if (error && (error as NodeJS.ErrnoException).code === 'ENOENT') {
          const message = `[agent-browser] binary not found (${CLI_BIN}). It ships with the npm package; the Docker image symlinks it onto PATH.`;
          telemetry.end({ outcome: 'spawn_error', error: message });
          reject(new Error(message));
          return;
        }
        // Timeout/abort kill the process, leaving no usable stdout.
        if (error && 'killed' in error && error.killed) {
          const aborted = Boolean(options.signal?.aborted);
          const message = aborted
            ? 'Browser command stopped by user'
            : `Command timed out after ${timeoutMs}ms`;
          telemetry.end({
            outcome: aborted ? 'aborted' : 'timeout',
            error: message,
          });
          resolve({ success: false, data: null, error: message });
          return;
        }

        const response = parseResponse(stdout, stderr);
        telemetry.end({
          outcome: response.success ? 'success' : 'command_error',
          error: response.error,
        });
        resolve(response);
      },
    );
  });
}
