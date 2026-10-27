import { normalizeOrigin } from '../lib/relay/policy';

// Pure helpers for scripts/build-extension.ts, kept apart so tests can import
// them without running a build.

export class ExtensionBuildError extends Error {}

const LOOPBACK_HOSTS = new Set(['127.0.0.1', 'localhost']);

/** An http origin on a loopback host, written exactly as its origin. */
function isLoopbackHttpOrigin(value: string): boolean {
  try {
    const parsed = new URL(value);
    return (
      parsed.protocol === 'http:' &&
      LOOPBACK_HOSTS.has(parsed.hostname) &&
      parsed.origin === value
    );
  } catch {
    return false;
  }
}

/**
 * The app origin the extension may call, from `APP_ORIGIN`, or '' when unset
 * outside a release build (the manifest then has no host_permissions).
 *
 * The value must be an exact origin: no path, query, fragment, userinfo,
 * default port or trailing slash. A release build (`EXTENSION_RELEASE=1`)
 * requires an https origin; other builds also accept a loopback http origin
 * so tests can point the extension at a local server.
 */
export function resolveAppOrigin(env: Record<string, string | undefined>) {
  const release = env.EXTENSION_RELEASE === '1';
  const value = env.APP_ORIGIN ?? '';
  if (value === '') {
    if (release) {
      throw new ExtensionBuildError(
        'APP_ORIGIN is required when EXTENSION_RELEASE=1 (an exact https origin, e.g. https://app.example.org)',
      );
    }
    return '';
  }
  if (normalizeOrigin(value) === value) return value;
  if (release) {
    throw new ExtensionBuildError(
      `APP_ORIGIN must be an exact https origin when EXTENSION_RELEASE=1 (no path, query, fragment, credentials, default port or trailing slash); got ${JSON.stringify(value)}`,
    );
  }
  if (isLoopbackHttpOrigin(value)) return value;
  throw new ExtensionBuildError(
    `APP_ORIGIN must be an exact https origin, or http://127.0.0.1:<port> / http://localhost:<port> outside a release build (no path, query, fragment, credentials, default port or trailing slash); got ${JSON.stringify(value)}`,
  );
}

/** The built manifest: the source manifest plus host_permissions for the app. */
export function buildManifest(
  source: Record<string, unknown>,
  appOrigin: string,
): Record<string, unknown> {
  if (appOrigin === '') return { ...source };
  return { ...source, host_permissions: [`${appOrigin}/*`] };
}

/**
 * Whether the build installs the bridge's test hooks (the e2e suites drive
 * the worker through them): only with `EXTENSION_TEST_HOOKS=1` or the
 * `--test-hooks` flag, and never in a release build.
 */
export function resolveTestHooks(
  env: Record<string, string | undefined>,
  argv: readonly string[],
): boolean {
  const hooks =
    env.EXTENSION_TEST_HOOKS === '1' || argv.includes('--test-hooks');
  if (hooks && env.EXTENSION_RELEASE === '1') {
    throw new ExtensionBuildError(
      'test hooks (EXTENSION_TEST_HOOKS=1 / --test-hooks) are not allowed when EXTENSION_RELEASE=1',
    );
  }
  return hooks;
}

/** Longest test hold accepted: the production hold is 60 s. */
const MAX_TEST_ORIGIN_HOLD_MS = 600_000;

/**
 * A shortened origin hold (ms) for a test-hooks build, from
 * `EXTENSION_TEST_ORIGIN_HOLD_MS`, so the e2e suites see a run that left its
 * origin end in seconds rather than after the bridge's 60 s (the total cap
 * is twice it). Undefined when unset: the bridge keeps its 60 s. Refused in a
 * build without the test hooks (and so in a release build, which refuses the
 * hooks): it is a build-time constant only, never a runtime setting.
 */
export function resolveOriginHoldMs(
  env: Record<string, string | undefined>,
  testHooks: boolean,
): number | undefined {
  const value = env.EXTENSION_TEST_ORIGIN_HOLD_MS ?? '';
  if (value === '') return undefined;
  if (!testHooks) {
    throw new ExtensionBuildError(
      'EXTENSION_TEST_ORIGIN_HOLD_MS is allowed only with the test hooks (EXTENSION_TEST_HOOKS=1 / --test-hooks)',
    );
  }
  const ms = /^[1-9]\d*$/.test(value) ? Number(value) : Number.NaN;
  if (!Number.isSafeInteger(ms) || ms > MAX_TEST_ORIGIN_HOLD_MS) {
    throw new ExtensionBuildError(
      `EXTENSION_TEST_ORIGIN_HOLD_MS must be a whole number of ms from 1 to ${MAX_TEST_ORIGIN_HOLD_MS}; got ${JSON.stringify(value)}`,
    );
  }
  return ms;
}

/**
 * esbuild argv bundling `entry` into `outFile` with APP_ORIGIN and
 * EXTENSION_TEST_HOOKS defined. The hooks flag is always defined (false by
 * default) so a plain build drops the hook code instead of testing for it
 * at run time. `originHoldMs` (see resolveOriginHoldMs) is defined as
 * `__ORIGIN_HOLD_MS__` only in a test-hooks build.
 */
export function esbuildArgs(
  entry: string,
  outFile: string,
  appOrigin: string,
  testHooks = false,
  originHoldMs?: number,
): string[] {
  const args = [
    entry,
    '--bundle',
    '--format=esm',
    `--outfile=${outFile}`,
    `--define:__APP_ORIGIN__=${JSON.stringify(appOrigin)}`,
    `--define:EXTENSION_TEST_HOOKS=${testHooks}`,
  ];
  if (testHooks && originHoldMs !== undefined) {
    args.push(`--define:__ORIGIN_HOLD_MS__=${originHoldMs}`);
  }
  return args;
}
