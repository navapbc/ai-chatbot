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

/** esbuild argv bundling `entry` into `outFile` with APP_ORIGIN defined. */
export function esbuildArgs(
  entry: string,
  outFile: string,
  appOrigin: string,
): string[] {
  return [
    entry,
    '--bundle',
    '--format=esm',
    `--outfile=${outFile}`,
    `--define:__APP_ORIGIN__=${JSON.stringify(appOrigin)}`,
  ];
}
