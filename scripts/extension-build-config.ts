import path from 'node:path';
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

/**
 * esbuild argv bundling the side panel (`extension/panel/main.tsx` under
 * `root`) into `outFile`, an ES module loaded by panel.html's
 * `<script type="module">`. `@/` resolves to the repo root as in the app,
 * except the exact specifier `@/lib/utils`, which components/ui leaf files
 * import for `cn`: the real lib/utils pulls in `ai` and the database schema,
 * so the panel gets its own copy (extension/panel/lib/cn.ts). esbuild picks
 * the longest matching alias, so the order of the two does not matter.
 * `metafile`, when given, records the import graph (the bundle test reads it).
 * `appOrigin` is defined as `__APP_ORIGIN__` for extension/src/config.ts, as
 * in the worker build.
 */
export function panelEsbuildArgs(
  root: string,
  outFile: string,
  metafile?: string,
  appOrigin = '',
): string[] {
  const args = [
    path.join(root, 'extension/panel/main.tsx'),
    '--bundle',
    '--format=esm',
    '--jsx=automatic',
    `--outfile=${outFile}`,
    `--define:process.env.NODE_ENV=${JSON.stringify('production')}`,
    `--define:__APP_ORIGIN__=${JSON.stringify(appOrigin)}`,
    `--alias:@/lib/utils=${path.join(root, 'extension/panel/lib/cn.ts')}`,
    `--alias:@=${root}`,
  ];
  if (metafile) args.push(`--metafile=${metafile}`);
  return args;
}

/** The index just past the `}` matching the `{` at `open`. */
function matchBrace(css: string, open: number): number {
  let depth = 0;
  for (let i = open; i < css.length; i++) {
    if (css[i] === '{') depth++;
    else if (css[i] === '}' && --depth === 0) return i + 1;
  }
  throw new ExtensionBuildError('unbalanced braces in app/globals.css');
}

/** Every top-level `<selector> { ... }` block of `css` whose selector matches. */
function blocks(css: string, selector: RegExp): string[] {
  const found: string[] = [];
  let i = 0;
  while (i < css.length) {
    const open = css.indexOf('{', i);
    if (open === -1) break;
    const close = matchBrace(css, open);
    const head = css
      .slice(i, open)
      .replace(/\/\*[\s\S]*?\*\//g, '')
      .trim();
    if (selector.test(head)) found.push(`${head} ${css.slice(open, close)}`);
    i = close;
  }
  return found;
}

/**
 * The app's design tokens for the side panel: the `:root` and `.dark`
 * variable blocks of the `@layer base` block in app/globals.css that holds
 * them, and nothing else from that file (no `@tailwind`, body or font rules,
 * which belong to the Next app). One source of truth: the build regenerates
 * extension/panel/tokens.css from it.
 */
export function extractTokens(globalsCss: string): string {
  const layer = blocks(globalsCss, /^@layer\s+base$/).find((b) =>
    /:root\s*\{/.test(b),
  );
  if (!layer) {
    throw new ExtensionBuildError(
      'app/globals.css has no @layer base block with the :root tokens',
    );
  }
  const body = layer.slice(layer.indexOf('{') + 1, layer.lastIndexOf('}'));
  const vars = blocks(body, /^(:root|\.dark)$/);
  return `@layer base {\n  ${vars.join('\n  ')}\n}\n`;
}
