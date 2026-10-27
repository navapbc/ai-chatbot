import { execFileSync } from 'node:child_process';
import {
  copyFileSync,
  mkdirSync,
  readFileSync,
  renameSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import path from 'node:path';
import {
  buildManifest,
  ExtensionBuildError,
  esbuildArgs,
  extractTokens,
  panelEsbuildArgs,
  resolveAppOrigin,
  resolveOriginHoldMs,
  resolveTestHooks,
} from './extension-build-config';

// Builds the browser bridge extension: bundles the service worker, builds the
// side panel (panel.html, panel.js, panel.css) and writes the manifest,
// adding host_permissions for APP_ORIGIN when it is set.
// Usage: tsx scripts/build-extension.ts [--out <dir>] [--test-hooks]
//   [--panel-metafile <file>]
//   (default extension/dist). --panel-metafile writes the panel bundle's
//   esbuild metafile (its import graph) to <file>, outside the extension. EXTENSION_TEST_HOOKS=1 or --test-hooks installs
//   the bridge's globalThis test hooks for the e2e suites; refused together
//   with EXTENSION_RELEASE=1. EXTENSION_TEST_ORIGIN_HOLD_MS=<ms> (test-hooks
//   builds only) shortens the bridge's 60 s origin hold for the e2e suites.

const ROOT = path.resolve(__dirname, '..');

function outDir(argv: string[]): string {
  const i = argv.indexOf('--out');
  if (i === -1) return path.join(ROOT, 'extension/dist');
  const dir = argv[i + 1];
  if (!dir) throw new ExtensionBuildError('--out needs a directory');
  return path.resolve(dir);
}

function panelMetafile(argv: string[]): string | undefined {
  const i = argv.indexOf('--panel-metafile');
  if (i === -1) return undefined;
  const file = argv[i + 1];
  if (!file) throw new ExtensionBuildError('--panel-metafile needs a file');
  return path.resolve(file);
}

/**
 * Builds the side panel into `out`. The tokens file is regenerated from
 * app/globals.css on every build and replaced with a rename, so builds
 * running at the same time (the e2e files do) never read a half-written one.
 */
function buildPanel(out: string, metafile: string | undefined) {
  const panel = path.join(ROOT, 'extension/panel');
  const tokens = path.join(panel, 'tokens.css');
  const pending = `${tokens}.${process.pid}.tmp`;
  try {
    writeFileSync(
      pending,
      extractTokens(readFileSync(path.join(ROOT, 'app/globals.css'), 'utf8')),
    );
    renameSync(pending, tokens);
  } finally {
    rmSync(pending, { force: true });
  }
  execFileSync(
    path.join(ROOT, 'node_modules/.bin/esbuild'),
    panelEsbuildArgs(ROOT, path.join(out, 'panel.js'), metafile),
    { cwd: ROOT, stdio: 'inherit' },
  );
  // No --postcss: the panel CSS uses no nesting, and the CLI's own pipeline
  // inlines the tokens.css @import.
  execFileSync(
    path.join(ROOT, 'node_modules/.bin/tailwindcss'),
    [
      '-c',
      path.join(panel, 'tailwind.config.ts'),
      '-i',
      path.join(panel, 'panel.css'),
      '-o',
      path.join(out, 'panel.css'),
    ],
    { cwd: ROOT, stdio: 'inherit' },
  );
  copyFileSync(path.join(panel, 'index.html'), path.join(out, 'panel.html'));
}

try {
  const appOrigin = resolveAppOrigin(process.env);
  const testHooks = resolveTestHooks(process.env, process.argv.slice(2));
  const originHoldMs = resolveOriginHoldMs(process.env, testHooks);
  const out = outDir(process.argv.slice(2));
  const metafile = panelMetafile(process.argv.slice(2));
  mkdirSync(out, { recursive: true });
  execFileSync(
    path.join(ROOT, 'node_modules/.bin/esbuild'),
    esbuildArgs(
      path.join(ROOT, 'extension/src/service-worker.ts'),
      path.join(out, 'service-worker.js'),
      appOrigin,
      testHooks,
      originHoldMs,
    ),
    { cwd: ROOT, stdio: 'inherit' },
  );
  buildPanel(out, metafile);
  const source = JSON.parse(
    readFileSync(path.join(ROOT, 'extension/manifest.json'), 'utf8'),
  );
  writeFileSync(
    path.join(out, 'manifest.json'),
    `${JSON.stringify(buildManifest(source, appOrigin), null, 2)}\n`,
  );
} catch (error) {
  if (error instanceof ExtensionBuildError) {
    console.error(`extension build failed: ${error.message}`);
    process.exit(1);
  }
  throw error;
}
