import { execFileSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import {
  buildManifest,
  ExtensionBuildError,
  esbuildArgs,
  resolveAppOrigin,
} from './extension-build-config';

// Builds the browser bridge extension: bundles the service worker and writes
// the manifest, adding host_permissions for APP_ORIGIN when it is set.
// Usage: tsx scripts/build-extension.ts [--out <dir>]   (default extension/dist)

const ROOT = path.resolve(__dirname, '..');

function outDir(argv: string[]): string {
  const i = argv.indexOf('--out');
  if (i === -1) return path.join(ROOT, 'extension/dist');
  const dir = argv[i + 1];
  if (!dir) throw new ExtensionBuildError('--out needs a directory');
  return path.resolve(dir);
}

try {
  const appOrigin = resolveAppOrigin(process.env);
  const out = outDir(process.argv.slice(2));
  mkdirSync(out, { recursive: true });
  execFileSync(
    path.join(ROOT, 'node_modules/.bin/esbuild'),
    esbuildArgs(
      path.join(ROOT, 'extension/src/service-worker.ts'),
      path.join(out, 'service-worker.js'),
      appOrigin,
    ),
    { cwd: ROOT, stdio: 'inherit' },
  );
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
