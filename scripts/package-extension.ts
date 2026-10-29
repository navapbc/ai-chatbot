import { execFileSync } from 'node:child_process';
import { createPrivateKey, type KeyObject } from 'node:crypto';
import {
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import {
  ExtensionBuildError,
  resolveAppOrigin,
  resolveOriginHoldMs,
  resolveTestHooks,
} from './extension-build-config';
import {
  buildCrx3,
  checkKeyMatchesManifest,
  checkReleaseFiles,
  checkReleaseManifest,
  PackageError,
  parsePackageArgs,
  RELEASE_FILES,
  resolvePemPath,
  updatesXml,
  writeZip,
} from './package-extension-lib';

// Packages the browser bridge extension for a managed (policy) install.
// Usage: tsx scripts/package-extension.ts --app-origin <https origin>
//   --out <dir> [--update-url <https://.../updates.xml>
//   --package-url <https://.../extension.crx>] [--version x.y.z[.w]]
//
// Runs the release build (EXTENSION_RELEASE=1) of scripts/build-extension.ts
// into a fresh temporary directory, checks the file set and the manifest,
// and writes into <dir>:
// - extension.zip: the five release files and nothing else.
// - updates.xml: Chrome's update manifest, only with --update-url and
//   --package-url (the built manifest then also carries update_url).
// - extension.crx: a CRX3 package, only when EXTENSION_PEM_PATH names the
//   extension's private key, an unencrypted PEM, in a file outside the
//   repository (a passphrase-protected key is refused). The key is read only
//   then, never written, and never printed. To produce the CRX for a release,
//   re-run this script locally with EXTENSION_PEM_PATH and the same flags;
//   nothing signs an existing zip.

const ROOT = path.resolve(__dirname, '..');
const OUTPUTS = ['extension.zip', 'updates.xml', 'extension.crx'];

/** The signing key, checked against the source manifest, or undefined. */
function loadSigningKey(
  env: Record<string, string | undefined>,
): KeyObject | undefined {
  const pemPath = env.EXTENSION_PEM_PATH ?? '';
  if (pemPath === '') return undefined;
  const real = resolvePemPath(pemPath, ROOT);
  let key: KeyObject;
  try {
    key = createPrivateKey(readFileSync(real));
  } catch {
    throw new PackageError(
      'EXTENSION_PEM_PATH is not a readable unencrypted private key',
    );
  }
  const source = JSON.parse(
    readFileSync(path.join(ROOT, 'extension/manifest.json'), 'utf8'),
  );
  checkKeyMatchesManifest(key, source.key);
  return key;
}

function main() {
  const argv = process.argv.slice(2);
  const args = parsePackageArgs(argv, process.env);
  const releaseEnv: NodeJS.ProcessEnv = {
    ...process.env,
    EXTENSION_RELEASE: '1',
    APP_ORIGIN: args.appOrigin,
  };
  // The same checks the build makes, run first so a refusal leaves nothing
  // behind and costs no build.
  const appOrigin = resolveAppOrigin(releaseEnv);
  resolveTestHooks(releaseEnv, argv);
  resolveOriginHoldMs(releaseEnv, false);
  const signingKey = loadSigningKey(process.env);

  const staging = mkdtempSync(path.join(tmpdir(), 'extension-release-'));
  try {
    const { EXTENSION_PEM_PATH: _pem, ...buildEnv } = releaseEnv;
    try {
      execFileSync(
        path.join(ROOT, 'node_modules/.bin/tsx'),
        ['scripts/build-extension.ts', '--out', staging],
        { cwd: ROOT, env: buildEnv, stdio: ['ignore', 'inherit', 'inherit'] },
      );
    } catch {
      throw new PackageError('the release build failed (see output above)');
    }
    checkReleaseFiles(readdirSync(staging));

    const manifestPath = path.join(staging, 'manifest.json');
    const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
    if (args.version !== undefined) manifest.version = args.version;
    if (args.updateUrl !== undefined) manifest.update_url = args.updateUrl;
    const expectedId = readFileSync(
      path.join(ROOT, 'extension/EXTENSION_ID'),
      'utf8',
    ).trim();
    const id = checkReleaseManifest(manifest, appOrigin, expectedId);
    writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);

    const zip = writeZip(
      RELEASE_FILES.map((name) => [
        name,
        readFileSync(path.join(staging, name)),
      ]),
    );
    const out = path.resolve(args.out);
    mkdirSync(out, { recursive: true });
    for (const name of OUTPUTS) rmSync(path.join(out, name), { force: true });
    const written = ['extension.zip'];
    writeFileSync(path.join(out, 'extension.zip'), zip);
    if (args.packageUrl !== undefined) {
      writeFileSync(
        path.join(out, 'updates.xml'),
        updatesXml(id, args.packageUrl, manifest.version),
      );
      written.push('updates.xml');
    }
    if (signingKey) {
      writeFileSync(
        path.join(out, 'extension.crx'),
        buildCrx3(zip, signingKey),
      );
      written.push('extension.crx');
    }
    console.log(
      `packaged extension ${id} version ${manifest.version} for ${appOrigin}: ${written.join(', ')}`,
    );
  } finally {
    rmSync(staging, { recursive: true, force: true });
  }
}

/** ` (CODE)` for a short Node error code, else ''. Never the message. */
function errorCodeSuffix(error: unknown): string {
  const code =
    typeof error === 'object' && error !== null
      ? (error as { code?: unknown }).code
      : undefined;
  return typeof code === 'string' && /^[A-Z0-9_]{1,40}$/.test(code)
    ? ` (${code})`
    : '';
}

try {
  main();
} catch (error) {
  if (error instanceof PackageError || error instanceof ExtensionBuildError) {
    console.error(`extension package failed: ${error.message}`);
    process.exit(1);
  }
  // Anything else gets a fixed line: an unexpected error must not echo a
  // key or a path into the log. A short Node error code (e.g. EACCES) is
  // safe to print; the message never is.
  console.error(
    `extension package failed: unexpected error${errorCodeSuffix(error)}`,
  );
  process.exit(1);
}
