import { createHash, createPublicKey, type KeyObject, sign } from 'node:crypto';
import { realpathSync } from 'node:fs';
import path from 'node:path';
import { crc32, deflateRawSync } from 'node:zlib';

// Pure helpers for scripts/package-extension.ts, kept apart so tests can
// import them without running a build: the release file set, the manifest
// checks, the extension id, a minimal zip writer, the CRX3 container and the
// Chrome update manifest. Nothing here prints, logs or writes a private key;
// errors carry fixed messages only.

export class PackageError extends Error {}

/** The only files a release may ship, in zip order. */
export const RELEASE_FILES = [
  'manifest.json',
  'service-worker.js',
  'panel.html',
  'panel.js',
  'panel.css',
] as const;

/**
 * Refuses a build directory listing that is not exactly RELEASE_FILES: a
 * missing file, a test page, a source map, a subdirectory or anything else.
 */
export function checkReleaseFiles(listing: readonly string[]): void {
  const expected = new Set<string>(RELEASE_FILES);
  const extra = listing.filter((name) => !expected.has(name));
  if (extra.length > 0) {
    throw new PackageError(
      `release build has unexpected files: ${extra.sort().join(', ')}`,
    );
  }
  const missing = RELEASE_FILES.filter((name) => !listing.includes(name));
  if (missing.length > 0) {
    throw new PackageError(
      `release build is missing files: ${missing.join(', ')}`,
    );
  }
}

/**
 * Whether `value` is a Chrome extension version: 3 or 4 dot-separated
 * integers, each 0 to 65535, with no leading zeros.
 */
export function isExtensionVersion(value: string): boolean {
  const parts = value.split('.');
  if (parts.length < 3 || parts.length > 4) return false;
  return parts.every(
    (part) => /^(0|[1-9]\d{0,4})$/.test(part) && Number(part) <= 65535,
  );
}

/** The DER SubjectPublicKeyInfo bytes of the manifest `key`. */
export function manifestKeyDer(key: unknown): Buffer {
  if (typeof key !== 'string' || !/^[A-Za-z0-9+/]+={0,2}$/.test(key)) {
    throw new PackageError('manifest key is missing or not base64');
  }
  const der = Buffer.from(key, 'base64');
  try {
    createPublicKey({ key: der, format: 'der', type: 'spki' });
  } catch {
    throw new PackageError('manifest key is not a public key');
  }
  return der;
}

/**
 * The first 16 bytes of SHA-256 over `publicKeyDer`: the CRX id, and, in
 * Chrome's a-p alphabet, the extension id.
 */
export function crxIdBytes(publicKeyDer: Buffer): Buffer {
  return createHash('sha256').update(publicKeyDer).digest().subarray(0, 16);
}

/** The extension id for a DER public key: each hex digit 0-f mapped to a-p. */
export function extensionIdFromDer(publicKeyDer: Buffer): string {
  return [...crxIdBytes(publicKeyDer).toString('hex')]
    .map((digit) => String.fromCharCode(97 + Number.parseInt(digit, 16)))
    .join('');
}

/**
 * Checks the built manifest for a release and returns its extension id: a
 * public `key`, `minimum_chrome_version`, a valid `version`, and host access
 * for the app origin only (`host_permissions` exactly `<origin>/*`, no
 * optional host permissions, no content scripts, no URL patterns among the
 * API permissions). The id must equal `expectedId` (extension/EXTENSION_ID).
 */
export function checkReleaseManifest(
  manifest: Record<string, unknown>,
  appOrigin: string,
  expectedId: string,
): string {
  const id = extensionIdFromDer(manifestKeyDer(manifest.key));
  if (id !== expectedId) {
    throw new PackageError(
      `extension id from the manifest key (${id}) does not match extension/EXTENSION_ID (${expectedId})`,
    );
  }
  const minimum = manifest.minimum_chrome_version;
  if (typeof minimum !== 'string' || !/^\d+(\.\d+){0,3}$/.test(minimum)) {
    throw new PackageError('manifest minimum_chrome_version is missing');
  }
  if (
    typeof manifest.version !== 'string' ||
    !isExtensionVersion(manifest.version)
  ) {
    throw new PackageError(
      'manifest version must be 3 or 4 numbers (0-65535) separated by dots',
    );
  }
  const hosts = manifest.host_permissions;
  if (
    !Array.isArray(hosts) ||
    hosts.length !== 1 ||
    hosts[0] !== `${appOrigin}/*`
  ) {
    throw new PackageError(
      'manifest host_permissions must be exactly the app origin',
    );
  }
  if ('optional_host_permissions' in manifest) {
    throw new PackageError('manifest must not have optional_host_permissions');
  }
  if ('content_scripts' in manifest) {
    throw new PackageError('manifest must not have content_scripts');
  }
  const permissions = manifest.permissions;
  if (
    Array.isArray(permissions) &&
    permissions.some(
      (p) => typeof p !== 'string' || p.includes('://') || p === '<all_urls>',
    )
  ) {
    throw new PackageError('manifest permissions must not name hosts');
  }
  return id;
}

/** An absolute https URL (no credentials), or a PackageError naming `flag`. */
export function checkHttpsUrl(value: string, flag: string): string {
  let parsed: URL;
  try {
    parsed = new URL(value);
  } catch {
    throw new PackageError(`${flag} must be an https URL`);
  }
  if (
    parsed.protocol !== 'https:' ||
    parsed.username !== '' ||
    parsed.password !== ''
  ) {
    throw new PackageError(`${flag} must be an https URL without credentials`);
  }
  return value;
}

/** Validates the update manifest URL: https, path ending in updates.xml. */
export function checkUpdateUrl(value: string): string {
  checkHttpsUrl(value, '--update-url');
  if (!new URL(value).pathname.endsWith('updates.xml')) {
    throw new PackageError('--update-url must end with updates.xml');
  }
  return value;
}

const XML_ESCAPES: Record<string, string> = {
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;',
  "'": '&apos;',
};

function xmlAttr(value: string): string {
  return value.replace(/[&<>"']/g, (c) => XML_ESCAPES[c]);
}

/** Chrome's gupdate update manifest for one extension. */
export function updatesXml(
  id: string,
  packageUrl: string,
  version: string,
): string {
  return [
    "<?xml version='1.0' encoding='UTF-8'?>",
    '<gupdate xmlns="http://www.google.com/update2/response" protocol="2.0">',
    `  <app appid="${xmlAttr(id)}">`,
    `    <updatecheck codebase="${xmlAttr(packageUrl)}" version="${xmlAttr(version)}"/>`,
    '  </app>',
    '</gupdate>',
    '',
  ].join('\n');
}

// MS-DOS date for 1980-01-01 with time 00:00: a fixed timestamp keeps the
// zip byte-identical across builds of the same files.
const DOS_DATE = (0 << 9) | (1 << 5) | 1;
const DOS_TIME = 0;

/**
 * A zip archive of `files` (name -> bytes) in the given order, every entry
 * deflated, with a fixed timestamp, no directory entries and no zip64 (the
 * extension is a few hundred KB).
 */
export function writeZip(files: ReadonlyArray<[string, Buffer]>): Buffer {
  const locals: Buffer[] = [];
  const centrals: Buffer[] = [];
  let offset = 0;
  for (const [name, data] of files) {
    const nameBytes = Buffer.from(name, 'utf8');
    const compressed = deflateRawSync(data, { level: 9 });
    const crc = crc32(data);

    const local = Buffer.alloc(30);
    local.writeUInt32LE(0x04034b50, 0);
    local.writeUInt16LE(20, 4); // version needed to extract
    local.writeUInt16LE(0, 6); // flags
    local.writeUInt16LE(8, 8); // deflate
    local.writeUInt16LE(DOS_TIME, 10);
    local.writeUInt16LE(DOS_DATE, 12);
    local.writeUInt32LE(crc, 14);
    local.writeUInt32LE(compressed.length, 18);
    local.writeUInt32LE(data.length, 22);
    local.writeUInt16LE(nameBytes.length, 26);
    local.writeUInt16LE(0, 28); // extra length
    locals.push(local, nameBytes, compressed);

    const central = Buffer.alloc(46);
    central.writeUInt32LE(0x02014b50, 0);
    central.writeUInt16LE(20, 4); // version made by
    central.writeUInt16LE(20, 6); // version needed to extract
    central.writeUInt16LE(0, 8);
    central.writeUInt16LE(8, 10);
    central.writeUInt16LE(DOS_TIME, 12);
    central.writeUInt16LE(DOS_DATE, 14);
    central.writeUInt32LE(crc, 16);
    central.writeUInt32LE(compressed.length, 20);
    central.writeUInt32LE(data.length, 24);
    central.writeUInt16LE(nameBytes.length, 28);
    // extra length, comment length, disk, internal and external attributes
    // stay 0.
    central.writeUInt32LE(offset, 42);
    centrals.push(central, nameBytes);

    offset += local.length + nameBytes.length + compressed.length;
  }
  const centralDir = Buffer.concat(centrals);
  const end = Buffer.alloc(22);
  end.writeUInt32LE(0x06054b50, 0);
  end.writeUInt16LE(files.length, 8);
  end.writeUInt16LE(files.length, 10);
  end.writeUInt32LE(centralDir.length, 12);
  end.writeUInt32LE(offset, 16);
  return Buffer.concat([...locals, centralDir, end]);
}

function varint(value: number): Buffer {
  const bytes: number[] = [];
  let v = value;
  while (v > 0x7f) {
    bytes.push((v & 0x7f) | 0x80);
    v >>>= 7;
  }
  bytes.push(v);
  return Buffer.from(bytes);
}

/** One length-delimited protobuf field (wire type 2). */
function protoBytes(field: number, data: Buffer): Buffer {
  return Buffer.concat([varint((field << 3) | 2), varint(data.length), data]);
}

const CRX3_SIGNED_DATA_PREFIX = Buffer.from('CRX3 SignedData\0', 'latin1');

/**
 * A CRX3 package of `zip` signed with `privateKey` (RSA), as Chromium's
 * crx_file reads it:
 *
 *   "Cr24" | LE32 3 | LE32 header length | CrxFileHeader | zip
 *
 * CrxFileHeader carries field 2 `sha256_with_rsa` (AsymmetricKeyProof:
 * 1 public_key as DER SubjectPublicKeyInfo, 2 signature) and field 10000
 * `signed_header_data` (SignedData: 1 crx_id, the first 16 bytes of SHA-256
 * over the public key). The signature is RSA PKCS#1 v1.5 with SHA-256 over
 * "CRX3 SignedData\0" | LE32 len(signed_header_data) | signed_header_data |
 * zip.
 */
export function buildCrx3(zip: Buffer, privateKey: KeyObject): Buffer {
  if (privateKey.asymmetricKeyType !== 'rsa') {
    throw new PackageError('the CRX signing key must be an RSA key');
  }
  const publicKeyDer = createPublicKey(privateKey).export({
    type: 'spki',
    format: 'der',
  });
  const signedHeaderData = protoBytes(1, crxIdBytes(publicKeyDer));
  const signedLength = Buffer.alloc(4);
  signedLength.writeUInt32LE(signedHeaderData.length, 0);
  const signature = sign(
    'sha256',
    Buffer.concat([
      CRX3_SIGNED_DATA_PREFIX,
      signedLength,
      signedHeaderData,
      zip,
    ]),
    privateKey,
  );
  const header = Buffer.concat([
    protoBytes(
      2,
      Buffer.concat([protoBytes(1, publicKeyDer), protoBytes(2, signature)]),
    ),
    protoBytes(10000, signedHeaderData),
  ]);
  const prefix = Buffer.alloc(12);
  prefix.write('Cr24', 0, 'latin1');
  prefix.writeUInt32LE(3, 4);
  prefix.writeUInt32LE(header.length, 8);
  return Buffer.concat([prefix, header, zip]);
}

function isInside(child: string, parent: string): boolean {
  const rel = path.relative(parent, child);
  return rel === '' || (!rel.startsWith('..') && !path.isAbsolute(rel));
}

/**
 * The real path of the signing key file, refused when it lies inside the
 * repository `root` (checked on the given path and on both real paths, so a
 * symlink cannot hide a key in the repo) or cannot be resolved. Reads
 * nothing from the file.
 */
export function resolvePemPath(pemPath: string, root: string): string {
  const inside =
    'EXTENSION_PEM_PATH must point outside the repository; refusing to use a key inside it';
  if (isInside(path.resolve(pemPath), path.resolve(root))) {
    throw new PackageError(inside);
  }
  let real: string;
  try {
    real = realpathSync(pemPath);
  } catch {
    throw new PackageError('EXTENSION_PEM_PATH does not name a readable file');
  }
  if (isInside(real, realpathSync(root))) throw new PackageError(inside);
  return real;
}

/**
 * Refuses a signing key whose public half is not the manifest `key`: a CRX
 * signed with another key would install under another extension id.
 */
export function checkKeyMatchesManifest(
  privateKey: KeyObject,
  manifestKey: unknown,
): void {
  const mismatch = 'the EXTENSION_PEM_PATH key does not match the manifest key';
  let publicDer: Buffer;
  try {
    publicDer = createPublicKey(privateKey).export({
      type: 'spki',
      format: 'der',
    });
  } catch {
    throw new PackageError(mismatch);
  }
  if (!publicDer.equals(manifestKeyDer(manifestKey))) {
    throw new PackageError(mismatch);
  }
}

/** Parsed command line for scripts/package-extension.ts. */
export interface PackageArgs {
  appOrigin: string;
  out: string;
  updateUrl?: string;
  packageUrl?: string;
  version?: string;
}

const VALUE_FLAGS = new Set([
  '--app-origin',
  '--out',
  '--update-url',
  '--package-url',
  '--version',
]);

/**
 * Parses the package script's flags. `--app-origin` falls back to
 * `APP_ORIGIN`; origin validation itself is resolveAppOrigin's. Unknown flags
 * are refused (including `--test-hooks`, which never belongs in a release).
 */
export function parsePackageArgs(
  argv: readonly string[],
  env: Record<string, string | undefined>,
): PackageArgs {
  const values = new Map<string, string>();
  for (let i = 0; i < argv.length; i++) {
    const flag = argv[i];
    if (flag === '--test-hooks') {
      throw new PackageError('test hooks are not allowed in a release package');
    }
    if (!VALUE_FLAGS.has(flag)) {
      throw new PackageError(`unknown argument ${JSON.stringify(flag)}`);
    }
    const value = argv[i + 1];
    if (value === undefined || value.startsWith('--')) {
      throw new PackageError(`${flag} needs a value`);
    }
    values.set(flag, value);
    i++;
  }
  const out = values.get('--out');
  if (!out) throw new PackageError('--out <dir> is required');
  const updateUrl = values.get('--update-url');
  const packageUrl = values.get('--package-url');
  if ((updateUrl === undefined) !== (packageUrl === undefined)) {
    throw new PackageError(
      '--update-url and --package-url must be given together',
    );
  }
  if (updateUrl !== undefined) checkUpdateUrl(updateUrl);
  if (packageUrl !== undefined) checkHttpsUrl(packageUrl, '--package-url');
  const version = values.get('--version');
  if (version !== undefined && !isExtensionVersion(version)) {
    throw new PackageError(
      '--version must be 3 or 4 numbers (0-65535) separated by dots',
    );
  }
  return {
    appOrigin: values.get('--app-origin') ?? env.APP_ORIGIN ?? '',
    out,
    updateUrl,
    packageUrl,
    version,
  };
}
