import { spawnSync } from 'node:child_process';
import {
  createPublicKey,
  generateKeyPairSync,
  type KeyObject,
  verify,
} from 'node:crypto';
import {
  existsSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { crc32, inflateRawSync } from 'node:zlib';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import {
  buildCrx3,
  checkKeyMatchesManifest,
  checkReleaseFiles,
  checkReleaseManifest,
  crxIdBytes,
  extensionIdFromDer,
  isExtensionVersion,
  manifestKeyDer,
  parsePackageArgs,
  RELEASE_FILES,
  resolvePemPath,
  updatesXml,
  writeZip,
} from '../../scripts/package-extension-lib';

// scripts/package-extension.ts: the release package for a managed install.
// Throwaway RSA keys generated here stand in for the real signing key, which
// is never in the repository.

const ROOT = path.resolve(__dirname, '../..');
const SOURCE = JSON.parse(
  readFileSync(path.join(ROOT, 'extension/manifest.json'), 'utf8'),
);
const EXTENSION_ID = readFileSync(
  path.join(ROOT, 'extension/EXTENSION_ID'),
  'utf8',
).trim();
const ORIGIN = 'https://app.example.org';
const UPDATE_URL = 'https://dl.example.org/ext/updates.xml';
const PACKAGE_URL = 'https://dl.example.org/ext/extension.crx?v=1&x=2';

const cleanups: string[] = [];
afterAll(() => {
  for (const dir of cleanups) rmSync(dir, { recursive: true, force: true });
});

function tempDir(): string {
  const dir = mkdtempSync(path.join(tmpdir(), 'package-extension-test-'));
  cleanups.push(dir);
  return dir;
}

/** The environment with every variable the script reads removed. */
function cleanEnv(): NodeJS.ProcessEnv {
  const {
    APP_ORIGIN: _o,
    EXTENSION_RELEASE: _r,
    EXTENSION_TEST_HOOKS: _h,
    EXTENSION_TEST_ORIGIN_HOLD_MS: _m,
    EXTENSION_PEM_PATH: _p,
    ...env
  } = process.env;
  return env;
}

function runScript(args: string[], extra: Record<string, string> = {}) {
  const result = spawnSync(
    path.join(ROOT, 'node_modules/.bin/tsx'),
    ['scripts/package-extension.ts', ...args],
    { cwd: ROOT, env: { ...cleanEnv(), ...extra }, encoding: 'utf8' },
  );
  return {
    status: result.status,
    output: `${result.stdout}${result.stderr}`,
  };
}

function rsaKey(): KeyObject {
  return generateKeyPairSync('rsa', { modulusLength: 2048 }).privateKey;
}

function pemOf(key: KeyObject): string {
  return key.export({ type: 'pkcs8', format: 'pem' }).toString();
}

/** The base64 body of a PEM, without its armor lines. */
function pemBody(pem: string): string {
  return pem
    .split('\n')
    .filter((line) => line !== '' && !line.startsWith('-----'))
    .join('');
}

/** Independent zip reader: EOCD, then the central directory, then data. */
function readZip(zip: Buffer): Map<string, Buffer> {
  let eocd = -1;
  for (let i = zip.length - 22; i >= 0; i--) {
    if (zip.readUInt32LE(i) === 0x06054b50) {
      eocd = i;
      break;
    }
  }
  expect(eocd).toBeGreaterThanOrEqual(0);
  const count = zip.readUInt16LE(eocd + 10);
  let at = zip.readUInt32LE(eocd + 16);
  const files = new Map<string, Buffer>();
  for (let n = 0; n < count; n++) {
    expect(zip.readUInt32LE(at)).toBe(0x02014b50);
    const method = zip.readUInt16LE(at + 10);
    const crc = zip.readUInt32LE(at + 16);
    const size = zip.readUInt32LE(at + 20);
    const nameLength = zip.readUInt16LE(at + 28);
    const extraLength = zip.readUInt16LE(at + 30);
    const commentLength = zip.readUInt16LE(at + 32);
    const local = zip.readUInt32LE(at + 42);
    const name = zip.toString('utf8', at + 46, at + 46 + nameLength);
    expect(zip.readUInt32LE(local)).toBe(0x04034b50);
    const start =
      local + 30 + zip.readUInt16LE(local + 26) + zip.readUInt16LE(local + 28);
    const data =
      method === 8
        ? inflateRawSync(zip.subarray(start, start + size))
        : zip.subarray(start, start + size);
    expect(crc32(data)).toBe(crc);
    files.set(name, data);
    at += 46 + nameLength + extraLength + commentLength;
  }
  return files;
}

/** Independent protobuf reader: length-delimited fields of one message. */
function readProto(buf: Buffer): Array<[number, Buffer]> {
  const fields: Array<[number, Buffer]> = [];
  let i = 0;
  const varint = () => {
    let value = 0;
    let shift = 0;
    for (;;) {
      const byte = buf[i++];
      value += (byte & 0x7f) * 2 ** shift;
      if (byte < 0x80) return value;
      shift += 7;
    }
  };
  while (i < buf.length) {
    const tag = varint();
    expect(tag & 7).toBe(2);
    const length = varint();
    fields.push([Math.floor(tag / 8), buf.subarray(i, i + length)]);
    i += length;
  }
  return fields;
}

describe('the source manifest', () => {
  it('sets minimum_chrome_version 120', () => {
    expect(SOURCE.minimum_chrome_version).toBe('120');
  });

  it('derives extension/EXTENSION_ID from the manifest key', () => {
    expect(extensionIdFromDer(manifestKeyDer(SOURCE.key))).toBe(EXTENSION_ID);
    expect(EXTENSION_ID).toBe('ppcenadbeddfnlgklnccjkgjegmbkkkl');
  });
});

describe('parsePackageArgs', () => {
  it('reads the flags and falls back to APP_ORIGIN', () => {
    expect(
      parsePackageArgs(
        [
          '--out',
          'o',
          '--update-url',
          UPDATE_URL,
          '--package-url',
          PACKAGE_URL,
          '--version',
          '1.2.3.4',
        ],
        { APP_ORIGIN: ORIGIN },
      ),
    ).toEqual({
      appOrigin: ORIGIN,
      out: 'o',
      updateUrl: UPDATE_URL,
      packageUrl: PACKAGE_URL,
      version: '1.2.3.4',
    });
  });

  it.each([
    [['--out', 'o', '--test-hooks'], /test hooks/],
    [['--app-origin', ORIGIN], /--out/],
    [['--out', 'o', '--bogus', 'x'], /unknown argument/],
    [['--out', 'o', '--update-url', UPDATE_URL], /together/],
    [['--out', 'o', '--package-url', PACKAGE_URL], /together/],
    [
      [
        '--out',
        'o',
        '--update-url',
        'http://x.org/updates.xml',
        '--package-url',
        PACKAGE_URL,
      ],
      /https/,
    ],
    [
      [
        '--out',
        'o',
        '--update-url',
        'https://x.org/update.json',
        '--package-url',
        PACKAGE_URL,
      ],
      /updates\.xml/,
    ],
    [
      [
        '--out',
        'o',
        '--update-url',
        UPDATE_URL,
        '--package-url',
        'http://x.org/e.crx',
      ],
      /https/,
    ],
    [['--out', 'o', '--version', '1.2'], /--version/],
    [['--out', 'o', '--version'], /needs a value/],
  ])('refuses %j', (argv, message) => {
    expect(() => parsePackageArgs(argv, {})).toThrow(message);
  });
});

describe('isExtensionVersion', () => {
  it.each(['0.1.0', '1.2.3', '65535.0.0.65535'])('accepts %s', (v) => {
    expect(isExtensionVersion(v)).toBe(true);
  });
  it.each([
    '1',
    '1.2',
    '1.2.3.4.5',
    '01.2.3',
    '1.2.65536',
    '1.2.x',
    '1..2',
    '',
  ])('rejects %s', (v) => {
    expect(isExtensionVersion(v)).toBe(false);
  });
});

describe('checkReleaseFiles', () => {
  it('accepts exactly the five release files', () => {
    expect(() => checkReleaseFiles([...RELEASE_FILES].reverse())).not.toThrow();
  });
  it.each([
    [[...RELEASE_FILES, 'test-page.html'], /test-page\.html/],
    [[...RELEASE_FILES, 'panel.js.map'], /panel\.js\.map/],
    [[...RELEASE_FILES, 'assets'], /assets/],
    [RELEASE_FILES.slice(1), /missing files: manifest\.json/],
  ])('refuses %j', (listing, message) => {
    expect(() => checkReleaseFiles(listing)).toThrow(message);
  });
});

describe('checkReleaseManifest', () => {
  const release = { ...SOURCE, host_permissions: [`${ORIGIN}/*`] };

  it('returns the id for a release manifest', () => {
    expect(checkReleaseManifest(release, ORIGIN, EXTENSION_ID)).toBe(
      EXTENSION_ID,
    );
  });

  it.each([
    [{ key: undefined }, /key/],
    [{ minimum_chrome_version: undefined }, /minimum_chrome_version/],
    [{ version: '1.2' }, /version/],
    [{ host_permissions: undefined }, /host_permissions/],
    [
      { host_permissions: [`${ORIGIN}/*`, 'https://other.org/*'] },
      /host_permissions/,
    ],
    [{ host_permissions: ['<all_urls>'] }, /host_permissions/],
    [{ optional_host_permissions: ['https://*/*'] }, /optional_host/],
    [{ content_scripts: [{ matches: ['<all_urls>'] }] }, /content_scripts/],
    [{ permissions: ['debugger', 'https://x.org/*'] }, /permissions/],
  ])('refuses %j', (patch, message) => {
    expect(() =>
      checkReleaseManifest({ ...release, ...patch }, ORIGIN, EXTENSION_ID),
    ).toThrow(message);
  });

  it('refuses an id that is not extension/EXTENSION_ID', () => {
    expect(() => checkReleaseManifest(release, ORIGIN, 'a'.repeat(32))).toThrow(
      /does not match extension\/EXTENSION_ID/,
    );
  });
});

describe('updatesXml', () => {
  it('names the id, the package URL (escaped) and the version', () => {
    expect(updatesXml(EXTENSION_ID, PACKAGE_URL, '1.2.3')).toBe(
      [
        "<?xml version='1.0' encoding='UTF-8'?>",
        '<gupdate xmlns="http://www.google.com/update2/response" protocol="2.0">',
        `  <app appid="${EXTENSION_ID}">`,
        '    <updatecheck codebase="https://dl.example.org/ext/extension.crx?v=1&amp;x=2" version="1.2.3"/>',
        '  </app>',
        '</gupdate>',
        '',
      ].join('\n'),
    );
  });
});

describe('writeZip', () => {
  it('round-trips through an independent reader', () => {
    const files: Array<[string, Buffer]> = [
      ['a.txt', Buffer.from('hello')],
      ['b.bin', Buffer.alloc(5000, 7)],
      ['empty', Buffer.alloc(0)],
    ];
    const read = readZip(writeZip(files));
    expect([...read.keys()]).toEqual(['a.txt', 'b.bin', 'empty']);
    for (const [name, data] of files) expect(read.get(name)).toEqual(data);
  });

  it('is byte-identical for the same input', () => {
    const files: Array<[string, Buffer]> = [['x', Buffer.from('y')]];
    expect(writeZip(files).equals(writeZip(files))).toBe(true);
  });
});

describe('buildCrx3', () => {
  it('writes a CRX3 whose signature verifies and whose id is the key hash', () => {
    const key = rsaKey();
    const zip = writeZip([['manifest.json', Buffer.from('{}')]]);
    const crx = buildCrx3(zip, key);

    expect(crx.toString('latin1', 0, 4)).toBe('Cr24');
    expect(crx.readUInt32LE(4)).toBe(3);
    const headerLength = crx.readUInt32LE(8);
    const header = crx.subarray(12, 12 + headerLength);
    expect(crx.subarray(12 + headerLength).equals(zip)).toBe(true);

    const fields = readProto(header);
    expect(fields.map(([n]) => n)).toEqual([2, 10000]);
    const proof = readProto(fields[0][1]);
    expect(proof.map(([n]) => n)).toEqual([1, 2]);
    const [publicKeyDer, signature] = [proof[0][1], proof[1][1]];
    const expectedDer = createPublicKey(key).export({
      type: 'spki',
      format: 'der',
    });
    expect(publicKeyDer.equals(expectedDer)).toBe(true);

    const signedHeaderData = fields[1][1];
    const signedData = readProto(signedHeaderData);
    expect(signedData).toHaveLength(1);
    expect(signedData[0][0]).toBe(1);
    expect(signedData[0][1].equals(crxIdBytes(expectedDer))).toBe(true);
    expect(signedData[0][1]).toHaveLength(16);

    const length = Buffer.alloc(4);
    length.writeUInt32LE(signedHeaderData.length, 0);
    const signed = Buffer.concat([
      Buffer.from('CRX3 SignedData\0', 'latin1'),
      length,
      signedHeaderData,
      zip,
    ]);
    const publicKey = createPublicKey({
      key: publicKeyDer,
      format: 'der',
      type: 'spki',
    });
    expect(verify('sha256', signed, publicKey, signature)).toBe(true);
    // A changed archive no longer verifies.
    const tampered = Buffer.from(signed);
    tampered[tampered.length - 1] ^= 1;
    expect(verify('sha256', tampered, publicKey, signature)).toBe(false);
  });

  it('refuses a non-RSA key', () => {
    const { privateKey } = generateKeyPairSync('ec', { namedCurve: 'P-256' });
    expect(() => buildCrx3(Buffer.alloc(0), privateKey)).toThrow(/RSA/);
  });
});

describe('the signing key checks', () => {
  it('refuses a key path inside the repository, even one that exists', () => {
    expect(() =>
      resolvePemPath(path.join(ROOT, 'extension/manifest.json'), ROOT),
    ).toThrow(/outside the repository/);
    expect(() => resolvePemPath('extension/missing.pem', ROOT)).toThrow(
      /outside the repository/,
    );
  });

  it('accepts a path outside the repository', () => {
    const file = path.join(tempDir(), 'k.pem');
    writeFileSync(file, 'not read');
    expect(resolvePemPath(file, ROOT)).toMatch(/k\.pem$/);
  });

  it('accepts a key whose public half is the manifest key', () => {
    const key = rsaKey();
    const manifestKey = createPublicKey(key)
      .export({ type: 'spki', format: 'der' })
      .toString('base64');
    expect(() => checkKeyMatchesManifest(key, manifestKey)).not.toThrow();
  });

  it('refuses a key that does not match, with a fixed message', () => {
    const key = rsaKey();
    let message = '';
    try {
      checkKeyMatchesManifest(key, SOURCE.key);
    } catch (error) {
      message = (error as Error).message;
    }
    expect(message).toBe(
      'the EXTENSION_PEM_PATH key does not match the manifest key',
    );
  });
});

describe('the package script refuses', () => {
  it('a missing APP_ORIGIN', () => {
    const { status, output } = runScript(['--out', tempDir()]);
    expect(status).toBe(1);
    expect(output).toMatch(/APP_ORIGIN is required/);
  });

  it('a non-https origin', () => {
    const { status, output } = runScript([
      '--app-origin',
      'http://localhost:3000',
      '--out',
      tempDir(),
    ]);
    expect(status).toBe(1);
    expect(output).toMatch(/exact https origin/);
  });

  it('test hooks by flag and by environment', () => {
    const out = tempDir();
    const byFlag = runScript([
      '--app-origin',
      ORIGIN,
      '--out',
      out,
      '--test-hooks',
    ]);
    expect(byFlag.status).toBe(1);
    expect(byFlag.output).toMatch(/test hooks/);
    const byEnv = runScript(['--app-origin', ORIGIN, '--out', out], {
      EXTENSION_TEST_HOOKS: '1',
    });
    expect(byEnv.status).toBe(1);
    expect(byEnv.output).toMatch(/test hooks/);
    expect(readdirSync(out)).toEqual([]);
  });

  it('a signing key inside the repository', () => {
    const out = tempDir();
    const { status, output } = runScript(
      ['--app-origin', ORIGIN, '--out', out],
      {
        EXTENSION_PEM_PATH: path.join(ROOT, 'extension/manifest.json'),
      },
    );
    expect(status).toBe(1);
    expect(output).toMatch(/outside the repository/);
    expect(readdirSync(out)).toEqual([]);
  });

  it('a signing key that does not match the manifest, printing no key', () => {
    const pem = pemOf(rsaKey());
    const keyFile = path.join(tempDir(), 'throwaway.pem');
    writeFileSync(keyFile, pem);
    const out = tempDir();
    const { status, output } = runScript(
      ['--app-origin', ORIGIN, '--out', out],
      {
        EXTENSION_PEM_PATH: keyFile,
      },
    );
    expect(status).toBe(1);
    expect(output).toContain(
      'the EXTENSION_PEM_PATH key does not match the manifest key',
    );
    expect(output).not.toContain('PRIVATE KEY');
    expect(output).not.toContain(pemBody(pem).slice(0, 40));
    expect(readdirSync(out)).toEqual([]);
  });
});

describe('a release package', () => {
  let out: string;
  let output: string;
  let zipFiles: Map<string, Buffer>;

  beforeAll(() => {
    out = tempDir();
    const result = runScript([
      '--app-origin',
      ORIGIN,
      '--out',
      out,
      '--update-url',
      UPDATE_URL,
      '--package-url',
      PACKAGE_URL,
      '--version',
      '1.2.3',
    ]);
    output = result.output;
    expect(result.status, output).toBe(0);
    zipFiles = readZip(readFileSync(path.join(out, 'extension.zip')));
  }, 120_000);

  it('writes the zip and updates.xml, and no CRX without a key', () => {
    expect(readdirSync(out).sort()).toEqual(['extension.zip', 'updates.xml']);
    expect(existsSync(path.join(out, 'extension.crx'))).toBe(false);
  });

  it('zips exactly the five release files: no test page, no maps', () => {
    expect([...zipFiles.keys()].sort()).toEqual([...RELEASE_FILES].sort());
    for (const data of zipFiles.values()) {
      expect(data.length).toBeGreaterThan(0);
    }
    expect(zipFiles.get('panel.js')?.toString()).not.toContain(
      'sourceMappingURL',
    );
  });

  it('ships a manifest with the key, the minimum Chrome, the version and only the app origin', () => {
    const manifest = JSON.parse(
      zipFiles.get('manifest.json')?.toString() ?? '',
    );
    expect(manifest.key).toBe(SOURCE.key);
    expect(manifest.minimum_chrome_version).toBe('120');
    expect(manifest.version).toBe('1.2.3');
    expect(manifest.host_permissions).toEqual([`${ORIGIN}/*`]);
    expect(manifest.update_url).toBe(UPDATE_URL);
    expect(manifest).not.toHaveProperty('optional_host_permissions');
    expect(manifest).not.toHaveProperty('content_scripts');
  });

  it('writes updates.xml with the derived id, the version and the package URL', () => {
    expect(readFileSync(path.join(out, 'updates.xml'), 'utf8')).toBe(
      updatesXml(EXTENSION_ID, PACKAGE_URL, '1.2.3'),
    );
  });

  it('prints no key material', () => {
    expect(output).toContain(
      `packaged extension ${EXTENSION_ID} version 1.2.3`,
    );
    expect(output).not.toMatch(/PRIVATE KEY|BEGIN /);
    for (const name of readdirSync(out)) {
      expect(readFileSync(path.join(out, name)).toString('latin1')).not.toMatch(
        /PRIVATE KEY|BEGIN /,
      );
    }
  });
});

describe('the Chrome policy template', () => {
  const text = readFileSync(
    path.join(ROOT, 'deploy/chrome-policy/extension-policy.template.json'),
    'utf8',
  );

  it('is valid JSON that force-installs the extension from the update URL', () => {
    const policy = JSON.parse(text);
    expect(policy.ExtensionInstallForcelist).toEqual([
      '__EXTENSION_ID__;__UPDATE_URL__',
    ]);
    expect(policy.ExtensionSettings).toEqual({
      __EXTENSION_ID__: {
        installation_mode: 'force_installed',
        update_url: '__UPDATE_URL__',
      },
    });
  });

  it('blocks no hosts, permissions or screenshots', () => {
    expect(text).not.toMatch(
      /runtime_blocked_hosts|blocked_permissions|DisableScreenshots/,
    );
  });

  it('still parses once the placeholders are filled', () => {
    const filled = text
      .replaceAll('__EXTENSION_ID__', EXTENSION_ID)
      .replaceAll('__UPDATE_URL__', UPDATE_URL);
    expect(JSON.parse(filled).ExtensionSettings[EXTENSION_ID].update_url).toBe(
      UPDATE_URL,
    );
  });
});
