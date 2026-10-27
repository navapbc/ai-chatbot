import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { describe, expect, it } from 'vitest';
import {
  buildManifest,
  esbuildArgs,
  resolveAppOrigin,
} from '../../scripts/extension-build-config';

const ROOT = path.resolve(__dirname, '../..');

describe('resolveAppOrigin', () => {
  it('returns an empty origin when APP_ORIGIN is unset outside a release', () => {
    expect(resolveAppOrigin({})).toBe('');
    expect(resolveAppOrigin({ APP_ORIGIN: '' })).toBe('');
  });

  it.each([
    'https://app.example.org',
    'https://app.example.org:8443',
    'https://localhost:3443',
  ])('accepts the exact https origin %s', (origin) => {
    expect(resolveAppOrigin({ APP_ORIGIN: origin })).toBe(origin);
    expect(
      resolveAppOrigin({ APP_ORIGIN: origin, EXTENSION_RELEASE: '1' }),
    ).toBe(origin);
  });

  it.each([
    'http://127.0.0.1:3000',
    'http://localhost:3000',
    'http://localhost',
  ])('accepts the loopback http origin %s outside a release', (origin) => {
    expect(resolveAppOrigin({ APP_ORIGIN: origin })).toBe(origin);
  });

  it.each([
    'https://app.example.org/',
    'https://app.example.org/path',
    'https://app.example.org?q=1',
    'https://app.example.org#frag',
    'https://user:pass@app.example.org',
    'https://app.example.org:443',
    'https://App.Example.org',
    'http://localhost:80',
    'http://127.0.0.1:3000/',
    'http://app.example.org',
    'http://10.0.0.1:3000',
    'chrome-extension://abc',
    'not a url',
  ])('rejects %s', (origin) => {
    expect(() => resolveAppOrigin({ APP_ORIGIN: origin })).toThrow(
      /APP_ORIGIN must be an exact https origin/,
    );
  });

  it('requires APP_ORIGIN in a release build', () => {
    expect(() => resolveAppOrigin({ EXTENSION_RELEASE: '1' })).toThrow(
      /APP_ORIGIN is required when EXTENSION_RELEASE=1/,
    );
  });

  it('rejects a loopback http origin in a release build', () => {
    expect(() =>
      resolveAppOrigin({
        APP_ORIGIN: 'http://127.0.0.1:3000',
        EXTENSION_RELEASE: '1',
      }),
    ).toThrow(/exact https origin when EXTENSION_RELEASE=1/);
  });

  it('treats only EXTENSION_RELEASE=1 as a release', () => {
    expect(resolveAppOrigin({ EXTENSION_RELEASE: 'true' })).toBe('');
    expect(
      resolveAppOrigin({
        APP_ORIGIN: 'http://127.0.0.1:3000',
        EXTENSION_RELEASE: 'true',
      }),
    ).toBe('http://127.0.0.1:3000');
  });
});

describe('buildManifest', () => {
  const source = JSON.parse(
    readFileSync(path.join(ROOT, 'extension/manifest.json'), 'utf8'),
  );

  it('adds no host_permissions without an origin', () => {
    const built = buildManifest(source, '');
    expect(built).toEqual(source);
    expect(built).not.toHaveProperty('host_permissions');
  });

  it('grants exactly the app origin and keeps everything else', () => {
    const before = structuredClone(source);
    const built = buildManifest(source, 'https://app.example.org');
    expect(built.host_permissions).toEqual(['https://app.example.org/*']);
    expect(built.key).toBe(source.key);
    expect(built.permissions).toEqual(source.permissions);
    const { host_permissions: _, ...rest } = built;
    expect(rest).toEqual(source);
    expect(source).toEqual(before);
  });
});

describe('esbuildArgs', () => {
  it('keeps the bundle flags and defines APP_ORIGIN', () => {
    expect(esbuildArgs('in.ts', 'out/sw.js', 'https://a.example')).toEqual([
      'in.ts',
      '--bundle',
      '--format=esm',
      '--outfile=out/sw.js',
      '--define:__APP_ORIGIN__="https://a.example"',
    ]);
  });

  it.each([
    ['https://app.example.org', 'https://app.example.org'],
    ['', ''],
  ])(
    'substitutes %j into extension/src/config.ts',
    async (origin, expected) => {
      const dir = mkdtempSync(path.join(tmpdir(), 'extension-build-test-'));
      try {
        const out = path.join(dir, 'config.mjs');
        execFileSync(
          path.join(ROOT, 'node_modules/.bin/esbuild'),
          esbuildArgs(path.join(ROOT, 'extension/src/config.ts'), out, origin),
          { stdio: 'ignore' },
        );
        expect(readFileSync(out, 'utf8')).not.toContain('__APP_ORIGIN__');
        const built = await import(pathToFileURL(out).href);
        expect(built.APP_ORIGIN).toBe(expected);
      } finally {
        rmSync(dir, { recursive: true, force: true });
      }
    },
  );
});
