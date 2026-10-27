import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { describe, expect, it } from 'vitest';
import {
  buildManifest,
  esbuildArgs,
  extractTokens,
  panelEsbuildArgs,
  resolveAppOrigin,
  resolveOriginHoldMs,
  resolveTestHooks,
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

describe('the source manifest', () => {
  const source = JSON.parse(
    readFileSync(path.join(ROOT, 'extension/manifest.json'), 'utf8'),
  );

  it('keeps the pinned key (it fixes the extension id)', () => {
    expect(source.key).toBe(
      'MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAwwc2pYVHPW/t+29jPeAaudSFX1xI1MLtnOcQo+wtpjYFO1aXXSVV6+ex79qw5cqbvGBMH74vcLTxNu6In+cRvYRYhAw6jowmmmTb+MdYUj0sANHABPzZfi4C7nI6YHSJXIptrXbSWz/RE+Ip1gu2k6Z/o/84qPd/8awQUuTAHXwCNSPVNfwRT06+dDh3msZfOLRTNnkXsLCCrK6WIZYMRHATWnz4nqfoEhHy007/MQ5z8T6aVMWX46WtBxUyOv5sXnSLDdfBuRr9+4pUFATjM+Rv5GO+2JRdzmpka2APaI5To2bXjBG++TsPyNJPnMz8HoO+qG/19bP+BXNVSabUbQIDAQAB',
    );
  });

  it('declares the side panel, the toolbar action and their permissions', () => {
    expect(source.permissions).toEqual([
      'debugger',
      'tabs',
      'storage',
      'sidePanel',
      'alarms',
    ]);
    expect(source.side_panel).toEqual({ default_path: 'panel.html' });
    expect(source.action).toEqual({ default_title: 'Labs ASP' });
    const built = buildManifest(source, 'https://app.example.org');
    expect(built.side_panel).toEqual(source.side_panel);
    expect(built.action).toEqual(source.action);
    expect(built.permissions).toEqual(source.permissions);
  });
});

describe('panelEsbuildArgs', () => {
  it('bundles the panel as an ESM module with the panel-only cn alias', () => {
    expect(panelEsbuildArgs('/repo', 'out/panel.js')).toEqual([
      '/repo/extension/panel/main.tsx',
      '--bundle',
      '--format=esm',
      '--jsx=automatic',
      '--outfile=out/panel.js',
      '--define:process.env.NODE_ENV="production"',
      '--define:__APP_ORIGIN__=""',
      '--alias:@/lib/utils=/repo/extension/panel/lib/cn.ts',
      '--alias:@=/repo',
    ]);
  });

  it('defines the app origin for extension/src/config.ts', () => {
    expect(
      panelEsbuildArgs('/repo', 'out/panel.js', undefined, 'https://app.test'),
    ).toContain('--define:__APP_ORIGIN__="https://app.test"');
  });

  it('writes a metafile only when asked', () => {
    expect(
      panelEsbuildArgs('/repo', 'out/panel.js', '/tmp/meta.json').at(-1),
    ).toBe('--metafile=/tmp/meta.json');
  });
});

describe('extractTokens', () => {
  const globals = readFileSync(path.join(ROOT, 'app/globals.css'), 'utf8');

  it('keeps the light and dark token blocks of app/globals.css', () => {
    const tokens = extractTokens(globals);
    expect(tokens).toMatch(/:root\s*\{[^}]*--primary:\s*320 47% 47%;/);
    expect(tokens).toMatch(/:root\s*\{[^}]*--radius:\s*0\.75rem;/);
    expect(tokens).toMatch(/\.dark\s*\{[^}]*--background:/);
  });

  it('drops the Next-only rules around them', () => {
    const tokens = extractTokens(globals);
    for (const rule of [
      'body',
      'font-family',
      '@apply',
      '@tailwind',
      '@media',
      '--foreground-rgb',
      '.skeleton',
      'scrollbar',
    ]) {
      expect(tokens).not.toContain(rule);
    }
  });

  it('refuses a stylesheet without the token block', () => {
    expect(() => extractTokens('body { color: red; }')).toThrow(
      /no @layer base/,
    );
  });
});

describe('esbuildArgs', () => {
  it('keeps the bundle flags and defines APP_ORIGIN and the test hooks', () => {
    expect(esbuildArgs('in.ts', 'out/sw.js', 'https://a.example')).toEqual([
      'in.ts',
      '--bundle',
      '--format=esm',
      '--outfile=out/sw.js',
      '--define:__APP_ORIGIN__="https://a.example"',
      '--define:EXTENSION_TEST_HOOKS=false',
    ]);
    expect(esbuildArgs('in.ts', 'out/sw.js', '', true).at(-1)).toBe(
      '--define:EXTENSION_TEST_HOOKS=true',
    );
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

describe('resolveTestHooks', () => {
  it('is on only for EXTENSION_TEST_HOOKS=1 or --test-hooks', () => {
    expect(resolveTestHooks({}, [])).toBe(false);
    expect(resolveTestHooks({ EXTENSION_TEST_HOOKS: 'true' }, [])).toBe(false);
    expect(resolveTestHooks({ EXTENSION_TEST_HOOKS: '0' }, [])).toBe(false);
    expect(resolveTestHooks({ EXTENSION_TEST_HOOKS: '1' }, [])).toBe(true);
    expect(resolveTestHooks({}, ['--test-hooks'])).toBe(true);
  });

  it('refuses test hooks in a release build', () => {
    expect(() =>
      resolveTestHooks(
        { EXTENSION_TEST_HOOKS: '1', EXTENSION_RELEASE: '1' },
        [],
      ),
    ).toThrow(/not allowed when EXTENSION_RELEASE=1/);
    expect(() =>
      resolveTestHooks({ EXTENSION_RELEASE: '1' }, ['--test-hooks']),
    ).toThrow(/not allowed when EXTENSION_RELEASE=1/);
  });
});

describe('resolveOriginHoldMs', () => {
  it('is undefined (the bridge keeps 60 s) when unset', () => {
    expect(resolveOriginHoldMs({}, false)).toBeUndefined();
    expect(resolveOriginHoldMs({}, true)).toBeUndefined();
    expect(
      resolveOriginHoldMs({ EXTENSION_TEST_ORIGIN_HOLD_MS: '' }, true),
    ).toBeUndefined();
  });

  it('reads a whole number of ms, only for a test-hooks build', () => {
    expect(
      resolveOriginHoldMs({ EXTENSION_TEST_ORIGIN_HOLD_MS: '1500' }, true),
    ).toBe(1500);
    expect(() =>
      resolveOriginHoldMs({ EXTENSION_TEST_ORIGIN_HOLD_MS: '1500' }, false),
    ).toThrow(/only with the test hooks/);
  });

  it.each(['0', '-5', '1.5', 'abc', '600001', '1e3'])('refuses %j', (value) => {
    expect(() =>
      resolveOriginHoldMs({ EXTENSION_TEST_ORIGIN_HOLD_MS: value }, true),
    ).toThrow(/EXTENSION_TEST_ORIGIN_HOLD_MS/);
  });

  it('defines the hold only when given, and only with the hooks', () => {
    expect(esbuildArgs('in.ts', 'out/sw.js', '', true, 1500).at(-1)).toBe(
      '--define:__ORIGIN_HOLD_MS__=1500',
    );
    expect(esbuildArgs('in.ts', 'out/sw.js', '', false, 1500)).not.toContain(
      '--define:__ORIGIN_HOLD_MS__=1500',
    );
    expect(
      esbuildArgs('in.ts', 'out/sw.js', '', true).some((a) =>
        a.includes('__ORIGIN_HOLD_MS__'),
      ),
    ).toBe(false);
  });
});

describe('the built service worker', () => {
  const HOOKS = ['bridgeHandleMessage', 'bridgeOnDebuggerDetach'];

  /** Runs scripts/build-extension.ts into a temp dir; returns the bundle. */
  const build = (extra: Record<string, string>, argv: string[] = []) => {
    const dir = mkdtempSync(path.join(tmpdir(), 'extension-build-test-'));
    try {
      const {
        APP_ORIGIN: _o,
        EXTENSION_RELEASE: _r,
        EXTENSION_TEST_HOOKS: _h,
        ...env
      } = process.env;
      execFileSync(
        path.join(ROOT, 'node_modules/.bin/tsx'),
        ['scripts/build-extension.ts', '--out', dir, ...argv],
        { cwd: ROOT, env: { ...env, ...extra }, stdio: 'ignore' },
      );
      return readFileSync(path.join(dir, 'service-worker.js'), 'utf8');
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  };

  it('has no test hooks in a plain build', () => {
    const bundle = build({});
    for (const hook of HOOKS) expect(bundle).not.toContain(hook);
    expect(bundle).not.toContain('EXTENSION_TEST_HOOKS');
  }, 30_000);

  it('has the test hooks with EXTENSION_TEST_HOOKS=1 or --test-hooks', () => {
    for (const bundle of [
      build({ EXTENSION_TEST_HOOKS: '1' }),
      build({}, ['--test-hooks']),
    ]) {
      for (const hook of HOOKS) expect(bundle).toContain(hook);
    }
  }, 30_000);

  it('bakes a shortened origin hold into a test-hooks build only', () => {
    const hooks = build({
      EXTENSION_TEST_HOOKS: '1',
      EXTENSION_TEST_ORIGIN_HOLD_MS: '1234',
    });
    expect(hooks).not.toContain('__ORIGIN_HOLD_MS__');
    expect(hooks).toMatch(/\b1234\b/);
    expect(() => build({ EXTENSION_TEST_ORIGIN_HOLD_MS: '1234' })).toThrow();
  }, 30_000);
});
