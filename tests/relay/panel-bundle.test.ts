import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';

// The side panel bundle (extension/panel) built by scripts/build-extension.ts:
// what it writes, and what its import graph may reach. The panel runs in the
// extension, so nothing from Next, the database or the model SDKs belongs in
// it; components/ui leaf files reach `cn` through the panel's own copy.

const ROOT = path.resolve(__dirname, '../..');
const PINNED_KEY = JSON.parse(
  readFileSync(path.join(ROOT, 'extension/manifest.json'), 'utf8'),
).key;

/** Input paths (relative to the repo root) the panel bundle may not reach. */
const FORBIDDEN_ANYWHERE = [
  /(^|\/)node_modules\/next\//,
  /(^|\/)node_modules\/server-only\//,
  /(^|\/)node_modules\/drizzle-orm\//,
  /(^|\/)node_modules\/@typesafe-ai\//,
  /(^|\/)node_modules\/ai\//,
];
/** Repo source paths the bundle may not reach (outside node_modules). */
const FORBIDDEN_SOURCE = [
  /^lib\/utils\.ts$/,
  /^lib\/db\//,
  /^lib\/ai\//,
  /next\//,
  /server-only/,
  /drizzle/,
];

const cleanups: string[] = [];
afterAll(() => {
  for (const dir of cleanups) rmSync(dir, { recursive: true, force: true });
});

/** Runs the extension build into a temp dir; returns the dir and metafile. */
function build(extra: Record<string, string>) {
  const dir = mkdtempSync(path.join(tmpdir(), 'panel-bundle-test-'));
  cleanups.push(dir);
  const out = path.join(dir, 'out');
  const metafile = path.join(dir, 'panel.meta.json');
  const {
    APP_ORIGIN: _o,
    EXTENSION_RELEASE: _r,
    EXTENSION_TEST_HOOKS: _h,
    ...env
  } = process.env;
  execFileSync(
    path.join(ROOT, 'node_modules/.bin/tsx'),
    ['scripts/build-extension.ts', '--out', out, '--panel-metafile', metafile],
    { cwd: ROOT, env: { ...env, ...extra }, stdio: 'pipe' },
  );
  return { out, metafile };
}

const read = (dir: string, file: string) =>
  readFileSync(path.join(dir, file), 'utf8');

describe('the panel bundle', () => {
  let plain: { out: string; metafile: string };

  beforeAll(() => {
    plain = build({});
  }, 60_000);

  it('writes panel.html, panel.js and panel.css next to the worker', () => {
    for (const file of [
      'panel.html',
      'panel.js',
      'panel.css',
      'service-worker.js',
      'manifest.json',
    ]) {
      expect(read(plain.out, file).length).toBeGreaterThan(0);
    }
  });

  it('keeps the metafile out of the extension directory', () => {
    expect(() => read(plain.out, 'panel.meta.json')).toThrow();
  });

  it('reaches components/ui through the panel cn, never the real lib/utils', () => {
    const inputs = Object.keys(
      JSON.parse(readFileSync(plain.metafile, 'utf8')).inputs,
    );
    // Guards against a vacuous pass over an empty or wrong graph.
    expect(inputs).toContain('extension/panel/main.tsx');
    expect(inputs).toContain('extension/panel/lib/cn.ts');
    expect(inputs).toContain('components/ui/button.tsx');
    expect(inputs.some((i) => /node_modules\/react\//.test(i))).toBe(true);

    const offending = inputs.filter(
      (input) =>
        FORBIDDEN_ANYWHERE.some((p) => p.test(input)) ||
        (!input.startsWith('node_modules/') &&
          FORBIDDEN_SOURCE.some((p) => p.test(input))),
    );
    expect(offending).toEqual([]);
  });

  it('defines NODE_ENV so React does not read process at run time', () => {
    expect(read(plain.out, 'panel.js')).not.toMatch(/process\.env\.NODE_ENV/);
  });

  it('loads only panel.js and panel.css, with no inline script or style', () => {
    const html = read(plain.out, 'panel.html');
    const srcs = [...html.matchAll(/\b(?:src|href)="([^"]*)"/g)].map(
      (m) => m[1],
    );
    expect(srcs.sort()).toEqual(['panel.css', 'panel.js']);
    expect(html).toMatch(/<script type="module" src="panel\.js"><\/script>/);
    for (const match of html.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/g)) {
      expect(match[1].trim()).toBe('');
    }
    expect(html).not.toMatch(/<style\b/);
    expect(html).not.toMatch(/\sstyle=/);
    expect(html).toContain('<div id="root"></div>');
  });

  it('has the app tokens and the utilities the panel and components/ui use', () => {
    const css = read(plain.out, 'panel.css');
    expect(css).toMatch(/--primary:\s*320 47% 47%/);
    expect(css).toMatch(/--radius:\s*0\.75rem/);
    expect(css).toMatch(/\.bg-primary\s*\{/);
    // The indeterminate bar pulses only when motion is allowed.
    expect(css).toContain('motion-safe');
    expect(css).toMatch(/--font-ibm-plex-mono:/);
    expect(css).not.toContain('@tailwind');
    expect(css).not.toContain('@import');
  });

  it('declares the side panel in the manifest and keeps the pinned key', () => {
    const manifest = JSON.parse(read(plain.out, 'manifest.json'));
    expect(manifest.key).toBe(PINNED_KEY);
    expect(manifest.side_panel).toEqual({ default_path: 'panel.html' });
    expect(manifest.action).toEqual({ default_title: 'Labs ASP' });
    expect(manifest.permissions).toEqual(
      expect.arrayContaining(['sidePanel', 'alarms']),
    );
  });

  it('builds the panel in a release build too', () => {
    const release = build({
      EXTENSION_RELEASE: '1',
      APP_ORIGIN: 'https://app.example.org',
    });
    for (const file of ['panel.html', 'panel.js', 'panel.css']) {
      expect(read(release.out, file).length).toBeGreaterThan(0);
    }
    expect(
      JSON.parse(read(release.out, 'manifest.json')).host_permissions,
    ).toEqual(['https://app.example.org/*']);
  }, 60_000);
});
