import { defineConfig } from 'vitest/config';
import path from 'node:path';
import react from '@vitejs/plugin-react';

const OPTIMIZE_DEPS = [
  'react',
  'react-dom',
  'react/jsx-dev-runtime',
  '@radix-ui/react-alert-dialog',
  '@radix-ui/react-checkbox',
  '@radix-ui/react-collapsible',
  '@radix-ui/react-dialog',
  '@radix-ui/react-dropdown-menu',
  '@radix-ui/react-label',
  '@radix-ui/react-radio-group',
  '@radix-ui/react-select',
  '@radix-ui/react-switch',
  '@radix-ui/react-tooltip',
];

const EXCLUDE = [
  '**/node_modules/**',
  '**/dist/**',
  '**/cypress/**',
  '**/.{idea,git,cache,output,temp}/**',
  '**/{karma,rollup,webpack,vite,vitest,jest,ava,babel,nyc,cypress,tsup,build}.config.*',
  '**/tests/e2e/**',
  '**/tests/routes/**',
  '**/lib/ai/models.test.ts', // This is not a test file, it exports mocks
  // Node-only tests (node:fs, eve/tools) — run via vitest.config.node.mjs.
  // They crash the browser runner on import, so keep them out of `pnpm test`.
  '**/tests/agent/**',
  '**/.eve/**', // transient `eve dev` runtime snapshots
];

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './'),
    },
  },
  // Pre-bundle UI deps so the optimizer doesn't discover them mid-run and force
  // a reload (which nulls React and breaks Radix-based component tests).
  // Repeated on the browser project below: a project's own Vite config wins
  // over the root's, so relying on inheritance here silently drops it.
  optimizeDeps: { include: OPTIMIZE_DEPS },
  test: {
    globals: true,
    exclude: EXCLUDE,
    projects: [
      {
        // Default: component tests run in a real browser.
        extends: true,
        optimizeDeps: { include: OPTIMIZE_DEPS },
        test: {
          name: 'browser',
          setupFiles: ['./tests/setup.ts'],
          include: ['tests/client/**/*.test.{ts,tsx}'],
          // A project-level `exclude` replaces the root one rather than
          // merging, so the shared entries have to be repeated here.
          // Server-only modules (node:child_process etc.) cannot resolve in
          // the browser bundle; those files opt into the `node` project below.
          exclude: [...EXCLUDE, 'tests/client/**/*.node.test.ts'],
          browser: {
            enabled: true,
            provider: 'playwright',
            instances: [{ browser: 'chromium' }],
          },
        },
      },
      {
        extends: true,
        test: {
          name: 'node',
          environment: 'node',
          include: ['tests/client/**/*.node.test.ts'],
          exclude: EXCLUDE,
        },
      },
    ],
  }
});
