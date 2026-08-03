import { defineConfig } from 'vitest/config';
import path from 'node:path';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './'),
    },
  },
  // Pre-bundle UI deps so the optimizer doesn't discover them mid-run and force
  // a reload (which nulls React and breaks Radix-based component tests).
  optimizeDeps: {
    include: [
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
    ],
  },
  test: {
    globals: true,
    exclude: [
      '**/node_modules/**',
      '**/dist/**',
      '**/cypress/**',
      '**/.{idea,git,cache,output,temp}/**',
      '**/{karma,rollup,webpack,vite,vitest,jest,ava,babel,nyc,cypress,tsup,build}.config.*',
      '**/tests/e2e/**',
      '**/tests/routes/**',
      '**/lib/ai/models.test.ts', // This is not a test file, it exports mocks
    ],
    projects: [
      {
        // Default: component tests run in a real browser.
        extends: true,
        test: {
          name: 'browser',
          setupFiles: ['./tests/setup.ts'],
          include: ['tests/client/**/*.test.{ts,tsx}'],
          // Server-only modules (node:child_process etc.) cannot resolve in
          // the browser bundle; those files opt into the `node` project below.
          exclude: ['tests/client/**/*.node.test.ts'],
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
        },
      },
    ],
  }
});
