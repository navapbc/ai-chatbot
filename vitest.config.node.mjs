import { defineConfig } from 'vitest/config';
import path from 'node:path';

export default defineConfig({
  resolve: {
    alias: {
      '@': path.resolve(process.cwd(), './'),
    },
  },
  test: {
    environment: 'node',
    globals: true,
    // tests/relay/** includes the browser e2e suite (tests/relay/e2e); run
    // tests/agent alone with `pnpm exec vitest run --config vitest.config.node.mjs tests/agent`.
    include: ['tests/agent/**/*.test.ts', 'tests/relay/**/*.test.ts'],
  },
});
