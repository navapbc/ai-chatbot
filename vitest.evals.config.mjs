import { defineConfig } from 'vitest/config';
import path from 'path';
import dotenv from 'dotenv';

// Load .env.local so Vertex AI credentials are available
dotenv.config({ path: path.resolve(__dirname, '.env.local') });

export default defineConfig({
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './'),
    },
  },
  test: {
    globals: true,
    include: ['tests/evals/**/*.eval.ts'],
    testTimeout: 120_000,
    // Evals hit real APIs — run sequentially to avoid rate limits
    pool: 'forks',
    poolOptions: {
      forks: { singleFork: true },
    },
  },
});
