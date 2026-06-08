// The vitest browser environment has no Node `process` global, but some app
// modules (e.g. lib/constants.ts) read `process.env.*` at import time. Provide
// a minimal shim so those modules load in tests.
(globalThis as unknown as { process?: { env: Record<string, string | undefined> } }).process ??= {
  env: {},
};
