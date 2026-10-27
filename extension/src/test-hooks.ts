import type { handleMessage, onDebuggerDetach } from './bridge';

// The worker cannot message itself, so e2e tests (via Playwright's worker
// handle) call the bridge through these globals. The detach hook lets tests
// simulate the user dismissing Chrome's debugging infobar, which cannot be
// clicked in headless Chromium.
//
// Only a build with EXTENSION_TEST_HOOKS=1 calls this (see service-worker.ts
// and scripts/build-extension.ts). The global names live only in this module
// so a plain build tree-shakes them out entirely.

export function installTestHooks(
  target: object,
  bridgeHandleMessage: typeof handleMessage,
  bridgeOnDebuggerDetach: typeof onDebuggerDetach,
): void {
  Object.assign(target, { bridgeHandleMessage, bridgeOnDebuggerDetach });
}
