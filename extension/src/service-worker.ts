/// <reference types="chrome" />
import {
  handleMessage,
  onDebuggerDetach,
  onDebuggerEvent,
  onTabRemoved,
  onTabUpdated,
  resumeRun,
} from './bridge';
import { installTestHooks } from './test-hooks';

// Listeners are registered synchronously at top level so a service worker
// woken by an event receives it.
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  // No content scripts exist; only the extension's own pages may start runs.
  if (sender.id !== chrome.runtime.id) return false;
  handleMessage(message).then(sendResponse, () =>
    sendResponse({ ok: false, error: 'bridge error' }),
  );
  return true;
});

chrome.debugger.onEvent.addListener(onDebuggerEvent);
chrome.debugger.onDetach.addListener(onDebuggerDetach);
chrome.tabs.onUpdated.addListener(onTabUpdated);
chrome.tabs.onRemoved.addListener(onTabRemoved);

// esbuild defines this (scripts/build-extension.ts): true only for a build
// with EXTENSION_TEST_HOOKS=1, false otherwise, so a plain build drops the
// hooks entirely. Undeclared outside a bundle (a unit test): no hooks.
declare const EXTENSION_TEST_HOOKS: boolean | undefined;

if (typeof EXTENSION_TEST_HOOKS !== 'undefined' && EXTENSION_TEST_HOOKS) {
  installTestHooks(globalThis, handleMessage, onDebuggerDetach);
}

void resumeRun();
