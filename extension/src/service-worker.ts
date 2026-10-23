/// <reference types="chrome" />
import {
  handleMessage,
  onDebuggerDetach,
  onDebuggerEvent,
  onTabRemoved,
  onTabUpdated,
  resumeRun,
} from './bridge';

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

// The worker cannot message itself, so tests (via Playwright's worker handle)
// and future extension pages in this context call the handler directly. Only
// code running inside the extension can reach this global.
(
  globalThis as { bridgeHandleMessage?: typeof handleMessage }
).bridgeHandleMessage = handleMessage;

void resumeRun();
