// The CDP methods agent-browser 0.33.2 sends for the commands the Eve
// `browser` tool documents, as exercised by tests/relay/e2e/allowlist.e2e.test.ts.
// That test fails when a listed method is neither measured nor justified, when
// a measured method is missing here, and when anything the spec denies (cookie,
// storage, Fetch, download-behavior, browser-level Target/Browser methods) is
// listed. It does not prove coverage of commands it does not run.
export const ALLOWED_CDP_METHODS: ReadonlySet<string> = new Set([
  'Runtime.evaluate',
  'Runtime.enable',
  'Runtime.callFunctionOn',
  'Runtime.runIfWaitingForDebugger',
  'Page.enable',
  'Page.navigate',
  'Page.captureScreenshot',
  'Page.handleJavaScriptDialog',
  'Page.getFrameTree',
  'Page.bringToFront',
  'Network.enable',
  'Target.setAutoAttach',
  'DOM.enable',
  'DOM.describeNode',
  'DOM.resolveNode',
  'DOM.getBoxModel',
  'DOM.scrollIntoViewIfNeeded',
  'Accessibility.enable',
  'Accessibility.getFullAXTree',
  'Input.insertText',
  'Input.dispatchKeyEvent',
  'Input.dispatchMouseEvent',
]);
