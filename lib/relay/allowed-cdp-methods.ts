// The CDP methods agent-browser 0.33.2 sends for every command the Eve
// `browser` tool documents, measured by tests/relay/e2e/allowlist.e2e.test.ts.
// That test fails if this set and the measured traffic drift apart. Cookie,
// storage, Fetch, download-behavior and browser-level Target/Browser methods
// stay out: the spec denies them.
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
