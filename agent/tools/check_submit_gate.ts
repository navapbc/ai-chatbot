import { defineTool } from 'eve/tools';
import { z } from 'zod';
import { runBrowserCommand } from '@/lib/kernel/eve-browser';

// PROBE_SCRIPT and FORCE_ENABLE_SCRIPT copied verbatim from
// lib/ai/tools/check-submit-gate.ts — do not modify the DOM logic.
const PROBE_SCRIPT = `(() => {
  const tokenInput = document.querySelector('[name="cf-turnstile-response"]');
  const tokenValue = tokenInput && 'value' in tokenInput ? String(tokenInput.value || '') : '';
  const turnstileWidget = document.querySelector('[data-callback], .cf-turnstile, [data-sitekey]');
  const callbackName = turnstileWidget?.getAttribute('data-callback') || null;
  const callbackDefined = callbackName ? typeof window[callbackName] === 'function' : false;

  // Submit candidates — prefer explicit ids, then type=submit, then submit-named buttons.
  const submitEl = document.querySelector('#btnSubmit, button[type="submit"], input[type="submit"], button[name="submit"], input[name="submit"]');
  const submitFound = !!submitEl;
  const submitDisabled = submitEl ? (submitEl.disabled === true || submitEl.hasAttribute('disabled')) : null;
  const submitSelector = submitEl ? (submitEl.id ? '#' + submitEl.id : submitEl.tagName.toLowerCase() + (submitEl.getAttribute('type') ? '[type="' + submitEl.getAttribute('type') + '"]' : '')) : null;

  // Generic collapsed/expand sections (IHSS Section 9 affirmation is a span.header + content pattern).
  const collapsed = Array.from(document.querySelectorAll('[aria-expanded="false"], details:not([open])')).length;
  const expandHeaders = Array.from(document.querySelectorAll('.header, .expand, [class*="collapse"]'))
    .filter(el => el.textContent && /expand|read|affirm/i.test(el.textContent)).length;

  // Required-field error indicator (IHSS surfaces #errorMsgDiv).
  const errorEl = document.querySelector('#errorMsgDiv, .error-message, [role="alert"]');
  const errorVisible = errorEl ? getComputedStyle(errorEl).display !== 'none' : false;

  return {
    turnstile: { tokenPresent: tokenValue.length > 0, tokenLength: tokenValue.length, callbackName, callbackDefined },
    submit: { found: submitFound, disabled: submitDisabled, selector: submitSelector },
    gates: { collapsedSections: collapsed, expandHeaders, errorVisible },
  };
})()`;

const FORCE_ENABLE_SCRIPT = (selector: string, callbackName: string | null) => `(() => {
  const results = { callbackInvoked: false, disabledRemoved: false };
  ${callbackName ? `
  try {
    const token = document.querySelector('[name="cf-turnstile-response"]')?.value;
    if (token && typeof window[${JSON.stringify(callbackName)}] === 'function') {
      window[${JSON.stringify(callbackName)}](token);
      results.callbackInvoked = true;
    }
  } catch (_) {}
  ` : ''}
  const btn = document.querySelector(${JSON.stringify(selector)});
  if (btn) {
    btn.disabled = false;
    btn.removeAttribute('disabled');
    btn.classList.remove('disabled');
    results.disabledRemoved = !btn.hasAttribute('disabled') && btn.disabled === false;
  }
  return results;
})()`;

export default defineTool({
  description:
    'On a page with a Cloudflare Turnstile widget where the submit button is stuck disabled, probe the DOM and (if forceEnable) force-enable the button so the caseworker can take control and submit. Never clicks submit. Do not call on pages without a Turnstile widget.',
  inputSchema: z.object({
    forceEnable: z.boolean().default(true),
  }),
  async execute({ forceEnable }, ctx) {
    try {
      const probe = await runBrowserCommand(ctx, { action: 'evaluate', script: PROBE_SCRIPT });
      if (!probe.success) return { success: false, error: probe.error ?? 'probe failed', state: null, action: null };
      if (!probe.data) return { success: false, error: 'probe returned no data', state: null, action: null };
      const state = typeof probe.data === 'string' ? JSON.parse(probe.data) : probe.data;

      if (!forceEnable || !state.submit?.found || state.submit?.disabled !== true) {
        return { success: true, state, action: null };
      }
      const enable = await runBrowserCommand(ctx, {
        action: 'evaluate',
        script: FORCE_ENABLE_SCRIPT(state.submit.selector, state.turnstile.callbackName),
      });
      const action = enable.success
        ? (typeof enable.data === 'string' ? JSON.parse(enable.data) : enable.data)
        : { error: enable.error ?? 'force-enable failed' };
      return { success: true, state, action };
    } catch (error: unknown) {
      return { success: false, error: error instanceof Error ? error.message : String(error), state: null, action: null };
    }
  },
});
