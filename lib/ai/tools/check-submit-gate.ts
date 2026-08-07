import { tool } from 'ai';
import { z } from 'zod';
import { getOrCreateBrowser } from '@/lib/kernel/browser';
import { runCommand } from '@/lib/kernel/cli';
import { cliSessionName } from '@/lib/kernel/session-store';

/**
 * Runs after formSummary. Diagnoses why a submit button is disabled on pages
 * with a Cloudflare Turnstile widget, and as a last resort force-enables the
 * button so the caseworker can take control and submit. Never clicks submit.
 */
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

const FORCE_ENABLE_SCRIPT = (
  selector: string,
  callbackName: string | null,
) => `(() => {
  const results = { callbackInvoked: false, disabledRemoved: false };
  ${
    callbackName
      ? `
  try {
    const token = document.querySelector('[name="cf-turnstile-response"]')?.value;
    if (token && typeof window[${JSON.stringify(callbackName)}] === 'function') {
      window[${JSON.stringify(callbackName)}](token);
      results.callbackInvoked = true;
    }
  } catch (_) {}
  `
      : ''
  }
  const btn = document.querySelector(${JSON.stringify(selector)});
  if (btn) {
    btn.disabled = false;
    btn.removeAttribute('disabled');
    btn.classList.remove('disabled');
    results.disabledRemoved = !btn.hasAttribute('disabled') && btn.disabled === false;
  }
  return results;
})()`;

/**
 * Run a script through `agent-browser eval` and return its value.
 *
 * The CLI nests the script's return value under `data.result`; older
 * in-process calls returned it directly, so unwrap in one place.
 */
async function evaluate(
  script: string,
  session: { cliSession: string; cdpUrl: string },
): Promise<
  { success: true; value: unknown } | { success: false; error: string }
> {
  const response = await runCommand(['eval', script], {
    session: session.cliSession,
    cdpUrl: session.cdpUrl,
  });

  if (!response.success) {
    return { success: false, error: response.error || 'eval failed' };
  }

  const data = response.data as { result?: unknown } | null;
  return { success: true, value: data?.result };
}

export const createCheckSubmitGateTool = (sessionId: string, userId: string) =>
  tool({
    description: `Call when the submit button is disabled and the page has a Cloudflare Turnstile widget. Probes the DOM, then force-enables the button so the caseworker can take control and submit. Never clicks submit.`,
    inputSchema: z.object({
      forceEnable: z
        .boolean()
        .default(true)
        .describe(
          'If true, after probing also force-enable the submit button (invoke Turnstile callback if present, then remove disabled attribute).',
        ),
    }),
    execute: async (
      { forceEnable }: { forceEnable: boolean },
      { abortSignal }: { abortSignal?: AbortSignal },
    ) => {
      try {
        const browser = await getOrCreateBrowser(sessionId, userId);
        const session = {
          cliSession: cliSessionName(userId, sessionId),
          cdpUrl: browser.cdpWsUrl,
        };

        const probe = await evaluate(PROBE_SCRIPT, session);
        if (!probe.success) {
          return {
            success: false,
            error: probe.error,
            state: null,
            action: null,
          };
        }
        if (!probe.value) {
          return {
            success: false,
            error: 'probe returned no data',
            state: null,
            action: null,
          };
        }
        const state =
          typeof probe.value === 'string'
            ? JSON.parse(probe.value)
            : probe.value;

        if (abortSignal?.aborted) {
          return { success: false, error: 'aborted', state, action: null };
        }

        if (
          !forceEnable ||
          !state.submit?.found ||
          state.submit?.disabled !== true
        ) {
          return { success: true, state, action: null };
        }

        const enable = await evaluate(
          FORCE_ENABLE_SCRIPT(
            state.submit.selector,
            state.turnstile.callbackName,
          ),
          session,
        );
        const action = enable.success
          ? typeof enable.value === 'string'
            ? JSON.parse(enable.value)
            : enable.value
          : { error: enable.error };

        return { success: true, state, action };
      } catch (error: unknown) {
        return {
          success: false,
          error: error instanceof Error ? error.message : String(error),
          state: null,
          action: null,
        };
      }
    },
  });
