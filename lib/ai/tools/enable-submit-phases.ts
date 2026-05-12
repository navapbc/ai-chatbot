import { z } from 'zod';
import type { EnableSubmitResult, EmitFn } from './enable-submit-types';
import { submitGates } from './enable-submit-gates';

export type RunCommand = (cmd: Record<string, unknown>) => Promise<{
  success: boolean;
  output?: string;
  error?: string | null;
}>;

export type GenerateTextFn = typeof import('ai').generateText;

export type PhaseInput = {
  runCommand: RunCommand;
  submitSelector?: string;
  abortSignal?: AbortSignal;
  // Underscore-prefixed fields are test-only injection points; production calls leave them undefined.
  _generateText?: GenerateTextFn;
  _model?: import('ai').LanguageModel;
};

export type Phase0Output = {
  outcome: EnableSubmitResult | null;
  submitSelector?: string;
};

const SUBMIT_LABEL_RE = /submit|apply|send|finish/i;

function parseRefFromSnapshot(snapshot: string): string | null {
  const candidates: { ref: string; disabled: boolean }[] = [];
  for (const rawLine of snapshot.split('\n')) {
    const line = rawLine.trimStart();
    const match = line.match(/^-?\s*button\s+"([^"]*)".*\[ref=(e\d+)\]/);
    if (!match) continue;
    const [, label, refNum] = match;
    if (!SUBMIT_LABEL_RE.test(label)) continue;
    const disabled = /\[disabled\]/i.test(line);
    candidates.push({ ref: `@${refNum}`, disabled });
  }
  if (candidates.length === 0) return null;
  const disabledOne = candidates.find((c) => c.disabled);
  return disabledOne ? disabledOne.ref : candidates[candidates.length - 1].ref;
}

function snapshotShowsDisabled(snapshot: string, selector: string): boolean {
  if (selector.startsWith('@e')) {
    const refNum = selector.slice(1);
    const pattern = new RegExp(`\\[ref=${refNum}\\b[^\\]]*\\]`);
    const line = snapshot.split('\n').find((l) => pattern.test(l));
    if (!line) return false;
    return /\[disabled\]/i.test(line);
  }
  return false;
}

export async function phase0LocateButton({
  runCommand,
  submitSelector,
}: PhaseInput): Promise<Phase0Output> {
  const snap = await runCommand({ action: 'snapshot', selector: 'form' });
  if (!snap.success || !snap.output) {
    return { outcome: { status: 'browser-error', error: snap.error ?? 'snapshot failed' } };
  }

  const selector = submitSelector ?? parseRefFromSnapshot(snap.output);
  if (!selector) {
    return {
      outcome: {
        status: 'blocked-unknown',
        diagnostic: { reason: 'submit-button-not-found' },
      },
    };
  }

  if (!snapshotShowsDisabled(snap.output, selector)) {
    return { outcome: { status: 'enabled' }, submitSelector: selector };
  }

  return { outcome: null, submitSelector: selector };
}

export type Phase1Output = { outcome: EnableSubmitResult | null };

const missingFieldsSchema = z.object({
  missing: z.array(z.string()).describe('Human-readable labels of required fields that are empty or have an error message.'),
});

const PHASE1_PROMPT = `You are reviewing a form snapshot to find required fields that are NOT filled in correctly.

Return the human-readable LABEL (not the ref) of each required field that:
- Is marked required (asterisk, "required" text, aria-required) AND is empty, OR
- Has a visible error message indicating an invalid or missing value.

Return ONLY labels. Ignore optional fields. Ignore CAPTCHA/Turnstile widgets.
If everything looks filled, return an empty list.`;

export async function phase1CheckRequiredFields({
  runCommand,
  _generateText,
  _model,
}: PhaseInput): Promise<Phase1Output> {
  const snap = await runCommand({ action: 'snapshot', selector: 'form' });
  if (!snap.success || !snap.output) {
    return { outcome: { status: 'browser-error', error: snap.error ?? 'snapshot failed' } };
  }

  const ai = await import('ai');
  const gen: GenerateTextFn = _generateText ?? ai.generateText;
  const model = _model ?? (await import('@/lib/ai/providers')).prepareStepModel;

  try {
    const result = await gen({
      model,
      prompt: `${PHASE1_PROMPT}\n\nSNAPSHOT:\n${snap.output}`,
      output: ai.Output.object({ schema: missingFieldsSchema }),
    });
    const missing = result.output.missing;
    if (missing.length > 0) {
      return { outcome: { status: 'blocked-missing-fields', fields: missing } };
    }
    return { outcome: null };
  } catch (err) {
    console.warn('[enable-submit] phase1 generateText failed; assuming no missing fields', err);
    return { outcome: null };
  }
}

export type Phase2Output = { outcome: EnableSubmitResult | null };

const expandSchema = z.object({
  refs: z.array(z.string()).describe('Refs (like @e5) of collapsible sections that look unopened.'),
});

const PHASE2_PROMPT = `Find collapsible sections in this form snapshot that look UNOPENED and likely need to be expanded for the form to submit.

Look for:
- Buttons/links with labels like "+ Expand", "Show more", "Please expand and read"
- Sections marked with chevrons or +/- icons in a closed state
- Acknowledgment blocks the user must open to read

Return the refs (like @e5) of each. If none, return an empty list.`;

export async function phase2ExpandSections({
  runCommand,
  _generateText,
  _model,
}: PhaseInput): Promise<Phase2Output> {
  const snap = await runCommand({ action: 'snapshot', selector: 'form' });
  if (!snap.success || !snap.output) {
    return { outcome: { status: 'browser-error', error: snap.error ?? 'snapshot failed' } };
  }

  const ai = await import('ai');
  const gen: GenerateTextFn = _generateText ?? ai.generateText;
  const model = _model ?? (await import('@/lib/ai/providers')).prepareStepModel;

  let refs: string[];
  try {
    const result = await gen({
      model,
      prompt: `${PHASE2_PROMPT}\n\nSNAPSHOT:\n${snap.output}`,
      output: ai.Output.object({ schema: expandSchema }),
    });
    refs = result.output.refs;
  } catch (err) {
    console.warn('[enable-submit] phase2 generateText failed; skipping expand', err);
    return { outcome: null };
  }

  if (refs.length === 0) return { outcome: null };

  for (const ref of refs) {
    await runCommand({ action: 'click', selector: ref });
  }
  await runCommand({ action: 'snapshot', selector: 'form' });
  return { outcome: null };
}

export type Phase3Opts = {
  tickMs?: number;
  maxTicks?: number;
  emit?: EmitFn;
  _sleep?: (ms: number) => Promise<void>;
};

const TOKEN_READ_SCRIPT = "document.querySelector('[name=cf-turnstile-response]')?.value || ''";

function disabledReadScript(selector: string): string {
  if (selector.startsWith('@')) {
    return `(function(){const els=document.querySelectorAll('button,input[type=submit]');for(const el of els){if(/submit|apply|send|finish/i.test(el.textContent||el.value||'')){return el.disabled?'true':'false';}}return 'unknown';})()`;
  }
  return `document.querySelector(${JSON.stringify(selector)})?.disabled === true ? 'true' : 'false'`;
}

export async function phase3WaitForTurnstile(
  input: PhaseInput & { submitSelector: string },
  opts: Phase3Opts = {},
): Promise<{ outcome: EnableSubmitResult | null }> {
  const tickMs = opts.tickMs ?? 8000;
  const maxTicks = opts.maxTicks ?? 4;
  const emit = opts.emit ?? (() => {});
  const sleep = opts._sleep ?? ((ms: number) => new Promise((r) => setTimeout(r, ms)));

  for (let tick = 1; tick <= maxTicks; tick++) {
    if (input.abortSignal?.aborted) throw new Error('stopped by user');
    emit(`Waiting for security check (${(tickMs * tick) / 1000}s)`);
    await sleep(tickMs);
    if (input.abortSignal?.aborted) throw new Error('stopped by user');

    const tokenRes = await input.runCommand({ action: 'evaluate', script: TOKEN_READ_SCRIPT });
    const disRes = await input.runCommand({
      action: 'evaluate',
      script: disabledReadScript(input.submitSelector),
    });

    if (disRes.success && disRes.output === 'false') {
      return { outcome: { status: 'enabled' } };
    }
    void tokenRes;
  }

  return { outcome: null };
}

export async function phase4Verify({
  runCommand,
  submitSelector,
}: PhaseInput & { submitSelector: string }): Promise<{ outcome: EnableSubmitResult | null }> {
  const snap = await runCommand({ action: 'snapshot', selector: 'form' });
  if (!snap.success || !snap.output) {
    return { outcome: { status: 'browser-error', error: snap.error ?? 'snapshot failed' } };
  }
  if (!snapshotShowsDisabled(snap.output, submitSelector)) {
    return { outcome: { status: 'enabled' } };
  }
  return { outcome: null };
}

const PENDING_TURNSTILE_MESSAGE = 'Turnstile token is still empty — wait ~30s and try again.';

export async function phase5Diagnose({
  runCommand,
}: PhaseInput): Promise<{ outcome: EnableSubmitResult | null; tokenPresent: boolean }> {
  const tokenRes = await runCommand({ action: 'evaluate', script: TOKEN_READ_SCRIPT });
  const tokenPresent = !!(tokenRes.success && tokenRes.output && tokenRes.output.length > 0);

  if (!tokenPresent) {
    return {
      outcome: { status: 'pending-turnstile', message: PENDING_TURNSTILE_MESSAGE },
      tokenPresent: false,
    };
  }
  return { outcome: null, tokenPresent: true };
}

export type Phase6Input = PhaseInput & {
  submitSelector: string;
  tokenPresent: boolean;
  lastSnapshot: string;
};

export async function phase6ForceEnable(input: Phase6Input): Promise<{ outcome: EnableSubmitResult | null }> {
  if (!input.tokenPresent) {
    return {
      outcome: {
        status: 'pending-turnstile',
        message: 'Turnstile token is still empty — refusing to force-enable.',
      },
    };
  }

  for (const gate of submitGates) {
    if (!gate.match(input.lastSnapshot)) continue;

    await input.runCommand({ action: 'evaluate', script: gate.script(input.submitSelector) });
    const snap = await input.runCommand({ action: 'snapshot', selector: 'form' });
    if (!snap.success || !snap.output) continue;

    if (!snapshotShowsDisabled(snap.output, input.submitSelector)) {
      return {
        outcome: {
          status: 'enabled-via-force',
          warning: `I enabled the button by satisfying client-side gating (pattern: ${gate.name}). The Turnstile token is present; the server should accept the submission.`,
        },
      };
    }
  }

  return {
    outcome: {
      status: 'blocked-unknown',
      diagnostic: {
        reason: 'no-gate-pattern-matched',
        tokenPresent: input.tokenPresent,
        submitSelector: input.submitSelector,
      },
    },
  };
}
