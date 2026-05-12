import { z } from 'zod';
import type { EnableSubmitResult } from './enable-submit-types';

export type RunCommand = (cmd: Record<string, unknown>) => Promise<{
  success: boolean;
  output?: string;
  error?: string | null;
}>;

export type GenerateTextFn = typeof import('ai').generateText;

export type PhaseInput = {
  runCommand: RunCommand;
  submitSelector?: string;
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
