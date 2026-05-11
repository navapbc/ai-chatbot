import type { EnableSubmitResult } from './enable-submit-types';

export type RunCommand = (cmd: Record<string, unknown>) => Promise<{
  success: boolean;
  output?: string;
  error?: string | null;
}>;

export type PhaseInput = {
  runCommand: RunCommand;
  submitSelector?: string;
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
