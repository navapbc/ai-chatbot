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
  for (const line of snapshot.split('\n')) {
    if (!line.includes('[button')) continue;
    if (!SUBMIT_LABEL_RE.test(line)) continue;
    const match = line.match(/^@e\d+/);
    if (match) return match[0];
  }
  return null;
}

function snapshotShowsDisabled(snapshot: string, ref: string): boolean {
  const line = snapshot.split('\n').find((l) => l.trim().startsWith(ref));
  if (!line) return false;
  return line.includes('disabled');
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
