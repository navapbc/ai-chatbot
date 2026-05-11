export type EnableSubmitResult =
  | { status: 'enabled' }
  | { status: 'enabled-via-force'; warning: string }
  | { status: 'blocked-missing-fields'; fields: string[] }
  | { status: 'pending-turnstile'; message: string }
  | { status: 'blocked-unknown'; diagnostic: Record<string, unknown> }
  | { status: 'browser-error'; error: string };

export type ProgressLabel =
  | 'Checking required fields'
  | 'Opening sections to acknowledge'
  | 'Trying to enable the submit button'
  | string;

export type EmitFn = (label: ProgressLabel) => void;

export type EnableSubmitContext = {
  sessionId: string;
  userId: string;
  submitSelector?: string;
  abortSignal?: AbortSignal;
  emit: EmitFn;
};
