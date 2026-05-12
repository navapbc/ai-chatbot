import { tool, type ToolExecutionOptions } from 'ai';
import { nanoid } from 'nanoid';
import { z } from 'zod';
import { executeCommand } from 'agent-browser/dist/actions.js';
import type { Command } from 'agent-browser/dist/types.js';
import { getOrCreateBrowser } from '@/lib/kernel/browser';
import { withSessionQueue } from './browser';
import type { EnableSubmitResult, EmitFn } from './enable-submit-types';
import {
  phase0LocateButton,
  phase1CheckRequiredFields,
  phase2ExpandSections,
  phase3WaitForTurnstile,
  phase4Verify,
  phase5Diagnose,
  phase6ForceEnable,
  type RunCommand,
  type Phase3Opts,
} from './enable-submit-phases';

const TOOL_TIMEOUT_MS = 90_000;

export type OrchestratorInput = {
  runCommand: RunCommand;
  emit: EmitFn;
  abortSignal: AbortSignal | undefined;
  submitSelector?: string;
  _generateText?: import('./enable-submit-phases').GenerateTextFn;
  _model?: import('ai').LanguageModel;
  phase3Opts?: Pick<Phase3Opts, 'tickMs' | 'maxTicks' | '_sleep'>;
};

export async function runEnableSubmit(input: OrchestratorInput): Promise<EnableSubmitResult> {
  const { runCommand, emit, submitSelector, _generateText, _model } = input;

  const p0 = await phase0LocateButton({ runCommand, submitSelector });
  if (p0.outcome) return p0.outcome;
  const selector = p0.submitSelector!;

  emit('Checking required fields');
  const p1 = await phase1CheckRequiredFields({ runCommand, _generateText, _model });
  if (p1.outcome) return p1.outcome;

  emit('Opening sections to acknowledge');
  await phase2ExpandSections({ runCommand, _generateText, _model });

  const p3 = await phase3WaitForTurnstile(
    { runCommand, submitSelector: selector },
    { emit, ...(input.phase3Opts ?? {}) },
  );
  if (p3.outcome) return p3.outcome;

  const p4 = await phase4Verify({ runCommand, submitSelector: selector });
  if (p4.outcome) return p4.outcome;

  const p5 = await phase5Diagnose({ runCommand });
  if (p5.outcome) return p5.outcome;

  emit('Trying to enable the submit button');
  const snap = await runCommand({ action: 'snapshot', selector: 'form' });
  const lastSnapshot = snap.output ?? '';
  const p6 = await phase6ForceEnable({
    runCommand,
    submitSelector: selector,
    tokenPresent: p5.tokenPresent,
    lastSnapshot,
  });
  return p6.outcome ?? { status: 'blocked-unknown', diagnostic: { reason: 'no-result' } };
}

export const createEnableSubmitTool = (sessionId: string, userId: string) =>
  tool({
    description: `Enable the final submit button on a benefits-application form so the caseworker can review and click it.

Runs a deterministic diagnose-and-enable sequence. Returns one of:
- "enabled": button is now enabled. Proceed to formSummary.
- "enabled-via-force": button is enabled but Turnstile may be incomplete. Relay the warning to the caseworker.
- "blocked-missing-fields": list of human-readable labels that need to be filled. Route through gapAnalysis.
- "pending-turnstile": Turnstile token has not populated. Relay the message; wait and retry.
- "blocked-unknown": could not enable. Surface the diagnostic to the caseworker.
- "browser-error": browser command failed.

This tool NEVER clicks the submit button. It only enables it.`,
    inputSchema: z.object({
      submitSelector: z
        .string()
        .optional()
        .describe('Optional ref or CSS selector for the submit button; auto-detected if omitted.'),
    }),
    execute: async (
      { submitSelector }: { submitSelector?: string },
      { abortSignal }: ToolExecutionOptions,
    ): Promise<EnableSubmitResult> => {
      return withSessionQueue(sessionId, async () => {
        try {
          const session = await getOrCreateBrowser(sessionId, userId);

          const runCommand: RunCommand = async (cmd) => {
            const command = { id: nanoid(), ...cmd } as Command;
            const response = await executeCommand(command, session.browserManager);
            if (response.success) {
              const output =
                typeof response.data === 'string'
                  ? response.data
                  : JSON.stringify(response.data);
              return { success: true, output };
            }
            return { success: false, error: response.error ?? 'unknown' };
          };

          const emit: EmitFn = () => {
            // Wired to dataStream in Task 10
          };

          return await Promise.race([
            runEnableSubmit({ runCommand, emit, abortSignal, submitSelector }),
            new Promise<EnableSubmitResult>((resolve) =>
              setTimeout(
                () =>
                  resolve({
                    status: 'blocked-unknown',
                    diagnostic: { reason: 'timeout', timeoutMs: TOOL_TIMEOUT_MS },
                  }),
                TOOL_TIMEOUT_MS,
              ),
            ),
          ]);
        } catch (err) {
          const message = err instanceof Error ? err.message : String(err);
          if (abortSignal?.aborted || message.includes('stopped by user')) {
            return { status: 'browser-error', error: 'stopped by user' };
          }
          return { status: 'browser-error', error: message };
        }
      });
    },
  });
