import { tool, type ToolExecutionOptions } from 'ai';
import { z } from 'zod';
import type { EnableSubmitResult } from './enable-submit-types';

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
      void submitSelector;
      void abortSignal;
      void sessionId;
      void userId;
      return { status: 'blocked-unknown', diagnostic: { reason: 'not-implemented' } };
    },
  });
