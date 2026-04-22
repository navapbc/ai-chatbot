import { generateText, tool, type ModelMessage } from 'ai';
import { z } from 'zod';
import { prepareStepModel } from '@/lib/ai/providers';
import { WORKING_MEMORY_PREFIX } from '@/lib/ai/working-memory';

// The agent triggers this; Haiku does the actual extraction. This keeps
// WM updates cheap (Sonnet spends only a tool-call's worth of output
// tokens on the trigger, not on serializing the full state itself).

const EXTRACTION_SYSTEM_PROMPT =
  'You are maintaining a structured working memory document for a ' +
  'benefits form-filling agent. Extract ALL participant data, caseworker ' +
  'answers, household members, and form-filling progress from the ' +
  'transcript below. Merge with any prior working memory and include ' +
  'EVERYTHING currently known — the agent replaces its entire WM with ' +
  'your output, so omissions are lost. Never fabricate values; use ' +
  '[UNKNOWN] for anything not explicitly present.';

const participantRecord = z.record(z.string(), z.unknown());

const workingMemorySchema = z.object({
  participant: participantRecord
    .optional()
    .describe('Primary applicant fields: name, DOB, SSN, gender, race, citizenship, contact info, CalWorks ID, etc.'),
  household: z
    .array(participantRecord)
    .optional()
    .describe('Other household members with their fields and relationships'),
  caseworkerInputs: participantRecord
    .optional()
    .describe('Answers the caseworker provided this session (gap fills, corrections, one-off responses)'),
  formState: z
    .object({
      formName: z.string().optional(),
      currentUrl: z.string().optional(),
      currentStep: z.string().optional(),
      completedFields: z.record(z.string(), z.string()).optional(),
      pendingFields: z.array(z.string()).optional(),
      keyDecisions: z.array(z.string()).optional(),
    })
    .optional()
    .describe('Current form-filling progress and key decisions'),
});

const writeWorkingMemory = tool({
  description: 'Internal tool used by updateWorkingMemory to force structured output.',
  inputSchema: workingMemorySchema,
  execute: async (input) => input,
});

function flattenForExtraction(msg: ModelMessage): string {
  const role = msg.role.toUpperCase();
  if (typeof msg.content === 'string') return `[${role}]: ${msg.content}`;
  if (!Array.isArray(msg.content)) return `[${role}]: ${JSON.stringify(msg.content)}`;
  const parts = (msg.content as any[]).map((part) => {
    if (!part) return '';
    // Strip heavy browser payloads — only status matters for WM extraction
    if (part.type === 'tool-result' && part.toolName === 'browser') {
      const raw = part.output ?? part.result ?? {};
      const r = raw && typeof raw === 'object' && 'value' in raw ? (raw as any).value : raw;
      return `[browser result: ${(r as any)?.success ? 'ok' : 'err'}]`;
    }
    if (part.type === 'tool-call' && part.toolName === 'browser') {
      const a = (part.input ?? part.args ?? {}) as Record<string, any>;
      return `[browser: ${a.action ?? '?'}${a.selector ? ` ${a.selector}` : a.url ? ` ${a.url}` : ''}]`;
    }
    const s = typeof part === 'string' ? part : (part.text ?? JSON.stringify(part) ?? '');
    return String(s);
  });
  return `[${role}]: ${parts.join('\n')}`;
}

function extractCurrentWm(messages: ModelMessage[]): string {
  // Find the most recent WM (seed message at top, or prior tool result)
  const first = messages[0];
  if (
    first &&
    typeof first.content === 'string' &&
    first.content.startsWith(WORKING_MEMORY_PREFIX)
  ) {
    return first.content;
  }
  return '(no prior working memory)';
}

export const updateWorkingMemory = tool({
  description:
    'Refresh the structured working memory with the latest participant data, ' +
    'caseworker answers, household members, and form-filling progress. ' +
    'Call this after meaningful state changes (caseworker answers, data ' +
    'fetches, completing a form section) — NOT after every browser action. ' +
    'A Haiku model reads the recent transcript and returns the updated WM, ' +
    'so you do not need to serialize the state yourself.',
  inputSchema: z.object({
    reason: z
      .string()
      .describe(
        'Short note on what just changed (e.g. "finished address section", "caseworker provided SSN"). Helps focus the extraction.',
      ),
  }),
  execute: async ({ reason }, options) => {
    const t0 = Date.now();
    const priorWm = extractCurrentWm(options.messages);
    const transcript = options.messages.map(flattenForExtraction).join('\n\n');

    try {
      const result = await generateText({
        model: prepareStepModel,
        maxOutputTokens: 4096,
        system: EXTRACTION_SYSTEM_PROMPT,
        tools: { writeWorkingMemory },
        toolChoice: { type: 'tool', toolName: 'writeWorkingMemory' },
        messages: [
          {
            role: 'user',
            content:
              `Trigger reason: ${reason}\n\n` +
              `Prior working memory:\n${priorWm}\n\n` +
              `Recent transcript:\n${transcript}`,
          },
        ],
      });
      const wm = result.toolResults?.[0]?.output as Record<string, unknown> | undefined;
      const elapsed = Date.now() - t0;
      console.log(
        `[working-memory] updated in ${elapsed}ms — keys: ${wm ? Object.keys(wm).join(',') : 'none'}`,
      );
      return wm ?? { error: 'no output from extraction' };
    } catch (err: any) {
      console.log('[working-memory] extraction failed:', err?.message ?? err);
      return { error: `extraction failed: ${err?.message ?? 'unknown'}` };
    }
  },
});
