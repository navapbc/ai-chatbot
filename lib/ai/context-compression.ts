import { generateText, type ModelMessage } from 'ai';
import { z } from 'zod';
import { tool } from 'ai';
import { prepareStepModel } from '@/lib/ai/providers';
import { WORKING_MEMORY_PREFIX, buildWorkingMemoryMessage } from '@/lib/ai/working-memory';

// Vertex AI rejects `compact-2026-01-12`, so native compaction is unavailable.
// `clear_tool_uses_20250919` handles tool-result bloat server-side; this
// module is the fallback when pruning alone isn't enough.
//
// When Vertex enables the compaction beta, this whole file can be deleted
// and replaced with a `compact_20260112` edit in providerOptions.anthropic.

const COMPACT_THRESHOLD_TOKENS = 150_000; // only above this do we summarize
const KEEP_RECENT = 8;
const SUMMARY_PREFIX = '[Session summary — earlier context compacted]';

const COMPACTION_SYSTEM_PROMPT =
  'You are creating a session handoff document for a benefits form-filling agent. ' +
  'Return BOTH a prose summary AND structured participant data.\n\n' +
  'SUMMARY should capture:\n' +
  '- SESSION STATE: form name, URL, current page/step.\n' +
  '- COMPLETED FIELDS: fields already filled with values.\n' +
  '- PENDING FIELDS: fields still needing input.\n' +
  '- CASEWORKER INPUTS: answers/corrections the caseworker provided.\n' +
  '- GAP ANALYSIS: identified gaps and reasons.\n' +
  '- KEY DECISIONS: clarifications made during the session.\n\n' +
  'WORKING MEMORY should extract all participant data (Apricot records, ' +
  'caseworker answers, household). Do not fabricate — use [UNKNOWN] for missing.\n\n' +
  'Do not include participant PII in the summary (it belongs in working memory). ' +
  'Do not include browser snapshots or raw HTML.';

const log = (...args: unknown[]) => console.log('[compressor]', ...args);

const recordSchema = z.record(z.string(), z.unknown());
const writeHandoff = tool({
  description: 'Write the session handoff summary and updated working memory.',
  inputSchema: z.object({
    summary: z.string().describe('Prose handoff summary (see system prompt)'),
    workingMemory: z
      .object({
        participant: recordSchema.optional(),
        household: z.array(recordSchema).optional(),
        caseworkerInputs: recordSchema.optional(),
        formState: z
          .object({
            formName: z.string().optional(),
            currentUrl: z.string().optional(),
            currentStep: z.string().optional(),
            completedFields: z.record(z.string(), z.string()).optional(),
            pendingFields: z.array(z.string()).optional(),
          })
          .optional(),
      })
      .optional()
      .describe('Structured participant data extracted from the transcript'),
  }),
  execute: async (input) => input,
});

function extractWorkingMemory(messages: ModelMessage[]): {
  wmMessage: ModelMessage | null;
  rest: ModelMessage[];
} {
  if (
    messages.length > 0 &&
    typeof messages[0].content === 'string' &&
    messages[0].content.startsWith(WORKING_MEMORY_PREFIX)
  ) {
    return { wmMessage: messages[0], rest: messages.slice(1) };
  }
  return { wmMessage: null, rest: messages };
}

function flattenMessage(msg: ModelMessage): string {
  const role = msg.role.toUpperCase();
  if (typeof msg.content === 'string') return `[${role}]: ${msg.content}`;
  if (!Array.isArray(msg.content)) return `[${role}]: ${JSON.stringify(msg.content)}`;

  const parts = (msg.content as any[]).map((part) => {
    if (!part) return '';
    if (part.type === 'tool-result' && part.toolName === 'browser') {
      const raw = part.output ?? part.result ?? {};
      const r = (raw && typeof raw === 'object' && 'value' in raw ? (raw as any).value : raw) as any;
      const status = r?.success ? 'success' : `error: ${r?.error ?? 'unknown'}`;
      return `[browser result: ${status}]`;
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

function buildSummaryMessage(summary: string): ModelMessage {
  return { role: 'assistant', content: `${SUMMARY_PREFIX}\n\n${summary}` };
}

/**
 * Stateless fallback compaction. Called once at the top of a request, never
 * inside the tool loop.
 *
 * Contract:
 * - If estimatedInputTokens is below the threshold, returns messages unchanged.
 * - If above, runs one Haiku call with a tool-forced schema that emits both
 *   summary and structured working memory.
 * - Returns `[wm?, summary, ...last 8 messages]` on success, or the original
 *   messages on any failure (compaction is best-effort, never fatal).
 *
 * Never splits a tool-call from its tool-result (Anthropic would reject).
 */
export async function prepareMessages(
  messages: ModelMessage[],
  estimatedInputTokens: number,
): Promise<{ messages: ModelMessage[]; compacted: boolean; summary?: string }> {
  if (estimatedInputTokens < COMPACT_THRESHOLD_TOKENS) {
    return { messages, compacted: false };
  }

  const { wmMessage: incomingWm, rest } = extractWorkingMemory(messages);
  if (rest.length <= KEEP_RECENT) {
    log(`over threshold but only ${rest.length} non-WM msgs — skip`);
    return { messages, compacted: false };
  }

  let splitAt = rest.length - KEEP_RECENT;
  while (splitAt > 0 && rest[splitAt]?.role === 'tool') splitAt -= 1;
  const oldMessages = rest.slice(0, splitAt);
  const recentMessages = rest.slice(splitAt);
  const transcript = oldMessages.map(flattenMessage).join('\n\n');

  log(
    `compacting — ${oldMessages.length} old msgs (${transcript.length} chars) → summary + WM, ` +
    `keeping ${recentMessages.length} recent`,
  );

  const t0 = Date.now();
  let toolResult:
    | { summary?: string; workingMemory?: Record<string, unknown> }
    | undefined;
  try {
    const result = await generateText({
      model: prepareStepModel,
      maxOutputTokens: 4096,
      system: COMPACTION_SYSTEM_PROMPT,
      tools: { writeHandoff },
      toolChoice: { type: 'tool', toolName: 'writeHandoff' },
      messages: [{ role: 'user', content: `Transcript:\n\n${transcript}` }],
    });
    toolResult = result.toolResults?.[0]?.output as typeof toolResult;
  } catch (err) {
    log('ERROR — compaction generateText failed:', err);
    return { messages, compacted: false };
  }

  const summary = toolResult?.summary?.trim();
  if (!summary) {
    log('ABORT — empty summary from Haiku');
    return { messages, compacted: false };
  }
  log(`compaction done in ${Date.now() - t0}ms — summary ${summary.length} chars`);

  const newWm =
    toolResult?.workingMemory && Object.keys(toolResult.workingMemory).length > 0
      ? buildWorkingMemoryMessage(toolResult.workingMemory)
      : incomingWm;

  const out: ModelMessage[] = [];
  if (newWm) out.push(newWm);
  out.push(buildSummaryMessage(summary));
  out.push(...recentMessages);

  return { messages: out, compacted: true, summary };
}
