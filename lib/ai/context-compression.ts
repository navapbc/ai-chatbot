import { generateText, type ModelMessage } from 'ai';
import { prepareStepModel } from '@/lib/ai/providers';
import { WORKING_MEMORY_PREFIX, buildWorkingMemoryMessage } from '@/lib/ai/working-memory';
import { updateWorkingMemory } from '@/lib/ai/tools/working-memory';

const MODEL_CONTEXT_WINDOW = 200_000; // claude-sonnet-4-6
const COMPACT_THRESHOLD_PCT = 0.75;
const COMPACT_THRESHOLD_TOKENS = MODEL_CONTEXT_WINDOW * COMPACT_THRESHOLD_PCT; // 150K
const KEEP_RECENT = 8;                // keep last N messages after compaction

const SUMMARY_PREFIX = '[Session summary — earlier context compacted]';

const COMPACTION_SYSTEM_PROMPT =
  'You are creating a session handoff document for a benefits form-filling agent. ' +
  'Participant field-value data is preserved separately in working memory — ' +
  'do NOT list individual participant field values in the summary.\n\n' +
  'Extract and preserve the following from the transcript:\n' +
  '- SESSION STATE: The current form name, URL, and which page/step we are on.\n' +
  '- COMPLETED FIELDS: Every field that has already been filled and its value.\n' +
  '- PENDING FIELDS: Every field still needing input.\n' +
  '- CASEWORKER INPUTS: Every answer or correction the caseworker provided.\n' +
  '- GAP ANALYSIS: Every field that has been identified as a gap and the reason why.\n' +
  '- GAP ANSWERS: Every answer or correction the caseworker provided to a gap analysis.\n' +
  '- KEY DECISIONS: Any decisions or clarifications made during the session.\n\n' +
  'CRITICAL RULES:\n' +
  '- Do NOT invent, infer, or fabricate any data that is not explicitly present in the transcript.\n' +
  '- If a field value appears truncated or unclear, write [UNKNOWN] rather than guessing.\n' +
  '- Do NOT include participant PII (names, DOB, SSN, address) — it is in working memory.\n' +
  'Do NOT include browser snapshot content or raw HTML.';

const log = (..._args: unknown[]) => {};

/**
 * Detect and extract a working memory message from the beginning of the
 * message list. The working memory message is always the first message and
 * starts with WORKING_MEMORY_PREFIX. It must be excluded from compaction
 * so the model always has ground-truth participant data.
 */
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

/**
 * Flatten a ModelMessage into a plain-text line for the transcript.
 * Strips browser snapshots/screenshots to keep the transcript manageable.
 */
function flattenMessage(msg: ModelMessage): string {
  const role = msg.role.toUpperCase();
  if (typeof msg.content === 'string') return `[${role}]: ${msg.content}`;
  if (!Array.isArray(msg.content)) return `[${role}]: ${JSON.stringify(msg.content)}`;

  const parts = (msg.content as any[]).map((part) => {
    if (!part) return '';
    // Browser tool results: keep status, strip the heavy output payload.
    // AI SDK v5 puts the tool return value on `output`; older shape used
    // `result`. Support both for safety.
    if (part.type === 'tool-result' && part.toolName === 'browser') {
      const raw = part.output ?? part.result ?? {};
      const r = (raw && typeof raw === 'object' && 'value' in raw ? (raw as any).value : raw) as any;
      const status = r?.success ? 'success' : `error: ${r?.error ?? 'unknown'}`;
      return `[browser result: ${status}]`;
    }
    // Browser tool calls: keep action + key param for context.
    // AI SDK v5 uses `input`; older shape used `args`. Support both.
    if (part.type === 'tool-call' && part.toolName === 'browser') {
      const a = (part.input ?? part.args ?? {}) as Record<string, any>;
      return `[browser: ${a.action ?? '?'}${a.selector ? ` ${a.selector}` : a.url ? ` ${a.url}` : ''}]`;
    }
    const s = typeof part === 'string' ? part : (part.text ?? JSON.stringify(part) ?? '');
    return String(s);
  });
  return `[${role}]: ${parts.join('\n')}`;
}

/**
 * Shared summarization: split messages into old + recent, then run two
 * parallel Haiku calls on the same transcript:
 *   1. Compaction summary (session state, actions, decisions)
 *   2. Working memory extraction (structured participant data via tool call)
 *
 * Returns null on failure (caller should fall back to original messages).
 */
async function summarizeMessages(
  messages: ModelMessage[],
  logPrefix: string,
  onCompacting?: () => void,
): Promise<{
  summary: string;
  workingMemory: Record<string, unknown> | null;
  recentMessages: ModelMessage[];
  splitAt: number;
} | null> {
  if (messages.length <= KEEP_RECENT) {
    log(`${logPrefix}over threshold but only ${messages.length} msgs (≤ ${KEEP_RECENT}), skipping`);
    return null;
  }

  // Never split between a tool-call assistant message and its following
  // tool-result message — Anthropic requires each tool_result to have a
  // tool_use in the previous message, so an orphan tool-role message in
  // recentMessages fails validation. Walk back past tool-role messages.
  let splitAt = messages.length - KEEP_RECENT;
  while (splitAt > 0 && messages[splitAt]?.role === 'tool') {
    splitAt -= 1;
  }
  const oldMessages = messages.slice(0, splitAt);
  const recentMessages = messages.slice(splitAt);

  log(
    `${logPrefix}COMPACTING — summarizing ${oldMessages.length} old msgs, ` +
    `keeping ${recentMessages.length} recent`
  );

  const transcript = oldMessages.map(flattenMessage).join('\n\n');
  log(`${logPrefix}transcript length: ${transcript.length} chars from ${oldMessages.length} msgs`);

  onCompacting?.();

  const t0 = Date.now();

  // Run compaction summary and working memory extraction in parallel on Haiku
  const [compactionResult, wmResult] = await Promise.all([
    // 1. Compaction summary
    generateText({
      model: prepareStepModel,
      maxOutputTokens: 4096,
      system: COMPACTION_SYSTEM_PROMPT,
      messages: [{ role: 'user', content: `Summarize this session transcript:\n\n${transcript}` }],
    }).catch((err) => {
      log(`${logPrefix}compaction ERROR:`, err);
      return null;
    }),

    // 2. Working memory extraction via tool call
    generateText({
      model: prepareStepModel,
      maxOutputTokens: 4096,
      tools: { updateWorkingMemory },
      toolChoice: { type: 'tool', toolName: 'updateWorkingMemory' },
      system:
        'Extract all participant data from this transcript. ' +
        'Include data from database records and caseworker answers. ' +
        'Only include data explicitly present — never fabricate.',
      messages: [{ role: 'user', content: transcript }],
    }).catch((err) => {
      log(`${logPrefix}working memory ERROR:`, err);
      return null;
    }),
  ]);

  const elapsed = Date.now() - t0;

  // Process compaction result
  const summary = compactionResult?.text?.trim();
  if (!summary) {
    log(`${logPrefix}ABORT — empty or failed compaction summary`);
    return null;
  }
  log(`${logPrefix}compaction done in ${elapsed}ms — summary: ${summary.length} chars`);

  // Process working memory result
  let workingMemory: Record<string, unknown> | null = null;
  if (wmResult?.toolResults?.length) {
    workingMemory = wmResult.toolResults[0].output as Record<string, unknown>;
    log(`${logPrefix}working memory extracted — ${Object.keys(workingMemory).length} keys`);
  } else {
    log(`${logPrefix}working memory extraction failed or empty — continuing without`);
  }

  return { summary, workingMemory, recentMessages, splitAt };
}

function buildSummaryMessage(summary: string): ModelMessage {
  return { role: 'assistant', content: `${SUMMARY_PREFIX}\n\n${summary}` };
}

/**
 * Stateless fallback compaction. Called ONCE at the top of a request, never
 * inside the tool loop.
 *
 * This replaces the old per-step `createMessageCompressor`, which re-extracted
 * and re-prepended working memory and re-applied compaction on every step.
 * That per-step rewriting shifted the message prefix between steps, so an
 * Anthropic cache breakpoint placed on the history never stayed byte-stable
 * long enough to earn cache reads. Keeping the prefix stable across the step
 * loop is what lets the sliding cache breakpoint (see cache-breakpoints.ts)
 * pay off; mid-run context growth is bounded server-side by the
 * `clear_tool_uses_20250919` context-management edit instead.
 *
 * Contract:
 * - Below the threshold, returns messages unchanged (cache-stable prefix).
 * - Above, runs one summarization pass (reusing `summarizeMessages`) and
 *   returns `[wm?, summary, ...recentMessages]`.
 * - On any failure, returns the original messages (compaction is best-effort).
 */
export async function prepareMessages(
  messages: ModelMessage[],
  estimatedInputTokens: number,
  onCompacting?: () => void,
): Promise<{ messages: ModelMessage[]; compacted: boolean; summary?: string }> {
  if (estimatedInputTokens < COMPACT_THRESHOLD_TOKENS) {
    return { messages, compacted: false };
  }

  // Working memory is always preserved verbatim — never folded into a summary.
  const { wmMessage: incomingWm, rest } = extractWorkingMemory(messages);

  const result = await summarizeMessages(rest, '', onCompacting);
  if (!result) {
    log(`over threshold (~${estimatedInputTokens} tokens) but compaction skipped/failed`);
    return { messages, compacted: false };
  }

  const wm = result.workingMemory
    ? buildWorkingMemoryMessage(result.workingMemory)
    : incomingWm;

  const out: ModelMessage[] = [];
  if (wm) out.push(wm);
  out.push(buildSummaryMessage(result.summary));
  out.push(...result.recentMessages);

  log(
    `compacted — ~${estimatedInputTokens} tokens → 1 summary + ` +
    `${result.recentMessages.length} recent${wm ? ' +WM' : ''}`
  );

  return { messages: out, compacted: true, summary: result.summary };
}
