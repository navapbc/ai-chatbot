import { describe, it, expect } from 'vitest';
import { generateText } from 'ai';
import { vertexAnthropic } from '@ai-sdk/google-vertex/anthropic';
import { llmJudge, containsSubstring } from './scorers/llm-judge';
import {
  compressionDataset,
  type CompressionTestCase,
} from './datasets/context-compression';

/**
 * Context Compression Evals
 *
 * These tests call the real compression prompt against a live model and use
 * LLM-as-judge scoring to verify that summaries preserve critical facts.
 *
 * Run with:  pnpm test:evals
 *
 * These are excluded from regular `pnpm test` because they hit real APIs
 * and are slower / non-deterministic.
 */

// The compaction prompt — duplicated from context-compression.ts so we can
// eval it in isolation without importing internal module state.
const COMPACTION_SYSTEM_PROMPT =
  'You are creating a session handoff document for a benefits form-filling agent. ' +
  'Extract and preserve ALL of the following — be explicit and complete:\n' +
  '- PARTICIPANT DATA: Every field-value pair from the database (Apricot record) and caseworker. Format as "Field: Value" lines.\n' +
  '- SESSION STATE: The current form name, URL, and which page/step we are on.\n' +
  '- COMPLETED FIELDS: Every field that has already been filled and its value.\n' +
  '- PENDING FIELDS: Every field still needing input.\n' +
  '- CASEWORKER INPUTS: Every answer or correction the caseworker provided.\n' +
  '- GAP ANALYSIS: Every field that has been identified as a gap and the reason why.\n' +
  '- GAP ANSWERS: Every answer or correction the caseworker provided to a gap analysis.\n' +
  'Do NOT summarize participant data — list every field and value explicitly. ' +
  'Do NOT include browser snapshot content or raw HTML.';

// Models to eval — add more to compare across providers
const MODELS_TO_EVAL = [
  { id: 'claude-haiku-4-5', provider: () => vertexAnthropic('claude-haiku-4-5') },
  // Uncomment to compare:
  // { id: 'claude-sonnet-4-6', provider: () => vertexAnthropic('claude-sonnet-4-6') },
];

/**
 * Flatten messages to a transcript string (mirrors context-compression.ts logic).
 * Prunes browser snapshots to keep the transcript manageable.
 */
function buildTranscript(testCase: CompressionTestCase): string {
  return testCase.messages
    .map((msg) => {
      const role = msg.role.toUpperCase();
      if (typeof msg.content === 'string') return `[${role}]: ${msg.content}`;
      if (!Array.isArray(msg.content)) return `[${role}]: ${JSON.stringify(msg.content)}`;

      const parts = (msg.content as any[]).map((part) => {
        if (!part) return '';
        if (part.type === 'tool-result' && part.toolName === 'browser') {
          const r = part.result;
          if (r?.snapshot || r?.accessibility_tree || r?.screenshot)
            return '[browser output: pruned]';
        }
        const s = typeof part === 'string' ? part : (part.text ?? JSON.stringify(part) ?? '');
        return String(s).slice(0, 500);
      });
      return `[${role}]: ${parts.join('\n')}`;
    })
    .join('\n\n');
}

// Minimum score to pass — 0.8 means at least 80% of required facts must be preserved
const PASS_THRESHOLD = 0.8;

for (const model of MODELS_TO_EVAL) {
  describe(`context-compression [${model.id}]`, () => {
    for (const testCase of compressionDataset) {
      it(
        `${testCase.name}: ${testCase.description}`,
        { timeout: 60_000 },
        async () => {
          // 1. Build transcript from messages
          const transcript = buildTranscript(testCase);

          // 2. Run compression prompt against the model
          const { text: summary } = await generateText({
            model: model.provider(),
            maxOutputTokens: 4096,
            system: COMPACTION_SYSTEM_PROMPT,
            messages: [
              {
                role: 'user',
                content: `Summarize this session transcript:\n\n${transcript}`,
              },
            ],
          });

          expect(summary.trim().length).toBeGreaterThan(0);

          // 3. LLM-as-judge scoring: are required facts preserved?
          const judgment = await llmJudge({
            summary,
            expectedPreserved: testCase.expectedPreserved,
          });

          console.log(
            `\n[${model.id}] ${testCase.name}:\n` +
              `  Score: ${judgment.score}\n` +
              `  Preserved: ${judgment.preserved.join(', ')}\n` +
              `  Missing: ${judgment.missing.join(', ') || '(none)'}\n` +
              `  Reasoning: ${judgment.reasoning}\n`
          );

          expect(judgment.score).toBeGreaterThanOrEqual(PASS_THRESHOLD);

          // 4. Negative checks: things that should NOT appear in summaries
          if (testCase.expectedOmitted) {
            for (const omitted of testCase.expectedOmitted) {
              expect(containsSubstring(summary, omitted)).toBe(false);
            }
          }
        },
      );
    }
  });
}
