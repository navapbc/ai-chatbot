import { generateObject } from 'ai';
import { z } from 'zod';
import { vertexAnthropic } from '@ai-sdk/google-vertex/anthropic';

/**
 * Model used for judging eval outputs.
 * Uses a different model than the one being evaluated to avoid self-bias.
 */
const judgeModel = vertexAnthropic('claude-haiku-4-5');

const JudgmentSchema = z.object({
  score: z.number().min(0).max(1).describe('0 = completely wrong, 1 = perfect'),
  preserved: z
    .array(z.string())
    .describe('Facts from the expected list that were found in the output'),
  missing: z
    .array(z.string())
    .describe('Facts from the expected list that were NOT found in the output'),
  reasoning: z.string().describe('Brief explanation of the judgment'),
});

export type Judgment = z.infer<typeof JudgmentSchema>;

/**
 * Use an LLM to judge whether a compression summary preserves required facts.
 *
 * Returns a structured judgment with a 0-1 score, lists of preserved/missing
 * facts, and reasoning.
 */
export async function llmJudge({
  summary,
  expectedPreserved,
  criteria,
}: {
  summary: string;
  expectedPreserved: string[];
  criteria?: string;
}): Promise<Judgment> {
  const { object } = await generateObject({
    model: judgeModel,
    schema: JudgmentSchema,
    prompt:
      `You are evaluating the quality of a session compression summary for a benefits form-filling agent.\n\n` +
      `## Criteria\n` +
      `${criteria ?? 'The summary must preserve all critical facts listed below. Each fact should be present either verbatim or semantically equivalent.'}\n\n` +
      `## Required facts\n` +
      expectedPreserved.map((f, i) => `${i + 1}. ${f}`).join('\n') +
      `\n\n## Summary to evaluate\n${summary}\n\n` +
      `Score 1.0 if ALL required facts are present, 0.0 if none are. ` +
      `Partial credit: score = (facts found) / (total facts). ` +
      `List each required fact as either preserved or missing.`,
  });

  return object;
}

/**
 * Simple exact-match check: does the summary contain a substring?
 * Useful for quick checks (e.g. raw HTML should NOT appear).
 */
export function containsSubstring(text: string, substring: string): boolean {
  return text.toLowerCase().includes(substring.toLowerCase());
}
