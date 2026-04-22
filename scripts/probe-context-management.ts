/**
 * Probe whether Vertex AI's Anthropic endpoint accepts the
 * context_management API (clear_tool_uses_20250919, compact_20260112)
 * via @ai-sdk/google-vertex/anthropic + providerOptions.anthropic.
 *
 * Run:
 *   cd client && pnpm tsx scripts/probe-context-management.ts
 *
 * Requires .env.local to be loaded (GOOGLE_VERTEX_PROJECT,
 * GOOGLE_VERTEX_LOCATION, GOOGLE_APPLICATION_CREDENTIALS).
 */
import 'dotenv/config';
import { config as loadDotenv } from 'dotenv';
import { generateText } from 'ai';
import { vertexAnthropic } from '@ai-sdk/google-vertex/anthropic';

loadDotenv({ path: '.env.local', override: false });

const model = vertexAnthropic('claude-haiku-4-5');

async function probe(label: string, providerOptions: any) {
  console.log(`\n─── ${label} ───`);
  try {
    const result = await generateText({
      model,
      maxOutputTokens: 40,
      messages: [
        { role: 'user', content: 'Say "probe ok" and nothing else.' },
      ],
      providerOptions,
    });
    console.log('  ✓ request succeeded');
    console.log('  text:', JSON.stringify(result.text));
    console.log('  providerMetadata:', JSON.stringify(result.providerMetadata, null, 2));
    console.log('  warnings:', JSON.stringify(result.warnings));
  } catch (err: any) {
    console.log('  ✗ request failed');
    console.log('  message:', err?.message);
    const cause = err?.cause ?? err?.data ?? err?.responseBody;
    if (cause) console.log('  cause:', JSON.stringify(cause).slice(0, 800));
  }
}

(async () => {
  // 1. Baseline: no provider options — sanity check that auth works
  await probe('baseline (no providerOptions)', {});

  // 2. contextManagement + compact
  await probe('contextManagement: compact_20260112', {
    anthropic: {
      contextManagement: {
        edits: [
          {
            type: 'compact_20260112',
            trigger: { type: 'input_tokens', value: 50 },
            instructions: 'Summarize concisely.',
          },
        ],
      },
    },
  });

  // 3. contextManagement + clear_tool_uses
  await probe('contextManagement: clear_tool_uses_20250919', {
    anthropic: {
      contextManagement: {
        edits: [
          {
            type: 'clear_tool_uses_20250919',
            trigger: { type: 'input_tokens', value: 50 },
            keep: { type: 'tool_uses', value: 2 },
          },
        ],
      },
    },
  });

  // 4. Both edits at once (what route.ts would actually use)
  await probe('contextManagement: both edits', {
    anthropic: {
      contextManagement: {
        edits: [
          {
            type: 'clear_tool_uses_20250919',
            trigger: { type: 'input_tokens', value: 50 },
            keep: { type: 'tool_uses', value: 2 },
          },
          {
            type: 'compact_20260112',
            trigger: { type: 'input_tokens', value: 100 },
          },
        ],
      },
    },
  });

  // 5. Explicit anthropicBeta to see if SDK forwards it
  await probe('explicit anthropicBeta header forwarding', {
    anthropic: {
      anthropicBeta: ['context-management-2025-06-27', 'compact-2026-01-12'],
    },
  });

  console.log('\n─── done ───');
})();
