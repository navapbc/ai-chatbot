/**
 * Pure, dependency-free input schema for the `browser` tool.
 *
 * Kept separate from `lib/ai/tools/browser.ts` (which reaches `server-only`
 * through `lib/kernel/browser.ts` → `lib/db/queries.ts`) so the eval suite and
 * node-mode tests can describe the tool without pulling that chain in.
 *
 * This matters more than it looks: `server-only`'s default export *throws* on
 * import, and only Next.js supplies the `react-server` export condition that
 * resolves it to a no-op. Any other bundler — notably the esbuild pass behind
 * `braintrust eval` — gets the throwing build, so a single transitive import
 * takes down every eval file at compile time.
 *
 * Same reasoning as `lib/kernel/session-store.ts`.
 */

import { z } from 'zod';

export const browserInputSchema = z
  .object({
    command: z
      .array(z.string())
      .min(1)
      .describe(
        'agent-browser CLI argv, e.g. ["click", "@e1"] or ["fill", "@e1", "John"]. One argument per array element; do not quote or escape values.',
      ),
  })
  .describe('An agent-browser CLI command as an argv array');
