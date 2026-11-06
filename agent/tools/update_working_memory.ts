import { defineState } from 'eve/context';
import { defineTool } from 'eve/tools';
import { z } from 'zod';

// Prototype for sub-project 4 (Task 4 / Q2 of the Eve spike): persist
// structured working-memory data (participant fields, caseworker inputs,
// form-fill progress) OUTSIDE the model's message history so it survives
// across turns even after compaction rewrites the transcript.
//
// Investigation result (see docs/eve-spike-findings.md ## Q2): eve DOES
// expose a native per-session store for exactly this purpose —
// `defineState` from `eve/context` (node_modules/eve/dist/src/public/
// definitions/state.d.ts). It returns a `StateHandle<T>` backed by a durable
// `ContextKey` that "survives across workflow step boundaries" and, per
// node_modules/eve/docs/guides/state.md, "is durable by default and does
// not reset between turns." This is a real native API, not an app-runtime
// fallback — so this prototype uses it directly instead of writing to the
// app's Postgres.
//
// Declared at module scope so every importer (this tool, and — if a
// dynamic-instructions resolver is added later to re-inject the memory into
// the system prompt every turn, matching (d) in the Q2 enumeration) shares
// the same durable slot.
export const workingMemoryState = defineState<Record<string, unknown>>(
  'labs-asp.working-memory',
  () => ({}),
);

export default defineTool({
  description:
    'Persist structured working-memory data (participant fields, caseworker ' +
    'inputs, form-fill progress) so it survives across turns, even across ' +
    'context compaction. Call with `data` to merge new fields into the ' +
    'stored memory (each key you pass overwrites the same key already ' +
    'stored; omit keys you are not updating). Call with no `data` to read ' +
    'back everything currently stored.',
  inputSchema: z
    .object({
      data: z
        .record(z.string(), z.unknown())
        .optional()
        .describe(
          'Fields to merge into working memory. Omit to just read the current stored state.',
        ),
    })
    .strict(),
  async execute({ data }: { data?: Record<string, unknown> }) {
    if (data && Object.keys(data).length > 0) {
      workingMemoryState.update((current) => ({ ...current, ...data }));
    }
    const current = workingMemoryState.get();
    return { ok: true, keys: Object.keys(current), data: current };
  },
});
