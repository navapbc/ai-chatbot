import { defineTool } from 'eve/tools';
import { readFile } from 'node:fs/promises';
import { resolve, normalize, join } from 'node:path';
import { z } from 'zod';

const REFERENCES_DIR = normalize(
  join(process.cwd(), 'lib/ai/prompts/references'),
);

// Pure, unit-testable core. The defineTool wrapper below delegates to this
// so the file-reading logic is verifiable without the Eve runtime.
export async function readReferenceFile(
  filePath: string,
): Promise<{ content: string } | { error: string }> {
  const cleaned = filePath.replace(/^references\//, '');
  const resolved = resolve(REFERENCES_DIR, cleaned);
  if (
    !resolved.startsWith(`${REFERENCES_DIR}/`) &&
    resolved !== REFERENCES_DIR
  ) {
    return { error: 'Access denied: path must be within references' };
  }
  try {
    const content = await readFile(resolved, 'utf-8');
    return { content };
  } catch {
    return { error: `File not found: ${filePath}` };
  }
}

// NOTE: previously this used a plain JSON-Schema `inputSchema` instead of
// zod, because the repo's zod (pinned to `^3.25.76`) lacked the
// `~standard.jsonSchema.input`/`.output` extension eve's runtime
// tool-schema serializer (`serializeInputSchema` in
// `eve/dist/src/shared/tool-schema.js`) requires on any schema carrying a
// `~standard` key. That extension is present starting with zod's v4 line
// (`zod@4.4.3`, matching what eve bundles internally under
// `#compiled/zod`). Now that the repo's `zod` dependency is bumped to
// `^4.4.3`, the zod inputSchema below works with eve's serializer — see
// docs/eve-spike-findings.md and the zod-migration report for the
// before/after verification.
export default defineTool({
  description:
    'Load a reference document. Use the path the instructions tell you to load (e.g. "field-patterns.md", "custom-dropdowns.md", "browser-commands.md").',
  inputSchema: z.object({
    path: z
      .string()
      .describe(
        'Filename within lib/ai/prompts/references (e.g. "field-patterns.md")',
      ),
  }),
  async execute(input: { path: string }) {
    return readReferenceFile(input.path);
  },
});
