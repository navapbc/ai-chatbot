import { defineTool } from 'eve/tools';
import { readFile } from 'node:fs/promises';
import { resolve, normalize, join } from 'node:path';

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

// NOTE: the brief's Step 4 sample used a `z.object(...)` inputSchema (zod
// imported from the repo's own `zod` dependency, pinned to `^3.25.76`).
// That crashes eve@0.27.0 at agent-graph-resolution time: eve's runtime
// tool-schema serializer (`serializeInputSchema` in
// `eve/dist/src/shared/tool-schema.js`) treats any schema with a
// `~standard` property as eve's own extended "StandardJSONSchemaV1" shape,
// which additionally requires `~standard.jsonSchema.input`/`.output`
// functions. That extension is only present starting with zod's newer v4
// line (confirmed present in the `zod@4.4.3` eve bundles internally under
// `#compiled/zod`, and absent from both `zod` and `zod/v4` as resolved
// from this repo's pinned `zod@3.25.76`). The repo's `"zod": "^3.25.76"`
// range cannot reach a version with this support (that would require a
// major bump — out of scope for this additive spike, same reasoning as
// the `ai` peer-version watch-item called out in the task brief).
//
// Per the brief's own escape hatch ("If the installed package differs,
// adjust the wrapper only — leave `readReferenceFile` ... unchanged"),
// this uses `defineTool`'s plain-JSON-Schema `inputSchema` overload
// instead of a zod schema. Eve rehydrates a JSON-Schema `inputSchema` into
// its own internal (compatible) zod instance for runtime validation, so
// this still validates `path` as a required string — it just avoids
// constructing the schema with the repo's zod at all.
export default defineTool({
  description:
    'Load a reference document. Use the path the instructions tell you to load (e.g. "field-patterns.md", "custom-dropdowns.md", "browser-commands.md").',
  inputSchema: {
    type: 'object',
    properties: {
      path: {
        type: 'string',
        description:
          'Filename within lib/ai/prompts/references (e.g. "field-patterns.md")',
      },
    },
    required: ['path'],
    additionalProperties: false,
  },
  async execute(input: Record<string, unknown>) {
    return readReferenceFile(input.path as string);
  },
});
