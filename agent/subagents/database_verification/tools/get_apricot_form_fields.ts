import { defineTool } from 'eve/tools';
import { z } from 'zod';

// Example tool. Production logic: lib/ai/tools/apricot/. Resolves field_NNNN -> label.
export default defineTool({
  description: 'Resolve the field_NNNN -> label map for an Apricot form, so raw field IDs can be trusted.',
  inputSchema: z.object({ formId: z.string() }),
  async execute({ formId }) {
    return { formId, labels: {}, note: 'stub — see lib/ai/tools/apricot' };
  },
});
