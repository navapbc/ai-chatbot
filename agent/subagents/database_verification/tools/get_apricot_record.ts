import { defineTool } from 'eve/tools';
import { z } from 'zod';

// Example tool. Production logic: lib/ai/tools/apricot/.
export default defineTool({
  description: 'Fetch a participant record (and linked records) from Apricot by participant ID.',
  inputSchema: z.object({ participantId: z.string() }),
  async execute({ participantId }) {
    // Demonstrative stub: production calls the Apricot API (lib/apricot-api.ts).
    return { participantId, fields: {}, note: 'stub — see lib/ai/tools/apricot' };
  },
});
