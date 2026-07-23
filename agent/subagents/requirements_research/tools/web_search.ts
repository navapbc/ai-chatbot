import { defineTool } from 'eve/tools';
import { z } from 'zod';

// Example tool. Production would use the app's web-search integration.
export default defineTool({
  description: 'Search the web for a benefits program application and its required fields.',
  inputSchema: z.object({ query: z.string() }),
  async execute({ query }) {
    return { query, results: [], note: 'stub — wire to production web search' };
  },
});
