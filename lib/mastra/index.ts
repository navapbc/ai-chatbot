// Import the built Mastra app from local client build
import * as mastraBundle from '../../.mastra/output/mastra.mjs';

// Find the mastra instance in the bundle exports
export const mastra = Object.values(mastraBundle).find(
  (value: any) => value && typeof value.getAgent === 'function'
) as any;
export type { Agent } from '@mastra/core/agent';
