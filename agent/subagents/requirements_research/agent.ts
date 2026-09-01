import { vertexAnthropic } from '@ai-sdk/google-vertex/anthropic';
import { defineAgent } from 'eve';
import { VERTEX_CONTEXT_WINDOW_TOKENS } from '@/lib/ai/eve/model-map';

export default defineAgent({
  // The `message` MUST carry the URL: a subagent never sees the parent's
  // history, so that is the only channel for it. Said here rather than only in
  // the skill because this description is always in the caller's context,
  // whereas benefits-application/SKILL.md has to be loaded first.
  description:
    "Research a benefits program's application up front and enumerate ALL fields it will require across every page, so gap analysis is complete before form-filling starts. Returns a field checklist. Your `message` MUST include the application's URL verbatim, plus the program name and locale — this subagent has no web search and will not look a URL up. Do not call it without a URL; ask the caseworker for one instead.",
  // Called directly on Vertex AI rather than through the Vercel AI Gateway,
  // matching agent/agent.ts. The explicit context window is required: Eve
  // cannot look a direct provider instance up in the gateway model catalog.
  model: vertexAnthropic('claude-sonnet-4-6'),
  modelContextWindowTokens: VERTEX_CONTEXT_WINDOW_TOKENS,
});
