import { vertexAnthropic } from '@ai-sdk/google-vertex/anthropic';
import { defineAgent } from 'eve';
import { VERTEX_CONTEXT_WINDOW_TOKENS } from '@/lib/ai/eve/model-map';

export default defineAgent({
  // Deliberately talks the caller OUT of this subagent for the common case.
  // Measured in preview: an IHSS run spent ~80s here (8s of work, ~71s of
  // eve's child-completion overhead) and got back generic program knowledge,
  // because riversideihss.org 403s its `web_fetch` while the Kernel browser
  // loads the same URL fine. The caller snapshots the real page regardless.
  //
  // Both the URL requirement and this steer live here, not only in
  // benefits-application/SKILL.md, because a tool description is always in the
  // caller's context whereas that skill has to be loaded first.
  description:
    "Enumerate the fields a benefits application requires across every page, for an UNFAMILIAR program or an unfamiliar county's variant of a familiar one. Returns a field checklist. PREFER YOUR OWN PROGRAM KNOWLEDGE over calling this: you navigate and snapshot the real application anyway, and most county/state URLs return 403 Forbidden to this subagent's web_fetch, so it usually returns only knowledge you already have — after a slow (~80s) round trip. Call it just when you genuinely do not know what the program asks for. Your `message` MUST then include the application's URL verbatim plus the program name and locale; it has no web search and will not look a URL up.",
  // Called directly on Vertex AI rather than through the Vercel AI Gateway,
  // matching agent/agent.ts. The explicit context window is required: Eve
  // cannot look a direct provider instance up in the gateway model catalog.
  model: vertexAnthropic('claude-haiku-4-5'),
  modelContextWindowTokens: VERTEX_CONTEXT_WINDOW_TOKENS,
});
