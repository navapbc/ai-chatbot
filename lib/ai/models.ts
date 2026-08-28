import { isProductionEnvironment } from '@/lib/constants';

export const DEFAULT_CHAT_MODEL: string = 'web-automation-model';

export interface ChatModel {
  id: string;
  name: string;
  description: string;
}

/**
 * The production model. Its id is deliberately NOT in
 * `lib/ai/eve/model-map.ts`, so selecting it sends no `x-eve-model` header and
 * Eve uses the fallback in `agent/agent.ts`.
 */
const productionChatModels: Array<ChatModel> = [
  {
    id: 'web-automation-model',
    name: 'Web Automation Agent',
    description: 'AI agent for web navigation and automation tasks',
  },
];

/**
 * Dev/eval-only overrides, hidden in production.
 *
 * Each id must exist in BOTH `lib/ai/providers.ts` (for the legacy transport)
 * and `MODEL_MAP` in `lib/ai/eve/model-map.ts` (which turns it into the AI
 * Gateway slug Eve routes through) — otherwise picking it silently falls back.
 * `/api/eve-chat` also refuses to honor an override in production, so keeping
 * these out of the picker there matches what the server will actually do.
 *
 * Eve reads the override at session start, so a change only affects new chats.
 */
const devOnlyChatModels: Array<ChatModel> = [
  {
    id: 'gpt-5.4-nano',
    name: 'GPT-5.4 Nano',
    description: 'Cheapest — smoke-testing the agent loop',
  },
  {
    id: 'gpt-5.4-mini',
    name: 'GPT-5.4 Mini',
    description: 'Cheap; usable for most form-filling runs',
  },
  {
    id: 'gpt-5.4',
    name: 'GPT-5.4',
    description: 'Mid-tier OpenAI model',
  },
  {
    id: 'gpt-5.4-pro',
    name: 'GPT-5.4 Pro',
    description: 'Highest-cost OpenAI model',
  },
  {
    id: 'claude-haiku-4-5',
    name: 'Claude Haiku 4.5',
    description: 'Cheapest Claude; fast, weaker at long tool loops',
  },
  {
    id: 'claude-sonnet-4-6',
    name: 'Claude Sonnet 4.6',
    description: "Balanced Claude; Eve's default when no override is sent",
  },
  {
    id: 'claude-sonnet-5',
    name: 'Claude Sonnet 5',
    description: 'Newest Sonnet; Vertex enablement unverified',
  },
  {
    id: 'claude-opus-4-7',
    name: 'Claude Opus 4.7',
    description: 'Strongest at long web-automation runs',
  },
  {
    id: 'claude-opus-4-8',
    name: 'Claude Opus 4.8',
    description: 'Previous-generation Opus',
  },
  {
    id: 'claude-opus-5',
    name: 'Claude Opus 5',
    description: 'Newest Opus; Vertex enablement unverified',
  },
];

export const chatModels: Array<ChatModel> = isProductionEnvironment
  ? productionChatModels
  : [...productionChatModels, ...devOnlyChatModels];

/** Ids the picker may offer, so entitlements can't drift from `chatModels`. */
export const selectableChatModelIds: Array<string> = chatModels.map(
  (m) => m.id,
);
