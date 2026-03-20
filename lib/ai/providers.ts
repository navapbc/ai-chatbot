import {
  customProvider,
  extractReasoningMiddleware,
  wrapLanguageModel,
} from 'ai';
import { xai } from '@ai-sdk/xai';
import { openai } from '@ai-sdk/openai';
import { google } from '@ai-sdk/google';
import { gateway } from '@ai-sdk/gateway';
import { createVertexAnthropic } from '@ai-sdk/google-vertex/anthropic';
import { createVertex } from '@ai-sdk/google-vertex';

const vertex = createVertex({
  location: process.env.GOOGLE_VERTEX_LOCATION ?? 'us-central1',
  project: process.env.GOOGLE_VERTEX_PROJECT ?? 'placeholder',
});
import {
  artifactModel,
  chatModel,
  reasoningModel,
  titleModel,
} from './models.test';
import { isTestEnvironment } from '../constants';

// Environment tag injected into every Vertex AI Anthropic request as
// `metadata.user_id` so BigQuery request-response logs can be filtered
// by environment (dev / prod / preview-pr-*).
const environment = process.env.NEXT_PUBLIC_ENVIRONMENT || process.env.ENVIRONMENT || 'unknown';

const vertexAnthropic = createVertexAnthropic({
  fetch: async (url, init) => {
    if (init?.body && typeof init.body === 'string') {
      try {
        const body = JSON.parse(init.body);
        body.metadata = { ...body.metadata, user_id: environment };
        init = { ...init, body: JSON.stringify(body) };
      } catch {
        // non-JSON body, pass through
      }
    }
    return fetch(url, init);
  },
});

// Anthropic model for web automation via Vertex AI
export const webAutomationModel = vertexAnthropic('claude-sonnet-4-6');
export const prepareStepModel = vertexAnthropic('claude-haiku-4-5');

export const myProvider = isTestEnvironment
  ? customProvider({
      languageModels: {
        'chat-model': chatModel,
        'chat-model-reasoning': reasoningModel,
        'title-model': titleModel,
        'artifact-model': artifactModel,
      },
    })
  : customProvider({
      languageModels: {
        'chat-model': openai('gpt-4o'),
        'chat-model-reasoning': wrapLanguageModel({
          model: openai('gpt-4o'),
          middleware: extractReasoningMiddleware({ tagName: 'think' }),
        }),
        'title-model': openai('gpt-4o-mini'),
        'artifact-model': openai('gpt-4o'),
        // Dev-only selectable models (shown in ModelSelectorButton, hidden in production)
        'gpt-5.4': openai('gpt-5.4'),
        'gpt-5.4-pro': openai('gpt-5.4-pro'),
        'gpt-5.4-mini': openai('gpt-5.4-mini'),
        'gpt-5.4-nano': openai('gpt-5.4-nano'),
        'claude-sonnet-4-6': vertexAnthropic('claude-sonnet-4-6'),
        'claude-haiku-4-5': vertexAnthropic('claude-haiku-4-5'),
        'grok-4': xai('grok-4-0709'),
        'grok-4-fast': xai('grok-4-fast-non-reasoning'),
        'grok-4.1-fast': xai('grok-4.1-fast-non-reasoning'),
        'gemini-2.5-pro': vertex('gemini-2.5-pro'),
        'gemini-2.5-flash': vertex('gemini-2.5-flash'),
        'gemini-2.5-flash-lite': vertex('gemini-2.5-flash-lite'),
        'gemini-3.1': google('gemini-3.1-pro-preview'),
        'gemini-3.1-flash': google('gemini-3.1-flash-lite-preview')
      },
      imageModels: {
        'small-model': openai.image('dall-e-3'),
      },
    });
