import {
  artifactModel,
  chatModel,
  reasoningModel,
  titleModel,
} from './models.test';
import {
  customProvider,
  extractReasoningMiddleware,
  wrapLanguageModel,
} from 'ai';

import { isTestEnvironment } from '../constants';
import { openai } from '@ai-sdk/openai';
import { vertexAnthropic } from '@ai-sdk/google-vertex/anthropic';

// Anthropic model for web automation via Vertex AI.
//
// Requires GOOGLE_VERTEX_LOCATION=global, which is what Cloud Run sets
// (terraform/cloud_run.tf). The opus base models have NO
// `online_prediction_input_tokens_per_minute_per_base_model` quota on `nava-labs`
// in any regional endpoint — us-east5, europe-west1, and asia-southeast1 all
// return 429 RESOURCE_EXHAUSTED even for a 10-token request, and only the global
// endpoint serves them. If this 429s locally, check GOOGLE_VERTEX_LOCATION
// first; sonnet-4-6 and haiku-4-5 are the only models with regional quota.
export const webAutomationModel = vertexAnthropic('claude-opus-4-7');
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
        'claude-opus-4-7': vertexAnthropic('claude-opus-4-7'),
        'claude-opus-4-8': vertexAnthropic('claude-opus-4-8'),
        'claude-sonnet-4-6': vertexAnthropic('claude-sonnet-4-6'),
        'claude-haiku-4-5': vertexAnthropic('claude-haiku-4-5'),
      },
      imageModels: {
        'small-model': openai.image('dall-e-3'),
      },
    });
