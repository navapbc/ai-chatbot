import {
  convertToModelMessages,
  createUIMessageStream,
  JsonToSseTransformStream,
  smoothStream,
  stepCountIs,
  streamText,
} from 'ai';
import { auth, type UserType } from '@/app/(auth)/auth';
import { type RequestHints, systemPrompt } from '@/lib/ai/prompts';
import {
  createStreamId,
  deleteChatById,
  getChatById,
  getMessageCountByUserId,
  getMessagesByChatId,
  saveChat,
  saveMessages,
} from '@/lib/db/queries';
import { convertToUIMessages, generateUUID } from '@/lib/utils';
import { generateTitleFromUserMessage } from '../../actions';
import { createDocument } from '@/lib/ai/tools/create-document';
import { updateDocument } from '@/lib/ai/tools/update-document';
import { requestSuggestions } from '@/lib/ai/tools/request-suggestions';
import { getWeather } from '@/lib/ai/tools/get-weather';
import { isProductionEnvironment } from '@/lib/constants';
import { myProvider } from '@/lib/ai/providers';
import { entitlementsByUserType } from '@/lib/ai/entitlements';
import { postRequestBodySchema, type PostRequestBody } from './schema';
import { geolocation } from '@vercel/functions';
import {
  createResumableStreamContext,
  type ResumableStreamContext,
} from 'resumable-stream';
import { after } from 'next/server';
import { ChatSDKError } from '@/lib/errors';
import type { ChatMessage } from '@/lib/types';
import type { ChatModel } from '@/lib/ai/models';
import type { VisibilityType } from '@/components/visibility-selector';

import { tool } from 'ai';
import { z } from 'zod';

export const maxDuration = 60;

let globalStreamContext: ResumableStreamContext | null = null;

export function getStreamContext() {
  if (!globalStreamContext) {
    try {
      globalStreamContext = createResumableStreamContext({
        waitUntil: after,
      });
    } catch (error: any) {
      if (error.message.includes('REDIS_URL')) {
        console.log(
          ' > Resumable streams are disabled due to missing REDIS_URL',
        );
      } else {
        console.error(error);
      }
    }
  }

  return globalStreamContext;
}

export async function POST(request: Request) {
  let requestBody: PostRequestBody;

  try {
    const json = await request.json();
    requestBody = postRequestBodySchema.parse(json);
  } catch (_) {
    return new ChatSDKError('bad_request:api').toResponse();
  }

  try {
    const {
      id,
      message,
      selectedChatModel,
      selectedVisibilityType,
    }: {
      id: string;
      message: ChatMessage;
      selectedChatModel: ChatModel['id'];
      selectedVisibilityType: VisibilityType;
    } = requestBody;

    const session = await auth();

    if (!session?.user) {
      return new ChatSDKError('unauthorized:chat').toResponse();
    }

    const userType: UserType = session.user.type;

    const messageCount = await getMessageCountByUserId({
      id: session.user.id,
      differenceInHours: 24,
    });

    if (messageCount > entitlementsByUserType[userType].maxMessagesPerDay) {
      return new ChatSDKError('rate_limit:chat').toResponse();
    }

    const chat = await getChatById({ id });

    if (!chat) {
      const title = await generateTitleFromUserMessage({
        message,
      });

      await saveChat({
        id,
        userId: session.user.id,
        title,
        visibility: selectedVisibilityType,
      });
    } else {
      if (chat.userId !== session.user.id) {
        return new ChatSDKError('forbidden:chat').toResponse();
      }
    }

    const messagesFromDb = await getMessagesByChatId({ id });
    const uiMessages = [...convertToUIMessages(messagesFromDb), message];

    const { longitude, latitude, city, country } = geolocation(request);

    const requestHints: RequestHints = {
      longitude,
      latitude,
      city,
      country,
    };

    await saveMessages({
      messages: [
        {
          chatId: id,
          id: message.id,
          role: 'user',
          parts: message.parts,
          attachments: [],
          createdAt: new Date(),
        },
      ],
    });

    const streamId = generateUUID();
    await createStreamId({ streamId, chatId: id });

  // Create web automation tool that uses Mastra's streamVNext with AI SDK compatibility
const webAutomationTool = tool({
  description: 'Automate web tasks using browser automation, including taking screenshots, filling forms, and saving data to the database',
  inputSchema: z.object({
    instruction: z.string().describe('The web automation task to perform'),
    url: z.string().optional().describe('The URL to navigate to (if needed)'),
  }),
  execute: async ({ instruction, url }) => {
    try {
      // Call the Mastra backend API for web automation using streamVNext
      const mastraApiUrl = process.env.MASTRA_API_URL || 'http://localhost:4111';
      const prompt = `${instruction}${url ? ` on ${url}` : ''}`;
      
      const response = await fetch(`${mastraApiUrl}/api/agents/webAutomationAgent/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-mastra-dev-playground': 'true', // Bypass auth for internal API calls
        },
        body: JSON.stringify({ 
          messages: prompt,
          memory: {
            thread: { id: `chat-${id}` },
            resource: session?.user?.id || 'anonymous'
          },
          temperature: 0.1,
          maxSteps: 10
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error(`Mastra API error: ${response.status} ${response.statusText}`, errorText);
        throw new Error(`Mastra API error: ${response.status} ${response.statusText}`);
      }

      // Stream the response and collect the final result
      let fullResult = '';
      const reader = response.body?.getReader();
      
      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          
          const chunk = new TextDecoder().decode(value);
          const lines = chunk.split('\n');
          
          for (const line of lines) {
            if (line.startsWith('0:')) {
              // Handle text chunk from Mastra stream (format: 0:"text content")
              try {
                const textMatch = line.match(/^0:"(.*)"/);
                if (textMatch) {
                  fullResult += textMatch[1];
                }
              } catch (parseError) {
                // Skip malformed lines
              }
            } else if (line.startsWith('e:')) {
              // Handle finish event (format: e:{"finishReason":"stop","usage":{...}})
              try {
                const eventData = JSON.parse(line.slice(2));
                if (eventData.finishReason) {
                  break; // Stream is complete
                }
              } catch (parseError) {
                // Skip malformed JSON lines
              }
            }
          }
        }
      }
      
      return { result: fullResult || 'Web automation completed successfully.' };
    } catch (error) {
      console.error('Web automation error:', error);
      return { 
        result: `Web automation is currently unavailable. Error: ${error instanceof Error ? error.message : 'Unknown error'}. Please try again later or contact support.` 
      };
    }
  },
});

  const stream = createUIMessageStream({
    execute: async ({ writer: dataStream }) => {
      // Check if this is a web automation request
      const lastMessage = uiMessages[uiMessages.length - 1];
      const messageText = lastMessage?.parts?.find(part => part.type === 'text')?.text?.toLowerCase() || '';
      
      const isWebAutomationRequest = 
        messageText.includes('screenshot') ||
        messageText.includes('navigate') ||
        messageText.includes('browser') ||
        messageText.includes('website') ||
        messageText.includes('web') ||
        messageText.includes('automation') ||
        messageText.includes('playwright') ||
        messageText.includes('fill form') ||
        messageText.includes('click');

      if (isWebAutomationRequest) {
        try {
          // Use streamText but with web automation tool prominently featured
          const result = streamText({
            model: myProvider.languageModel(selectedChatModel),
            system: `${systemPrompt({ selectedChatModel, requestHints })}

You are enhanced with web automation capabilities. When users request web automation tasks like taking screenshots, navigating websites, or interacting with web pages, use the web-automation tool.`,
            messages: convertToModelMessages(uiMessages),
            stopWhen: stepCountIs(5),
            experimental_activeTools: ['web-automation'],
            experimental_transform: smoothStream({ chunking: 'word' }),
            tools: {
              'web-automation': webAutomationTool,
            },
            experimental_telemetry: {
              isEnabled: isProductionEnvironment,
              functionId: 'web-automation-enhanced-chat',
            },
          });

          result.consumeStream();
          dataStream.merge(result.toUIMessageStream({ sendReasoning: true }));
        } catch (error) {
          console.error('Web automation streaming error:', error);
          // Fallback to regular chat with error message
          const result = streamText({
            model: myProvider.languageModel(selectedChatModel),
            system: `${systemPrompt({ selectedChatModel, requestHints })} 

Note: Web automation is currently unavailable due to a technical error. Please try again later.`,
            messages: convertToModelMessages(uiMessages),
            stopWhen: stepCountIs(5),
            experimental_transform: smoothStream({ chunking: 'word' }),
          });

          result.consumeStream();
          dataStream.merge(result.toUIMessageStream({ sendReasoning: true }));
        }
      } else {
        // Use regular chat with enhanced tools including web automation
        const result = streamText({
          model: myProvider.languageModel(selectedChatModel),
          system: systemPrompt({ selectedChatModel, requestHints }),
          messages: convertToModelMessages(uiMessages),
          stopWhen: stepCountIs(5),
          experimental_activeTools:
            selectedChatModel === 'chat-model-reasoning'
              ? []
              : [
                  'getWeather',
                  'createDocument',
                  'updateDocument',
                  'requestSuggestions',
                  'web-automation',
                ],
          experimental_transform: smoothStream({ chunking: 'word' }),
          tools: {
            getWeather,
            createDocument: createDocument({ session, dataStream }),
            updateDocument: updateDocument({ session, dataStream }),
            requestSuggestions: requestSuggestions({
              session,
              dataStream,
            }),
            'web-automation': webAutomationTool,
          },
          experimental_telemetry: {
            isEnabled: isProductionEnvironment,
            functionId: 'stream-text',
          },
        });

        result.consumeStream();

        dataStream.merge(
          result.toUIMessageStream({
            sendReasoning: true,
          }),
        );
      }
    },
      generateId: generateUUID,
      onFinish: async ({ messages }) => {
        await saveMessages({
          messages: messages.map((message) => ({
            id: message.id,
            role: message.role,
            parts: message.parts,
            createdAt: new Date(),
            attachments: [],
            chatId: id,
          })),
        });
      },
      onError: () => {
        return 'Oops, an error occurred!';
      },
    });

    const streamContext = getStreamContext();

    if (streamContext) {
      return new Response(
        await streamContext.resumableStream(streamId, () =>
          stream.pipeThrough(new JsonToSseTransformStream()),
        ),
      );
    } else {
      return new Response(stream.pipeThrough(new JsonToSseTransformStream()));
    }
  } catch (error) {
    if (error instanceof ChatSDKError) {
      return error.toResponse();
    }
    
    console.error('Unexpected error in chat API:', error);
    return new ChatSDKError('internal_server_error:api').toResponse();
  }
}

export async function DELETE(request: Request) {
  const { searchParams } = new URL(request.url);
  const id = searchParams.get('id');

  if (!id) {
    return new ChatSDKError('bad_request:api').toResponse();
  }

  const session = await auth();

  if (!session?.user) {
    return new ChatSDKError('unauthorized:chat').toResponse();
  }

  const chat = await getChatById({ id });

  if (chat.userId !== session.user.id) {
    return new ChatSDKError('forbidden:chat').toResponse();
  }

  const deletedChat = await deleteChatById({ id });

  return Response.json(deletedChat, { status: 200 });
}
