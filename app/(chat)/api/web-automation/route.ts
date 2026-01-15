import {
  convertToModelMessages,
  createUIMessageStream,
  JsonToSseTransformStream,
  streamText,
  tool,
  stepCountIs,
} from 'ai';
import { openai } from '@ai-sdk/openai';
import { z } from 'zod';
import { auth, type UserType } from '@/app/(auth)/auth';
import { entitlementsByUserType } from '@/lib/ai/entitlements';
import {
  createStreamId,
  getChatById,
  getMessageCountByUserId,
  getMessagesByChatId,
  saveChat,
  saveMessages,
} from '@/lib/db/queries';
import { convertToUIMessages, generateUUID } from '@/lib/utils';
import { generateTitleFromUserMessage } from '../../actions';
import { ChatSDKError } from '@/lib/errors';
import type { ChatMessage } from '@/lib/types';
import type { VisibilityType } from '@/components/visibility-selector';
import { Stagehand } from '@browserbasehq/stagehand';
import { google } from '@ai-sdk/google';

export const maxDuration = 300; // 5 minutes for web automation tasks

// Store active Stagehand sessions
const activeSessions = new Map<string, { stagehand: Stagehand; liveViewUrl: string }>();

// Clean up session when done
async function closeSession(sessionId: string) {
  const session = activeSessions.get(sessionId);
  if (session) {
    console.log(`[WebAutomation] Closing Stagehand session: ${sessionId}`);
    try {
      await session.stagehand.close();
    } catch (e) {
      console.error(`[WebAutomation] Error closing session:`, e);
    }
    activeSessions.delete(sessionId);
  }
}

// Create Stagehand browser tools for a session
function createBrowserTools(sessionId: string) {
  const getStagehand = (): Stagehand => {
    const session = activeSessions.get(sessionId);
    if (!session) {
      throw new Error(`No Stagehand session found for: ${sessionId}`);
    }
    return session.stagehand;
  };

  return {
    browser_navigate: tool({
      description: 'Navigate to a URL in the browser and wait for page to load',
      inputSchema: z.object({
        url: z.string().describe('The URL to navigate to'),
      }),
      execute: async ({ url }) => {
        const stagehand = getStagehand();
        const page = stagehand.context.pages()[0];
        await page.goto(url, { waitUntil: 'domcontentloaded' });
        // Wait a bit for dynamic content
        await new Promise(resolve => setTimeout(resolve, 2000));
        return { success: true, url };
      },
    }),

    browser_act: tool({
      description: 'Perform an action in the browser using natural language. Examples: "click the login button", "type hello in the search box", "scroll down"',
      inputSchema: z.object({
        action: z.string().describe('Natural language description of the action to perform'),
      }),
      execute: async ({ action }) => {
        const stagehand = getStagehand();
        const result = await stagehand.act(action);
        return {
          success: result.success,
          message: result.message || (result.success ? 'Action completed' : 'Action failed'),
        };
      },
    }),

    browser_extract: tool({
      description: 'Extract specific information from the current page using natural language. Returns the extracted data.',
      inputSchema: z.object({
        instruction: z.string().describe('What information to extract from the page'),
      }),
      execute: async ({ instruction }) => {
        const stagehand = getStagehand();
        try {
          // Wait for page to be ready
          const page = stagehand.context.pages()[0];
          await page.waitForLoadState('domcontentloaded');

          const result = await stagehand.extract({
            instruction,
            schema: z.object({
              content: z.string().describe('The extracted content'),
            }),
          });
          console.log('[browser_extract] Raw result:', JSON.stringify(result, null, 2));
          return { success: true, data: result.content || JSON.stringify(result) };
        } catch (error: unknown) {
          console.error('[browser_extract] Error:', error);
          const message = error instanceof Error ? error.message : 'Extraction failed';
          return { success: false, data: message };
        }
      },
    }),

    browser_observe: tool({
      description: 'Observe the current page and identify interactive elements or answer questions about what is visible. Returns a list of elements with descriptions and selectors.',
      inputSchema: z.object({
        instruction: z.string().describe('What to observe or look for on the page'),
      }),
      execute: async ({ instruction }) => {
        const stagehand = getStagehand();
        try {
          // Wait for page to be ready
          const page = stagehand.context.pages()[0];
          await page.waitForLoadState('domcontentloaded');

          const result = await stagehand.observe({ instruction });
          console.log('[browser_observe] Raw result:', JSON.stringify(result, null, 2));

          if (!result || result.length === 0) {
            return {
              success: true,
              observations: [{ description: 'No elements found matching the instruction' }],
            };
          }

          return {
            success: true,
            observations: result.map((r: { description?: string; text?: string; selector?: string }) => ({
              description: r.description || r.text || JSON.stringify(r),
              selector: r.selector,
            })),
          };
        } catch (error: unknown) {
          console.error('[browser_observe] Error:', error);
          const message = error instanceof Error ? error.message : 'Observation failed';
          return { success: false, observations: [{ description: message }] };
        }
      },
    }),

    browser_screenshot: tool({
      description: 'Take a screenshot of the current page',
      inputSchema: z.object({
        fullPage: z.boolean().optional().describe('Whether to capture the full scrollable page'),
      }),
      execute: async ({ fullPage }) => {
        const stagehand = getStagehand();
        const page = stagehand.context.pages()[0];
        await page.screenshot({
          fullPage: fullPage ?? false,
          path: `/tmp/screenshot-${sessionId}-${Date.now()}.png`,
        });
        return { success: true, message: 'Screenshot captured' };
      },
    }),

    browser_get_url: tool({
      description: 'Get the current URL of the browser',
      inputSchema: z.object({}),
      execute: async () => {
        const stagehand = getStagehand();
        const page = stagehand.context.pages()[0];
        return { url: page.url() };
      },
    }),

    browser_close: tool({
      description: 'Close the browser session',
      inputSchema: z.object({}),
      execute: async () => {
        await closeSession(sessionId);
        return { success: true };
      },
    }),
  };
}

// Web automation system prompt
const webAutomationSystemPrompt = `You are a web automation assistant that helps users interact with websites using browser automation.

You have access to browser automation tools that let you:
- Navigate to URLs (browser_navigate)
- Perform actions like clicking, typing, scrolling (browser_act)
- Extract information from pages (browser_extract)
- Observe what's on the page (browser_observe)
- Take screenshots (browser_screenshot)
- Get the current URL (browser_get_url)
- Close the browser when done (browser_close)

When the user asks you to do something on a website:
1. First navigate to the appropriate URL if not already there
2. Use observe to understand what's on the page
3. Use act to perform actions (click buttons, fill forms, etc.)
4. Use extract to get information the user needs
5. Report your findings clearly

Always describe what you're doing and what you see. Be helpful and thorough.`;

export async function POST(request: Request) {
  let sessionId: string | undefined;

  try {
    const json = await request.json();
    const { messages, threadId, resourceId } = json;

    if (!messages || !Array.isArray(messages) || messages.length === 0) {
      return new ChatSDKError('bad_request:api', 'Messages array is required').toResponse();
    }

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

    // Use threadId as chat ID
    const chatId = threadId;
    sessionId = `${threadId}-${resourceId}`;

    console.log(`[WebAutomation] Starting session: ${sessionId}`);

    // Get or create chat
    const chat = await getChatById({ id: chatId });

    if (!chat) {
      const userMessage = messages.find((m: { role: string }) => m.role === 'user');
      const messageText = userMessage?.parts?.find((p: { type: string; text?: string }) => p.type === 'text')?.text ||
                         userMessage?.content || 'Web Automation';
      const title = await generateTitleFromUserMessage({
        message: { id: generateUUID(), role: 'user', parts: [{ type: 'text', text: messageText }] },
      });

      await saveChat({
        id: chatId,
        userId: session.user.id,
        title,
        visibility: 'private' as VisibilityType,
      });
    } else if (chat.userId !== session.user.id) {
      return new ChatSDKError('forbidden:chat').toResponse();
    }

    // Get existing messages from DB and merge with new
    const messagesFromDb = await getMessagesByChatId({ id: chatId });
    const existingUIMessages = convertToUIMessages(messagesFromDb);

    // Convert incoming messages to ChatMessage format
    const incomingMessages: ChatMessage[] = messages.map((m: { id?: string; role: string; parts?: Array<{ type: string; text?: string }>; content?: string }) => ({
      id: m.id || generateUUID(),
      role: m.role as 'user' | 'assistant',
      parts: (m.parts || [{ type: 'text' as const, text: m.content || '' }]) as ChatMessage['parts'],
      metadata: {
        createdAt: new Date().toISOString(),
      },
    }));

    // Save user message
    const userMessage = incomingMessages.find(m => m.role === 'user');
    if (userMessage) {
      await saveMessages({
        messages: [{
          chatId,
          id: userMessage.id,
          role: 'user',
          parts: userMessage.parts,
          attachments: [],
          createdAt: new Date(),
        }],
      });
    }

    // Combine messages for context
    const allMessages = [...existingUIMessages, ...incomingMessages];

    const streamId = generateUUID();
    await createStreamId({ streamId, chatId });

    // Create Stagehand session FIRST to get liveViewUrl
    console.log(`[WebAutomation] Creating Stagehand session...`);
    const stagehand = new Stagehand({
      env: 'BROWSERBASE',
      apiKey: process.env.BROWSERBASE_API_KEY,
      projectId: process.env.BROWSERBASE_PROJECT_ID!,
      llmClient: {
        type: 'aisdk',
        model: google('gemini-3-pro-preview'),
      } as any,
      verbose: 2,
      disablePino: true,
      logger: (logLine) => {
        console.log(`[Stagehand] ${logLine.category || ''}: ${logLine.message}`, logLine.auxiliary || '');
      },
    });

    await stagehand.init();
    const liveViewUrl = stagehand.browserbaseDebugURL || '';
    console.log(`[WebAutomation] Stagehand initialized with liveViewUrl: ${liveViewUrl}`);

    // Store session
    activeSessions.set(sessionId, { stagehand, liveViewUrl });

    // Create browser tools for this session
    const browserTools = createBrowserTools(sessionId);

    // Create the UI message stream
    const stream = createUIMessageStream({
      execute: async ({ writer: dataStream }) => {
        // Send liveViewUrl immediately as a data stream part
        // This uses the same format as the data-stream-handler expects
        dataStream.write({
          type: 'data-liveViewUrl',
          data: liveViewUrl,
        });

        // Now run the AI with browser tools
        const result = streamText({
          model: openai('gpt-4o'),
          system: webAutomationSystemPrompt,
          messages: convertToModelMessages(allMessages),
          tools: browserTools,
          stopWhen: stepCountIs(50),
          experimental_telemetry: {
            isEnabled: true,
            functionId: 'web-automation',
          },
        });

        result.consumeStream();

        dataStream.merge(
          result.toUIMessageStream({
            sendReasoning: true,
          }),
        );
      },
      generateId: generateUUID,
      onFinish: async ({ messages: responseMessages }) => {
        // Save assistant messages
        await saveMessages({
          messages: responseMessages.map((message) => ({
            id: message.id,
            role: message.role,
            parts: message.parts,
            createdAt: new Date(),
            attachments: [],
            chatId,
          })),
        });
        // Clean up session
        await closeSession(sessionId!);
      },
      onError: (error: unknown) => {
        console.error('[WebAutomation] Stream error:', error);
        closeSession(sessionId!);
        return 'An error occurred during web automation.';
      },
    });

    return new Response(stream.pipeThrough(new JsonToSseTransformStream()));
  } catch (error) {
    console.error('[WebAutomation] Error:', error);
    if (sessionId) {
      await closeSession(sessionId);
    }

    if (error instanceof ChatSDKError) {
      return error.toResponse();
    }

    return new ChatSDKError('internal_server_error:api').toResponse();
  }
}

// Handle stop requests
export async function DELETE(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const threadId = searchParams.get('threadId');
    const resourceId = searchParams.get('resourceId');

    if (!threadId || !resourceId) {
      return new Response(
        JSON.stringify({ error: 'threadId and resourceId are required' }),
        { status: 400, headers: { 'Content-Type': 'application/json' } }
      );
    }

    const sessionId = `${threadId}-${resourceId}`;
    await closeSession(sessionId);

    return new Response(
      JSON.stringify({ status: 'ok', message: 'Session closed', sessionId }),
      { status: 200, headers: { 'Content-Type': 'application/json' } }
    );
  } catch (error) {
    console.error('[WebAutomation] Error stopping session:', error);
    return new Response(
      JSON.stringify({ error: 'Failed to stop session' }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    );
  }
}
