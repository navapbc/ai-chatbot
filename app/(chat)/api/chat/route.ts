import {
  convertToModelMessages,
  createUIMessageStream,
  JsonToSseTransformStream,
  stepCountIs,
  streamText,
} from 'ai';
import { auth, type UserType } from '@/app/(auth)/auth';
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
import { isProductionEnvironment } from '@/lib/constants';
import { myProvider, webAutomationModel } from '@/lib/ai/providers';
import { entitlementsByUserType } from '@/lib/ai/entitlements';
import { postRequestBodySchema, type PostRequestBody } from './schema';
import {
  createResumableStreamContext,
  type ResumableStreamContext,
} from 'resumable-stream';
import { after } from 'next/server';
import { ChatSDKError } from '@/lib/errors';
import { apricotTools } from '@/lib/ai/tools/apricot';
import { createBrowserTool } from '@/lib/ai/tools/browser';
import { createCheckSubmitGateTool } from '@/lib/ai/tools/check-submit-gate';
import { gapAnalysis } from '@/lib/ai/tools/gap-analysis';
import { formSummary } from '@/lib/ai/tools/form-summary';
import { actionLabel } from '@/lib/ai/tools/action-label';
import { getWebAutomationSystemPrompt, getCurrentDateString } from '@/lib/ai/prompts/web-automation';
import { readReference } from '@/lib/ai/tools/read-reference';
import { prepareMessages } from '@/lib/ai/context-compression';
import { withSlidingCacheBreakpoint } from '@/lib/ai/cache-breakpoints';
import { registerChatAbort, clearChatAbort } from '@/lib/chat-abort-registry';

export const maxDuration = 300; // 5 minutes for web automation tasks

let globalStreamContext: ResumableStreamContext | null = null;

export function getStreamContext() {
  if (!globalStreamContext) {
    try {
      globalStreamContext = createResumableStreamContext({
        waitUntil: after,
      });
    } catch (error: any) {
      if (!error.message.includes('REDIS_URL')) {
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
      modelOverride,
      selectedVisibilityType,
    } = requestBody;

    // Only honour modelOverride in non-production environments.
    const resolvedModelOverride = !isProductionEnvironment ? modelOverride : undefined;

    const session = await auth();

    if (!session?.user) {
      return new ChatSDKError('unauthorized:chat').toResponse();
    }

    const userType: UserType = session.user.type ?? 'regular';

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
    const existingMessageIds = new Set(uiMessages.map((m) => m.id));

    // Save only messages generated during this request (not already in DB).
    const saveNewMessages = async (messages: Array<{ id: string; role: string; parts: unknown }>) => {
      const newMessages = messages.filter((m) => !existingMessageIds.has(m.id));
      if (newMessages.length > 0) {
        await saveMessages({
          messages: newMessages.map((m) => ({
            id: m.id,
            role: m.role,
            parts: m.parts,
            createdAt: new Date(),
            attachments: [],
            chatId: id,
          })),
        });
      }
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

    // Create session ID for browser isolation
    // sessionId includes both chatId and userId to ensure global uniqueness
    const sessionId = `${id}-${session.user.id}`;

    // Register an AbortController the client can trigger via
    // POST /api/chat/stop. Cloud Run HTTP/1.1 does not propagate client
    // disconnects to request.signal, so this explicit channel is the
    // only reliable way to abort an in-flight run from the browser.
    const chatAbort = registerChatAbort(id);

    const stream = createUIMessageStream({
      execute: async ({ writer: dataStream }) => {
        const rawModelMessages = await convertToModelMessages(uiMessages);

        const activeModel = resolvedModelOverride
          ? myProvider.languageModel(resolvedModelOverride)
          : webAutomationModel;
        const isAnthropic = activeModel.provider.includes('anthropic');

        // Cheap char-based token estimate (~4 chars/token) to decide whether
        // to run the one-shot fallback compactor before the step loop starts.
        // Overshooting slightly is fine — the server-side clear_tool_uses edit
        // below also bounds context growth during the run.
        const estimatedInputTokens = Math.ceil(
          rawModelMessages.reduce(
            (n, m) => n + JSON.stringify(m.content ?? '').length,
            0,
          ) / 4,
        );

        // Fallback compaction (Haiku) — runs at most once, only if the request
        // already starts near the window. Vertex doesn't yet accept the
        // Anthropic `compact_20260112` beta header, so we summarize in-app.
        const { messages: preparedMessages, compacted, summary } =
          await prepareMessages(rawModelMessages, estimatedInputTokens, () => {
            dataStream.write({
              type: 'data-compacting',
              data: { timestamp: Date.now() },
              transient: true,
            });
          });
        if (compacted) {
          dataStream.write({
            type: 'data-checkpoint',
            data: {
              stepNumber: 0,
              inputTokens: estimatedInputTokens,
              timestamp: Date.now(),
              summary,
            },
            transient: true,
          });
        }

        // Sliding cache breakpoint on the last message. Combined with the
        // system-prompt breakpoint below, every step (and every later turn)
        // reads system + tools + prior history from cache instead of
        // reprocessing it at full price. No-op for non-Anthropic providers.
        const historyMessages = isAnthropic
          ? withSlidingCacheBreakpoint(preparedMessages)
          : preparedMessages;

        // Anthropic server-side context editing. `clear_tool_uses_20250919`
        // replaces old tool_result payloads with a placeholder once input
        // crosses 80K tokens, keeping the last 4 tool_uses intact — so the
        // step loop stays bounded without us rewriting (and invalidating) the
        // cached message prefix in JS. `updateWorkingMemory` is excluded
        // because its payload is load-bearing participant data.
        // Verified accepted on Vertex via scripts/probe-context-management.ts;
        // when Google enables `compact-2026-01-12` we can add a compact edit.
        const contextManagementOptions = isAnthropic
          ? {
              anthropic: {
                contextManagement: {
                  edits: [
                    {
                      type: 'clear_tool_uses_20250919' as const,
                      trigger: { type: 'input_tokens' as const, value: 80_000 },
                      keep: { type: 'tool_uses' as const, value: 4 },
                      clearAtLeast: {
                        type: 'input_tokens' as const,
                        value: 10_000,
                      },
                      excludeTools: ['updateWorkingMemory'],
                    },
                  ],
                },
              },
            }
          : undefined;

        const result = streamText({
          model: activeModel,
          messages: [
            {
              role: 'system',
              content: getWebAutomationSystemPrompt(),
              providerOptions: {
                anthropic: { cacheControl: { type: 'ephemeral' } },
              },
            },
            {
              role: 'system',
              content: getCurrentDateString(),
            },
            ...historyMessages,
          ],
          providerOptions: contextManagementOptions,
          tools: {
            ...apricotTools,
            gapAnalysis,
            formSummary,
            actionLabel,
            browser: createBrowserTool(sessionId, session.user.id),
            checkSubmitGate: createCheckSubmitGateTool(sessionId, session.user.id),
            readReference,
          },
          // request.signal.aborted is checked at each step boundary so the
          // tool loop halts even before Node's write-failure-based abort
          // detection fires. Without this, streamText keeps running until
          // a write to the closed socket fails — which can be seconds of
          // extra tool calls after the user hits stop.
          // Abort is checked at step boundaries via stopWhen — not
          // passed as abortSignal. Mid-tool abort would leave a
          // tool-call with no matching tool-result, triggering
          // AI_MissingToolResultsError on the next turn.
          stopWhen: [stepCountIs(500), () => chatAbort.signal.aborted],
          // Context growth during the loop is bounded server-side by the
          // clear_tool_uses edit above (no per-step JS compaction, so the
          // cached message prefix stays byte-stable across steps). The
          // one-shot prepareMessages call before streamText handles requests
          // that already start near the window.
          //
          // Emit cumulative token usage after each step so the client can
          // display it in real-time via the Context component, and log
          // Anthropic cache hit/write counts for verification.
          onStepFinish: ({ usage, providerMetadata }) => {
            const cache = (providerMetadata?.anthropic?.usage ?? {}) as Record<
              string,
              number
            >;
            if (cache.cache_read_input_tokens || cache.cache_creation_input_tokens) {
              console.log(
                `[cache] read=${cache.cache_read_input_tokens ?? 0} ` +
                  `write=${cache.cache_creation_input_tokens ?? 0} ` +
                  `uncached=${usage.inputTokens ?? '?'}`,
              );
            }
            dataStream.write({
              type: 'data-token-usage',
              data: usage,
              transient: true,
            });
          },
          experimental_telemetry: {
            isEnabled: isProductionEnvironment,
            functionId: 'web-automation-agent',
          },
        });

        dataStream.merge(result.toUIMessageStream());
      },
      generateId: generateUUID,
      onFinish: async ({ messages }) => {
        clearChatAbort(id, chatAbort);
        await saveNewMessages(messages);
      },
      onError: () => {
        return 'Oops, an error occurred!';
      },
    });

    const streamContext = getStreamContext();

    if (streamContext) {
      return new Response(
        await streamContext.resumableStream(streamId, () =>
          stream.pipeThrough(new JsonToSseTransformStream())
        )
      );
    }
    return new Response(stream.pipeThrough(new JsonToSseTransformStream()));
  } catch (error) {
    if (error instanceof ChatSDKError) {
      return error.toResponse();
    }
    
    console.error('Unexpected error in chat API:', {
      chatId: requestBody?.id,
      error,
    });
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

  if (!chat) {
    return new ChatSDKError('not_found:chat').toResponse();
  }

  if (chat.userId !== session.user.id) {
    return new ChatSDKError('forbidden:chat').toResponse();
  }

  const deletedChat = await deleteChatById({ id });

  return Response.json(deletedChat, { status: 200 });
}
