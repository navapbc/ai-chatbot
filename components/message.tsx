'use client';
import cx from 'classnames';
import { AnimatePresence, motion } from 'framer-motion';
import { memo, useState, useRef, useEffect } from 'react';
import type { Vote } from '@/lib/db/schema';
import { DocumentToolCall, DocumentToolResult } from './document';
import { PencilEditIcon, SparklesIcon } from './icons';
import { Markdown } from './markdown';
import { MessageActions } from './message-actions';
import { PreviewAttachment } from './preview-attachment';
import { Weather } from './weather';
import equal from 'fast-deep-equal';
import { cn, sanitizeText } from '@/lib/utils';
import { Button } from './ui/button';
import { Tooltip, TooltipContent, TooltipTrigger } from './ui/tooltip';
import { MessageEditor } from './message-editor';
import { DocumentPreview } from './document-preview';
import { MessageReasoning } from './message-reasoning';
import type { UseChatHelpers } from '@ai-sdk/react';
import type { ChatMessage } from '@/lib/types';
import { useDataStream } from './data-stream-provider';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from './ui/collapsible';
import { ChevronDown, CheckCircle2 } from 'lucide-react';
import { CollapsibleWrapper } from './ui/collapsible-wrapper';
import { getToolDisplayInfo } from './tool-icon';
import { Spinner } from './ui/spinner';
import {
  subscribeToBrowserAction,
  type BrowserActionEvent,
} from './data-stream-handler';

// Responsive min-height calculation that accounts for side-chat-header height
// This ensures the last message has enough space to scroll properly with the header
const RESPONSIVE_MIN_HEIGHT = 'min-h-[calc(100vh-22rem)] md:min-h-[calc(100vh-24rem)] lg:min-h-[calc(100vh-26rem)]';

// Real-time browser action indicator component
function LiveBrowserActionIndicator() {
  const [action, setAction] = useState<BrowserActionEvent | null>(null);

  useEffect(() => {
    return subscribeToBrowserAction(setAction);
  }, []);

  // Show the current action if running, otherwise show generic "Thinking..."
  const displayText = action?.status === 'running' ? action.action : 'Thinking...';

  return (
    <motion.div
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex items-center gap-2 py-1.5"
    >
      <div className="text-[10px] leading-[150%] font-ibm-plex-mono text-[#9D5A8C] flex items-center gap-2">
        <Spinner className="size-3 shrink-0 text-custom-purple" />
        <span>{displayText}</span>
      </div>
    </motion.div>
  );
}

// Parse partner data from XML-wrapped content in user messages
function parsePartnerData(text: string): { participantData: any; taskText: string } | null {
  const match = text.match(/<partner_context>[\s\S]*?<participant_data>([\s\S]*?)<\/participant_data>[\s\S]*?<\/partner_context>\s*([\s\S]*)/);
  if (!match) return null;

  const jsonData = match[1].trim();
  const taskText = match[2].trim();

  let parsedData;
  try {
    parsedData = JSON.parse(jsonData);
    delete parsedData.task;
    delete parsedData.request;
  } catch {
    parsedData = jsonData;
  }

  return { participantData: parsedData, taskText };
}

// Type narrowing is handled by TypeScript's control flow analysis
// The AI SDK provides proper discriminated unions for tool calls

const PurePreviewMessage = ({
  chatId,
  message,
  vote,
  isLoading,
  setMessages,
  regenerate,
  isReadonly,
  requiresScrollPadding,
}: {
  chatId: string;
  message: ChatMessage;
  vote: Vote | undefined;
  isLoading: boolean;
  setMessages: UseChatHelpers<ChatMessage>['setMessages'];
  regenerate: UseChatHelpers<ChatMessage>['regenerate'];
  isReadonly: boolean;
  requiresScrollPadding: boolean;
}) => {
  const [mode, setMode] = useState<'view' | 'edit'>('view');

  const attachmentsFromMessage = message.parts.filter(
    (part) => part.type === 'file',
  );

  useDataStream();

  return (
    <AnimatePresence>
      <motion.div
        data-testid={`message-${message.role}`}
        className="w-full mx-auto max-w-3xl px-4 group/message"
        initial={{ y: 5, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        data-role={message.role}
      >
        <div
          className={cn(
            'flex gap-4 w-full group-data-[role=user]/message:ml-auto group-data-[role=user]/message:max-w-2xl',
            {
              'w-full': mode === 'edit',
              'group-data-[role=user]/message:w-fit': mode !== 'edit',
            },
          )}
        >

          <div
            className={cn('flex flex-col gap-4 w-full', {
              [RESPONSIVE_MIN_HEIGHT]: message.role === 'assistant' && requiresScrollPadding,
            })}
          >
            {attachmentsFromMessage.length > 0 && (
              <div
                data-testid={`message-attachments`}
                className="flex flex-row justify-end gap-2"
              >
                {attachmentsFromMessage.map((attachment) => (
                  <PreviewAttachment
                    key={attachment.url}
                    attachment={{
                      name: attachment.filename ?? 'file',
                      contentType: attachment.mediaType,
                      url: attachment.url,
                    }}
                  />
                ))}
              </div>
            )}

            {message.parts?.map((part, index) => {
              const { type } = part;
              const key = `message-${message.id}-part-${index}`;

              if (type === 'reasoning' && part.text?.trim().length > 0) {
                return (
                  <MessageReasoning
                    key={key}
                    isLoading={isLoading}
                    reasoning={part.text}
                  />
                );
              }

              if (type === 'text') {
                if (mode === 'view') {
                  const partnerData = parsePartnerData(part.text);

                  if (partnerData && message.role === 'user') {
                    return (
                      <div key={key} className="flex flex-col gap-2 items-end w-full">
                        {partnerData.taskText && (
                          <div
                            data-testid="message-content"
                            className="bg-[#EFD9E9] dark:bg-slate-800 text-black dark:text-slate-100 px-[18px] py-[18px] rounded-xl text-xs leading-[18px] font-inter"
                          >
                            <Markdown>{sanitizeText(partnerData.taskText)}</Markdown>
                          </div>
                        )}
                        <CollapsibleWrapper
                          displayName="Participant data from partner"
                          output={partnerData.participantData}
                        />
                      </div>
                    );
                  }

                  return (
                    <div key={key} className="flex flex-row gap-2 items-start">
                      {/* {message.role === 'user' && !isReadonly && (
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Button
                              data-testid="message-edit-button"
                              variant="ghost"
                              className="px-2 h-fit rounded-full text-muted-foreground opacity-0 group-hover/message:opacity-100"
                              onClick={() => {
                                setMode('edit');
                              }}
                            >
                              <PencilEditIcon />
                            </Button>
                          </TooltipTrigger>
                          <TooltipContent>Edit message</TooltipContent>
                        </Tooltip>
                      )} */}

                      <div
                        data-testid="message-content"
                        className={cn('flex flex-col gap-4', {
                          'bg-[#EFD9E9] dark:bg-slate-800 text-black dark:text-slate-100 px-[18px] py-[18px] rounded-xl text-xs leading-[18px] font-inter':
                            message.role === 'user',
                          'assistant-message-bubble font-source-serif':
                            message.role === 'assistant',
                        })}
                      >
                        <Markdown>{sanitizeText(part.text)}</Markdown>
                      </div>
                    </div>
                  );
                }

                if (mode === 'edit') {
                  return (
                    <div key={key} className="flex flex-row gap-2 items-start">
                      <div className="size-8" />

                      <MessageEditor
                        key={message.id}
                        message={message}
                        setMode={setMode}
                        setMessages={setMessages}
                        regenerate={regenerate}
                      />
                    </div>
                  );
                }
              }

              if (type === 'tool-getWeather') {
                const { toolCallId, state } = part;

                if (state === 'input-available') {
                  return (
                    <div key={toolCallId} className="skeleton">
                      <Weather />
                    </div>
                  );
                }

                if (state === 'output-available') {
                  const { output } = part;
                  return (
                    <div key={toolCallId}>
                      <Weather weatherAtLocation={output} />
                    </div>
                  );
                }
              }

              if (type === 'tool-createDocument') {
                const { toolCallId, state } = part;

                if (state === 'input-available') {
                  const { input } = part;
                  return (
                    <div key={toolCallId}>
                      <DocumentPreview isReadonly={isReadonly} args={input} />
                    </div>
                  );
                }

                if (state === 'output-available') {
                  const { output } = part;

                  if ('error' in output) {
                    return (
                      <div
                        key={toolCallId}
                        className="text-red-500 p-2 border rounded"
                      >
                        Error: {String(output.error)}
                      </div>
                    );
                  }

                  return (
                    <div key={toolCallId}>
                      <DocumentPreview
                        isReadonly={isReadonly}
                        result={output}
                      />
                    </div>
                  );
                }
              }

              if (type === 'tool-updateDocument') {
                const { toolCallId, state } = part;

                if (state === 'input-available') {
                  const { input } = part;

                  return (
                    <div key={toolCallId}>
                      <DocumentToolCall
                        type="update"
                        args={input}
                        isReadonly={isReadonly}
                      />
                    </div>
                  );
                }

                if (state === 'output-available') {
                  const { output } = part;

                  if ('error' in output) {
                    return (
                      <div
                        key={toolCallId}
                        className="text-red-500 p-2 border rounded"
                      >
                        Error: {String(output.error)}
                      </div>
                    );
                  }

                  return (
                    <div key={toolCallId}>
                      <DocumentToolResult
                        type="update"
                        result={output}
                        isReadonly={isReadonly}
                      />
                    </div>
                  );
                }
              }

              if (type === 'tool-requestSuggestions') {
                const { toolCallId, state } = part;

                if (state === 'input-available') {
                  const { input } = part;
                  return (
                    <div key={toolCallId}>
                      <DocumentToolCall
                        type="request-suggestions"
                        args={input}
                        isReadonly={isReadonly}
                      />
                    </div>
                  );
                }

                if (state === 'output-available') {
                  const { output } = part;

                  if ('error' in output) {
                    return (
                      <div
                        key={toolCallId}
                        className="text-red-500 p-2 border rounded"
                      >
                        Error: {String(output.error)}
                      </div>
                    );
                  }

                  return (
                    <div key={toolCallId}>
                      <DocumentToolResult
                        type="request-suggestions"
                        result={output}
                        isReadonly={isReadonly}
                      />
                    </div>
                  );
                }
              }

              // Handle any other tool calls (including web automation tools)
              if (type.startsWith('tool-') && !['tool-getWeather', 'tool-createDocument', 'tool-updateDocument', 'tool-requestSuggestions'].includes(type)) {
                const { toolCallId, state } = part as any;

                if (state === 'input-available') {
                  const { input } = part as any;
                  const { text: displayName, icon: Icon } = getToolDisplayInfo(type, input);

                  // Hide noisy tools
                  if (displayName === 'Updated memory' || displayName === 'Running script') {
                    return null;
                  }

                  // Use CollapsibleWrapper for participant data
                  if (displayName === 'Loading participant data') {
                    return (
                      <CollapsibleWrapper key={toolCallId} displayName={displayName} input={input} icon={Icon} />
                    );
                  }

                  // For all other tools in input-available state, show with spinner (executing)
                  return (
                    <motion.div
                      key={toolCallId}
                      initial={{ opacity: 0, y: 4 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="flex items-center gap-2 py-1.5"
                    >
                      <div className="text-[10px] leading-[150%] font-ibm-plex-mono text-[#767676] flex items-center gap-2">
                        <Spinner className="size-3 shrink-0 text-custom-purple" />
                        <span className="text-[#9D5A8C]">{displayName}</span>
                      </div>
                    </motion.div>
                  );
                }

                if (state === 'output-available') {
                  const { output, input } = part as any;
                  const { text: displayName, icon: Icon } = getToolDisplayInfo(type, input);

                  // Hide noisy tools
                  if (displayName === 'Updated memory' || displayName === 'Running script') {
                    return null;
                  }

                  // Use CollapsibleWrapper for participant data
                  if (displayName === 'Loading participant data') {
                    const participantHasError = output && 'error' in output && output.error !== null && output.error !== undefined;
                    return (
                      <CollapsibleWrapper
                        key={toolCallId}
                        displayName={participantHasError ? 'Failed to load participant data' : 'Loaded participant data'}
                        input={input}
                        output={output}
                        isError={participantHasError}
                        icon={Icon}
                      />
                    );
                  }

                  // Check for actual error
                  const hasError = output && 'error' in output && output.error !== null && output.error !== undefined;

                  // For completed tools, show with checkmark or error indicator
                  return (
                    <motion.div
                      key={toolCallId}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      className="flex items-center gap-2 py-1.5"
                    >
                      <div className={`text-[10px] leading-[150%] font-ibm-plex-mono flex items-center gap-2 ${hasError ? 'text-red-500' : 'text-[#767676]'}`}>
                        {hasError ? (
                          <span className="size-3 shrink-0 text-red-500">✕</span>
                        ) : (
                          <CheckCircle2 size={12} className="shrink-0 text-green-600" />
                        )}
                        <span>{displayName}</span>
                        {hasError && <span className="text-red-500">(failed)</span>}
                      </div>
                    </motion.div>
                  );
                }
              }
            })}

            {isLoading && <LiveBrowserActionIndicator />}

            {/* {!isReadonly && (
              <MessageActions
                key={`action-${message.id}`}
                chatId={chatId}
                message={message}
                vote={vote}
                isLoading={isLoading}
              />
            )} */}
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
};

export const PreviewMessage = memo(
  PurePreviewMessage,
  (prevProps, nextProps) => {
    if (prevProps.isLoading !== nextProps.isLoading) return false;
    if (prevProps.message.id !== nextProps.message.id) return false;
    if (prevProps.requiresScrollPadding !== nextProps.requiresScrollPadding)
      return false;
    if (!equal(prevProps.message.parts, nextProps.message.parts)) return false;
    if (!equal(prevProps.vote, nextProps.vote)) return false;

    return false;
  },
);

export const ThinkingMessage = () => {
  const role = 'assistant';
  const messageRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (messageRef.current) {
      messageRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, []);

  return (
    <motion.div
      ref={messageRef}
      data-testid="message-assistant-loading"
      className={`w-full mx-auto max-w-3xl px-4 group/message ${RESPONSIVE_MIN_HEIGHT}`}
      initial={{ y: 5, opacity: 0 }}
      animate={{ y: 0, opacity: 1, transition: { delay: 1 } }}
      data-role={role}
    >
      <div
        className={cx(
          'flex gap-4 group-data-[role=user]/message:px-3 w-full group-data-[role=user]/message:w-fit group-data-[role=user]/message:ml-auto group-data-[role=user]/message:max-w-2xl group-data-[role=user]/message:py-2 rounded-xl',
          {
            'group-data-[role=user]/message:bg-muted': true,
          },
        )}
      >

        <div className="flex flex-col gap-2 w-full">
          <div className="flex flex-col gap-4 assistant-message-bubble">
            Hmm...
          </div>
        </div>
      </div>
    </motion.div>
  );
};