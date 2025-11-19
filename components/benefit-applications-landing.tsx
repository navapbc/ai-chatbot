'use client';

import type { ChatMessage, Attachment } from '@/lib/types';
import type { UseChatHelpers } from '@ai-sdk/react';
import type { VisibilityType } from './visibility-selector';
import type { Dispatch, SetStateAction } from 'react';
import { MultimodalInput } from './multimodal-input';
import { Alert, AlertDescription } from './ui/alert';
import { Button } from './ui/button';
import { useRouter } from 'next/navigation';
import type { Session } from 'next-auth';
import { guestRegex } from '@/lib/constants';

interface BenefitApplicationsLandingProps {
  input: string;
  setInput: Dispatch<SetStateAction<string>>;
  isReadonly: boolean;
  chatId: string;
  sendMessage: UseChatHelpers<ChatMessage>['sendMessage'];
  selectedVisibilityType: VisibilityType;
  status: UseChatHelpers<ChatMessage>['status'];
  stop: () => void;
  attachments: Array<Attachment>;
  setAttachments: Dispatch<SetStateAction<Array<Attachment>>>;
  messages: ChatMessage[];
  setMessages: UseChatHelpers<ChatMessage>['setMessages'];
  session: Session | null;
}

export function BenefitApplicationsLanding({
  input,
  setInput,
  isReadonly,
  chatId,
  sendMessage,
  selectedVisibilityType,
  status,
  stop,
  attachments,
  setAttachments,
  messages,
  setMessages,
  session,
}: BenefitApplicationsLandingProps) {
  const router = useRouter();
  const isGuest = !session || guestRegex.test(session?.user?.email ?? '');
  
  return (
    <div className="flex-1 flex flex-col items-center justify-center p-8 bg-chat-background">
      <div className="max-w-4xl w-full text-left">
        {/* Main Title */}
        <h1 className="text-[64px] font-source-serif leading-[1.15] text-black dark:text-white mb-12">
          Let&apos;s start a new application.
        </h1>

        {/* Subheader */}
        <h2 className="text-2xl font-inter text-black dark:text-white mb-12">
          What&apos;s your client&apos;s name and which program do they need?
        </h2>

        {/* Login Warning Alert */}
        {isGuest && (
          <Alert variant="warning" className="mb-6 bg-custom-purple/10 border-custom-purple/30">
            <AlertDescription className="flex items-center justify-between">
              <div className="flex flex-col gap-1 font-inter">
                <span className="text-base font-medium">Log in to get started</span>
                <span className="text-sm text-gray-600 dark:text-gray-400">You'll be able to complete applications once you're logged in.</span>
              </div>
              <Button
                onClick={() => router.push('/login')}
                className="bg-custom-purple hover:bg-custom-purple/90 text-white px-4 py-2 shrink-0"
              >
                Log in
              </Button>
            </AlertDescription>
          </Alert>
        )}

        {/* Input Form */}
        <div className="mb-8 max-w-4xl mx-auto">
          <MultimodalInput
            chatId={chatId}
            input={input}
            setInput={setInput}
            status={status}
            stop={stop}
            attachments={attachments}
            setAttachments={setAttachments}
            messages={messages}
            setMessages={setMessages}
            sendMessage={sendMessage}
            selectedVisibilityType={selectedVisibilityType}
            showStopButton={false}
            placeholder="Ex. Fill out the WIC form for Jane Doe"
            isDisabled={isGuest}
          />
        </div>
      </div>
    </div>
  );
}
