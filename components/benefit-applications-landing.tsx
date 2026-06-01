'use client';

import { useState, useRef, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { ChevronsUpDown } from 'lucide-react';
import type { ChatMessage, Attachment } from '@/lib/types';
import type { UseChatHelpers } from '@ai-sdk/react';
import type { VisibilityType } from './visibility-selector';
import type { Dispatch, SetStateAction } from 'react';
import type { Session } from 'next-auth';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { MultimodalInput } from '@/components/multimodal-input';

const PROGRAMS = [
  { id: 'wic', name: 'Apply 4 WIC Form', website: 'https://www.ruhealth.org/appointments/apply-4-wic-form' },
  { id: 'ihss', name: 'In-Home Supportive Services (IHSS): Intake Application', website: 'https://riversideihss.org/IntakeApp' },
  { id: 'benefits-cal', name: 'BenefitsCal', website: 'https://benefitscal.com/' },
  // { id: 'head-start', name: 'RCOE Early Head Start: 0-3; Head Start: 3-5', website: 'https://app.informedk12.com/link_campaigns/rcoe-head-start-ehs-application-english-electronic-form?token=RX3jMrVUfWLz3aQnjpiQpseu' },
  { id: 'nurse-family-partnership', name: 'Nurse-Family Partnership', website: 'https://forms.office.com/Pages/ResponsePage.aspx?id=yqoVt4-WGUe7BO0xcOCKaQVow3g6-R9Mh0F8VizNQzhUQ1hCNENFTFBMOVg4SElRSldIWk5BRUkxUi4u' },
  { id: 'pregnancy-planning', name: 'RCOE Referrals System', website: 'https://rrrcoe.nohosoftware.com/online_referrals/' },
];

type TabId = 'select' | 'describe';

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
  chatId,
  sendMessage,
  session,
  status,
  stop,
  attachments,
  setAttachments,
  messages,
  setMessages,
  selectedVisibilityType,
}: BenefitApplicationsLandingProps) {
  const router = useRouter();
  const [activeTab, setActiveTab] = useState<TabId>('select');
  const [clientId, setClientId] = useState('');
  const [program, setProgram] = useState<(typeof PROGRAMS)[number] | null>(null);
  const [query, setQuery] = useState('');
  const [isComboOpen, setIsComboOpen] = useState(false);
  const comboRef = useRef<HTMLDivElement>(null);
  const isLoggedIn = !!session;

  const isUrl = (s: string) => {
    try { new URL(s); return true; } catch { return false; }
  };

  const filteredPrograms = PROGRAMS.filter((p) =>
    p.name.toLowerCase().includes(query.toLowerCase()),
  );

  const queryRef = useRef(query);
  queryRef.current = query;

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (comboRef.current && !comboRef.current.contains(e.target as Node)) {
        setIsComboOpen(false);
        if (!isUrl(queryRef.current)) {
          setQuery(program?.name ?? '');
        }
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [program]);

  const submitMessage = (text: string) => {
    window.history.replaceState({}, '', `/chat/${chatId}`);
    sendMessage({ role: 'user', parts: [{ type: 'text', text }] });
  };

  const handleStartAutoFilling = () => {
    if (!isLoggedIn || !clientId || (!program && !isUrl(query))) return;
    const target = program ? `${program.name} at ${program.website}` : query;
    submitMessage(`Retrieve ID #${clientId} and apply for ${target}`);
  };

  const loginAlert = (
    <Alert className="border-primary/30 bg-primary/10">
      <AlertDescription className="flex flex-col items-stretch gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex flex-col gap-1 font-inter">
          <span className="text-base font-medium">Log in to get started</span>
          <span className="text-sm text-muted-foreground">
            You&apos;ll be able to complete applications once you&apos;re logged in.
          </span>
        </div>
        <Button
          onClick={() => router.push('/login')}
          className="shrink-0 bg-primary hover:bg-primary/90 text-primary-foreground"
        >
          Log in
        </Button>
      </AlertDescription>
    </Alert>
  );

  const tabButtonClass = (active: boolean) =>
    `flex-1 rounded-[10px] px-2.5 py-1.5 font-inter text-sm font-medium leading-6 transition-colors ${
      active
        ? 'bg-card text-foreground shadow-[0px_1px_1.5px_rgba(0,0,0,0.1),0px_1px_1px_rgba(0,0,0,0.1)]'
        : 'text-foreground/80 hover:text-foreground'
    }`;

  return (
    <div className="flex flex-1 flex-col items-center overflow-y-auto bg-chat-background px-4 sm:px-6">
      <div className="my-auto flex w-full max-w-[648px] flex-col gap-6 py-10">
        <h1 className="font-source-serif text-3xl leading-[1.15] text-foreground text-left sm:text-4xl">
          Let&apos;s start a new application.
        </h1>

        {!isLoggedIn && loginAlert}

        <div className="flex w-full flex-col rounded-xl bg-card text-card-foreground shadow-sm md:min-h-[538px] md:w-[648px]">
          {/* Tabs */}
          <div className="px-6 pt-6 sm:px-8">
            <div
              role="tablist"
              aria-label="Application entry mode"
              className="flex items-center gap-2 rounded-xl bg-[#f5f5f5] p-2"
            >
              <button
                type="button"
                role="tab"
                aria-selected={activeTab === 'select' ? 'true' : 'false'}
                onClick={() => setActiveTab('select')}
                className={tabButtonClass(activeTab === 'select')}
              >
                Select program
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={activeTab === 'describe' ? 'true' : 'false'}
                onClick={() => setActiveTab('describe')}
                className={tabButtonClass(activeTab === 'describe')}
              >
                Describe what you need
              </button>
            </div>
          </div>

          {/* Tab content */}
          {activeTab === 'select' ? (
            <div className="flex-1 px-6 py-6 sm:px-8">
              {/* Client ID */}
              <div className="mb-12">
                <p className="font-source-serif text-lg font-semibold text-foreground sm:text-xl">Client ID</p>
                <p className="mt-1 font-source-serif text-sm text-muted-foreground sm:text-base">
                  Enter the client&apos;s Apricot 360 ID.
                </p>
                <input
                  type="text"
                  placeholder="00000"
                  value={clientId}
                  onChange={(e) => setClientId(e.target.value)}
                  disabled={!isLoggedIn}
                  className="mt-3 h-[52px] w-[129px] rounded-[10px] border border-[#b5b5b5] px-4 font-inter text-base placeholder:text-[#b5b5b5] focus:border-primary focus:shadow-[0px_0px_8px_0px_rgba(177,64,146,0.25)] focus:outline-none disabled:cursor-not-allowed disabled:opacity-40"
                />
              </div>

              {/* Application */}
              <div>
                <p className="font-source-serif text-lg font-semibold text-foreground sm:text-xl">Application</p>
                <p className="mt-1 font-source-serif text-sm text-muted-foreground sm:text-base">
                  Select a program or paste an application URL.
                </p>
                <div ref={comboRef} className="relative mt-3">
                  <div
                    className={`flex h-[52px] w-full items-center rounded-[10px] border bg-card px-4 transition-colors ${
                      isComboOpen ? 'border-primary ring-2 ring-primary/20' : 'border-[#b5b5b5]'
                    } ${!isLoggedIn ? 'cursor-not-allowed opacity-40' : ''}`}
                  >
                    <input
                      type="text"
                      value={query}
                      onChange={(e) => {
                        const val = e.target.value;
                        setQuery(val);
                        if (isUrl(val)) {
                          setIsComboOpen(false);
                          if (program) setProgram(null);
                        } else {
                          setIsComboOpen(true);
                          if (program && val !== program.name) setProgram(null);
                        }
                      }}
                      onFocus={() => { if (!isUrl(query)) setIsComboOpen(true); }}
                      placeholder="Select a program or paste URL"
                      disabled={!isLoggedIn}
                      className="flex-1 bg-transparent font-inter text-base text-foreground placeholder:text-[#8e8e8e] focus:outline-none disabled:cursor-not-allowed truncate"
                    />
                    {query ? (
                      <button
                        type="button"
                        onMouseDown={(e) => {
                          e.preventDefault();
                          setProgram(null);
                          setQuery('');
                        }}
                        aria-label="Clear selection"
                        className="shrink-0 text-[#8e8e8e] hover:text-foreground"
                      >
                        &#x2715;
                      </button>
                    ) : (
                      <button
                        type="button"
                        onMouseDown={(e) => {
                          e.preventDefault();
                          if (!isLoggedIn) return;
                          setIsComboOpen((prev) => !prev);
                        }}
                        aria-label="Toggle program list"
                        className="shrink-0"
                      >
                        <ChevronsUpDown className="h-5 w-5 text-[#8e8e8e]" />
                      </button>
                    )}
                  </div>
                  {isComboOpen && (
                    <div className="absolute top-full z-30 mt-1 max-h-[280px] w-full overflow-y-auto rounded-[10px] border border-[#b5b5b5] bg-card shadow-md">
                      {filteredPrograms.length === 0 ? (
                        <p className="px-4 py-3 font-inter text-sm text-muted-foreground">
                          No matching programs found. <br />
                          <span className="text-sm text-muted-foreground">You can paste an application URL instead.</span>
                        </p>
                      ) : (
                        filteredPrograms.map((p) => (
                          <button
                            key={p.id}
                            type="button"
                            onMouseDown={(e) => {
                              e.preventDefault();
                              setProgram(p);
                              setQuery(p.name);
                              setIsComboOpen(false);
                            }}
                            className={`block w-full px-4 py-2 text-left font-inter text-sm hover:bg-primary/10 truncate sm:text-base ${
                              program?.id === p.id ? 'bg-primary/5 text-primary' : 'text-foreground'
                            }`}
                          >
                            {p.name}
                          </button>
                        ))
                      )}
                    </div>
                  )}
                </div>
              </div>
            </div>
          ) : (
            <div className="flex flex-1 flex-col px-6 py-6 sm:px-8">
              <p className="font-source-serif text-lg font-semibold text-foreground sm:text-xl">
                Describe what you need
              </p>
              <p className="mt-1 font-source-serif text-sm text-muted-foreground sm:text-base">
                Use this for clients without an Apricot 360 ID, multiple programs, or programs not in the list.
              </p>
              <div className="mt-3 flex flex-1 flex-col">
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
                  fullWidthSubmit
                  placeholder="Apply [ID ####] for [Program URL]"
                  session={session}
                />
              </div>
            </div>
          )}

          {/* Footer button — Select program tab only */}
          {activeTab === 'select' && (
            <div className="border-t border-border px-6 py-5 sm:px-8">
              <button
                type="button"
                onClick={handleStartAutoFilling}
                disabled={!isLoggedIn || !clientId || (!program && !isUrl(query))}
                className="w-full rounded-lg bg-primary px-4 py-2.5 font-inter text-sm font-medium tracking-[0.08px] text-primary-foreground transition-colors hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-50"
              >
                Start auto-filling
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
