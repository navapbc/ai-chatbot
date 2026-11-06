'use client';

export default function HandoffDiagramPage() {
  return (
    <div className="min-h-screen bg-chat-background">
      <div className="mx-auto max-w-[720px] px-6 py-20">
        {/* Title */}
        <p className="font-source-serif text-4xl leading-[1.15] text-foreground">
          Agent Handoff
        </p>
        <p className="mt-3 font-source-serif text-xl text-[#787878]">
          How control transfers between the AI agent and the caseworker. The
          agent fills but never submits. The caseworker always has the final
          say.
        </p>

        {/* Rectangular cycle diagram */}
        <div className="mt-14 mx-auto max-w-lg">
          <div className="rounded-xl bg-[#ffffff] border border-primary/10 px-6 py-6">
            {/* Top connector: Agent mode → Take control */}
            <div className="flex items-center mb-3 px-8">
              <div className="w-[10px] shrink-0" />
              <div className="h-[1.5px] flex-1 bg-primary" />
              <span className="mx-2 inline-flex items-center gap-1 rounded-md bg-primary px-3 py-1.5 font-inter text-[11px] font-medium text-white shadow-sm pointer-events-none shrink-0">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="13"
                  height="13"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="M9 2l-5 12 7.5-1 1 7 5-12-7.5 1-1-7z" />
                </svg>
                Take control
              </span>
              <div className="h-[1.5px] flex-1 bg-primary" />
              <svg
                width="10"
                height="10"
                viewBox="0 0 10 10"
                className="shrink-0 -ml-px"
              >
                <polygon points="0,1 10,5 0,9" fill="hsl(320 47% 47%)" />
              </svg>
            </div>

            <div className="grid grid-cols-2 gap-4">
              {/* Top-left: Agent mode */}
              <div className="rounded-lg border border-primary/25 bg-white px-5 py-4 shadow-sm">
                <div className="flex items-center gap-2 mb-2">
                  <span className="h-2 w-2 rounded-full bg-[#b5b5b5] shrink-0" />
                  <p className="font-source-serif text-sm font-bold text-foreground">
                    Agent mode
                  </p>
                </div>
                <p className="font-inter text-[10px] text-[#787878] mb-2">
                  Agent controls the browser. Caseworker watches.
                </p>
                <div className="space-y-1">
                  <div className="flex items-start gap-2">
                    <span className="mt-[5px] h-1 w-1 rounded-full bg-primary/40 shrink-0" />
                    <code className="font-mono text-[9px] text-primary/70">
                      pointer-events: none
                    </code>
                  </div>
                  <div className="flex items-start gap-2">
                    <span className="mt-[5px] h-1 w-1 rounded-full bg-primary/40 shrink-0" />
                    <code className="font-mono text-[9px] text-primary/70">
                      readOnly=true
                    </code>
                  </div>
                  <div className="flex items-start gap-2">
                    <span className="mt-[5px] h-1 w-1 rounded-full bg-primary/40 shrink-0" />
                    <span className="font-inter text-[9px] text-[#787878]">
                      Iframe visible but locked
                    </span>
                  </div>
                </div>
              </div>

              {/* Top-right: Take control */}
              <div className="rounded-lg border border-primary/25 bg-white px-5 py-4 shadow-sm">
                <div className="flex items-center gap-2 mb-2">
                  <span className="h-2 w-2 rounded-full bg-amber-500 shrink-0" />
                  <p className="font-source-serif text-sm font-bold text-foreground">
                    Take control
                  </p>
                </div>
                <p className="font-inter text-[10px] text-[#787878] mb-2">
                  Abort the agent, unlock the browser.
                </p>
                <div className="space-y-1">
                  <div className="flex items-start gap-2">
                    <span className="mt-[5px] h-1 w-1 rounded-full bg-primary/40 shrink-0" />
                    <span className="font-inter text-[9px] text-[#787878]">
                      <code className="text-primary/70">stop()</code> aborts AI
                      stream
                    </span>
                  </div>
                  <div className="flex items-start gap-2">
                    <span className="mt-[5px] h-1 w-1 rounded-full bg-primary/40 shrink-0" />
                    <span className="font-inter text-[9px] text-[#787878]">
                      <code className="text-primary/70">abortSignal</code> races
                      Playwright
                    </span>
                  </div>
                  <div className="flex items-start gap-2">
                    <span className="mt-[5px] h-1 w-1 rounded-full bg-primary/40 shrink-0" />
                    <span className="font-inter text-[9px] text-[#787878]">
                      In-flight commands reject cleanly
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Side connectors */}
            <div className="grid grid-cols-2 gap-4 my-1">
              <div className="flex justify-center">
                <svg width="20" height="28" viewBox="0 0 20 28">
                  <line
                    x1="10"
                    y1="24"
                    x2="10"
                    y2="8"
                    stroke="hsl(320 47% 47%)"
                    strokeWidth="1.5"
                  />
                  <polygon points="6,8 10,0 14,8" fill="hsl(320 47% 47%)" />
                </svg>
              </div>
              <div className="flex justify-center">
                <svg width="20" height="28" viewBox="0 0 20 28">
                  <line
                    x1="10"
                    y1="0"
                    x2="10"
                    y2="20"
                    stroke="hsl(320 47% 47%)"
                    strokeWidth="1.5"
                  />
                  <polygon points="6,20 10,28 14,20" fill="hsl(320 47% 47%)" />
                </svg>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              {/* Bottom-left: Give back control */}
              <div className="rounded-lg border border-primary/25 bg-white px-5 py-4 shadow-sm">
                <div className="flex items-center gap-2 mb-2">
                  <span className="h-2 w-2 rounded-full bg-green-500 shrink-0" />
                  <p className="font-source-serif text-sm font-bold text-foreground">
                    Give back control
                  </p>
                </div>
                <p className="font-inter text-[10px] text-[#787878] mb-2">
                  Caseworker hands back to the agent.
                </p>
                <div className="space-y-1">
                  <div className="flex items-start gap-2">
                    <span className="mt-[5px] h-1 w-1 rounded-full bg-primary/40 shrink-0" />
                    <span className="font-inter text-[9px] text-[#787878]">
                      <code className="text-primary/70">sendMessage()</code>{' '}
                      notifies agent to resume
                    </span>
                  </div>
                  <div className="flex items-start gap-2">
                    <span className="mt-[5px] h-1 w-1 rounded-full bg-primary/40 shrink-0" />
                    <span className="font-inter text-[9px] text-[#787878]">
                      Agent takes fresh snapshot, continues
                    </span>
                  </div>
                </div>
              </div>

              {/* Bottom-right: User mode */}
              <div className="rounded-lg border border-primary/25 bg-white px-5 py-4 shadow-sm">
                <div className="flex items-center gap-2 mb-2">
                  <span className="h-2 w-2 rounded-full bg-red-500 shrink-0" />
                  <p className="font-source-serif text-sm font-bold text-foreground">
                    User mode
                  </p>
                </div>
                <p className="font-inter text-[10px] text-[#787878] mb-2">
                  Caseworker has full control. Agent is paused.
                </p>
                <div className="space-y-1">
                  <div className="flex items-start gap-2">
                    <span className="mt-[5px] h-1 w-1 rounded-full bg-primary/40 shrink-0" />
                    <code className="font-mono text-[9px] text-primary/70">
                      pointer-events: auto
                    </code>
                  </div>
                  <div className="flex items-start gap-2">
                    <span className="mt-[5px] h-1 w-1 rounded-full bg-primary/40 shrink-0" />
                    <code className="font-mono text-[9px] text-primary/70">
                      readOnly
                    </code>
                    <span className="font-inter text-[9px] text-[#787878]">
                      param removed
                    </span>
                  </div>
                  <div className="flex items-start gap-2">
                    <span className="mt-[5px] h-1 w-1 rounded-full bg-primary/40 shrink-0" />
                    <span className="font-inter text-[9px] text-[#787878]">
                      Mouse + keyboard enabled
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Bottom connector: User mode → Give back */}
            <div className="flex items-center mt-3 px-8">
              <svg
                width="10"
                height="10"
                viewBox="0 0 10 10"
                className="shrink-0 -mr-px"
              >
                <polygon points="10,1 0,5 10,9" fill="hsl(320 47% 47%)" />
              </svg>
              <div className="h-[1.5px] flex-1 bg-primary" />
              <span className="mx-2 inline-flex items-center gap-1 rounded-md border border-primary/25 bg-white px-3 py-1.5 font-inter text-[11px] font-medium text-foreground shadow-sm pointer-events-none shrink-0">
                Give back control
              </span>
              <div className="h-[1.5px] flex-1 bg-primary" />
              <div className="w-[10px] shrink-0" />
            </div>
          </div>
        </div>

        {/* Safety Guarantees */}
        <div className="mt-14 mx-auto max-w-lg">
          <p className="font-source-serif text-xl font-bold text-foreground">
            Safety Guarantees
          </p>
          <p className="mt-1 font-source-serif text-sm text-[#787878]">
            Three layers ensure the agent never submits on behalf of the
            caseworker.
          </p>

          <div className="mt-6 rounded-lg border border-primary/25 bg-white px-6 py-6 shadow-sm space-y-4">
            <div>
              <p className="font-source-serif text-sm font-bold text-foreground">
                System prompt
              </p>
              <ul className="mt-1 space-y-1">
                <li className="flex items-center gap-2 font-inter text-xs text-[#787878]">
                  <span className="h-1 w-1 rounded-full bg-primary/40 shrink-0" />
                  Agent is instructed to never click submit buttons
                </li>
              </ul>
            </div>
            <div>
              <p className="font-source-serif text-sm font-bold text-foreground">
                Communication skill
              </p>
              <ul className="mt-1 space-y-1">
                <li className="flex items-center gap-2 font-inter text-xs text-[#787878]">
                  <span className="h-1 w-1 rounded-full bg-primary/40 shrink-0" />
                  Form summary tool renders a review card instead of submitting
                </li>
              </ul>
            </div>
            <div>
              <p className="font-source-serif text-sm font-bold text-foreground">
                Browser iframe
              </p>
              <ul className="mt-1 space-y-1">
                <li className="flex items-center gap-2 font-inter text-xs text-[#787878]">
                  <span className="h-1 w-1 rounded-full bg-primary/40 shrink-0" />
                  <code className="text-primary/70 text-[10px]">
                    pointer-events: none
                  </code>{' '}
                  until caseworker explicitly takes control
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
