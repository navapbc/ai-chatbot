'use client';

const loopSteps = [
  {
    label: 'Snapshot',
    mono: '{ action: "snapshot" }',
    detail: 'Accessibility tree with refs. 200 tokens vs 3,000 for raw HTML.',
    response: `textbox "First Name" [ref=@e2] [id=firstNameTxt]
textbox "SSN"        [ref=@e3] [id=ssnTxt]
button  "Next"       [ref=@e4]`,
  },
  {
    label: 'Act',
    mono: '{ action: "fill", selector: "@e2",\n  value: "Maria" }',
    detail:
      '<fill> for plain text, <type> for masked fields (SSN, dates, phone)',
    response: '{ success: true }',
  },
  {
    label: 'Verify',
    mono: '{ action: "snapshot" }',
    detail:
      'Re-snapshot to confirm values stuck. Retry masked fields if needed.',
    response: null,
  },
  {
    label: 'Advance',
    mono: '{ action: "click", selector: "@e4" }',
    detail: 'Next page, fresh snapshot, repeat.',
    response: null,
  },
];

export default function BrowserDiagramPage() {
  return (
    <div className="min-h-screen bg-chat-background">
      <div className="mx-auto max-w-[720px] px-6 py-20">
        {/* Title */}
        <p className="font-source-serif text-4xl leading-[1.15] text-foreground">
          The Browser
        </p>
        <p className="mt-3 font-source-serif text-xl text-[#787878]">
          Kernel hosts a real Chrome in the cloud. One API call returns two
          URLs&mdash;one for the agent to control, one for the caseworker to
          watch. Same session, two consumers.
        </p>

        {/* Two-path diagram */}
        <div className="mt-14 mx-auto max-w-lg rounded-2xl bg-[#ffffff] p-6">
          <div className="rounded-xl outline outline-[10px] outline-[#e5e5e5]/25 bg-chat-background p-4">
            {/* Browser center box */}
            <div className="rounded-lg border border-primary/25 bg-white px-4 py-2 shadow-sm text-center">
              <p className="font-source-serif text-sm font-bold text-foreground leading-tight">
                Cloud Browser
              </p>
              <p className="font-inter text-[10px] text-[#b5b5b5] leading-tight">
                SOC 2 Type II &middot; stealth mode &middot; 10-min idle timeout
              </p>
            </div>

            {/* Connecting lines from center box */}
            <div className="grid grid-cols-2 gap-3">
              <div className="flex justify-center">
                <div className="h-5 w-px bg-primary/25" />
              </div>
              <div className="flex justify-center">
                <div className="h-5 w-px bg-primary/25" />
              </div>
            </div>

            {/* Two paths */}
            <div className="grid grid-cols-2 gap-3">
              {/* Agent path */}
              <div className="flex flex-col items-center">
                <div className="rounded-lg border border-primary/25 bg-white px-4 py-4 shadow-sm w-full h-full">
                  <p className="font-source-serif text-sm font-bold text-foreground">
                    Agent path
                  </p>
                  <div className="mt-3 space-y-0">
                    {[
                      {
                        mono: 'kernel.browsers.create()',
                        label: 'Create browser',
                      },
                      {
                        mono: 'cdp_ws_url',
                        label: 'Playwright connects via CDP',
                      },
                      {
                        mono: 'executeCommand()',
                        label: 'fill / click / type / snapshot',
                      },
                      {
                        mono: '{ success, output }',
                        label: 'Structured response',
                      },
                    ].map((step, i, arr) => (
                      <div key={step.mono}>
                        <div className="grid grid-cols-[16px_1fr] gap-3 items-center">
                          <span className="flex h-4 w-4 items-center justify-center rounded-full bg-primary/10 font-mono text-[9px] font-bold text-primary shrink-0">
                            {i + 1}
                          </span>
                          <code className="font-mono text-[10px] leading-none text-foreground/70">
                            {step.mono}
                          </code>
                        </div>
                        <div className="grid grid-cols-[16px_1fr] gap-3">
                          <div className="flex justify-center">
                            {i < arr.length - 1 && (
                              <div className="h-full w-px bg-primary/10" />
                            )}
                          </div>
                          <p className="pb-3 font-inter text-[10px] text-[#b5b5b5] leading-snug">
                            {step.label}
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Caseworker path */}
              <div className="flex flex-col items-center">
                <div className="rounded-lg border border-primary/25 bg-white px-4 py-4 shadow-sm w-full h-full">
                  <p className="font-source-serif text-sm font-bold text-foreground">
                    Caseworker path
                  </p>
                  <div className="mt-3 space-y-0">
                    {[
                      {
                        mono: 'browser_live_view_url',
                        label: 'Live-view URL returned',
                      },
                      { mono: '<iframe>', label: 'Embedded in chat UI' },
                      {
                        mono: 'real-time',
                        label: 'Watches every click and fill',
                      },
                      { mono: 'keepalive', label: 'iframe IS the keepalive' },
                    ].map((step, i, arr) => (
                      <div key={step.mono}>
                        <div className="grid grid-cols-[16px_1fr] gap-3 items-center">
                          <span className="flex h-4 w-4 items-center justify-center rounded-full bg-primary/10 font-mono text-[9px] font-bold text-primary/60 shrink-0">
                            {i + 1}
                          </span>
                          <code className="font-mono text-[10px] leading-none text-foreground/70">
                            {step.mono}
                          </code>
                        </div>
                        <div className="grid grid-cols-[16px_1fr] gap-3">
                          <div className="flex justify-center">
                            {i < arr.length - 1 && (
                              <div className="h-full w-px bg-primary/10" />
                            )}
                          </div>
                          <p className="pb-3 font-inter text-[10px] text-[#b5b5b5] leading-snug">
                            {step.label}
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* The Loop — single slide layout */}
        <div className="mt-20 mx-auto max-w-lg">
          <p className="font-source-serif text-xl font-bold text-foreground">
            The Loop
          </p>
          <p className="mt-1 font-source-serif text-sm text-[#787878]">
            Snapshot, act, verify, advance. Every page, every form.
          </p>

          <div className="mt-6 rounded-2xl bg-[#ffffff] p-6">
            <div className="rounded-xl border-[4px] border-[#ffffff] outline outline-[10px] outline-[#e5e5e5]/25 bg-[#ffffff] p-0">
              <div className="rounded-lg bg-[#ffffff] px-3 py-3">
                {/* Steps */}
                <div className="grid grid-cols-2 gap-x-8 gap-y-5">
                  {loopSteps.map((step, i) => (
                    <div key={step.label} className="flex flex-col">
                      <div className="flex items-center gap-2 mb-2">
                        <span className="flex h-5 w-5 items-center justify-center rounded-full bg-primary/10 font-inter text-[10px] font-bold text-primary shrink-0">
                          {i + 1}
                        </span>
                        <span className="font-source-serif text-sm font-bold text-foreground">
                          {step.label}
                        </span>
                      </div>
                      <pre className="font-mono text-[9px] text-foreground/60 leading-snug whitespace-pre-wrap">
                        {step.mono}
                      </pre>
                      <p className="mt-1 font-inter text-[9px] text-[#b5b5b5] leading-snug">
                        {step.detail.split(/(<\w+>)/).map((part, j) => {
                          const match = part.match(/^<(\w+)>$/);
                          if (match)
                            return (
                              <code key={part} className="text-primary/70">
                                {match[1]}
                              </code>
                            );
                          return part;
                        })}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Slide 2: Repeat */}
        <div className="mt-10 mx-auto max-w-lg rounded-lg border border-primary/25 bg-white px-8 py-8 shadow-sm">
          <p className="font-source-serif text-xl font-bold text-foreground">
            Key Mechanics
          </p>
          <p className="mt-1 font-source-serif text-sm text-[#787878]">
            How the agent stays fast, safe, and stateful across 100+ steps.
          </p>

          <div className="mt-5 space-y-4">
            <div>
              <p className="font-source-serif text-sm font-bold text-foreground">
                fill vs type
              </p>
              <ul className="mt-1 space-y-1 font-inter text-xs text-[#787878] leading-relaxed">
                <li className="flex items-center gap-2">
                  <span className="h-1 w-1 rounded-full bg-primary/40 shrink-0" />
                  <code className="text-primary/70">fill</code> sets values
                  programmatically, fast for plain text
                </li>
                <li className="flex items-center gap-2">
                  <span className="h-1 w-1 rounded-full bg-primary/40 shrink-0" />
                  <code className="text-primary/70">type</code> sends real
                  keystrokes, required for SSN, dates, phone, zip
                </li>
              </ul>
            </div>
            <div>
              <p className="font-source-serif text-sm font-bold text-foreground">
                Parallel safety
              </p>
              <ul className="mt-1 space-y-1 font-inter text-xs text-[#787878] leading-relaxed">
                <li className="flex items-center gap-2">
                  <span className="h-1 w-1 rounded-full bg-primary/40 shrink-0" />
                  Agent fires multiple tool calls in parallel after a snapshot
                </li>
                <li className="flex items-center gap-2">
                  <span className="h-1 w-1 rounded-full bg-primary/40 shrink-0" />
                  Per-session mutex serializes Playwright commands so they never
                  race
                </li>
              </ul>
            </div>
            <div>
              <p className="font-source-serif text-sm font-bold text-foreground">
                The browser is the state
              </p>
              <ul className="mt-1 space-y-1 font-inter text-xs text-[#787878] leading-relaxed">
                <li className="flex items-center gap-2">
                  <span className="h-1 w-1 rounded-full bg-primary/40 shrink-0" />
                  Session affinity pins every request to the same server
                  instance
                </li>
                <li className="flex items-center gap-2">
                  <span className="h-1 w-1 rounded-full bg-primary/40 shrink-0" />
                  Playwright connection and ref maps persist in memory, no
                  external store
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
