'use client';

import { useState } from 'react';

type FileEntry = {
  name: string;
  indent: number;
  type: 'dir' | 'file';
  tier?: number;
  annotation?: string;
};

const fileTree: FileEntry[] = [
  { name: 'skills/', indent: 0, type: 'dir' },
  { name: 'agent-browser/', indent: 1, type: 'dir' },
  {
    name: 'skill.ts',
    indent: 2,
    type: 'file',
    tier: 0,
    annotation: 'always loaded',
  },
  {
    name: 'SKILL.md',
    indent: 2,
    type: 'file',
    tier: 1,
    annotation: 'loaded once',
  },
  { name: 'index.ts', indent: 2, type: 'file' },
  { name: 'references/', indent: 2, type: 'dir' },
  {
    name: 'form-submission.md',
    indent: 3,
    type: 'file',
    tier: 2,
    annotation: 'on demand',
  },
  {
    name: 'modals.md',
    indent: 3,
    type: 'file',
    tier: 2,
    annotation: 'on demand',
  },
  {
    name: 'form-automation.md',
    indent: 3,
    type: 'file',
    tier: 2,
    annotation: 'on demand',
  },
  {
    name: 'commands.md',
    indent: 3,
    type: 'file',
    tier: 2,
    annotation: 'on demand',
  },
  {
    name: 'snapshot-refs.md',
    indent: 3,
    type: 'file',
    tier: 2,
    annotation: 'on demand',
  },
  { name: 'caseworker-communication/', indent: 1, type: 'dir' },
  {
    name: 'SKILL.md',
    indent: 2,
    type: 'file',
    tier: 1,
    annotation: 'loaded once',
  },
];

const tiers = [
  {
    label: 'Always loaded',
    trigger: '9 rules in system prompt',
    detail: 'Injected into every request. Tiny footprint, non-negotiable.',
    example: `1. ALWAYS use snapshot refs (@e1, @e2) OR CSS IDs
2. Use type (NOT fill) for masked fields
3. NEVER mention technical terms to caseworkers
4. Call multiple tools in parallel when independent
5. evaluate is for workarounds, not form filling`,
    file: 'skill.ts',
  },
  {
    label: 'Loaded once',
    trigger: 'loadSkill("agent-browser")',
    detail: 'Full playbook cached before first browser action.',
    example: `## Snapshot Strategy (CRITICAL)
1. Full snapshot first → scope on complex pages
2. Re-snapshot after every DOM change

## Form Submission Protocol
1. Check missing fields → 2. Expand sections
3. Wait for auto-solver → 4. Verify → 5. Debug`,
    file: 'SKILL.md',
  },
  {
    label: 'On demand',
    trigger: 'readSkillFile("references/...")',
    detail: 'Specific chapters pulled when the agent hits a situation.',
    example: `# references/form-submission.md
## Step 5: Find the page's JavaScript source
{ action: "evaluate", script:
  "document.querySelectorAll('script[src]')..." }

Look for: gating variables, callbacks,
event handlers bound to CSS classes`,
    file: 'references/*.md',
  },
];

const annotationStyles: Record<
  number,
  { dot: string; text: string; bg: string }
> = {
  0: { dot: 'bg-primary', text: 'text-primary', bg: 'bg-primary/5' },
  1: { dot: 'bg-primary/60', text: 'text-primary/80', bg: 'bg-primary/[0.03]' },
  2: { dot: 'bg-primary/35', text: 'text-primary/55', bg: 'bg-primary/[0.02]' },
};

export default function SkillsDemoPage() {
  const [expanded, setExpanded] = useState<number | null>(null);

  return (
    <div className="min-h-screen bg-chat-background">
      <div className="mx-auto max-w-[720px] px-6 py-20">
        {/* Title */}
        <p className="font-source-serif text-4xl leading-[1.15] text-foreground">
          Agent Skills
        </p>
        <p className="mt-3 font-source-serif text-xl text-[#787878]">
          Context windows are finite. On a 500-step form fill, every extra token
          compounds. We teach the agent through progressive
          disclosure&mdash;three tiers of instructions, loaded only when needed.
        </p>

        {/* File tree */}
        <div className="mt-14 mx-auto max-w-md rounded-lg border border-border bg-white px-6 py-5 shadow-sm">
          <p className="mb-4 font-inter text-[11px] font-semibold uppercase tracking-[0.08em] text-[#787878]">
            lib/ai/skills
          </p>
          <div className="font-mono text-[13px] leading-[1.8]">
            {fileTree.map((entry, i) => {
              const style =
                entry.tier !== undefined ? annotationStyles[entry.tier] : null;
              return (
                <div
                  key={`${entry.name}-${entry.indent}`}
                  className={`flex items-center rounded-[4px] px-2 -mx-2 ${style?.bg ?? ''}`}
                  style={{ paddingLeft: `${entry.indent * 20 + 8}px` }}
                >
                  <span className="text-[#b5b5b5] select-none mr-2">
                    {entry.type === 'dir' ? (
                      <svg
                        width="14"
                        height="14"
                        viewBox="0 0 16 16"
                        fill="none"
                        className="inline-block -mt-px"
                      >
                        <path
                          d="M1.5 3.5C1.5 2.67 2.17 2 3 2h3.17a1.5 1.5 0 0 1 1.06.44l.94.94a.5.5 0 0 0 .36.12H13c.83 0 1.5.67 1.5 1.5v7c0 .83-.67 1.5-1.5 1.5H3c-.83 0-1.5-.67-1.5-1.5v-8.5Z"
                          stroke="currentColor"
                          strokeWidth="1.2"
                        />
                      </svg>
                    ) : (
                      <svg
                        width="14"
                        height="14"
                        viewBox="0 0 16 16"
                        fill="none"
                        className="inline-block -mt-px"
                      >
                        <path
                          d="M4 1.5h5.17a1 1 0 0 1 .7.3l3.34 3.33a1 1 0 0 1 .29.7V13a1.5 1.5 0 0 1-1.5 1.5H4A1.5 1.5 0 0 1 2.5 13V3A1.5 1.5 0 0 1 4 1.5Z"
                          stroke="currentColor"
                          strokeWidth="1.2"
                        />
                        <path
                          d="M9.5 1.5V5a.5.5 0 0 0 .5.5h3.5"
                          stroke="currentColor"
                          strokeWidth="1.2"
                        />
                      </svg>
                    )}
                  </span>
                  <span
                    className={`${style?.text ?? 'text-foreground'} ${entry.type === 'dir' ? 'font-semibold' : ''}`}
                  >
                    {entry.name}
                  </span>
                  {entry.annotation && style && (
                    <>
                      <span
                        className={`mx-2 inline-block h-[5px] w-[5px] rounded-full ${style.dot} shrink-0`}
                      />
                      <span className="font-inter text-[10px] font-medium uppercase tracking-[0.06em] text-[#b5b5b5]">
                        {entry.annotation}
                      </span>
                    </>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* Tiers */}
        <div className="mt-10 mx-auto max-w-md space-y-3">
          {tiers.map((tier, i) => {
            const isOpen = expanded === i;
            const style = annotationStyles[i];
            return (
              <div
                key={tier.label}
                className={`group w-full rounded-lg border bg-white px-6 py-5 text-left shadow-sm transition-colors ${
                  isOpen
                    ? 'border-primary/30'
                    : 'border-border hover:border-primary/15'
                }`}
              >
                <button
                  type="button"
                  aria-expanded={isOpen}
                  onClick={() => setExpanded(isOpen ? null : i)}
                  className="w-full text-left"
                >
                  <div className="flex items-center gap-3">
                    <span
                      className={`inline-block h-2 w-2 rounded-full ${style.dot} shrink-0`}
                    />
                    <span className="font-source-serif text-lg font-bold text-foreground">
                      {tier.label}
                    </span>
                    <span className="font-mono text-xs text-[#b5b5b5]">
                      {tier.trigger}
                    </span>
                  </div>
                  <p className="mt-1 pl-5 font-inter text-sm text-[#787878]">
                    {tier.detail}
                  </p>
                </button>

                {isOpen && (
                  <div className="mt-5 pl-5">
                    <div className="flex items-center gap-2 mb-2">
                      <span className="font-mono text-[10px] font-medium uppercase tracking-[0.08em] text-[#b5b5b5]">
                        {tier.file}
                      </span>
                    </div>
                    <pre className="overflow-x-auto rounded-md border border-border bg-[#fafafa] px-4 py-3 font-mono text-xs leading-[1.7] text-foreground/80">
                      <code>{tier.example}</code>
                    </pre>
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Flow */}
        <div className="mt-14 flex items-stretch gap-3">
          {tiers.map((tier, i) => {
            const style = annotationStyles[i];
            return (
              <div
                key={tier.label}
                className="flex flex-1 flex-col items-center"
              >
                <div
                  className={`flex h-8 w-8 items-center justify-center rounded-full ${style.bg} border border-primary/10`}
                >
                  <span
                    className={`font-inter text-sm font-bold ${style.text}`}
                  >
                    {i + 1}
                  </span>
                </div>
                <div className="mt-3 text-center">
                  <p className="font-source-serif text-sm font-bold text-foreground">
                    {tier.label}
                  </p>
                  <p className="mt-1 font-inter text-xs text-[#b5b5b5] leading-relaxed">
                    {tier.detail}
                  </p>
                </div>
              </div>
            );
          })}
        </div>

        <p className="mt-14 text-center font-inter text-xs text-[#b5b5b5]">
          New protocols for new sites are just a markdown file&mdash;no system
          prompt changes needed.
        </p>
      </div>
    </div>
  );
}
