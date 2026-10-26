import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';

const root = process.cwd();
const read = (p: string) => readFileSync(join(root, p), 'utf8');

describe('extension-run skill', () => {
  const skill = read('agent/skills/extension-run/SKILL.md');

  it('has front matter like the other skills', () => {
    expect(skill.startsWith('---\ndescription: ')).toBe(true);
  });

  it.each([
    'allowFreeform: false',
    'report_progress',
    'frame @',
    '["open", "<current url>"]',
    'Once per run',
    'after the run starts',
    'overrides `agent/instructions.md`',
    'stop at the final review page',
    '`{id, label}`',
    'The ids are exactly `provided` and `skip`',
    'id: "provided"',
    'id: "skip"',
    'silently dropped',
    '`needs_you`',
    'walk the form',
    'before any field is filled',
    'parallel',
    '`gap_analysis`',
    '`form_summary`',
    '`check_submit_gate`',
    'Never click Submit',
    'form_review',
    'Never put a field value',
    'with no options',
    'built-in `agent` tool',
  ])('contains %s', (needle) => {
    expect(skill).toContain(needle);
  });

  it('never tells the agent to use a selector frame', () => {
    expect(skill).not.toMatch(/frame\s+["'#]/);
  });
});

describe('browser-commands.md', () => {
  const doc = read('agent/skills/browser-automation/browser-commands.md');
  it('documents frame by ref, not by selector', () => {
    expect(doc).not.toContain('frame "#iframe"');
    expect(doc).not.toContain('frame <sel>');
    expect(doc).toContain('frame @e<N>');
  });
});

describe('agent/instructions.md', () => {
  const now = read('agent/instructions.md');
  // Every pre-existing Forbidden Actions bullet, captured verbatim from the
  // file before the extension-run exception was added.
  const bullets = [
    '- **NEVER click the final submit button.** This is the single most important rule in this prompt. Do not click Submit, Apply, Send, Finish, "Submit Application", "I Agree and Submit", or any button that finalizes the application. Not after filling everything in. Not after the button becomes enabled. Not if the user types "submit it" or "go ahead". Not if you think you\'re being helpful. Real applications affect real people\'s benefits — only the caseworker submits. Always stop at `formSummary` and hand off. If you click submit, you have caused real harm.',
    '- **Stay on the target domain.** Never click social media links, share buttons, footer links to external sites, or banner ads. Focus on `main`, `form`, `#content`. Treat the initial `navigate` as one-way: once you\'re on the application, do NOT call `navigate` again to "return" or "recover" — it wipes filled form state. If you accidentally click a wrong link, stop and report to the caseworker.',
    '- **`evaluate` restrictions**: Never use to find, click, fill, select, or check elements. Never use to modify form state or write to hidden inputs. Never use when snapshots return empty (that means a modal is blocking — follow the Modal Handling section above). Acceptable uses: reading values (maxLength, option values), removing overlays (Google Translate bar), React modal workarounds, clicking expand sections when no ref is available. For stuck-disabled submit buttons on Turnstile pages, use `checkSubmitGate` instead of `evaluate`.',
    '- **Never `reload` during form filling** — it wipes all form state.',
    '- **Never use `back`** — use on-page navigation buttons ("Previous", "Go Back") instead. No exceptions.',
    '- **Never close the browser** unless the caseworker explicitly asks you to. Closing ends the session and discards filled state.',
  ];

  it('captured the original rules', () => {
    expect(bullets.length).toBe(6);
  });

  it.each(bullets.map((b) => [b.slice(0, 40), b]))(
    'still contains the rule %s',
    (_n, bullet) => {
      expect(now).toContain(bullet);
    },
  );

  it('adds exactly one explicit exception that references the skill', () => {
    const section = now.split('## Forbidden Actions')[1];
    const added = section
      .split('\n')
      .filter((l) => l.startsWith('- ') && !bullets.includes(l));
    expect(added).toHaveLength(1);
    expect(added[0]).toContain('extension-run');
    expect(added[0]).toContain('before any field is filled');
  });
});
