import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { POLICY_AGENT_MESSAGE } from '@/extension/src/bridge';
import {
  COMMAND_REFUSED,
  relayCommandRefusal,
} from '@/lib/relay/command-policy';

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
    "When the caseworker's text answer is exactly `skip`, do not fill that field",
    'leave whatever is already in it and move on',
    'never type `skip`',
    'never a free-text question',
    'A text answer of exactly `provided` means the caseworker entered the value in the page themselves: do not ask again and do not try to read it.',
    'If a command answers "This browser is restricted by your organization\'s policy for this site.", stop using the browser, do not retry, and end your turn saying the caseworker\'s IT must allow the extension on this site.',
    'During that origin hold the `browser` tool waits and then tells you what to do: do not end your turn on the refusal. If the hold ends the run, you will be told.',
  ])('contains %s', (needle) => {
    expect(skill).toContain(needle);
  });

  it('quotes the fixed policy message exactly', () => {
    expect(skill).toContain(`"${POLICY_AGENT_MESSAGE}"`);
  });

  it('never tells the agent to use a selector frame', () => {
    expect(skill).not.toMatch(/frame\s+["'#]/);
  });
});

describe('extension-run skill: masked verification and ref-only commands', () => {
  const skill = read('agent/skills/extension-run/SKILL.md');

  it.each([
    '[masked]',
    'eval',
    'verify',
    'snapshot',
    '@eN',
    'wait --load',
    'get url',
    COMMAND_REFUSED,
    '`provided`',
    '`skip`',
    'allowFreeform: false',
    'You never type a sensitive value',
    'The screenshot was refused because sensitive fields could not be masked.',
    'browser-automation',
    'benefits-application',
    'field-patterns',
    'browser-commands',
    'get value',
    '`batch`',
    'tab new',
    'wait --text',
    '#id',
    '`find`',
  ])('contains %s', (needle) => {
    expect(skill).toContain(needle);
  });

  const isRefusedLine = (line: string) => /^\s*(?:[-*]\s*)?Refused:/.test(line);

  const argvExamples = skill
    .split('\n')
    .filter((l) => !isRefusedLine(l))
    .flatMap((l) => l.match(/\[\s*"[^\]\n]*\]/g) ?? [])
    .map(
      (m) =>
        JSON.parse(
          m.replaceAll('<current url>', 'https://example.test/'),
        ) as string[],
    )
    .filter((a) => Array.isArray(a) && a.every((x) => typeof x === 'string'));

  it('has argv examples to check', () => {
    expect(argvExamples.length).toBeGreaterThanOrEqual(6);
  });

  it.each(argvExamples.map((a) => [JSON.stringify(a), a]))(
    'example %s is allowed by the command policy',
    (_n, argv) => {
      expect(relayCommandRefusal(argv as string[])).toBeNull();
    },
  );

  it('marked Refused: examples really are refused', () => {
    const refused = skill
      .split('\n')
      .filter(isRefusedLine)
      .flatMap((l) => l.match(/\[\s*"[^\]\n]*\]/g) ?? [])
      .map((m) => JSON.parse(m) as string[]);
    expect(refused.length).toBeGreaterThan(0);
    for (const argv of refused) {
      expect(relayCommandRefusal(argv)).not.toBeNull();
    }
  });

  it('never recommends refused commands outside Refused: lines', () => {
    const live = skill
      .split('\n')
      .filter(
        (l) =>
          !isRefusedLine(l) && !/\b(refused|never|not|do not|don't)\b/i.test(l),
      )
      .join('\n');
    expect(live).not.toMatch(
      /get value|`eval|\bfind\b|#id|wait --text|wait --url/,
    );
  });

  it('stays concise', () => {
    expect(skill.split('\n').length).toBeLessThanOrEqual(120);
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
    '- Web chats only (refused on extension runs): **`evaluate` restrictions**: Never use to find, click, fill, select, or check elements. Never use to modify form state or write to hidden inputs. Never use when snapshots return empty (that means a modal is blocking — follow the Modal Handling section above). Acceptable uses: reading values (maxLength, option values), removing overlays (Google Translate bar), React modal workarounds, clicking expand sections when no ref is available. For stuck-disabled submit buttons on Turnstile pages, use `checkSubmitGate` instead of `evaluate`.',
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
