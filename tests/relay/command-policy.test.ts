import { describe, expect, it } from 'vitest';
import {
  COMMAND_REFUSED,
  relayCommandRefusal,
} from '@/lib/relay/command-policy';

const refused = (command: readonly string[]) =>
  relayCommandRefusal(command) === COMMAND_REFUSED;
const allowed = (command: readonly string[]) =>
  relayCommandRefusal(command) === null;

describe('relayCommandRefusal: refused (Review Focus 4)', () => {
  it.each([
    [['eval', 'document.title']],
    [['EVAL', 'document.title']],
    [['Eval', '1']],
    [[' eval', '1']],
    [['eval ', '1']],
    [['eval​', '1']],
    [['еval', '1']], // Cyrillic e
    [['eval', '-b', 'ZG9jdW1lbnQudGl0bGU=']],
    [['eval', '--stdin']],
    [['get', 'value', '@e1']],
    [['get', 'text', '@e1']],
    [['get', 'html', '@e1']],
    [['get', 'attr', '@e1', 'value']],
    [['get', 'styles', '@e1']],
    [['get', 'box', '@e1']],
    [['get', 'cdp-url']],
    [['get']],
    [['get', 'URL']],
    [['batch', 'eval document.title']],
    [['batch', '--bail', 'click @e1', 'get text @e2']],
    [['batch', '--json']],
    [['batch']],
    [['--cdp', 'ws://evil.example', 'snapshot']],
    [['--session', 'other', 'snapshot']],
    [['--allowed-domains', '*', 'snapshot']],
    [['--headers', '{}', 'snapshot']],
    [['--json', 'snapshot']],
    [['--profile', 'Default', 'snapshot']],
    [['--state', '/tmp/x.json', 'snapshot']],
    [['--executable-path', '/bin/sh', 'snapshot']],
    [['--proxy', 'http://evil.example', 'snapshot']],
    [['--args', '--no-sandbox', 'snapshot']],
    [['-p', 'kernel', 'snapshot']],
    [['find', 'role', 'textbox', 'text']],
    [['find', 'label', 'SSN', 'html']],
    [['find', 'label', 'SSN', 'value']],
    [['find', 'label', 'SSN', 'attr', 'value']],
    [['find', 'text', 'x', 'text']],
    [['find']],
    [['wait', '--fn', 'document.readyState']],
    [['wait', '--fn=1']],
    [['wait', '--download']],
    [['storage', 'local']],
    [['cookies']],
    [['console']],
    [['errors']],
    [['clipboard', 'read']],
    [['read']],
    [['read', 'https://benefits.example.gov']],
    [['snapshot', '--json-evil']],
    [['snapshot', '-x']],
    [['snapshot', '--json']],
    [['snapshot', '-i', '--cdp', 'x']],
    [['screenshot', '--annotate']],
    [['screenshot', 'out.png']],
    [['screenshot', '/etc/passwd']],
    [['open', 'javascript:alert(1)']],
    [['open', 'JavaScript:alert(1)']],
    [['open', ' javascript:alert(1)']],
    [['open', 'java\tscript:alert(1)']],
    [['open', 'data:text/html,<script>1</script>']],
    [['open', 'file:///etc/passwd']],
    [['open', 'blob:https://x/1']],
    [['open', 'https://a.example', 'https://b.example']],
    [['open']],
    [['tab', 'new']],
    [['tab', 'close']],
    [['tab', '2']],
    [['tab', 'list', 'x']],
    [['is', 'focused', '@e1']],
    [['is', 'visible']],
    [['click', '@e1', '--new-tab']],
    [['click', '@e1', '--cdp', 'x']],
    [['click', '--cdp', 'x', '@e1']],
    [['click', '--cdp=ws://x', '@e1']],
    [['click']],
    [['close']],
    [['dblclick', '@e1']],
    [['upload', '@e1', '/etc/passwd']],
    [['mcp']],
    [[]],
    [['']],
    [['click', '']],
    [['eval\n', '1']],
    [['click', '@e1\nget text @e2']],
    [['click', '@e1\0']],
    [['fill', '@e1', 'a\nb']],
  ])('refuses %j', (command) => {
    expect(relayCommandRefusal(command)).toBe(COMMAND_REFUSED);
  });

  it('refuses non-string elements and a non-array', () => {
    expect(refused([1 as unknown as string])).toBe(true);
    expect(refused(['click', null as unknown as string])).toBe(true);
    expect(refused(undefined as unknown as string[])).toBe(true);
  });

  it('refuses names that exist on Object.prototype', () => {
    for (const name of [
      'constructor',
      '__proto__',
      'toString',
      'hasOwnProperty',
    ]) {
      expect(refused([name])).toBe(true);
    }
  });

  it('refuses a flag in the position of a text value, as the CLI would consume it', () => {
    // agent-browser takes global flags in any position (`fill @e1 --cdp abc`
    // fails with "Invalid CDP value"), so text that looks like one is refused.
    expect(refused(['fill', '@e1', '--cdp', 'ws://evil.example'])).toBe(true);
    expect(refused(['fill', '@e1', '--cdp'])).toBe(true);
    expect(refused(['type', '@e1', '--session', 'x'])).toBe(true);
    expect(refused(['press', '--cdp'])).toBe(true);
    expect(refused(['select', '@e1', '--headers', '{}'])).toBe(true);
    expect(refused(['fill', '@e1', 'ok', '-p', 'kernel'])).toBe(true);
    expect(refused(['find', 'label', 'A', 'fill', '--cdp'])).toBe(true);
    expect(refused(['dialog', 'accept', '--cdp', 'x'])).toBe(true);
  });

  it('never echoes the argv', () => {
    expect(COMMAND_REFUSED).not.toMatch(/eval|--cdp|ssn/i);
    expect(relayCommandRefusal(['eval', 'SECRET-123-45-6789'])).not.toContain(
      '6789',
    );
  });
});

describe('relayCommandRefusal: allowed', () => {
  it.each([
    [['open', 'https://benefits.example.gov/apply?step=1']],
    [['open', 'benefits.example.gov']],
    [['snapshot']],
    [['snapshot', '-i']],
    [['snapshot', '-i', '-c']],
    [['snapshot', '-c', '-i', '-u']],
    [['snapshot', '-d', '3']],
    [['snapshot', '-s', 'form']],
    [['snapshot', '-s', '#main', '-i']],
    [['snapshot', '-s', '@e3']],
    [['click', '@e1']],
    [['click', '#submit-nope']],
    [['fill', '@e1', 'Ada']],
    [['fill', '@e1', 'Ada Lovelace']],
    [['fill', '@e1', '']],
    [['fill', '@e1', '-5']],
    [['fill', '@e1', 'a', 'b']],
    [['type', '@e2', 'Lovelace']],
    [['press', 'Tab']],
    [['press', 'Control+a']],
    [['select', '@e3', 'CA']],
    [['select', '@e3', 'a', 'b']],
    [['check', '@e4']],
    [['uncheck', '@e4']],
    [['focus', '@e1']],
    [['hover', '@e5']],
    [['scrollintoview', '@e5']],
    [['scroll', 'down']],
    [['scroll', 'down', '500']],
    [['scroll', 'up', '300']],
    [['scroll', 'left', '200']],
    [['scroll', 'right', '200']],
    [['scroll', 'down', '500', '--selector', 'div.content']],
    [['wait', '@e5']],
    [['wait', '#plain']],
    [['wait', '200']],
    [['wait', '--text', 'Welcome']],
    [['wait', '--url', '**/dashboard']],
    [['wait', '--load', 'networkidle']],
    [['wait', '--load', 'load']],
    [['wait', '--load', 'domcontentloaded']],
    [['tab']],
    [['tab', 'list']],
    [['back']],
    [['forward']],
    [['reload']],
    [['screenshot']],
    [['screenshot', '--full']],
    [['get', 'url']],
    [['get', 'title']],
    [['get', 'count', '.item']],
    [['is', 'visible', '@e1']],
    [['is', 'enabled', '@e1']],
    [['is', 'checked', '@e4']],
    [['find', 'role', 'textbox', 'click']],
    [['find', 'role', 'button', 'click', '--name', 'Plain action']],
    [['find', 'role', 'button', 'click', '--name', 'Go', '--exact']],
    [['find', 'label', 'First name', 'fill', 'Grace']],
    [['find', 'placeholder', 'Search', 'type', 'query']],
    [['find', 'text', 'Sign In', 'click']],
    [['find', 'text', 'Sign In', 'click', '--exact']],
    [['find', 'alt', 'Logo', 'hover']],
    [['find', 'title', 'Close', 'click']],
    [['find', 'testid', 'submit-btn', 'focus']],
    [['find', 'first', '.item', 'click']],
    [['find', 'last', '.item', 'check']],
    [['find', 'nth', '2', 'a', 'hover']],
    [['frame', '@e7']],
    [['frame', 'main']],
    [['dialog', 'accept']],
    [['dialog', 'accept', 'prompt answer']],
    [['dialog', 'dismiss']],
  ])('allows %j', (command) => {
    expect(relayCommandRefusal(command)).toBeNull();
  });

  it('allows free text that merely contains flag-like words', () => {
    expect(allowed(['fill', '@e1', 'the --cdp flag'])).toBe(true);
    expect(allowed(['fill', '@e1', 'Mary-Jane'])).toBe(true);
    expect(allowed(['fill', '@e1', '-'])).toBe(true);
  });
});

describe('every command the extension-run skill and the measured list use', () => {
  // tests/relay/e2e/allowlist.e2e.test.ts `driveDocumentedCommands`, minus the
  // value-returning ones, which must be refused (see below).
  it('is allowed', () => {
    for (const command of [
      ['open', 'https://benefits.example.gov/'],
      ['snapshot'],
      ['snapshot', '-s', 'form'],
      ['snapshot', '-i'],
      ['click', '@e1'],
      ['fill', '@e1', 'Ada'],
      ['type', '@e2', 'Lovelace'],
      ['select', '@e3', 'CA'],
      ['find', 'label', 'First name', 'fill', 'Grace'],
      ['find', 'role', 'button', 'click', '--name', 'Plain action'],
      ['press', 'Tab'],
      ['press', 'Control+a'],
      ['hover', '@e5'],
      ['check', '@e4'],
      ['uncheck', '@e4'],
      ['scrollintoview', '@e5'],
      ['wait', '@e5'],
      ['wait', '#plain'],
      ['wait', '200'],
      ['wait', '--load', 'networkidle'],
      ['get', 'url'],
      ['get', 'title'],
      ['scroll', 'down', '500'],
      ['scroll', 'up', '500'],
      ['screenshot'],
      ['dialog', 'accept'],
      ['dialog', 'dismiss'],
      ['frame', '#child'],
      ['frame', 'main'],
      ['frame', '@e7'],
    ]) {
      expect(allowed(command), JSON.stringify(command)).toBe(true);
    }
  });

  it('refuses the value-returning commands of the measured list', () => {
    for (const command of [
      ['get', 'text', '@e5'],
      ['get', 'value', '@e1'],
      ['eval', 'document.title'],
      ['eval', 'setTimeout(() => alert("a"), 50); 1'],
      ['batch', '--bail', 'click @e1', 'press 1', 'wait 100', 'press 2'],
    ]) {
      expect(refused(command), JSON.stringify(command)).toBe(true);
    }
  });
});

describe('top-level commands outside the allowlist', () => {
  // Every command named by `agent-browser --help` (0.33.2) and its aliases.
  // Only the ones below the allowlist line may be allowed.
  const ALL_CLI_COMMANDS = [
    'skills',
    'open',
    'goto',
    'navigate',
    'read',
    'click',
    'dblclick',
    'type',
    'fill',
    'press',
    'keyboard',
    'keydown',
    'keyup',
    'hover',
    'focus',
    'check',
    'uncheck',
    'select',
    'drag',
    'upload',
    'download',
    'scroll',
    'scrollintoview',
    'scrollinto',
    'wait',
    'screenshot',
    'pdf',
    'snapshot',
    'eval',
    'connect',
    'close',
    'quit',
    'exit',
    'back',
    'forward',
    'reload',
    'get',
    'is',
    'find',
    'mouse',
    'set',
    'network',
    'cookies',
    'storage',
    'tab',
    'window',
    'frame',
    'dialog',
    'diff',
    'trace',
    'profiler',
    'record',
    'console',
    'errors',
    'highlight',
    'inspect',
    'clipboard',
    'stream',
    'react',
    'vitals',
    'a11y',
    'pushstate',
    'addinitscript',
    'removeinitscript',
    'batch',
    'auth',
    'plugin',
    'confirm',
    'deny',
    'session',
    'mcp',
    'chat',
    'dashboard',
    'install',
    'upgrade',
    'doctor',
    'profiles',
    'state',
    'pdf',
  ];
  const ALLOWED_COMMANDS = new Set([
    'open',
    'snapshot',
    'click',
    'fill',
    'type',
    'press',
    'select',
    'check',
    'uncheck',
    'focus',
    'hover',
    'scrollintoview',
    'scroll',
    'wait',
    'tab',
    'back',
    'forward',
    'reload',
    'screenshot',
    'get',
    'is',
    'find',
    'frame',
    'dialog',
  ]);

  it.each(ALL_CLI_COMMANDS.filter((c) => !ALLOWED_COMMANDS.has(c)))(
    'refuses %s with no arguments and with plausible ones',
    (name) => {
      expect(refused([name])).toBe(true);
      expect(refused([name, '@e1'])).toBe(true);
      expect(refused([name, 'x', 'y'])).toBe(true);
    },
  );

  it('allows no command that is not in the allowlist set', () => {
    for (const name of ALL_CLI_COMMANDS.filter(
      (c) => !ALLOWED_COMMANDS.has(c),
    )) {
      expect(allowed([name])).toBe(false);
    }
  });
});
