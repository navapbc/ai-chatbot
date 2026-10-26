// Deny-by-default policy for the agent-browser argv the model may send in an
// extension run. The browser tool passes the model's argv to the CLI as-is
// (lib/kernel/cli.ts `buildArgs` validates nothing), so without this a model
// could read a sensitive field's real value through a command that returns one
// (`eval`, `get value|text|html|attr|styles`, `find ... text`, `wait --fn`,
// `batch "eval ..."`, `console`, `clipboard read`, `storage`, ...) or change
// the connection with a global flag (`--cdp`, `--session`, ...).
//
// The allowlist below is data: command -> the argument shapes it may take. A
// command or shape not listed is refused. Pure and dependency-free so the
// Eve bundle and the tests can both import it.
//
// CLI parsing facts this relies on (agent-browser 0.33.2, checked by running
// the CLI against an unknown ref, no browser needed):
// - Global flags are recognised in ANY argv position, not only before the
//   command: `click @e1 --cdp abc` fails with "Invalid CDP value", and
//   `fill @e1 -p bogus` with "Unknown provider". So even free text typed into
//   a field cannot safely start with `--` or be a short flag: the CLI would
//   take it as a flag. Text slots therefore refuse those, rather than treating
//   flags in text as text.
// - A value-less trailing `--cdp` and the `--flag=value` spelling were not
//   rejected by the probe; both start with `--` and are refused here anyway.
// The refusal never echoes the argv.

export const COMMAND_REFUSED =
  'This command is not available in extension runs. Verify values from the snapshot, which masks sensitive fields.';

/** Short global flags of the CLI (`-p`, `-v`, `-q`, `-V`, `-h`, `-b`). */
const SHORT_GLOBAL_FLAGS: ReadonlySet<string> = new Set([
  '-p',
  '-v',
  '-q',
  '-V',
  '-h',
  '-b',
]);

type Slot =
  /** One of these exact words. */
  | { readonly is: readonly string[] }
  /** A snapshot ref, `@e12`: the only way to name an element. */
  | 'ref'
  /** A non-negative integer. */
  | 'number'
  /** An http(s) URL to open. */
  | 'url'
  /** Free text typed into a field: anything that cannot be a CLI flag. */
  | 'text'
  /** A key the agent may press. */
  | 'key';

/** An argument shape: the slots in order, then optionally any number of text. */
interface Shape {
  readonly slots: readonly Slot[];
  readonly rest?: 'text';
}

interface CommandSpec {
  readonly shapes?: readonly Shape[];
  /** Flag-only commands: flag -> its value slot (null: no value). Any order. */
  readonly flags?: Readonly<Record<string, Slot | null>>;
}

const lit = (...words: string[]): Slot => ({ is: words });
const shape = (slots: readonly Slot[], rest?: 'text'): Shape => ({
  slots,
  rest,
});

// Elements are named by snapshot ref only. A CSS or XPath selector, a text
// match (`wait --text`, `find text`), a URL pattern (`wait --url`) or a count
// (`get count`) can each be probed with a guess ("does an input whose value
// starts with 9 exist?") and so read a masked value back one character at a
// time. Refs name an element without testing its content.
const DIRECTIONS = lit('up', 'down', 'left', 'right');
const LOAD_STATES = lit('load', 'domcontentloaded', 'networkidle');

const refOnly = { shapes: [shape(['ref'])] };
const noArgs = { shapes: [shape([])] };

/**
 * The allowlist. Nothing that returns page content, tests page content, or
 * reaches outside the tab.
 */
const ALLOWED: ReadonlyMap<string, CommandSpec> = new Map<string, CommandSpec>([
  ['open', { shapes: [shape(['url'])] }],
  [
    'snapshot',
    { flags: { '-i': null, '-c': null, '-u': null, '-d': 'number' } },
  ],
  ['click', refOnly],
  ['fill', { shapes: [shape(['ref'], 'text')] }],
  ['type', { shapes: [shape(['ref'], 'text')] }],
  ['press', { shapes: [shape(['key'])] }],
  ['select', { shapes: [shape(['ref', 'text'], 'text')] }],
  ['check', refOnly],
  ['uncheck', refOnly],
  ['focus', refOnly],
  ['hover', refOnly],
  ['scrollintoview', refOnly],
  ['scroll', { shapes: [shape([DIRECTIONS]), shape([DIRECTIONS, 'number'])] }],
  [
    'wait',
    {
      shapes: [
        shape(['ref']),
        shape(['number']),
        shape([lit('--load'), LOAD_STATES]),
      ],
    },
  ],
  ['tab', { shapes: [shape([]), shape([lit('list')])] }],
  ['back', noArgs],
  ['forward', noArgs],
  ['reload', noArgs],
  ['screenshot', { shapes: [shape([]), shape([lit('--full')])] }],
  ['get', { shapes: [shape([lit('url')]), shape([lit('title')])] }],
  ['is', { shapes: [shape([lit('visible', 'enabled', 'checked'), 'ref'])] }],
  ['frame', { shapes: [shape(['ref']), shape([lit('main')])] }],
  [
    'dialog',
    { shapes: [shape([lit('dismiss')]), shape([lit('accept')], 'text')] },
  ],
]);

const NAMED_KEYS = new Set([
  'Enter',
  'Tab',
  'Escape',
  'Backspace',
  'Delete',
  'Space',
  'ArrowUp',
  'ArrowDown',
  'ArrowLeft',
  'ArrowRight',
  'Home',
  'End',
  'PageUp',
  'PageDown',
  'Control+a',
  'Meta+a',
]);

/** http(s) only, no userinfo, bounded. A bare host gets https:// like the CLI. */
const isOpenableUrl = (arg: string): boolean => {
  if (arg.length === 0 || arg.length > 2048 || /[\s\\]/.test(arg)) return false;
  const candidate = arg.includes(':') ? arg : `https://${arg}`;
  let url: URL;
  try {
    url = new URL(candidate);
  } catch {
    return false;
  }
  return (
    (url.protocol === 'http:' || url.protocol === 'https:') &&
    url.username === '' &&
    url.password === '' &&
    url.hostname !== ''
  );
};

const matches = (slot: Slot, arg: string): boolean => {
  if (typeof slot === 'object') return slot.is.includes(arg);
  switch (slot) {
    case 'ref':
      return /^@e\d{1,6}$/.test(arg);
    case 'number':
      return /^\d{1,7}$/.test(arg);
    case 'url':
      return isOpenableUrl(arg);
    case 'key':
      return NAMED_KEYS.has(arg) || /^[\x21-\x7e]$/.test(arg);
    case 'text':
      return !arg.startsWith('--') && !SHORT_GLOBAL_FLAGS.has(arg);
  }
};

const matchesShape = (s: Shape, args: readonly string[]): boolean => {
  if (args.length < s.slots.length) return false;
  if (args.length > s.slots.length && !s.rest) return false;
  return args.every((arg, i) =>
    matches(i < s.slots.length ? s.slots[i] : (s.rest as Slot), arg),
  );
};

const matchesFlags = (
  flags: NonNullable<CommandSpec['flags']>,
  args: readonly string[],
): boolean => {
  for (let i = 0; i < args.length; i += 1) {
    const arg = args[i];
    if (!Object.hasOwn(flags, arg)) return false;
    const value = flags[arg];
    if (value === null) continue;
    i += 1;
    if (i >= args.length || !matches(value, args[i])) return false;
  }
  return true;
};

const WELL_FORMED = (arg: unknown): arg is string =>
  typeof arg === 'string' && !arg.includes('\0');

/**
 * `COMMAND_REFUSED` when the argv is not on the allowlist, null when it is.
 * Commands match case-sensitively and exactly: `EVAL`, ` eval` and lookalikes
 * are simply not in the map.
 */
export function relayCommandRefusal(command: readonly string[]): string | null {
  if (!Array.isArray(command) || command.length === 0) return COMMAND_REFUSED;
  if (!command.every(WELL_FORMED)) return COMMAND_REFUSED;
  const [name, ...args] = command;
  const spec = ALLOWED.get(name);
  if (!spec) return COMMAND_REFUSED;
  const allowed = spec.flags
    ? matchesFlags(spec.flags, args)
    : (spec.shapes ?? []).some((s) => matchesShape(s, args));
  return allowed ? null : COMMAND_REFUSED;
}
