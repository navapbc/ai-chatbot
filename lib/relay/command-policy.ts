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
  /** One non-empty word that is not flag-shaped: a ref, selector or key name. */
  | 'word'
  /** A non-negative integer. */
  | 'number'
  /** A URL to open. */
  | 'url'
  /** Free text: anything that cannot be mistaken for a CLI flag. */
  | 'text';

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

const DIRECTIONS = lit('up', 'down', 'left', 'right');
const LOAD_STATES = lit('load', 'domcontentloaded', 'networkidle');
const LOCATORS = [
  'role',
  'label',
  'placeholder',
  'alt',
  'title',
  'testid',
  'text',
];
// `text`, `html`, `value` and `attr` are the actions that return page content:
// never in this list.
const FIND_ACTIONS = ['click', 'hover', 'focus', 'check', 'uncheck'];
const FIND_TEXT_ACTIONS = ['fill', 'type'];

// `find <locator> <query> <action> [value]`, plus `first <sel>` and
// `nth <n> <sel>`. Click-like actions take only `--name`/`--exact`; fill and
// type take the text.
const findShapes = (): Shape[] => {
  const prefixes: Slot[][] = [
    ...LOCATORS.map((l): Slot[] => [lit(l), 'word']),
    [lit('first', 'last'), 'word'],
    [lit('nth'), 'number', 'word'],
  ];
  const tails: Slot[][] = [
    [],
    [lit('--exact')],
    [lit('--name'), 'text'],
    [lit('--name'), 'text', lit('--exact')],
  ];
  const shapes: Shape[] = [];
  for (const prefix of prefixes) {
    for (const tail of tails) {
      shapes.push(shape([...prefix, lit(...FIND_ACTIONS), ...tail]));
    }
    shapes.push(shape([...prefix, lit(...FIND_TEXT_ACTIONS)], 'text'));
  }
  return shapes;
};

const targetOnly = { shapes: [shape(['word'])] };
const noArgs = { shapes: [shape([])] };

/**
 * The allowlist. Everything the extension-run skill and the measured command
 * list (tests/relay/e2e/allowlist.e2e.test.ts) use is here, and nothing that
 * returns page content or reaches outside the tab.
 */
const ALLOWED: ReadonlyMap<string, CommandSpec> = new Map<string, CommandSpec>([
  ['open', { shapes: [shape(['url'])] }],
  [
    'snapshot',
    {
      flags: {
        '-i': null,
        '-c': null,
        '-u': null,
        '-d': 'number',
        '-s': 'word',
      },
    },
  ],
  ['click', targetOnly],
  ['fill', { shapes: [shape(['word'], 'text')] }],
  ['type', { shapes: [shape(['word'], 'text')] }],
  ['press', { shapes: [shape(['text'])] }],
  ['select', { shapes: [shape(['word', 'text'], 'text')] }],
  ['check', targetOnly],
  ['uncheck', targetOnly],
  ['focus', targetOnly],
  ['hover', targetOnly],
  ['scrollintoview', targetOnly],
  [
    'scroll',
    {
      shapes: [
        shape([DIRECTIONS]),
        shape([DIRECTIONS, 'number']),
        shape([DIRECTIONS, 'number', lit('--selector'), 'word']),
      ],
    },
  ],
  [
    'wait',
    {
      shapes: [
        shape(['word']),
        shape(['number']),
        shape([lit('--text'), 'text']),
        shape([lit('--url'), 'text']),
        shape([lit('--load'), LOAD_STATES]),
      ],
    },
  ],
  ['tab', { shapes: [shape([]), shape([lit('list')])] }],
  ['back', noArgs],
  ['forward', noArgs],
  ['reload', noArgs],
  ['screenshot', { shapes: [shape([]), shape([lit('--full')])] }],
  [
    'get',
    {
      shapes: [
        shape([lit('url')]),
        shape([lit('title')]),
        shape([lit('count'), 'word']),
      ],
    },
  ],
  ['is', { shapes: [shape([lit('visible', 'enabled', 'checked'), 'word'])] }],
  ['find', { shapes: findShapes() }],
  ['frame', targetOnly],
  [
    'dialog',
    {
      shapes: [shape([lit('dismiss')]), shape([lit('accept')], 'text')],
    },
  ],
]);

const BLOCKED_URL_SCHEME = /^(?:javascript|data|file|blob|vbscript):/i;

const matches = (slot: Slot, arg: string): boolean => {
  if (typeof slot === 'object') return slot.is.includes(arg);
  switch (slot) {
    case 'word':
      return arg.length > 0 && !arg.startsWith('-');
    case 'number':
      return /^\d{1,7}$/.test(arg);
    case 'url':
      return (
        arg.length > 0 &&
        arg.trim() === arg &&
        // Spaces, tabs and the like inside a scheme can hide it from the match.
        !/\s/.test(arg) &&
        !BLOCKED_URL_SCHEME.test(arg)
      );
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
  typeof arg === 'string' && !/[\0\n\r]/.test(arg);

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
