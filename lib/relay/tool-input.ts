// What the client stream may show of a tool call's input on a relay run.
// Pure and import-free. The browser tool's argv carries what the agent types
// into the caseworker's tab, so only the verb, a snapshot ref and (for
// `open`) the origin and path survive. Everything else becomes a placeholder.

const HIDDEN = '[hidden]';
const REF = /^@e\d+$/;
const VERB = /^[a-z][a-z0-9-]{0,30}$/i;

function safeUrl(raw: string): string {
  try {
    const u = new URL(raw);
    if (u.protocol !== 'https:' && u.protocol !== 'http:') return HIDDEN;
    return `${u.origin}${u.pathname}`;
  } catch {
    return HIDDEN;
  }
}

function redactArgv(argv: unknown[]): string[] {
  return argv.map((el, i) => {
    if (typeof el !== 'string') return HIDDEN;
    if (i === 0) return VERB.test(el) ? el : HIDDEN;
    if (i === 1) {
      if (REF.test(el)) return el;
      if (argv[0] === 'open') return safeUrl(el);
    }
    return HIDDEN;
  });
}

export function redactToolInput(toolName: string, input: unknown): unknown {
  if (toolName === 'report_progress') return input;
  if (toolName !== 'browser') return {};
  if (typeof input !== 'object' || input === null) return {};
  const rec = input as Record<string, unknown>;
  // The tool's schema names the argv `command`; `argv` is accepted too.
  for (const key of ['command', 'argv']) {
    if (!Object.hasOwn(rec, key)) continue;
    const value = rec[key];
    if (!Array.isArray(value)) return {};
    return { [key]: redactArgv(value) };
  }
  return {};
}
