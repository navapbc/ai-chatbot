/**
 * Execute agent-browser CLI argv against a local Playwright page.
 *
 * The production tool ships argv to the agent-browser binary, which drives a
 * remote Kernel browser. Evals run offline against fixture HTML, so this
 * interprets the same argv locally — the agent emits identical commands in both
 * places, and evals stay meaningful without a Kernel session.
 *
 * Only the commands the eval fixtures exercise are implemented; anything else
 * returns an explicit "unsupported" error rather than silently passing, so a
 * gap shows up as a failing eval instead of a false green.
 */

import type { Page } from 'playwright';

export interface CommandResult {
  success: boolean;
  output: string | null;
  error: string | null;
}

/**
 * Per-action cap, mirroring the real CLI's 25s default. Applied by the caller
 * via `page.setDefaultTimeout` so every locator action inherits it: without one
 * Playwright waits 30s, and a selector the agent got wrong would stall the eval
 * instead of returning an error it can react to.
 */
export const ACTION_TIMEOUT_MS = 25_000;

const ok = (data: unknown): CommandResult => ({
  success: true,
  output: typeof data === 'string' ? data : JSON.stringify(data),
  error: null,
});

const fail = (error: string): CommandResult => ({
  success: false,
  output: null,
  error,
});

/** Resolve an agent-browser selector. `@eN` refs map to the ref table. */
function resolveSelector(selector: string, refs: Map<string, string>): string {
  if (selector.startsWith('@')) {
    const resolved = refs.get(selector.slice(1));
    if (!resolved) throw new Error(`Unknown ref ${selector}`);
    return resolved;
  }
  return selector;
}

/**
 * Build an accessibility-tree snapshot with `@eN` refs, mirroring the shape the
 * real CLI returns: a text tree the model reads, plus a ref table it clicks by.
 */
async function snapshot(
  page: Page,
  refs: Map<string, string>,
  scope: string | undefined,
  interactiveOnly: boolean,
): Promise<CommandResult> {
  const nodes = await page.evaluate(
    ({ scope, interactiveOnly }) => {
      const root = scope ? document.querySelector(scope) : document.body;
      if (!root) return null;

      const INTERACTIVE =
        'a,button,input,select,textarea,[role=button],[role=link]';
      const elements = interactiveOnly
        ? Array.from(root.querySelectorAll(INTERACTIVE))
        : Array.from(root.querySelectorAll('*'));

      return elements
        .map((el) => {
          const style = getComputedStyle(el);
          if (style.display === 'none' || style.visibility === 'hidden')
            return null;

          const tag = el.tagName.toLowerCase();
          const input = el as HTMLInputElement;
          const name =
            el.getAttribute('aria-label') ||
            (el.id &&
              document.querySelector(`label[for="${el.id}"]`)?.textContent) ||
            el.getAttribute('placeholder') ||
            (tag === 'button' || tag === 'a' ? el.textContent : '') ||
            '';

          // A stable, unique selector so refs survive until the next snapshot.
          const selector = el.id
            ? `#${CSS.escape(el.id)}`
            : input.name
              ? `${tag}[name="${CSS.escape(input.name)}"]`
              : null;
          if (!selector) return null;

          return {
            tag,
            role: el.getAttribute('role') || tag,
            name: (name || '').trim().slice(0, 80),
            selector,
            value: 'value' in el ? String(input.value ?? '') : '',
          };
        })
        .filter((n): n is NonNullable<typeof n> => n !== null);
    },
    { scope, interactiveOnly },
  );

  if (nodes === null) return fail(`No element matches selector: ${scope}`);

  refs.clear();
  const lines = nodes.map((node, i) => {
    const ref = `e${i + 1}`;
    refs.set(ref, node.selector);
    const value = node.value ? ` value="${node.value}"` : '';
    return `- ${node.role} "${node.name}"${value} [ref=${ref}]`;
  });

  return ok(lines.join('\n'));
}

/**
 * Interpret one agent-browser command against `page`.
 *
 * `refs` persists across calls for the lifetime of a session, exactly as the
 * CLI daemon's ref map does.
 */
export async function executeCliCommand(
  page: Page,
  refs: Map<string, string>,
  argv: readonly string[],
): Promise<CommandResult> {
  const [command, ...args] = argv;
  const sel = (i: number) => resolveSelector(args[i], refs);

  try {
    switch (command) {
      case 'open':
      case 'goto':
      case 'navigate':
        await page.goto(args[0], { waitUntil: 'domcontentloaded' });
        return ok({ url: page.url(), title: await page.title() });

      case 'snapshot': {
        const scopeIdx = args.findIndex(
          (a) => a === '-s' || a === '--selector',
        );
        return snapshot(
          page,
          refs,
          scopeIdx === -1 ? undefined : args[scopeIdx + 1],
          args.includes('-i') || args.includes('--interactive'),
        );
      }

      case 'click':
        await page.click(sel(0));
        return ok('clicked');

      case 'fill':
        await page.fill(sel(0), args[1] ?? '');
        return ok('filled');

      case 'type':
        await page.type(sel(0), args[1] ?? '');
        return ok('typed');

      case 'select':
        await page.selectOption(sel(0), args.slice(1));
        return ok('selected');

      case 'check':
        await page.check(sel(0));
        return ok('checked');

      case 'uncheck':
        await page.uncheck(sel(0));
        return ok('unchecked');

      case 'hover':
        await page.hover(sel(0));
        return ok('hovered');

      case 'press':
        await page.keyboard.press(args[0]);
        return ok('pressed');

      case 'scrollintoview':
        await page.locator(sel(0)).scrollIntoViewIfNeeded();
        return ok('scrolled into view');

      case 'back':
        await page.goBack();
        return ok({ url: page.url() });

      case 'forward':
        await page.goForward();
        return ok({ url: page.url() });

      case 'eval':
        return ok({ result: await page.evaluate(args[0]) });

      case 'get': {
        const [what] = args;
        if (what === 'url') return ok({ url: page.url() });
        if (what === 'title') return ok({ title: await page.title() });
        if (what === 'text')
          return ok({ text: await page.textContent(sel(1)) });
        if (what === 'value')
          return ok({ value: await page.inputValue(sel(1)) });
        if (what === 'html') return ok({ html: await page.innerHTML(sel(1)) });
        return fail(`Unsupported: get ${what}`);
      }

      case 'wait': {
        const loadIdx = args.findIndex((a) => a === '--load');
        if (loadIdx !== -1) {
          await page.waitForLoadState(
            args[loadIdx + 1] as 'load' | 'domcontentloaded' | 'networkidle',
          );
          return ok('load state reached');
        }
        const ms = Number(args[0]);
        if (Number.isFinite(ms)) {
          await page.waitForTimeout(ms);
          return ok('waited');
        }
        await page.waitForSelector(sel(0));
        return ok('element visible');
      }

      case 'find': {
        // find <locator> <value> [action] [text]
        const [locator, value, action, text] = args;
        const target =
          locator === 'label'
            ? page.getByLabel(value)
            : locator === 'text'
              ? page.getByText(value)
              : locator === 'placeholder'
                ? page.getByPlaceholder(value)
                : locator === 'testid'
                  ? page.getByTestId(value)
                  : locator === 'role'
                    ? page.getByRole(
                        value as Parameters<Page['getByRole']>[0],
                        args.includes('--name')
                          ? { name: args[args.indexOf('--name') + 1] }
                          : undefined,
                      )
                    : null;
        if (!target) return fail(`Unsupported locator: ${locator}`);

        if (action === 'click') await target.click();
        else if (action === 'fill') await target.fill(text ?? '');
        else if (action === 'check') await target.check();
        else if (action === 'hover') await target.hover();
        else if (action === 'text')
          return ok({ text: await target.textContent() });
        else return fail(`Unsupported find action: ${action}`);
        return ok(`${action} via ${locator}`);
      }

      default:
        return fail(`Unsupported command in eval harness: ${command}`);
    }
  } catch (error: unknown) {
    return fail(error instanceof Error ? error.message : String(error));
  }
}
