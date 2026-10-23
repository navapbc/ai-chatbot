import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { ALLOWED_CDP_METHODS } from '../../../lib/relay/allowed-cdp-methods';
import {
  type AgentResult,
  agentBrowser,
  bridgeMessage,
  cdpUrl,
  closeAgentSessions,
  type ExtensionBrowser,
  type Fixtures,
  launchExtensionBrowser,
  mintRunTokens,
  type Relay,
  serveFixtures,
  startRelay,
} from '../helpers/harness';

// Measures the CDP methods agent-browser sends for every command the Eve
// `browser` tool documents, and pins ALLOWED_CDP_METHODS to that set. The
// relay enforces the allowlist as in production: a method that is not listed
// is refused, and the hub logs the refusal (method name only), so the
// measured set is what was forwarded plus what was refused for being unlisted.
// Run alone with `pnpm extension:build && pnpm test:relay allowlist`.

/** The spec denies these: reading storage, request interception, downloads. */
const NEVER_ALLOWED: ReadonlyArray<string | RegExp> = [
  /^Storage\./,
  /^Fetch\./,
  /^Network\.get(All)?Cookies$/,
  /^Network\.(set|delete)Cookies?$/,
  'Network.clearBrowserCookies',
  'Network.getResponseBody',
  'Network.getRequestPostData',
  /^Browser\./,
  'Page.setDownloadBehavior',
  'Target.createTarget',
  'Target.attachToTarget',
  'Target.sendMessageToTarget',
];

/** Allowed for the relay's own needs rather than agent-browser's traffic. */
const KEPT_FOR_RELAY: Record<string, string> = {
  'Runtime.callFunctionOn': 'ruling R6: kept for ref actions on resolved nodes',
};

const neverAllowed = (method: string) =>
  NEVER_ALLOWED.some((n) =>
    typeof n === 'string' ? n === method : n.test(method),
  );

let fixtures: Fixtures;
let relay: Relay;
let browser: ExtensionBrowser;
let cdp: string;
let failures: string[] = [];

const run = async (args: string[]): Promise<AgentResult> => {
  const result = await agentBrowser(args, cdp);
  // `batch` answers with an array of per-command results.
  const steps = Array.isArray(result) ? (result as AgentResult[]) : [result];
  const failed = steps.find((step) => !step.success);
  if (failed) failures.push(`${args.join(' ')} -> ${failed.error}`);
  return result;
};

const pause = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Looks a ref up in a `snapshot -i` result by role and accessible name. */
const refFor = (snap: AgentResult, role: string, name: string): string => {
  const refs = (
    snap.data as { refs: Record<string, { role: string; name: string }> }
  ).refs;
  const hit = Object.entries(refs).find(
    ([, r]) => r.role === role && r.name.trim() === name,
  );
  if (!hit) throw new Error(`no ${role} "${name}" in snapshot`);
  return `@${hit[0]}`;
};

beforeAll(async () => {
  // Leaves dialogs open so `dialog accept|dismiss` has something to answer;
  // a short default timeout keeps a command that cannot succeed quick.
  process.env.AGENT_BROWSER_NO_AUTO_DIALOG = '1';
  process.env.AGENT_BROWSER_DEFAULT_TIMEOUT = '4000';
  fixtures = await serveFixtures();
  relay = await startRelay();
  browser = await launchExtensionBrowser();
  const page = await browser.context.newPage();
  // Playwright dismisses dialogs nobody listens for; leave them to the agent.
  page.on('dialog', () => {});
  await page.goto(fixtures.formUrl);
  const tabId = (await browser.sw.evaluate(
    async (u) => (await chrome.tabs.query({ url: u }))[0]?.id,
    fixtures.formUrl,
  )) as number;
  const tokens = mintRunTokens(relay.secret, fixtures.origin);
  const started = await bridgeMessage(browser.sw, {
    type: 'startRun',
    relayUrl: relay.url,
    token: tokens.extension,
    tabId,
    allowedFrameOrigins: [fixtures.frameOrigin],
  });
  expect(started).toEqual({ ok: true });
  for (
    let i = 0;
    i < 200 && !relay.logs.some((l) => l.event === 'registered');
    i++
  ) {
    await new Promise((r) => setTimeout(r, 50));
  }
  cdp = cdpUrl(relay, tokens.agent);
}, 60_000);

afterAll(async () => {
  await closeAgentSessions();
  await bridgeMessage(browser.sw, { type: 'stopRun' }).catch(() => {});
  await browser?.stop();
  await relay?.stop();
  await fixtures?.stop();
}, 60_000);

/** Runs every command the Eve browser tool documents, once or more. */
async function driveDocumentedCommands() {
  await run(['open', fixtures.formUrl]);
  await run(['snapshot']);
  await run(['snapshot', '-s', 'form']);
  const snap = await run(['snapshot', '-i']);
  const first = refFor(snap, 'textbox', 'First name');
  const last = refFor(snap, 'textbox', 'Last name');
  const state = refFor(snap, 'combobox', 'State');
  const agree = refFor(snap, 'checkbox', 'I agree');
  const plain = refFor(snap, 'button', 'Plain action');

  await run(['click', first]);
  await run(['fill', first, 'Ada']);
  await run(['type', last, 'Lovelace']);
  await run(['select', state, 'CA']);
  await run(['find', 'label', 'First name', 'fill', 'Grace']);
  await run(['find', 'role', 'button', 'click', '--name', 'Plain action']);
  await run(['press', 'Tab']);
  await run(['press', 'Control+a']);
  await run(['hover', plain]);
  await run(['check', agree]);
  await run(['uncheck', agree]);
  await run(['scrollintoview', plain]);
  await run(['wait', plain]);
  await run(['wait', '#plain']);
  await run(['wait', '200']);
  await run(['wait', '--load', 'networkidle']);
  await run(['get', 'text', plain]);
  await run(['get', 'value', first]);
  await run(['get', 'url']);
  await run(['get', 'title']);
  await run(['scroll', 'down', '500']);
  await run(['scroll', 'up', '500']);
  await run(['screenshot']);
  await run(['eval', 'document.title']);
  await run([
    'batch',
    '--bail',
    `click ${first}`,
    'press 1',
    'wait 100',
    'press 2',
  ]);

  // Dialogs: the page opens one after the command returns.
  await run(['eval', 'setTimeout(() => alert("a"), 50); 1']);
  await pause(300);
  await run(['dialog', 'accept']);
  await run(['eval', 'setTimeout(() => confirm("b"), 50); 1']);
  await pause(300);
  await run(['dialog', 'dismiss']);

  // Frames: a same-origin iframe, then the cross-site one, then back.
  await run(['frame', '#child']);
  await run(['snapshot', '-i']);
  await run(['frame', 'main']);
  await run(['frame', '#xo']);
  await run(['snapshot', '-i']);
  await run(['frame', 'main']);

  // History: navigate in scope, then go back and forward.
  await run(['open', `${fixtures.origin}/page2.html`]);
  await run(['back']);
  await run(['forward']);
  await run(['back']);

  // Tabs. `tab new` is refused for scope; it still shows the method used.
  await run(['tab']);
  await run(['tab', 't1']);
  await run(['tab', 'new']);
  await run(['tab', 'close']);
}

/**
 * Commands that are expected to fail, and why. Their CDP traffic still counts:
 * the hub logs a method whether or not the command then succeeds.
 */
const EXPECTED_FAILURES = [
  // agent-browser refuses to close the only tab before sending anything.
  /^tab close /,
  // Refused for scope: the url is not the run origin.
  /^tab new /,
  // agent-browser 0.33.2 times out on `wait @ref` even against plain Chrome.
  /^wait @e\d+ /,
  // Resolving a cross-site iframe by selector fails through the relay.
  /^frame #xo /,
];

describe('CDP allowlist', { timeout: 90_000 }, () => {
  it('matches what agent-browser sends for the documented commands', async () => {
    failures = [];
    await driveDocumentedCommands();

    type Line = {
      event?: string;
      method?: string;
      level?: string;
      reason?: string;
    };
    const lines = relay.logs as Line[];
    const session = (l: Line) => l.level === 'session' && l.method;
    const attempted = new Set(
      lines
        .filter(
          (l) =>
            session(l) &&
            (l.event === 'command' ||
              l.event === 'command-failed' ||
              (l.event === 'command-denied' && l.reason === 'method')),
        )
        .map((l) => l.method as string),
    );
    const sorted = [...attempted].sort();

    // Report aid: shows in the failure message only.
    const eventMethods = [
      ...new Set(
        relay.frames
          .map((f) => JSON.parse(f) as { type?: string; method?: string })
          .filter((f) => f.type === 'event' && typeof f.method === 'string')
          .map((f) => f.method as string),
      ),
    ].sort();
    if (process.env.ALLOWLIST_VERBOSE === '1') {
      const browserLevel = [
        ...new Set(
          lines
            .filter((l) => l.level === 'browser' && l.method)
            .map((l) => `${l.event}:${l.method}`),
        ),
      ].sort();
      console.log(
        JSON.stringify(
          { sorted, browserLevel, eventMethods, failures },
          null,
          1,
        ),
      );
    }

    const forbidden = sorted.filter(neverAllowed);
    expect(
      forbidden,
      `agent-browser sent methods the spec denies: ${forbidden.join(', ')}`,
    ).toEqual([]);

    const missing = sorted.filter((m) => !ALLOWED_CDP_METHODS.has(m));
    expect(missing, `measured but not in ALLOWED_CDP_METHODS`).toEqual([]);

    const unexpected = failures.filter(
      (f) => !EXPECTED_FAILURES.some((e) => e.test(f)),
    );
    expect(unexpected, 'documented commands that failed').toEqual([]);

    const unmeasured = [...ALLOWED_CDP_METHODS]
      .filter((m) => !attempted.has(m) && !(m in KEPT_FOR_RELAY))
      .sort();
    expect(
      unmeasured,
      'in ALLOWED_CDP_METHODS but neither measured nor in KEPT_FOR_RELAY',
    ).toEqual([]);
    for (const kept of Object.keys(KEPT_FOR_RELAY)) {
      expect(ALLOWED_CDP_METHODS.has(kept), `${kept} is not allowed`).toBe(
        true,
      );
    }
  });
});
