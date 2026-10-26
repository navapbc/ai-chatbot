import { once } from 'node:events';
import { mkdtempSync, rmSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import type { Page } from 'playwright';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import WebSocket from 'ws';
import {
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

// End-to-end redaction: what the extension sends up the relay for a page
// holding sensitive values. Run alone with
// `pnpm extension:build && pnpm test:relay e2e/redaction`.
//
// Every value below is fake. Assertions report which value leaked by its
// index in the list, never the value, so a failure never prints one.

const MASK = '[masked]';

// Values held in sensitive fields (field masking protects these).
const FIELD_VALUES = [
  'Zq7-fake-pass', // password typed by script, label "Code"
  '900-00-1234', // SSN value attribute
  '900-00-5678', // SSN typed by script
  '900-00-2468', // value-shaped placeholder on an empty SSN field
  '900-55-0001', // aria-label of a bank account field
  '900-55-0002', // aria-description of a bank account field
  '900-55-0003', // bank account value attribute
  '4000 0000 0000 0002', // card number in the cross-site iframe
];
// Echoed in ordinary text; protected only once registered.
const KNOWN = '900-77-4321';

const forms = (value: string) => [value, value.replace(/[\s-]/g, '')];

/** Indices of `values` found (in any form) in any of `texts`. */
const leaked = (texts: string[], values: string[]) =>
  values.flatMap((value, i) =>
    forms(value).some((f) => texts.some((t) => t.includes(f))) ? [i] : [],
  );

let fixtures: Fixtures;
let relay: Relay;
let browser: ExtensionBrowser;
let page: Page;
let tabId: number;
let sensitiveUrl: string;

/**
 * What the agent sees in a field, without echoing a value into a failure
 * message: the mask, empty, or `clear` (any other text).
 */
const kind = (value: string | undefined) =>
  value === undefined
    ? 'missing'
    : value === MASK
      ? 'masked'
      : value === ''
        ? 'empty'
        : 'clear';

/** Everything that crossed the relay since `mark`, both directions. */
const seenSince = (mark: number) => [
  ...relay.frames.slice(mark),
  ...relay.logs.map((l) => JSON.stringify(l)),
];

/**
 * Splits what crossed the relay since `mark` into DOM.describeNode replies
 * and everything else (logs included), pairing replies with their commands.
 */
const splitDescribeReplies = (mark: number) => {
  const methods = new Map<string, string>();
  const describe: string[] = [];
  const rest: string[] = relay.logs.map((l) => JSON.stringify(l));
  for (const frame of relay.frames.slice(mark)) {
    const msg = JSON.parse(frame) as {
      t?: string;
      cmd?: { id?: string; method?: string };
      type?: string;
      id?: string;
    };
    if (msg.t === 'cmd' && msg.cmd?.id && msg.cmd.method) {
      methods.set(msg.cmd.id, msg.cmd.method);
    }
    const isDescribeReply =
      msg.type === 'reply' &&
      msg.id !== undefined &&
      methods.get(msg.id) === 'DOM.describeNode';
    (isDescribeReply ? describe : rest).push(frame);
  }
  return { describe, rest };
};

async function until(check: () => boolean | Promise<boolean>, ms = 10_000) {
  const deadline = Date.now() + ms;
  while (!(await check())) {
    if (Date.now() > deadline) throw new Error('condition not met in time');
    await new Promise((r) => setTimeout(r, 50));
  }
}

async function startRun(): Promise<{ cdp: string; runId: string }> {
  const tokens = mintRunTokens(relay.secret, fixtures.origin);
  const registered = relay.logs.length;
  const started = await bridgeMessage(browser.sw, {
    type: 'startRun',
    relayUrl: relay.url,
    token: tokens.extension,
    tabId,
    allowedFrameOrigins: [fixtures.frameOrigin],
  });
  expect(started).toEqual({ ok: true });
  await until(() =>
    relay.logs
      .slice(registered)
      .some((l) => l.event === 'registered' && l.runId === tokens.runId),
  );
  return { cdp: cdpUrl(relay, tokens.agent), runId: tokens.runId };
}

const addKnownValue = (value: unknown) =>
  bridgeMessage(browser.sw, { type: 'addKnownValue', value });

interface CdpMessage {
  id?: number;
  sessionId?: string;
  method?: string;
  params?: Record<string, unknown>;
  result?: Record<string, unknown>;
  error?: { message: string };
}

/** A raw CDP client on the relay's /cdp socket. */
async function rawAgent(url: string) {
  const ws = new WebSocket(url);
  await once(ws, 'open');
  const events: CdpMessage[] = [];
  const replies = new Map<number, (m: CdpMessage) => void>();
  const watchers: Array<{
    match: (m: CdpMessage) => boolean;
    resolve: (m: CdpMessage) => void;
  }> = [];
  ws.on('message', (data) => {
    const msg = JSON.parse(data.toString()) as CdpMessage;
    if (typeof msg.id === 'number') {
      replies.get(msg.id)?.(msg);
      return;
    }
    events.push(msg);
    for (const w of [...watchers]) {
      if (!w.match(msg)) continue;
      watchers.splice(watchers.indexOf(w), 1);
      w.resolve(msg);
    }
  });
  let next = 0;
  const call = (
    method: string,
    params: Record<string, unknown> = {},
    sessionId?: string,
  ) =>
    new Promise<CdpMessage>((resolve) => {
      next += 1;
      replies.set(next, resolve);
      ws.send(
        JSON.stringify({
          id: next,
          method,
          params,
          ...(sessionId ? { sessionId } : {}),
        }),
      );
    });
  const waitFor = (match: (m: CdpMessage) => boolean) => {
    const seen = events.find(match);
    if (seen) return Promise.resolve(seen);
    return new Promise<CdpMessage>((resolve) =>
      watchers.push({ match, resolve }),
    );
  };
  const attach = async () => {
    const targets = await call('Target.getTargets');
    const infos = targets.result?.targetInfos as Array<{ targetId: string }>;
    const attached = await call('Target.attachToTarget', {
      targetId: infos[0].targetId,
      flatten: true,
    });
    return attached.result?.sessionId as string;
  };
  /** Auto-attaches and returns the cross-site iframe's child session. */
  const childSession = async (sessionId: string) => {
    await call(
      'Target.setAutoAttach',
      { autoAttach: true, flatten: true, waitForDebuggerOnStart: false },
      sessionId,
    );
    const event = await waitFor(
      (m) =>
        m.method === 'Target.attachedToTarget' &&
        (m.params?.targetInfo as { type?: string })?.type === 'iframe',
    );
    return event.params?.sessionId as string;
  };
  const closed = new Promise<{ code: number; reason: string }>((resolve) =>
    ws.once('close', (code, reason) =>
      resolve({ code, reason: reason.toString() }),
    ),
  );
  const close = async () => {
    if (ws.readyState === WebSocket.CLOSED) return;
    ws.close();
    await once(ws, 'close');
  };
  return { call, waitFor, attach, childSession, events, closed, close };
}

type Agent = Awaited<ReturnType<typeof rawAgent>>;

interface AXNode {
  role?: { value?: string };
  value?: { value?: unknown };
  backendDOMNodeId?: number;
}

/**
 * Each textbox of a session's AX tree, keyed by its element id (read back
 * through DOM.describeNode, which keeps `id` on a sensitive node), mapped to
 * the AX value the agent sees.
 */
async function textboxValues(agent: Agent, sessionId: string) {
  const tree = await agent.call('Accessibility.getFullAXTree', {}, sessionId);
  expect(tree.error).toBeUndefined();
  const nodes = (tree.result?.nodes ?? []) as AXNode[];
  const values = new Map<string, string>();
  for (const node of nodes) {
    if (node.role?.value !== 'textbox') continue;
    if (typeof node.backendDOMNodeId !== 'number') continue;
    const described = await agent.call(
      'DOM.describeNode',
      { backendNodeId: node.backendDOMNodeId },
      sessionId,
    );
    const attrs = ((described.result?.node as { attributes?: string[] })
      ?.attributes ?? []) as string[];
    const at = attrs.indexOf('id');
    if (at < 0 || at % 2 !== 0) continue;
    values.set(attrs[at + 1], String(node.value?.value ?? ''));
  }
  return values;
}

const tabIdFor = (url: string) =>
  browser.sw.evaluate(
    async (u) => (await chrome.tabs.query({ url: u }))[0]?.id,
    url,
  ) as Promise<number>;

/** Everything the extension holds in chrome.storage, in every area. */
const storedAnywhere = () =>
  browser.sw.evaluate(async () =>
    JSON.stringify([
      await chrome.storage.session.get(null),
      await chrome.storage.local.get(null),
      await chrome.storage.sync.get(null),
    ]),
  ) as Promise<string>;

// --- screenshot helpers ----------------------------------------------------------

const REFUSED =
  'The screenshot was refused because sensitive fields could not be masked.';
const STYLED = 'width: 9em; color: rgb(20, 20, 20) !important;';

/**
 * Adds a row of extra fields for the screenshot tests: a sensitive input
 * that already has an inline style, a select, a contenteditable, a textarea,
 * and a same-origin srcdoc iframe with one sensitive and one ordinary field.
 * Returns a function that removes them again.
 */
async function addScreenshotFields() {
  await page.evaluate((styled) => {
    const box = document.createElement('div');
    box.id = 'shotbox';
    box.innerHTML = `
      <input id="styled" aria-label="Passport number" value="X1234567" style="${styled}">
      <select id="sid" aria-label="State ID number"><option>D1234567</option></select>
      <span id="cardlbl">Card number</span>
      <div id="cardce" contenteditable aria-labelledby="cardlbl" style="display: inline-block; width: 10em">4111 1111</div>
      <textarea id="pinnote" aria-label="PIN" rows="1">8642</textarea>
      <iframe id="so" title="same origin" style="height: 40px" srcdoc="<label for=sp>PIN</label> <input id=sp value=5555> <label for=sq>Pet</label> <input id=sq value=Rex>"></iframe>`;
    document.body.append(box);
  }, STYLED);
  await page.frameLocator('#so').locator('#sq').waitFor({ state: 'attached' });
  return () =>
    page.evaluate(() => document.getElementById('shotbox')?.remove());
}

/** The tab, the srcdoc frame and the cross-site frame, by test name. */
const frameName = (url: string) =>
  url === 'about:srcdoc'
    ? 'srcdoc'
    : url.includes('oopif-sensitive')
      ? 'xo'
      : 'main';

/** `frame#id` -> the style attribute of every element with an id. */
async function styleAttributes() {
  const out = new Map<string, string | null>();
  for (const frame of page.frames()) {
    const name = frameName(frame.url());
    const entries = await frame.evaluate(() =>
      Array.from(document.querySelectorAll('[id]'), (el) => [
        el.id,
        el.getAttribute('style'),
      ]),
    );
    for (const [id, style] of entries) out.set(`${name}#${id}`, style);
  }
  return out;
}

const blurAll = () =>
  Promise.all(
    page
      .frames()
      .map((f) =>
        f.evaluate(() =>
          (document.activeElement as HTMLElement | null)?.blur(),
        ),
      ),
  );

// Sensitive fields whose pixels must vanish; every other field in
// fieldRects must render identically masked and unmasked.
const SENSITIVE_IN_SHOT = [
  'main#code',
  'main#ssn',
  'main#ssn2',
  'main#spouse',
  'main#acct',
  'main#styled',
  'main#sid',
  'main#cardce',
  'main#pinnote',
  'srcdoc#sp',
  'xo#cc',
];

interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/**
 * The inner area of each field in viewport pixels, inset past the border
 * (and past the select's arrow and the textarea's grip, which are not text).
 */
async function fieldRects(): Promise<Record<string, Rect>> {
  const locate = {
    'main#code': page.locator('#code'),
    'main#ssn': page.locator('#ssn'),
    'main#ssn2': page.locator('#ssn2'),
    'main#spouse': page.locator('#spouse'),
    'main#acct': page.locator('#acct'),
    'main#city': page.locator('#city'),
    'main#styled': page.locator('#styled'),
    'main#sid': page.locator('#sid'),
    'main#cardce': page.locator('#cardce'),
    'main#pinnote': page.locator('#pinnote'),
    'srcdoc#sp': page.frameLocator('#so').locator('#sp'),
    'srcdoc#sq': page.frameLocator('#so').locator('#sq'),
    'xo#cc': page.frameLocator('#xo').locator('#cc'),
    'xo#qty': page.frameLocator('#xo').locator('#qty'),
  };
  const out: Record<string, Rect> = {};
  for (const [name, locator] of Object.entries(locate)) {
    const box = await locator.boundingBox();
    if (!box) throw new Error(`no box for ${name}`);
    // The select's arrow and the textarea's resize grip are drawn in the
    // field's corner and are not text.
    const right = name === 'main#sid' ? 24 : name === 'main#pinnote' ? 14 : 3;
    out[name] = {
      x: Math.ceil(box.x + 3),
      y: Math.ceil(box.y + 3),
      width: Math.floor(box.width - 3 - right),
      height: Math.floor(box.height - 6),
    };
  }
  return out;
}

/** An unmasked capture of the tab, straight through chrome.debugger. */
const rawCapture = async () =>
  (
    (await browser.sw.evaluate(
      (id) =>
        chrome.debugger.sendCommand({ tabId: id }, 'Page.captureScreenshot', {
          format: 'png',
        }),
      tabId,
    )) as { data: string }
  ).data;

/**
 * Decodes a PNG (in the extension worker: createImageBitmap + OffscreenCanvas)
 * and summarizes each region: `ink` counts pixels darker than 200 in every
 * channel's minimum (text, bullets, placeholder grey), `hash` fingerprints
 * the region's exact pixels.
 */
const regions = (data: string, rects: Rect[]) =>
  browser.sw.evaluate(
    async ({ data, rects }) => {
      const bytes = Uint8Array.from(atob(data), (c) => c.charCodeAt(0));
      const bitmap = await createImageBitmap(
        new Blob([bytes], { type: 'image/png' }),
      );
      const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
      const ctx = canvas.getContext('2d');
      if (!ctx) throw new Error('no 2d context');
      ctx.drawImage(bitmap, 0, 0);
      return rects.map((r) => {
        const px = ctx.getImageData(r.x, r.y, r.width, r.height).data;
        let ink = 0;
        let hash = 0;
        for (let i = 0; i < px.length; i += 4) {
          if (Math.min(px[i], px[i + 1], px[i + 2]) < 200) ink += 1;
          hash =
            (Math.imul(hash, 31) +
              px[i] * 65536 +
              px[i + 1] * 256 +
              px[i + 2]) >>>
            0;
        }
        return { ink, hash };
      });
    },
    { data, rects },
  );

/**
 * Wraps chrome.debugger.sendCommand in the worker (the bridge looks it up on
 * every call) to inject a fault into the bridge's own mask commands. Test
 * only: nothing in the shipped extension has a fault hook.
 */
const patchSendCommand = (mode: 'failChildEvaluate' | 'reloadBeforeCapture') =>
  browser.sw.evaluate((mode) => {
    const dbg = chrome.debugger as unknown as {
      sendCommand: (...args: unknown[]) => Promise<unknown>;
    };
    const g = globalThis as unknown as { originalSendCommand?: unknown };
    const original = dbg.sendCommand;
    g.originalSendCommand = original;
    let armed = true;
    dbg.sendCommand = async (...args: unknown[]) => {
      const [target, method, params] = args as [
        { tabId: number; sessionId?: string },
        string,
        { objectGroup?: string } | undefined,
      ];
      if (
        mode === 'failChildEvaluate' &&
        target.sessionId !== undefined &&
        method === 'Runtime.evaluate' &&
        String(params?.objectGroup ?? '').startsWith('mask-')
      ) {
        throw new Error('forced failure');
      }
      if (
        mode === 'reloadBeforeCapture' &&
        armed &&
        method === 'Page.captureScreenshot'
      ) {
        armed = false;
        await original.call(dbg, { tabId: target.tabId }, 'Page.reload', {});
        await new Promise((r) => setTimeout(r, 1_000));
      }
      return original.apply(dbg, args);
    };
  }, mode);

const unpatchSendCommand = () =>
  browser.sw.evaluate(() => {
    const g = globalThis as unknown as { originalSendCommand?: unknown };
    if (g.originalSendCommand) {
      (chrome.debugger as unknown as { sendCommand: unknown }).sendCommand =
        g.originalSendCommand;
    }
  });

beforeAll(async () => {
  fixtures = await serveFixtures();
  sensitiveUrl = `${fixtures.origin}/sensitive.html`;
  relay = await startRelay();
  browser = await launchExtensionBrowser();
  page = await browser.context.newPage();
  await page.goto(sensitiveUrl);
  await page
    .frameLocator('#xo')
    .locator('#cc')
    .waitFor({ state: 'attached', timeout: 10_000 });
  tabId = await tabIdFor(sensitiveUrl);
}, 60_000);

afterAll(async () => {
  try {
    await closeAgentSessions();
    if (browser) {
      await Promise.race([
        bridgeMessage(browser.sw, { type: 'stopRun' }).catch(() => {}),
        new Promise((resolve) => setTimeout(resolve, 3_000)),
      ]);
    }
  } finally {
    await browser?.stop();
    await relay?.stop();
    await fixtures?.stop();
  }
}, 60_000);

describe('redaction e2e', { timeout: 60_000 }, () => {
  it('no sensitive string appears in any frame the relay saw after snapshot', async () => {
    const mark = relay.frames.length;
    const { cdp } = await startRun();

    // Raw CDP: every page-data method the agent can call, tab and iframe.
    // (Run first: a raw client cannot attach while a daemon's socket lingers.)
    const agent = await rawAgent(cdp);
    const session = await agent.attach();
    await agent.call('DOM.enable', {}, session);
    const tabFields = await textboxValues(agent, session);
    expect(tabFields.size).toBeGreaterThanOrEqual(7);
    const tree = await agent.call('Accessibility.getFullAXTree', {}, session);
    for (const node of (tree.result?.nodes ?? []) as AXNode[]) {
      if (node.role?.value !== 'textbox') continue;
      const resolved = await agent.call(
        'DOM.resolveNode',
        { backendNodeId: node.backendDOMNodeId },
        session,
      );
      const objectId = (resolved.result?.object as { objectId?: string })
        ?.objectId;
      expect(typeof objectId).toBe('string');
      await agent.call(
        'Runtime.callFunctionOn',
        {
          objectId,
          functionDeclaration: 'function () { return this.outerHTML.length }',
          returnByValue: true,
        },
        session,
      );
    }
    const child = await agent.childSession(session);
    const childFields = await textboxValues(agent, child);
    expect([...childFields.keys()].sort()).toEqual(['cc', 'qty']);
    await agent.close();

    // agent-browser's own snapshot, click and fill paths, by ref (as the
    // agent drives it) and by selector.
    const snap = await agentBrowser(['snapshot', '-i'], cdp, 'red');
    expect(snap.success).toBe(true);
    const refs = Object.entries(
      (snap.data as { refs: Record<string, { role: string; name: string }> })
        .refs,
    );
    const ref = (match: (r: { role: string; name: string }) => boolean) => {
      const hit = refs.find(([, r]) => match(r));
      if (!hit) throw new Error('ref not found in snapshot');
      return `@${hit[0]}`;
    };
    const masked = ref((r) => r.role === 'textbox' && r.name === MASK);
    const middle = ref(
      (r) => r.role === 'textbox' && r.name.trim() === 'Middle name',
    );
    expect((await agentBrowser(['click', masked], cdp, 'red')).success).toBe(
      true,
    );
    expect(
      (await agentBrowser(['fill', middle, 'Quinn'], cdp, 'red')).success,
    ).toBe(true);
    expect((await agentBrowser(['click', '#ssn'], cdp, 'red')).success).toBe(
      true,
    );
    expect(
      (await agentBrowser(['fill', '#middle', 'Quinn'], cdp, 'red')).success,
    ).toBe(true);
    expect((await agentBrowser(['snapshot'], cdp, 'red')).success).toBe(true);
    await closeAgentSessions();

    const { describe, rest } = splitDescribeReplies(mark);
    expect(leaked(rest, FIELD_VALUES)).toEqual([]);
    expect(leaked(describe, FIELD_VALUES)).toEqual([]);
    // Positive control: the ordinary value the agent typed did go up.
    expect(seenSince(mark).some((f) => f.includes('Quinn'))).toBe(true);
  });

  it('DOM.describeNode replies leak no sensitive field value, aria-label included', async () => {
    const { cdp } = await startRun();
    const agent = await rawAgent(cdp);
    const session = await agent.attach();
    const mark = relay.frames.length;
    expect((await textboxValues(agent, session)).size).toBeGreaterThan(0);
    await agent.close();
    const { describe, rest } = splitDescribeReplies(mark);
    expect(describe.length).toBeGreaterThan(0);
    expect(leaked(describe, FIELD_VALUES)).toEqual([]);
    expect(leaked(rest, FIELD_VALUES)).toEqual([]);
  });

  it('the agent can tell a filled sensitive field from an empty one', async () => {
    const { cdp } = await startRun();
    const agent = await rawAgent(cdp);
    const session = await agent.attach();
    const fields = await textboxValues(agent, session);
    expect({
      code: kind(fields.get('code')),
      ssn: kind(fields.get('ssn')),
      ssn2: kind(fields.get('ssn2')),
      acct: kind(fields.get('acct')),
      spouse: kind(fields.get('spouse')),
    }).toEqual({
      code: 'masked',
      ssn: 'masked',
      ssn2: 'masked',
      acct: 'masked',
      spouse: 'empty',
    });
    await agent.close();
  });

  it('an ordinary field keeps its value', async () => {
    await page.fill('#middle', '');
    const { cdp } = await startRun();
    const agent = await rawAgent(cdp);
    const session = await agent.attach();
    const fields = await textboxValues(agent, session);
    expect(fields.get('city')).toBe('Springfield');
    expect(kind(fields.get('middle'))).toBe('empty');
    await agent.close();
  });

  it('iframe sensitive field is masked (describe bound to the child session)', async () => {
    const { cdp } = await startRun();
    const agent = await rawAgent(cdp);
    const session = await agent.attach();
    const child = await agent.childSession(session);
    const fields = await textboxValues(agent, child);
    // The card field's only signal is its autocomplete attribute, so it is
    // classified through describe(); the ordinary field keeps its value only
    // if describe() answered from the child's own session. Backend node ids
    // are per renderer, so a describe() sent through the tab session answers
    // for whichever main-frame node shares the id; the field order in
    // oopif-sensitive.html makes that fail both assertions (checked by
    // binding describe() to the tab session).
    expect(kind(fields.get('cc'))).toBe('masked');
    expect(fields.get('qty')).toBe('Blue widget');
    await agent.close();
  });

  it('addKnownValue is refused without a run or for a non-string', async () => {
    await bridgeMessage(browser.sw, { type: 'stopRun' });
    const noRun = await addKnownValue(KNOWN);
    expect(noRun.ok).toBe(false);
    expect(JSON.stringify(noRun)).not.toContain(KNOWN);
    await startRun();
    expect(await addKnownValue(12345)).toEqual({
      ok: false,
      error: 'known value not registered',
    });
    expect(await addKnownValue('x'.repeat(600))).toEqual({
      ok: false,
      error: 'known value not registered',
    });
  });

  it('a value registered with addKnownValue is scrubbed from a snapshot, a console event and a dialog event', async () => {
    const { cdp } = await startRun();
    expect(await addKnownValue(KNOWN)).toEqual({ ok: true });
    const mark = relay.frames.length;
    const agent = await rawAgent(cdp);
    const session = await agent.attach();
    await agent.call('Page.enable', {}, session);
    await agent.call('Runtime.enable', {}, session);

    page.once('dialog', (dialog) => void dialog.dismiss());
    await page.evaluate((v) => {
      console.log(`Echo ${v}`);
      setTimeout(() => alert(`Confirm ${v}`), 0);
    }, KNOWN);
    const consoleEvent = await agent.waitFor(
      (m) =>
        m.method === 'Runtime.consoleAPICalled' &&
        JSON.stringify(m.params).includes('Echo'),
    );
    const dialogEvent = await agent.waitFor(
      (m) => m.method === 'Page.javascriptDialogOpening',
    );
    expect(JSON.stringify(consoleEvent.params)).toContain(MASK);
    expect(JSON.stringify(dialogEvent.params)).toContain(MASK);

    const tree = await agent.call('Accessibility.getFullAXTree', {}, session);
    expect(JSON.stringify(tree.result)).toContain('You entered');
    await agent.close();
    expect((await agentBrowser(['snapshot'], cdp, 'known')).success).toBe(true);
    await closeAgentSessions();

    expect(leaked(seenSince(mark), [KNOWN])).toEqual([]);
    // Values live in worker memory only.
    expect(leaked([await storedAnywhere()], [KNOWN])).toEqual([]);
  });

  it('known values are gone after the run stops and a new run starts', async () => {
    const { cdp } = await startRun();
    expect(await addKnownValue(KNOWN)).toEqual({ ok: true });
    expect(await bridgeMessage(browser.sw, { type: 'stopRun' })).toEqual({
      ok: true,
    });
    expect(leaked([await storedAnywhere()], [KNOWN])).toEqual([]);
    expect((await agentBrowser(['get', 'title'], cdp, 'gone')).success).toBe(
      false,
    );
    await closeAgentSessions();

    const mark = relay.frames.length;
    const next = await startRun();
    const agent = await rawAgent(next.cdp);
    const session = await agent.attach();
    await agent.call('Accessibility.getFullAXTree', {}, session);
    await agent.close();
    // The echo text is ordinary, so with nothing registered it goes up.
    expect(leaked(seenSince(mark), [KNOWN])).toEqual([0]);
  });

  it('scope/leave-origin still ends the run when a known value collides with the URL', async () => {
    const { cdp, runId } = await startRun();
    // Collides with the run origin's host and with the hash set below.
    expect(await addKnownValue('localhost')).toEqual({ ok: true });
    expect(await addKnownValue(KNOWN)).toEqual({ ok: true });
    const mark = relay.frames.length;
    const agent = await rawAgent(cdp);
    const session = await agent.attach();
    await agent.call('Page.enable', {}, session);

    // In scope: routing reads the raw URL, so the run survives even though
    // the URL the agent sees is masked.
    await page.evaluate((v) => {
      location.hash = `ref-${v}`;
    }, KNOWN);
    await agent.waitFor((m) => m.method === 'Page.navigatedWithinDocument');
    const alive = await agent.call(
      'Runtime.evaluate',
      { expression: '1 + 1', returnByValue: true },
      session,
    );
    expect(alive.error).toBeUndefined();
    expect(leaked(seenSince(mark), [KNOWN])).toEqual([]);

    // Out of scope: still ends the run.
    const away = `${fixtures.frameOrigin}/away.html`;
    await page.evaluate((url) => {
      location.href = url;
    }, away);
    expect(await agent.closed).toEqual({ code: 1000, reason: 'run stopped' });
    expect(
      relay.logs.some(
        (l) =>
          l.event === 'run-stopped' &&
          l.runId === runId &&
          l.reason === 'left-origin',
      ),
    ).toBe(true);
    await page.goto(sensitiveUrl);
  });

  // --- screenshots (Task 5) ---------------------------------------------------

  it("a screenshot of the sensitive fixture succeeds and the field's inline style is unchanged afterward", async () => {
    const remove = await addScreenshotFields();
    try {
      const { cdp } = await startRun();
      const agent = await rawAgent(cdp);
      const session = await agent.attach();
      await agent.childSession(session);
      const before = await styleAttributes();
      // Pre-existing inline styles (with and without !important) and none.
      expect(before.get('main#styled')).toBe(STYLED);
      expect(before.get('main#ssn')).toBeNull();

      const shot = await agent.call(
        'Page.captureScreenshot',
        { format: 'png' },
        session,
      );
      expect(shot.error).toBeUndefined();
      expect(typeof shot.result?.data).toBe('string');
      expect(await styleAttributes()).toEqual(before);
      await agent.close();
    } finally {
      await remove();
    }
  });

  it('a masked screenshot hides every sensitive field (tab, srcdoc and cross-site frames) and leaves ordinary fields pixel-identical', async () => {
    const remove = await addScreenshotFields();
    try {
      const { cdp } = await startRun();
      const agent = await rawAgent(cdp);
      const session = await agent.attach();
      await agent.childSession(session);
      await blurAll();
      const rects = await fieldRects();

      // Control: the same tab captured through chrome.debugger directly,
      // bypassing the bridge, so nothing is masked.
      const control = await rawCapture();
      const shot = await agent.call(
        'Page.captureScreenshot',
        { format: 'png' },
        session,
      );
      expect(shot.error).toBeUndefined();
      const masked = shot.result?.data as string;

      const names = Object.keys(rects);
      const unmaskedRegions = await regions(
        control,
        names.map((n) => rects[n]),
      );
      const maskedRegions = await regions(
        masked,
        names.map((n) => rects[n]),
      );
      const result = Object.fromEntries(
        names.map((name, i) => {
          const sensitive = SENSITIVE_IN_SHOT.includes(name);
          const u = unmaskedRegions[i];
          const m = maskedRegions[i];
          return [
            name,
            sensitive
              ? // Visible text without the mask, none with it.
                u.ink > 0 && m.ink === 0
                ? 'hidden'
                : `not hidden (ink ${u.ink} -> ${m.ink})`
              : u.hash === m.hash
                ? 'identical'
                : 'changed',
          ];
        }),
      );
      expect(result).toEqual(
        Object.fromEntries(
          names.map((n) => [
            n,
            SENSITIVE_IN_SHOT.includes(n) ? 'hidden' : 'identical',
          ]),
        ),
      );
      await agent.close();
    } finally {
      await remove();
    }
  });

  it('a screenshot is refused when the mask cannot be applied (a child session whose evaluate is forced to fail)', async () => {
    const { cdp } = await startRun();
    const agent = await rawAgent(cdp);
    const session = await agent.attach();
    await agent.childSession(session);
    const before = await styleAttributes();
    await patchSendCommand('failChildEvaluate');
    try {
      const shot = await agent.call('Page.captureScreenshot', {}, session);
      expect(shot.result).toBeUndefined();
      expect(shot.error?.message).toBe(REFUSED);
      // The tab session was masked first; it is back as it was.
      expect(await styleAttributes()).toEqual(before);
    } finally {
      await unpatchSendCommand();
    }
    const again = await agent.call('Page.captureScreenshot', {}, session);
    expect(again.error).toBeUndefined();
    await agent.close();
  });

  it('a same-origin srcdoc iframe field is masked, and a same-site cross-origin iframe no session covers refuses the screenshot', async () => {
    const framePort = new URL(fixtures.frameOrigin).port;
    const { cdp } = await startRun();
    const agent = await rawAgent(cdp);
    const session = await agent.attach();
    await agent.childSession(session);

    const remove = await addScreenshotFields();
    try {
      await blurAll();
      const rects = await fieldRects();
      const shot = await agent.call('Page.captureScreenshot', {}, session);
      expect(shot.error).toBeUndefined();
      const [pin, pet] = await regions(shot.result?.data as string, [
        rects['srcdoc#sp'],
        rects['srcdoc#sq'],
      ]);
      expect(pin.ink).toBe(0);
      expect(pet.ink).toBeGreaterThan(0);
    } finally {
      await remove();
    }

    // Same site (localhost), other port: cross-origin, so the page cannot
    // look inside it, but in the tab's process, so it gets no child session.
    await page.evaluate((src) => {
      const frame = document.createElement('iframe');
      frame.id = 'samesite';
      frame.src = src;
      document.body.append(frame);
    }, `https://localhost:${framePort}/oopif-sensitive.html`);
    try {
      await page
        .frameLocator('#samesite')
        .locator('#cc')
        .waitFor({ state: 'attached', timeout: 10_000 });
      await new Promise((r) => setTimeout(r, 300));
      expect(
        agent.events.some(
          (m) =>
            m.method === 'Target.attachedToTarget' &&
            String(
              (m.params?.targetInfo as { url?: string })?.url ?? '',
            ).includes(`localhost:${framePort}`),
        ),
      ).toBe(false);
      const before = await styleAttributes();
      const shot = await agent.call('Page.captureScreenshot', {}, session);
      expect(shot.error?.message).toBe(REFUSED);
      expect(await styleAttributes()).toEqual(before);
    } finally {
      await page.evaluate(() => document.getElementById('samesite')?.remove());
    }
    const again = await agent.call('Page.captureScreenshot', {}, session);
    expect(again.error).toBeUndefined();
    await agent.close();
  });

  it('a navigation between apply and capture still leaves no masked style behind', async () => {
    const { cdp } = await startRun();
    const agent = await rawAgent(cdp);
    const session = await agent.attach();
    await agent.childSession(session);
    const loaded = page.waitForEvent('load');
    // Reloads the tab after the masks are applied, before the capture: the
    // image would show a document that was never masked.
    await patchSendCommand('reloadBeforeCapture');
    try {
      const shot = await agent.call('Page.captureScreenshot', {}, session);
      expect(shot.result).toBeUndefined();
      expect(shot.error?.message).toBe(REFUSED);
    } finally {
      await unpatchSendCommand();
    }
    await loaded;
    await page
      .frameLocator('#xo')
      .locator('#cc')
      .waitFor({ state: 'attached', timeout: 10_000 });
    const masks = await Promise.all(
      page
        .frames()
        .map((f) =>
          f.evaluate(
            () =>
              document.querySelectorAll('[style*="text-fill-color"]').length,
          ),
        ),
    );
    expect(masks.every((n) => n === 0)).toBe(true);
    await agent.close();
  });

  it('agent-browser screenshot through the real CLI succeeds against the fixture', async () => {
    const { cdp } = await startRun();
    const dir = mkdtempSync(path.join(tmpdir(), 'relay-e2e-shot-'));
    try {
      const file = path.join(dir, 'shot.png');
      const before = await styleAttributes();
      const shot = await agentBrowser(['screenshot', file], cdp, 'shot');
      expect(shot.success).toBe(true);
      expect(statSync(file).size).toBeGreaterThan(1000);
      expect(await styleAttributes()).toEqual(before);
    } finally {
      await closeAgentSessions();
      rmSync(dir, { recursive: true, force: true });
    }
  });

  // Kept last: Playwright reports no handle for the restarted worker, so
  // `browser.sw` is dead afterwards (modelled on bridge.e2e.test.ts).
  it('known values are gone after the worker restarts; fields stay masked', async () => {
    const { cdp, runId } = await startRun();
    expect(await addKnownValue(KNOWN)).toEqual({ ok: true });
    expect(leaked([await storedAnywhere()], [KNOWN])).toEqual([]);

    const devtools = await browser.context.newCDPSession(page);
    await devtools.send('ServiceWorker.enable');
    await devtools.send('ServiceWorker.stopAllWorkers');
    await until(() =>
      relay.logs.some(
        (l) => l.event === 'extension-detached' && l.runId === runId,
      ),
    );
    const waker = await browser.context.newPage();
    await waker.goto('about:blank');
    await until(() =>
      relay.logs.some(
        (l) => l.event === 'extension-restarted' && l.runId === runId,
      ),
    );

    const mark = relay.frames.length;
    const snap = await agentBrowser(['snapshot'], cdp, 'restarted');
    expect(snap.success).toBe(true);
    expect((snap.data as { snapshot: string }).snapshot).toContain(MASK);
    await closeAgentSessions();
    const { describe, rest } = splitDescribeReplies(mark);
    expect(leaked([...describe, ...rest], FIELD_VALUES)).toEqual([]);
    // The rebuilt run's redactor is empty: the echo text now goes up. Values
    // are never persisted, so the restart dropped the registration.
    expect(leaked(rest, [KNOWN])).toEqual([0]);

    await devtools.detach();
    await waker.close();
  });
});
