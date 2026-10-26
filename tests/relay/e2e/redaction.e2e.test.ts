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
      <style>#faded { transition: all 2s; }</style>
      <input id="faded" aria-label="Routing number" value="011000015">
      <label for="ref">Reference</label> <input id="ref" value="900-00-1234">
      <input id="field_37" name="field_37" value="900 00 1234">
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
  'main#faded',
  'main#ref',
  'main#field_37',
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
    'main#faded': page.locator('#faded'),
    'main#ref': page.locator('#ref'),
    'main#field_37': page.locator('#field_37'),
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
 * and summarizes each region: `ink` counts pixels whose darkest channel is
 * under 200 (text, bullets, placeholder grey), `nonBg` counts pixels that
 * differ by more than 24 in any channel from the region's most common colour
 * (its background: any text, selection highlight or glyph), `hash`
 * fingerprints the exact pixels.
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
        const counts = new Map<number, number>();
        for (let i = 0; i < px.length; i += 4) {
          const c = px[i] * 65536 + px[i + 1] * 256 + px[i + 2];
          counts.set(c, (counts.get(c) ?? 0) + 1);
        }
        const bg = [...counts].sort((a, b) => b[1] - a[1])[0]?.[0] ?? 0;
        const [br, bgr, bb] = [bg >> 16, (bg >> 8) & 255, bg & 255];
        let nonBg = 0;
        for (let i = 0; i < px.length; i += 4) {
          const min = Math.min(px[i], px[i + 1], px[i + 2]);
          if (min < 200) ink += 1;
          if (
            Math.max(
              Math.abs(px[i] - br),
              Math.abs(px[i + 1] - bgr),
              Math.abs(px[i + 2] - bb),
            ) > 24
          ) {
            nonBg += 1;
          }
          hash =
            (Math.imul(hash, 31) +
              px[i] * 65536 +
              px[i + 1] * 256 +
              px[i + 2]) >>>
            0;
        }
        return { ink, nonBg, hash };
      });
    },
    { data, rects },
  );

/**
 * Wraps chrome.debugger.sendCommand in the worker (the bridge looks it up on
 * every call) to inject a fault into the bridge's own mask commands. Test
 * only: nothing in the shipped extension has a fault hook.
 */
const patchSendCommand = (
  mode:
    | 'failChildEvaluate'
    | 'reloadBeforeCapture'
    | 'slowCapture'
    | 'detachAfterMask'
    | 'logDelayMask',
) =>
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
      if (
        mode === 'slowCapture' &&
        armed &&
        method === 'Page.captureScreenshot'
      ) {
        armed = false;
        await new Promise((r) => setTimeout(r, 1_500));
      }
      const fn = String(
        (params as { functionDeclaration?: string } | undefined)
          ?.functionDeclaration ?? '',
      );
      if (mode === 'logDelayMask') {
        // Logs every command (mask steps by name, the agent's evaluates by
        // expression) and when it settles; holds the mask step 600ms.
        const store = globalThis as unknown as { sendLog?: string[] };
        if (!store.sendLog) store.sendLog = [];
        const log = store.sendLog;
        const expr = (params as { expression?: string } | undefined)
          ?.expression;
        const label = fn
          ? (/this\.(\w+)\(/.exec(fn)?.[1] ?? 'fn')
          : method === 'Runtime.evaluate' && expr && !expr.startsWith('(()')
            ? expr.replace(/^"|"$/g, '').slice(0, 11)
            : method.replace(/^\w+\./, '');
        log.push(`send ${label}`);
        if (fn.includes('this.mask(')) {
          await new Promise((r) => setTimeout(r, 600));
        }
        try {
          return await original.apply(dbg, args);
        } finally {
          log.push(`done ${label}`);
          log.push(label);
        }
      }
      if (mode === 'detachAfterMask' && armed && fn.includes('this.mask(')) {
        armed = false;
        // The page's self-restore, shortened from 25s for the test.
        const p = params as unknown as {
          arguments: Array<{ value?: unknown }>;
        };
        p.arguments[1] = { value: 1_500 };
        const out = await original.apply(dbg, args);
        await chrome.debugger.detach({ tabId: target.tabId });
        return out;
      }
      return original.apply(dbg, args);
    };
  }, mode);

/** Mask leftovers in every frame: marker attributes, mask styles, sheets. */
const leftovers = async () => {
  let n = 0;
  for (const frame of page.frames()) {
    n += await frame.evaluate(() => {
      let found = document.adoptedStyleSheets.length;
      for (const el of document.querySelectorAll('*')) {
        if (
          el.getAttributeNames().some((a) => a.startsWith('data-labs-mask'))
        ) {
          found += 1;
        }
        if ((el.getAttribute('style') ?? '').includes('text-fill-color')) {
          found += 1;
        }
      }
      return found;
    });
  }
  return n;
};

/** Reloads the fixture (after a test that changed its DOM). */
async function reloadFixture() {
  await page.goto(sensitiveUrl);
  await page
    .frameLocator('#xo')
    .locator('#cc')
    .waitFor({ state: 'attached', timeout: 10_000 });
}

/** A locator's box inset by `inset` pixels on every side. */
async function insetBox(
  locator: { boundingBox(): Promise<Rect | null> },
  inset = 3,
): Promise<Rect> {
  const box = await locator.boundingBox();
  if (!box) throw new Error('no box');
  return {
    x: Math.ceil(box.x + inset),
    y: Math.ceil(box.y + inset),
    width: Math.floor(box.width - 2 * inset),
    height: Math.floor(box.height - 2 * inset),
  };
}

/**
 * Masked and unmasked captures of the same moment, each region summarized:
 * `hidden` when the control shows something and the masked capture shows
 * only background.
 */
async function hiddenIn(
  agent: Agent,
  session: string,
  rects: Record<string, Rect>,
) {
  const names = Object.keys(rects);
  const control = await rawCapture();
  const shot = await agent.call('Page.captureScreenshot', {}, session);
  if (shot.error) return { error: shot.error.message };
  const u = await regions(
    control,
    names.map((n) => rects[n]),
  );
  const m = await regions(
    shot.result?.data as string,
    names.map((n) => rects[n]),
  );
  return Object.fromEntries(
    names.map((n, i) => [
      n,
      u[i].nonBg > 0 && m[i].nonBg === 0
        ? 'hidden'
        : `visible (${u[i].nonBg} -> ${m[i].nonBg})`,
    ]),
  );
}

/** The command log of the 'logDelayMask' patch, as one string. */
const sendLog = () =>
  browser.sw.evaluate(() =>
    ((globalThis as unknown as { sendLog?: string[] }).sendLog ?? []).join(
      '\n',
    ),
  ) as Promise<string>;

const unpatchSendCommand = () =>
  browser.sw.evaluate(() => {
    const g = globalThis as unknown as { originalSendCommand?: unknown };
    (globalThis as unknown as { sendLog?: string[] }).sendLog = [];
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

  it('a value registered with addKnownValue is scrubbed from a snapshot, console events never reach the agent and dialog text is blank', async () => {
    const { cdp } = await startRun();
    expect(await addKnownValue(KNOWN)).toEqual({ ok: true });
    const mark = relay.frames.length;
    const agent = await rawAgent(cdp);
    const session = await agent.attach();
    await agent.call('Page.enable', {}, session);
    await agent.call('Runtime.enable', {}, session);

    const dialogShown = new Promise<void>((resolve) =>
      page.once('dialog', (dialog) => {
        void dialog.dismiss();
        resolve();
      }),
    );
    await page.evaluate((v) => {
      console.log(`Echo ${v}`);
      setTimeout(() => alert(`Confirm ${v}`), 0);
    }, KNOWN);
    await dialogShown;
    // The dialog opened: the agent hears that it did, with no text.
    const opening = await agent.waitFor(
      (m) => m.method === 'Page.javascriptDialogOpening',
    );
    expect(opening.params).toEqual({
      type: 'alert',
      hasBrowserHandler: true,
      url: '',
      message: '',
      defaultPrompt: '',
    });
    // Console output is not on the event allowlist: an allowed event after it
    // proves delivery works while it never arrives.
    await page.reload();
    await agent.waitFor(
      (m) => m.method === 'Page.frameNavigated' && m.sessionId === session,
    );
    expect(
      agent.events.filter((m) => m.method === 'Runtime.consoleAPICalled'),
    ).toEqual([]);

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
      await agent.call('Runtime.enable', {}, session);
      await agent.childSession(session);
      const before = await styleAttributes();
      // Pre-existing inline styles (with and without !important) and none.
      expect(before.get('main#styled')).toBe(STYLED);
      expect(before.get('main#ssn')).toBeNull();

      const started = Date.now();
      const shot = await agent.call(
        'Page.captureScreenshot',
        { format: 'png' },
        session,
      );
      // Well inside the relay's 30s command timeout.
      expect(Date.now() - started).toBeLessThan(10_000);
      expect(shot.error).toBeUndefined();
      expect(typeof shot.result?.data).toBe('string');
      expect(await styleAttributes()).toEqual(before);
      expect(await leftovers()).toBe(0);
      // The mask's isolated worlds are never announced to the agent.
      expect(
        agent.events.some(
          (m) =>
            m.method === 'Runtime.executionContextCreated' &&
            JSON.stringify(m.params).includes('__labs_screenshot_mask__'),
        ),
      ).toBe(false);
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
              ? // Something visible without the mask, only background with it.
                u.nonBg > 0 && m.nonBg === 0
                ? 'hidden'
                : `not hidden (${u.nonBg} -> ${m.nonBg})`
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
      expect(await leftovers()).toBe(0);
    } finally {
      await unpatchSendCommand();
    }
    const again = await agent.call('Page.captureScreenshot', {}, session);
    expect(again.error).toBeUndefined();
    await agent.close();
  });

  it('a same-site cross-origin iframe is masked through the frame tree; an out-of-process iframe with no child session refuses', async () => {
    const framePort = new URL(fixtures.frameOrigin).port;
    // Same site (localhost), other port: cross-origin, so page script could
    // not look inside it, but in the tab's process and frame tree.
    await page.evaluate((src) => {
      const frame = document.createElement('iframe');
      frame.id = 'samesite';
      frame.src = src;
      document.body.append(frame);
    }, `https://localhost:${framePort}/oopif-sensitive.html`);
    try {
      const inner = page.frameLocator('#samesite').locator('#cc');
      await inner.waitFor({ state: 'attached', timeout: 10_000 });
      const { cdp } = await startRun();
      const agent = await rawAgent(cdp);
      const session = await agent.attach();
      await agent.childSession(session);
      await blurAll();
      expect(
        await hiddenIn(agent, session, { samesite: await insetBox(inner) }),
      ).toEqual({ samesite: 'hidden' });
      await agent.close();
    } finally {
      await page.evaluate(() => document.getElementById('samesite')?.remove());
    }

    // No Target.setAutoAttach: the cross-site #xo has no child session.
    const { cdp } = await startRun();
    const agent = await rawAgent(cdp);
    const session = await agent.attach();
    const before = await styleAttributes();
    const shot = await agent.call('Page.captureScreenshot', {}, session);
    expect(shot.error?.message).toBe(REFUSED);
    expect(await styleAttributes()).toEqual(before);
    expect(await leftovers()).toBe(0);
    await agent.close();
  });

  it('an <object> and an <embed> holding an SSN field are masked (P2)', async () => {
    const framePort = new URL(fixtures.frameOrigin).port;
    const src = `https://localhost:${framePort}/oopif-sensitive.html`;
    await page.evaluate((url) => {
      const o = document.createElement('object');
      o.id = 'ob';
      o.data = url;
      o.width = '300';
      o.height = '60';
      const e = document.createElement('embed');
      e.id = 'em';
      e.src = url;
      e.width = '300';
      e.height = '60';
      document.body.append(o, e);
    }, src);
    try {
      await until(
        () =>
          page.frames().filter((f) => f.url() === src).length === 2 &&
          Promise.all(
            page
              .frames()
              .filter((f) => f.url() === src)
              .map((f) => f.locator('#cc').count()),
          ).then((c) => c.every((n) => n === 1)),
      );
      const frames = page.frames().filter((f) => f.url() === src);
      const { cdp } = await startRun();
      const agent = await rawAgent(cdp);
      const session = await agent.attach();
      await agent.childSession(session);
      await blurAll();
      expect(
        await hiddenIn(agent, session, {
          first: await insetBox(frames[0].locator('#cc')),
          second: await insetBox(frames[1].locator('#cc')),
        }),
      ).toEqual({ first: 'hidden', second: 'hidden' });
      await agent.close();
    } finally {
      await page.evaluate(() => {
        document.getElementById('ob')?.remove();
        document.getElementById('em')?.remove();
      });
    }
  });

  it('a closed shadow root holding a field refuses the screenshot (P3)', async () => {
    await page.evaluate(() => {
      const host = document.createElement('div');
      host.id = 'closedhost';
      const root = host.attachShadow({ mode: 'closed' });
      root.innerHTML = '<label>SSN <input value="900-00-1234"></label>';
      document.body.append(host);
    });
    const { cdp } = await startRun();
    const agent = await rawAgent(cdp);
    const session = await agent.attach();
    await agent.childSession(session);
    try {
      const shot = await agent.call('Page.captureScreenshot', {}, session);
      expect(shot.error?.message).toBe(REFUSED);
      expect(await leftovers()).toBe(0);
    } finally {
      await page.evaluate(() =>
        document.getElementById('closedhost')?.remove(),
      );
    }
    const again = await agent.call('Page.captureScreenshot', {}, session);
    expect(again.error).toBeUndefined();
    await agent.close();
  });

  it('a page that overrides querySelectorAll, Set.has, Array.push and getPropertyValue still gets masked (P7)', async () => {
    await page.evaluate(() => {
      Document.prototype.querySelectorAll = () => {
        throw new Error('nope');
      };
      Set.prototype.has = () => false;
      Array.prototype.push = () => 0;
      CSSStyleDeclaration.prototype.getPropertyValue = () => 'transparent';
    });
    try {
      const { cdp } = await startRun();
      const agent = await rawAgent(cdp);
      const session = await agent.attach();
      await agent.childSession(session);
      expect(
        await hiddenIn(agent, session, {
          ssn: await insetBox(page.locator('#ssn')),
        }),
      ).toEqual({ ssn: 'hidden' });
      await agent.close();
    } finally {
      await reloadFixture();
    }
  });

  /**
   * Installs a page watcher that runs `body` (with `el`, the element) every
   * time the mask lands on #ssn, i.e. inside every mask window; `once` stops
   * after the first.
   */
  const disruptOnMask = (body: string, once: boolean) =>
    page.evaluate(
      ([code, onlyOnce]) => {
        const run = new Function('el', code as string);
        const watcher = new MutationObserver((records) => {
          for (const r of records) {
            const el = r.target as HTMLElement;
            if (el.id !== 'ssn') continue;
            if (!(el.getAttribute('style') ?? '').includes('text-fill-color')) {
              continue;
            }
            if (onlyOnce) watcher.disconnect();
            run(el);
            return;
          }
        });
        watcher.observe(document.body, {
          subtree: true,
          attributes: true,
          attributeFilter: ['style'],
        });
      },
      [body, once] as const,
    );

  for (const [name, script] of [
    [
      'a React-style re-render that replaces the masked input (P1)',
      `const fresh = document.createElement('input');
       fresh.id = 'ssn'; fresh.value = el.value;
       el.replaceWith(fresh);`,
    ],
    [
      'a clone swapped in for the capture and swapped back (P1b)',
      `const clone = el.cloneNode(true);
       clone.removeAttribute('style');
       for (const a of clone.getAttributeNames()) if (a.startsWith('data-labs-mask')) clone.removeAttribute(a);
       el.replaceWith(clone);
       setTimeout(() => clone.replaceWith(el), 300);`,
    ],
    [
      'a srcdoc iframe with an SSN field appended during the window',
      `const f = document.createElement('iframe');
       f.srcdoc = '<label for=s>SSN</label><input id=s value=900-00-1234>';
       document.body.append(f);`,
    ],
  ] as const) {
    it(`refuses ${name}, on every retry`, async () => {
      await disruptOnMask(script, false);
      try {
        const { cdp } = await startRun();
        const agent = await rawAgent(cdp);
        const session = await agent.attach();
        await agent.childSession(session);
        const shot = await agent.call('Page.captureScreenshot', {}, session);
        expect(shot.result).toBeUndefined();
        expect(shot.error?.message).toBe(REFUSED);
        await agent.close();
        await until(async () => (await leftovers()) === 0, 3_000);
      } finally {
        await reloadFixture();
      }
    });
  }

  it('a one-off re-render during the window is retried, and the retry masks the new input', async () => {
    await disruptOnMask(
      `const fresh = document.createElement('input');
       fresh.id = 'ssn'; fresh.value = el.value;
       el.replaceWith(fresh);`,
      true,
    );
    try {
      const { cdp } = await startRun();
      const agent = await rawAgent(cdp);
      const session = await agent.attach();
      await agent.childSession(session);
      await blurAll();
      const rect = await insetBox(page.locator('#ssn'));
      const shot = await agent.call('Page.captureScreenshot', {}, session);
      expect(shot.error).toBeUndefined();
      const control = await rawCapture();
      const [u] = await regions(control, [rect]);
      const [m] = await regions(shot.result?.data as string, [rect]);
      expect(u.nonBg).toBeGreaterThan(0);
      expect(m.nonBg).toBe(0);
      await agent.close();
      expect(await leftovers()).toBe(0);
    } finally {
      await reloadFixture();
    }
  });

  it('text churn during the window does not refuse; a toast with a registered value refuses (I-B)', async () => {
    const OTHER = '900-88-1357';
    // A clock ticking every 20ms: text changes, no registered value.
    await page.evaluate(() => {
      const clock = document.createElement('p');
      clock.id = 'clock';
      document.body.append(clock);
      window.setInterval(() => {
        clock.textContent = new Date().toISOString();
      }, 20);
    });
    try {
      const { cdp } = await startRun();
      expect(await addKnownValue(OTHER)).toEqual({ ok: true });
      const agent = await rawAgent(cdp);
      const session = await agent.attach();
      await agent.childSession(session);
      const shot = await agent.call('Page.captureScreenshot', {}, session);
      expect(shot.error).toBeUndefined();

      // A toast saying the registered value appears inside the window.
      await page.evaluate((v) => {
        const target = document.getElementById('ssn') as HTMLElement;
        const watcher = new MutationObserver(() => {
          watcher.disconnect();
          const toast = document.createElement('p');
          toast.textContent = `Saved ${v}`;
          document.body.prepend(toast);
        });
        watcher.observe(target, {
          attributes: true,
          attributeFilter: ['style'],
        });
      }, OTHER);
      const second = await agent.call('Page.captureScreenshot', {}, session);
      expect(second.error?.message).toBe(REFUSED);
      await agent.close();
      await until(async () => (await leftovers()) === 0, 3_000);
    } finally {
      await reloadFixture();
    }
  });

  it('a registered value split by markup in page text refuses the screenshot (I-B, E1)', async () => {
    const OTHER = '900-88-1357';
    await page.evaluate(() => {
      const p = document.createElement('p');
      p.id = 'split';
      p.innerHTML = 'Reference <span>900</span>-88-1357 confirmed';
      document.body.prepend(p);
    });
    try {
      const { cdp } = await startRun();
      expect(await addKnownValue(OTHER)).toEqual({ ok: true });
      const agent = await rawAgent(cdp);
      const session = await agent.attach();
      await agent.childSession(session);
      const shot = await agent.call('Page.captureScreenshot', {}, session);
      expect(shot.error?.message).toBe(REFUSED);
      expect(await leftovers()).toBe(0);
      await agent.close();
    } finally {
      await page.evaluate(() => document.getElementById('split')?.remove());
    }
  });

  it('a command issued during the mask window runs only after the restore (I-A)', async () => {
    const { cdp } = await startRun();
    const agent = await rawAgent(cdp);
    const session = await agent.attach();
    await agent.childSession(session);
    await patchSendCommand('logDelayMask');
    try {
      const shot = agent.call('Page.captureScreenshot', {}, session);
      // Sent at once, and again from inside the (delayed) mask window.
      const early = agent.call(
        'Runtime.evaluate',
        { expression: '"probe-early"', returnByValue: true },
        session,
      );
      await new Promise((r) => setTimeout(r, 300));
      const late = agent.call(
        'Runtime.evaluate',
        { expression: '"probe-late"', returnByValue: true },
        session,
      );
      await Promise.all([shot, early, late]);
      const log = await sendLog();
      const firstMask = log.indexOf('createIsolatedWorld');
      const lastRestore = log.lastIndexOf('restore');
      expect(firstMask).toBeGreaterThanOrEqual(0);
      for (const probe of ['probe-early', 'probe-late']) {
        const sent = log.indexOf(`send ${probe}`);
        const done = log.indexOf(`done ${probe}`);
        // Wholly before the mask window, or after it.
        expect(done < firstMask || sent > lastRestore).toBe(true);
      }
      expect(log.indexOf('send probe-late')).toBeGreaterThan(lastRestore);
    } finally {
      await unpatchSendCommand();
    }
    await agent.close();
  });

  it('a command in flight when a screenshot starts is drained first (I-A)', async () => {
    const { cdp } = await startRun();
    const agent = await rawAgent(cdp);
    const session = await agent.attach();
    await agent.childSession(session);
    await patchSendCommand('logDelayMask');
    try {
      const slow = agent.call(
        'Runtime.evaluate',
        {
          expression:
            'new Promise((r) => setTimeout(() => r("probe-slow"), 800))',
          awaitPromise: true,
          returnByValue: true,
        },
        session,
      );
      // The command has reached chrome.debugger (it is in flight).
      await until(async () => (await sendLog()).includes('send new Promise'));
      const shot = await agent.call('Page.captureScreenshot', {}, session);
      expect(shot.error).toBeUndefined();
      await slow;
      const log = await sendLog();
      expect(log.indexOf('done new Promise')).toBeGreaterThanOrEqual(0);
      expect(log.indexOf('done new Promise')).toBeLessThan(
        log.indexOf('createIsolatedWorld'),
      );
    } finally {
      await unpatchSendCommand();
    }
    await agent.close();
  });

  it('a stop while a screenshot drains ends promptly and applies no mask (I-A)', async () => {
    const { cdp } = await startRun();
    const agent = await rawAgent(cdp);
    const session = await agent.attach();
    await agent.childSession(session);
    await patchSendCommand('logDelayMask');
    try {
      void agent.call(
        'Runtime.evaluate',
        {
          expression: 'new Promise((r) => setTimeout(() => r(1), 3000))',
          awaitPromise: true,
        },
        session,
      );
      // The command has reached chrome.debugger (it is in flight).
      await until(async () => (await sendLog()).includes('send new Promise'));
      void agent.call('Page.captureScreenshot', {}, session);
      await new Promise((r) => setTimeout(r, 200));
      const started = Date.now();
      expect(await bridgeMessage(browser.sw, { type: 'stopRun' })).toEqual({
        ok: true,
      });
      expect(Date.now() - started).toBeLessThan(7_000);
      await new Promise((r) => setTimeout(r, 500));
      expect((await sendLog()).includes('createIsolatedWorld')).toBe(false);
      expect(await leftovers()).toBe(0);
    } finally {
      await unpatchSendCommand();
    }
    await agent.close();
  });

  it('a hung command makes the screenshot refuse instead of wedging the tab (I-A)', async () => {
    const { cdp } = await startRun();
    const agent = await rawAgent(cdp);
    const session = await agent.attach();
    await agent.childSession(session);
    await patchSendCommand('logDelayMask');
    try {
      void agent.call(
        'Runtime.evaluate',
        { expression: 'new Promise(() => {})', awaitPromise: true },
        session,
      );
      // The command has reached chrome.debugger (it is in flight).
      await until(async () => (await sendLog()).includes('send new Promise'));
      const shot = await agent.call('Page.captureScreenshot', {}, session);
      expect(shot.error?.message).toBe(REFUSED);
      expect(await leftovers()).toBe(0);
      // Not wedged: other commands still run.
      const after = await agent.call(
        'Runtime.evaluate',
        { expression: '1 + 1', returnByValue: true },
        session,
      );
      expect(after.result?.result).toEqual({
        type: 'number',
        value: 2,
        description: '2',
      });
    } finally {
      await unpatchSendCommand();
      await agent.close();
      // The reload settles the hung evaluate.
      await reloadFixture();
    }
  });

  it('hides the selection, placeholders and descendants against page rules (C4)', async () => {
    await page.evaluate(() => {
      const style = document.createElement('style');
      style.id = 'c4style';
      style.textContent = `
        ::selection { color: #000; background: #fff; }
        #ssn::selection { -webkit-text-fill-color: #000; }
        ::placeholder { -webkit-text-fill-color: #000; }
        .v { -webkit-text-fill-color: #000 !important; }`;
      document.head.append(style);
      const box = document.createElement('div');
      box.id = 'c4box';
      box.innerHTML = `<span id="c4l">Card number</span>
        <div id="c4ce" contenteditable aria-labelledby="c4l" style="display: inline-block; width: 10em"><span class="v">4111 1111</span></div>`;
      document.body.append(box);
    });
    try {
      const { cdp } = await startRun();
      const agent = await rawAgent(cdp);
      const session = await agent.attach();
      await agent.childSession(session);
      await blurAll();
      await page.focus('#ssn');
      await page.keyboard.press('ControlOrMeta+a');
      expect(
        await hiddenIn(agent, session, {
          selected: await insetBox(page.locator('#ssn')),
          placeholder: await insetBox(page.locator('#spouse')),
          child: await insetBox(page.locator('#c4ce')),
        }),
      ).toEqual({ selected: 'hidden', placeholder: 'hidden', child: 'hidden' });
      await agent.close();
    } finally {
      await blurAll();
      await page.evaluate(() => {
        document.getElementById('c4style')?.remove();
        document.getElementById('c4box')?.remove();
      });
    }
  });

  it('a registered value shown as page text refuses the screenshot; one not on the page does not (I2, P21)', async () => {
    const { cdp } = await startRun();
    // Not on the page: the screenshot goes ahead.
    expect(await addKnownValue('900-88-1357')).toEqual({ ok: true });
    const agent = await rawAgent(cdp);
    const session = await agent.attach();
    await agent.childSession(session);
    const fine = await agent.call('Page.captureScreenshot', {}, session);
    expect(fine.error).toBeUndefined();
    // The fixture's #echo shows this one.
    expect(await addKnownValue(KNOWN)).toEqual({ ok: true });
    const before = await styleAttributes();
    const shot = await agent.call('Page.captureScreenshot', {}, session);
    expect(shot.error?.message).toBe(REFUSED);
    expect(await styleAttributes()).toEqual(before);
    expect(await leftovers()).toBe(0);
    await agent.close();
  });

  it('other commands of the tab wait while a screenshot is in flight (I3)', async () => {
    const { cdp } = await startRun();
    const agent = await rawAgent(cdp);
    const session = await agent.attach();
    await agent.childSession(session);
    const order: string[] = [];
    await patchSendCommand('slowCapture');
    try {
      const shot = agent
        .call('Page.captureScreenshot', {}, session)
        .then((r) => order.push(r.error ? 'shot failed' : 'shot'));
      await new Promise((r) => setTimeout(r, 300));
      const other = agent
        .call(
          'Runtime.evaluate',
          { expression: '1', returnByValue: true },
          session,
        )
        .then(() => order.push('evaluate'));
      await Promise.all([shot, other]);
      expect(order).toEqual(['shot', 'evaluate']);
    } finally {
      await unpatchSendCommand();
    }
    await agent.close();
  });

  it('two runs on the same tab cannot leave the page masked (P16)', async () => {
    const first = await startRun();
    const a = await rawAgent(first.cdp);
    const sa = await a.attach();
    await a.childSession(sa);
    await patchSendCommand('slowCapture');
    try {
      const pending = a.call('Page.captureScreenshot', {}, sa);
      await new Promise((r) => setTimeout(r, 300));
      // Replaces the first run while its screenshot's masks are on the page.
      const second = await startRun();
      const b = await rawAgent(second.cdp);
      const sb = await b.attach();
      await b.childSession(sb);
      const shot = await b.call('Page.captureScreenshot', {}, sb);
      expect(shot.error).toBeUndefined();
      await Promise.race([pending, a.closed]);
      await b.close();
    } finally {
      await unpatchSendCommand();
      await a.close();
    }
    expect(await leftovers()).toBe(0);
  });

  it('a page whose debugger detaches after the mask restores itself (I4)', async () => {
    const { cdp } = await startRun();
    const agent = await rawAgent(cdp);
    const session = await agent.attach();
    await agent.childSession(session);
    await patchSendCommand('detachAfterMask');
    try {
      const shot = await agent.call('Page.captureScreenshot', {}, session);
      expect(shot.result).toBeUndefined();
      // Masked right after (the worker could not restore)...
      expect(await leftovers()).toBeGreaterThan(0);
    } finally {
      await unpatchSendCommand();
    }
    // ...and back within the shortened self-restore.
    await until(async () => (await leftovers()) === 0, 5_000);
    await agent.close();
  });

  it('a selected sensitive field (focused, text selected, as click + Control+a leaves it) is hidden', async () => {
    const { cdp } = await startRun();
    const agent = await rawAgent(cdp);
    const session = await agent.attach();
    await agent.childSession(session);
    await blurAll();
    await page.focus('#ssn');
    await page.keyboard.press('ControlOrMeta+a');
    try {
      expect(
        await hiddenIn(agent, session, {
          ssn: await insetBox(page.locator('#ssn')),
        }),
      ).toEqual({ ssn: 'hidden' });
    } finally {
      await blurAll();
    }
    await agent.close();
  });

  it('a screenshot sent on a child session is masked the same way and leaves no style behind', async () => {
    const { cdp } = await startRun();
    const agent = await rawAgent(cdp);
    const session = await agent.attach();
    const child = await agent.childSession(session);
    const before = await styleAttributes();
    const shot = await agent.call('Page.captureScreenshot', {}, child);
    // Chromium does not capture from an iframe target; either way the
    // bridge masked and restored every session around the attempt.
    expect(shot.error === undefined || shot.error.message !== REFUSED).toBe(
      true,
    );
    expect(await styleAttributes()).toEqual(before);
    expect(await leftovers()).toBe(0);
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
    let shot: CdpMessage;
    try {
      // The first attempt's verify sees the navigation and refuses; the
      // retry masks the new document.
      shot = await agent.call('Page.captureScreenshot', {}, session);
    } finally {
      await unpatchSendCommand();
    }
    await loaded;
    await page
      .frameLocator('#xo')
      .locator('#cc')
      .waitFor({ state: 'attached', timeout: 10_000 });
    expect(await leftovers()).toBe(0);
    expect(shot.error).toBeUndefined();
    await blurAll();
    const rect = await insetBox(page.locator('#ssn'));
    const [u] = await regions(await rawCapture(), [rect]);
    const [m] = await regions(shot.result?.data as string, [rect]);
    expect(u.nonBg).toBeGreaterThan(0);
    expect(m.nonBg).toBe(0);
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
      expect(await leftovers()).toBe(0);
    } finally {
      await closeAgentSessions();
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('other painted text showing a registered value refuses: a button value, ::before content, an img alt, <title>, a closed shadow root, a transient toast', async () => {
    const OTHER = '900-88-1357';
    const cases: Array<[string, string]> = [
      [
        'button',
        `const i = document.createElement('input'); i.type = 'button'; i.value = '${OTHER}'; i.className = 'r4'; document.body.append(i);`,
      ],
      [
        'before',
        `const s = document.createElement('style'); s.className = 'r4'; s.textContent = '.r4b::before { content: "${OTHER}"; }'; document.head.append(s); const p = document.createElement('p'); p.className = 'r4 r4b'; document.body.append(p);`,
      ],
      [
        'alt',
        `const i = document.createElement('img'); i.alt = '${OTHER}'; i.className = 'r4'; document.body.append(i);`,
      ],
      [
        'title',
        `window.r4title = document.title; document.title = '${OTHER}';`,
      ],
      [
        'closed',
        `const h = document.createElement('div'); h.className = 'r4'; h.attachShadow({ mode: 'closed' }).innerHTML = '<p>Ordinary words</p>'; document.body.append(h);`,
      ],
    ];
    const { cdp } = await startRun();
    expect(await addKnownValue(OTHER)).toEqual({ ok: true });
    const agent = await rawAgent(cdp);
    const session = await agent.attach();
    await agent.childSession(session);
    const results: Record<string, string> = {};
    try {
      for (const [name, code] of cases) {
        await page.evaluate((c) => new Function(c)(), code);
        const shot = await agent.call('Page.captureScreenshot', {}, session);
        results[name] = shot.error?.message === REFUSED ? 'refused' : 'taken';
        await page.evaluate(() => {
          for (const el of document.querySelectorAll('.r4')) el.remove();
          const w = window as unknown as { r4title?: string };
          if (w.r4title !== undefined) document.title = w.r4title;
        });
      }
      // A toast added and removed again inside the window.
      await page.evaluate((v) => {
        const target = document.getElementById('ssn') as HTMLElement;
        const watcher = new MutationObserver(() => {
          watcher.disconnect();
          const t = document.createElement('p');
          t.textContent = `Saved ${v}`;
          document.body.append(t);
          setTimeout(() => t.remove(), 0);
        });
        watcher.observe(target, {
          attributes: true,
          attributeFilter: ['style'],
        });
      }, OTHER);
      const toast = await agent.call('Page.captureScreenshot', {}, session);
      results.toast = toast.error?.message === REFUSED ? 'refused' : 'taken';
      // Nothing showing it: taken.
      const clean = await agent.call('Page.captureScreenshot', {}, session);
      results.clean = clean.error ? 'refused' : 'taken';
      expect(results).toEqual({
        button: 'refused',
        before: 'refused',
        alt: 'refused',
        title: 'refused',
        closed: 'refused',
        toast: 'refused',
        clean: 'taken',
      });
      expect(await leftovers()).toBe(0);
    } finally {
      await agent.close();
      await reloadFixture();
    }
  });

  it('ordinary text in a closed shadow root: the screenshot succeeds with no known value registered and is refused once one is', async () => {
    await page.evaluate(() => {
      const host = document.createElement('div');
      host.id = 'closedtext';
      host.attachShadow({ mode: 'closed' }).innerHTML = '<p>Ordinary words</p>';
      document.body.append(host);
    });
    try {
      const { cdp } = await startRun();
      const agent = await rawAgent(cdp);
      const session = await agent.attach();
      await agent.childSession(session);
      const before = await agent.call('Page.captureScreenshot', {}, session);
      expect(before.error).toBeUndefined();
      expect(await addKnownValue(KNOWN)).toEqual({ ok: true });
      const after = await agent.call('Page.captureScreenshot', {}, session);
      expect(after.error?.message).toBe(REFUSED);
      expect(await leftovers()).toBe(0);
      await agent.close();
    } finally {
      await page.evaluate(() =>
        document.getElementById('closedtext')?.remove(),
      );
    }
  });

  it('a registered value shown transiently in an alt attribute, or inside a listbox, refuses the screenshot (round 5)', async () => {
    const OTHER = '900-88-1357';
    await page.evaluate(() => {
      const img = document.createElement('img');
      img.id = 'r5img';
      img.alt = 'logo';
      document.body.append(img);
    });
    const { cdp } = await startRun();
    expect(await addKnownValue(OTHER)).toEqual({ ok: true });
    const agent = await rawAgent(cdp);
    const session = await agent.attach();
    await agent.childSession(session);
    try {
      // Set inside the mask window and put back at once.
      await page.evaluate((v) => {
        const target = document.getElementById('ssn') as HTMLElement;
        const img = document.getElementById('r5img') as HTMLImageElement;
        const watcher = new MutationObserver(() => {
          watcher.disconnect();
          img.alt = v;
          setTimeout(() => {
            img.alt = 'logo';
          }, 0);
        });
        watcher.observe(target, {
          attributes: true,
          attributeFilter: ['style'],
        });
      }, OTHER);
      const transient = await agent.call('Page.captureScreenshot', {}, session);
      expect(transient.error?.message).toBe(REFUSED);
      await page.evaluate((v) => {
        const box = document.createElement('div');
        box.id = 'r5list';
        box.setAttribute('role', 'listbox');
        box.setAttribute('aria-label', 'Choices');
        box.innerHTML = `<div role="option">${v}</div>`;
        document.body.append(box);
      }, OTHER);
      const listbox = await agent.call('Page.captureScreenshot', {}, session);
      expect(listbox.error?.message).toBe(REFUSED);
      expect(await leftovers()).toBe(0);
    } finally {
      await agent.close();
      await page.evaluate(() => {
        document.getElementById('r5img')?.remove();
        document.getElementById('r5list')?.remove();
      });
    }
  });

  it('a registered value shown transiently in an alt attribute, or inside a listbox, refuses the screenshot (round 5)', async () => {
    const OTHER = '900-88-1357';
    await page.evaluate(() => {
      const img = document.createElement('img');
      img.id = 'r5img';
      img.alt = 'logo';
      document.body.append(img);
    });
    const { cdp } = await startRun();
    expect(await addKnownValue(OTHER)).toEqual({ ok: true });
    const agent = await rawAgent(cdp);
    const session = await agent.attach();
    await agent.childSession(session);
    try {
      // Set inside the mask window and put back at once.
      await page.evaluate((v) => {
        const target = document.getElementById('ssn') as HTMLElement;
        const img = document.getElementById('r5img') as HTMLImageElement;
        const watcher = new MutationObserver(() => {
          watcher.disconnect();
          img.alt = v;
          setTimeout(() => {
            img.alt = 'logo';
          }, 0);
        });
        watcher.observe(target, {
          attributes: true,
          attributeFilter: ['style'],
        });
      }, OTHER);
      const transient = await agent.call('Page.captureScreenshot', {}, session);
      expect(transient.error?.message).toBe(REFUSED);
      await page.evaluate((v) => {
        const box = document.createElement('div');
        box.id = 'r5list';
        box.setAttribute('role', 'listbox');
        box.setAttribute('aria-label', 'Choices');
        box.innerHTML = `<div role="option">${v}</div>`;
        document.body.append(box);
      }, OTHER);
      const listbox = await agent.call('Page.captureScreenshot', {}, session);
      expect(listbox.error?.message).toBe(REFUSED);
      expect(await leftovers()).toBe(0);
    } finally {
      await agent.close();
      await page.evaluate(() => {
        document.getElementById('r5img')?.remove();
        document.getElementById('r5list')?.remove();
      });
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
