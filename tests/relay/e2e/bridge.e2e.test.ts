import { randomBytes } from 'node:crypto';
import { once } from 'node:events';
import type { AddressInfo } from 'node:net';
import type { Dialog, Frame, Page } from 'playwright';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import WebSocket, { WebSocketServer } from 'ws';
import {
  type AgentResult,
  agentBrowser,
  bridgeMessage,
  bridgeMessageFromPage,
  cdpUrl,
  closeAgentSessions,
  EXTENSION_ID,
  type ExtensionBrowser,
  type Fixtures,
  launchExtensionBrowser,
  mintRunTokens,
  type Relay,
  serveFixtures,
  startRelay,
} from '../helpers/harness';

// End-to-end: agent-browser -> relay -> extension -> the caseworker's tab.
// Run alone with `pnpm extension:build && pnpm test:relay e2e`.

const OUT_OF_SCOPE = 'tab is outside the allowed origin';
const FRAME_OUT_OF_SCOPE = 'frame is outside the allowed origins';

let fixtures: Fixtures;
let relay: Relay;
let browser: ExtensionBrowser;
let page: Page;
let tabId: number;

const tabIdFor = (url: string) =>
  browser.sw.evaluate(
    async (u) => (await chrome.tabs.query({ url: u }))[0]?.id,
    url,
  ) as Promise<number>;

/** Starts a run on `page`'s tab; returns the agent's CDP URL. */
async function startRun(
  opts: { allowedFrameOrigins?: string[]; tab?: number } = {},
): Promise<{ cdp: string; runId: string }> {
  const tokens = mintRunTokens(relay.secret, fixtures.origin);
  const registered = relay.logs.length;
  const started = await bridgeMessage(browser.sw, {
    type: 'startRun',
    relayUrl: relay.url,
    token: tokens.extension,
    tabId: opts.tab ?? tabId,
    allowedFrameOrigins: opts.allowedFrameOrigins ?? [fixtures.frameOrigin],
  });
  expect(started).toEqual({ ok: true });
  await until(() =>
    relay.logs
      .slice(registered)
      .some((l) => l.event === 'registered' && l.runId === tokens.runId),
  );
  return { cdp: cdpUrl(relay, tokens.agent), runId: tokens.runId };
}

/** Polls a condition that has no event to await. */
async function until(check: () => boolean | Promise<boolean>, ms = 10_000) {
  const deadline = Date.now() + ms;
  while (!(await check())) {
    if (Date.now() > deadline) throw new Error('condition not met in time');
    await new Promise((r) => setTimeout(r, 50));
  }
}

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

const crossFrame = (p: Page): Frame => {
  const frame = p
    .frames()
    .find((f) => f.url() === `${fixtures.frameOrigin}/oopif.html`);
  if (!frame) throw new Error('cross-origin frame not found');
  return frame;
};

const crossValue = (p: Page) =>
  crossFrame(p).evaluate(
    () => (document.querySelector('#x') as HTMLInputElement).value,
  );

/**
 * Adds a cross-site iframe after the agent has auto-attached (agent-browser
 * asks for waitForDebuggerOnStart) and resolves once its field is usable, so
 * a paused child frame shows up as a timeout.
 */
async function lateFrameLoads(p: Page) {
  const src = `${fixtures.frameOrigin}/oopif.html?late`;
  await p.evaluate((url) => {
    const frame = document.createElement('iframe');
    frame.id = 'late';
    frame.src = url;
    document.body.append(frame);
  }, src);
  await until(() => p.frames().some((f) => f.url() === src), 5_000);
  const frame = p.frames().find((f) => f.url() === src) as Frame;
  await frame.waitForSelector('#x', { timeout: 3_000 });
  await p.evaluate(() => document.getElementById('late')?.remove());
}

/** Whether the extension still holds chrome.debugger on a tab. */
const extensionAttached = (id: number) =>
  browser.sw.evaluate(async (tab) => {
    // getTargets' `attached` also counts Playwright's own session.
    try {
      await chrome.debugger.sendCommand({ tabId: tab }, 'Runtime.evaluate', {
        expression: '1',
      });
      return true;
    } catch {
      return false;
    }
  }, id);

/**
 * Makes a child session emit events by enabling Runtime on it straight
 * through chrome.debugger, bypassing the bridge's command checks.
 */
const stirChild = (child: string) =>
  browser.sw.evaluate(
    ({ tab, sessionId }) =>
      chrome.debugger.sendCommand({ tabId: tab, sessionId }, 'Runtime.enable'),
    { tab: tabId, sessionId: child },
  );

interface FrameTree {
  frame: { id: string; url: string };
  childFrames?: FrameTree[];
}

interface CdpMessage {
  id?: number;
  sessionId?: string;
  method?: string;
  params?: Record<string, unknown>;
  result?: Record<string, unknown>;
  error?: { message: string };
}

/** A raw CDP client on the relay's /cdp socket that records every frame. */
async function rawAgent(url: string) {
  const ws = new WebSocket(url);
  await once(ws, 'open');
  const frames: string[] = [];
  const events: CdpMessage[] = [];
  const replies = new Map<number, (m: CdpMessage) => void>();
  const watchers: Array<{
    match: (m: CdpMessage) => boolean;
    resolve: (m: CdpMessage) => void;
  }> = [];
  ws.on('message', (data) => {
    const text = data.toString();
    frames.push(text);
    const msg = JSON.parse(text) as CdpMessage;
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
  /** Auto-attaches and returns the cross-origin iframe's child session. */
  const childSession = async (sessionId: string) => {
    await call(
      'Target.setAutoAttach',
      { autoAttach: true, flatten: true, waitForDebuggerOnStart: false },
      sessionId,
    );
    // Matched by type: a disallowed child's URL is blanked.
    const event = await waitFor(
      (m) =>
        m.method === 'Target.attachedToTarget' &&
        (m.params?.targetInfo as { type?: string })?.type === 'iframe',
    );
    return {
      child: event.params?.sessionId as string,
      targetInfo: event.params?.targetInfo as { url: string; title: string },
    };
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
  return {
    call,
    waitFor,
    attach,
    childSession,
    frames,
    events,
    closed,
    close,
  };
}

beforeAll(async () => {
  fixtures = await serveFixtures();
  relay = await startRelay();
  browser = await launchExtensionBrowser();
  page = await browser.context.newPage();
  await page.goto(fixtures.formUrl);
  tabId = await tabIdFor(fixtures.formUrl);
}, 60_000);

afterAll(async () => {
  try {
    await closeAgentSessions();
    // The service-worker restart test leaves `browser.sw` dead; an evaluate
    // on it may reject or never settle, so bound it.
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

describe('bridge e2e', { timeout: 30_000 }, () => {
  it('pins the extension ID from the manifest key', () => {
    expect(new URL(browser.sw.url()).host).toBe(EXTENSION_ID);
  });

  it('refuses a non-loopback ws: relay URL and accepts ws://127.0.0.1', async () => {
    const tokens = mintRunTokens(relay.secret, fixtures.origin);
    const start = (relayUrl: string, extra: Record<string, unknown> = {}) =>
      bridgeMessage(browser.sw, {
        type: 'startRun',
        relayUrl,
        token: tokens.extension,
        tabId,
        ...extra,
      });

    const refused = await start('ws://example.com/x');
    expect(refused.ok).toBe(false);
    expect(JSON.stringify(refused)).not.toContain(tokens.extension);
    expect((await start('http://127.0.0.1:1')).ok).toBe(false);
    expect(
      (await start(relay.url, { allowedFrameOrigins: ['https://a.gov/x'] })).ok,
    ).toBe(false);
    expect(
      (await start(relay.url, { allowedFrameOrigins: ['http://a.gov'] })).ok,
    ).toBe(false);

    const other = mintRunTokens(relay.secret, 'https://elsewhere.example');
    const wrongTab = await bridgeMessage(browser.sw, {
      type: 'startRun',
      relayUrl: relay.url,
      token: other.extension,
      tabId,
    });
    expect(wrongTab).toEqual({ ok: false, error: OUT_OF_SCOPE });

    expect(await start(relay.url)).toEqual({ ok: true });
    await until(() =>
      relay.logs.some(
        (l) => l.event === 'registered' && l.runId === tokens.runId,
      ),
    );
    expect(await bridgeMessage(browser.sw, { type: 'stopRun' })).toEqual({
      ok: true,
    });
  });

  describe('driven by agent-browser', () => {
    let cdp: string;

    beforeAll(async () => {
      ({ cdp } = await startRun());
    });

    afterAll(async () => {
      await closeAgentSessions();
    });

    it('snapshot -i lists the form and both iframes', async () => {
      const snap = await agentBrowser(['snapshot', '-i'], cdp);
      expect(snap.success).toBe(true);
      const text = (snap.data as { snapshot: string }).snapshot;
      expect(text).toContain('textbox "First name"');
      expect(text).toContain('combobox "State"');
      expect(text).toContain('checkbox " I agree"');
      expect(text).toContain('Iframe "child"');
      expect(text).toContain('textbox "Child field"');
      // agent-browser 0.33.2 lists an out-of-process iframe but not its
      // fields, even on a native CDP endpoint; child sessions are covered
      // with raw CDP below.
      expect(text).toContain('Iframe "cross"');
    });

    it('fill, select, check and batch land values', async () => {
      const snap = await agentBrowser(['snapshot', '-i'], cdp);
      const first = refFor(snap, 'textbox', 'First name');
      const child = refFor(snap, 'textbox', 'Child field');
      expect((await agentBrowser(['fill', first, 'Ada'], cdp)).success).toBe(
        true,
      );
      expect(
        (
          await agentBrowser(
            ['select', refFor(snap, 'combobox', 'State'), 'NY'],
            cdp,
          )
        ).success,
      ).toBe(true);
      expect(
        (
          await agentBrowser(
            ['check', refFor(snap, 'checkbox', 'I agree')],
            cdp,
          )
        ).success,
      ).toBe(true);
      const batch = await agentBrowser(
        ['batch', '--bail', 'fill #last Lovelace', `fill ${child} Byron`],
        cdp,
      );
      expect(batch).toHaveLength(2);
      for (const step of batch as unknown as AgentResult[]) {
        expect(step.success).toBe(true);
      }

      const values = await page.evaluate(() => {
        const get = (id: string) =>
          (document.getElementById(id) as HTMLInputElement).value;
        const childDoc = (document.getElementById('child') as HTMLIFrameElement)
          .contentDocument;
        return {
          first: get('first'),
          last: get('last'),
          state: get('state'),
          agree: (document.getElementById('agree') as HTMLInputElement).checked,
          child: (childDoc?.getElementById('c') as HTMLInputElement).value,
        };
      });
      expect(values).toEqual({
        first: 'Ada',
        last: 'Lovelace',
        state: 'NY',
        agree: true,
        child: 'Byron',
      });
    });

    it('eval document.title works', async () => {
      const result = await agentBrowser(['eval', 'document.title'], cdp);
      expect(result.success).toBe(true);
      expect(JSON.stringify(result.data)).toContain('Bridge Form');
    });

    it('open to another origin is refused and the tab stays put', async () => {
      const result = await agentBrowser(
        ['open', `${fixtures.frameOrigin}/away.html`],
        cdp,
      );
      expect(result.success).toBe(false);
      expect(page.url()).toBe(fixtures.formUrl);
    });

    it('a cross-site iframe added mid-run still loads', async () => {
      await lateFrameLoads(page);
    });

    it('tab new is refused', async () => {
      const pages = browser.context.pages().length;
      const logs = relay.logs.length;
      const result = await agentBrowser(['tab', 'new'], cdp);
      expect(result.success).toBe(false);
      // Refused by the relay for scope, not by a timeout somewhere.
      expect(relay.logs.slice(logs)).toContainEqual(
        expect.objectContaining({
          event: 'command-denied',
          method: 'Target.createTarget',
          level: 'browser',
          reason: 'scope',
        }),
      );
      expect(browser.context.pages().length).toBe(pages);
    });
  });

  describe('over raw CDP', () => {
    it('refuses Network.getAllCookies', async () => {
      const { cdp } = await startRun();
      const agent = await rawAgent(cdp);
      const session = await agent.attach();
      const reply = await agent.call('Network.getAllCookies', {}, session);
      expect(reply.error?.message).toMatch(/not allowed/);
      expect(reply.result).toBeUndefined();
      await agent.close();
    });

    it('never forwards cookie, authorization or body strings', async () => {
      const { cdp } = await startRun();
      const agent = await rawAgent(cdp);
      const session = await agent.attach();
      await agent.call('Network.enable', {}, session);
      const relayFrames = relay.frames.length;

      const auth = `auth-${randomBytes(12).toString('hex')}`;
      const body = `body-${randomBytes(12).toString('hex')}`;
      await page.evaluate(
        async ({ auth, body }) => {
          const headers = { Authorization: `Bearer ${auth}` };
          await fetch('/set-cookie', { headers });
          // Sends the cookie just set, plus the header and a body.
          await fetch('/api', { method: 'POST', headers, body });
        },
        { auth, body },
      );
      const api = await agent.waitFor(
        (m) =>
          m.method === 'Network.requestWillBeSent' &&
          (m.params?.request as { url?: string })?.url ===
            `${fixtures.origin}/api`,
      );
      await agent.waitFor(
        (m) =>
          m.method === 'Network.loadingFinished' &&
          m.params?.requestId === api.params?.requestId,
      );

      const seen = [...agent.frames, ...relay.frames.slice(relayFrames)].join(
        '\n',
      );
      expect(seen).toContain(`${fixtures.origin}/api`);
      expect(seen).not.toContain(fixtures.cookieSentinel);
      expect(seen).not.toContain(auth);
      expect(seen).not.toContain(body);
      expect(
        agent.events.filter((e) => e.method?.endsWith('ExtraInfo')),
      ).toEqual([]);
      await agent.close();
    });

    it('refuses a child session outside allowedFrameOrigins', async () => {
      const { cdp } = await startRun({ allowedFrameOrigins: [] });
      const agent = await rawAgent(cdp);
      const session = await agent.attach();
      const { child, targetInfo } = await agent.childSession(session);
      // Lifecycle bookkeeping arrives, without the frame's URL.
      expect(targetInfo.url).toBe('');
      expect(targetInfo.title).toBe('');

      const fill = await agent.call(
        'Runtime.evaluate',
        { expression: 'document.querySelector("#x").value = "blocked"' },
        child,
      );
      expect(fill.error?.message).toBe(FRAME_OUT_OF_SCOPE);
      expect(await crossValue(page)).toBe('');

      // Events the child emits never reach the agent.
      await stirChild(child);

      // The top-level document is still driven normally.
      const top = await agent.call(
        'Runtime.evaluate',
        { expression: 'document.title', returnByValue: true },
        session,
      );
      expect((top.result?.result as { value: string }).value).toBe(
        'Bridge Form',
      );
      expect(
        agent.events.filter(
          (e) => e.sessionId === child && !e.method?.startsWith('Target.'),
        ),
      ).toEqual([]);
      expect(
        agent.frames.some((f) => f.includes(`${fixtures.frameOrigin}/oopif`)),
      ).toBe(false);
      await agent.close();
    });

    it('a disallowed iframe added mid-run still loads', async () => {
      const { cdp } = await startRun({ allowedFrameOrigins: [] });
      expect(
        (await agentBrowser(['snapshot', '-i'], cdp, 'late')).success,
      ).toBe(true);
      await lateFrameLoads(page);
      await closeAgentSessions();
    });

    it('drives a child session in allowedFrameOrigins', async () => {
      const { cdp } = await startRun({
        allowedFrameOrigins: [fixtures.frameOrigin],
      });
      const agent = await rawAgent(cdp);
      const session = await agent.attach();
      const { child, targetInfo } = await agent.childSession(session);
      expect(targetInfo.url).toBe(`${fixtures.frameOrigin}/oopif.html`);

      // Positive control for the disallowed case: an allowed child's events
      // are forwarded.
      await stirChild(child);
      await agent.waitFor(
        (m) =>
          m.sessionId === child &&
          m.method === 'Runtime.executionContextCreated',
      );

      const fill = await agent.call(
        'Runtime.evaluate',
        { expression: 'document.querySelector("#x").value = "allowed"' },
        child,
      );
      expect(fill.error).toBeUndefined();
      expect(await crossValue(page)).toBe('allowed');
      await crossFrame(page).evaluate(() => {
        (document.querySelector('#x') as HTMLInputElement).value = '';
      });
      await agent.close();
    });

    it('fails a child command in flight when the page removes its frame', async () => {
      const { cdp } = await startRun();
      const agent = await rawAgent(cdp);
      const session = await agent.attach();
      const { child: cross } = await agent.childSession(session);

      const src = `${fixtures.frameOrigin}/oopif.html?removed`;
      await page.evaluate((url) => {
        const frame = document.createElement('iframe');
        frame.id = 'removed';
        frame.src = url;
        document.body.append(frame);
      }, src);
      await until(() => page.frames().some((f) => f.url() === src), 5_000);
      const frame = page.frames().find((f) => f.url() === src) as Frame;
      await frame.waitForSelector('#x', { timeout: 3_000 });
      // Earlier tests can leave other iframe sessions; find this frame's by
      // asking each one where it is. A successful reply also shows the child
      // is in scope and driven, so a later error is not a refusal.
      const hrefOf = async (sessionId: string) => {
        const reply = await agent.call(
          'Runtime.evaluate',
          { expression: 'location.href', returnByValue: true },
          sessionId,
        );
        return (reply.result?.result as { value?: string })?.value;
      };
      let child = '';
      await until(async () => {
        const candidates = agent.events
          .filter(
            (m) =>
              m.method === 'Target.attachedToTarget' &&
              (m.params?.targetInfo as { type?: string })?.type === 'iframe' &&
              m.params?.sessionId !== cross,
          )
          .map((m) => m.params?.sessionId as string);
        for (const candidate of candidates) {
          if ((await hrefOf(candidate)) === src) child = candidate;
        }
        return child !== '';
      }, 5_000);

      // Never settles by itself; the flag shows Chrome is running it.
      const inFlight = agent.call(
        'Runtime.evaluate',
        {
          expression: 'window.inFlight = 1; new Promise(() => {})',
          awaitPromise: true,
        },
        child,
      );
      await frame.waitForFunction(
        () => (window as unknown as { inFlight?: number }).inFlight === 1,
        undefined,
        // Interval polling: an offscreen iframe gets no animation frames.
        { polling: 100, timeout: 3_000 },
      );
      await page.evaluate(() => document.getElementById('removed')?.remove());

      // Well under the relay's 30 s command timeout, which used to answer it.
      const reply = await Promise.race([
        inFlight,
        new Promise<'pending'>((r) => setTimeout(() => r('pending'), 5_000)),
      ]);
      expect(reply).not.toBe('pending');
      expect((reply as CdpMessage).error?.message).toBe('session detached');
      await agent.close();
    });
  });

  it('refuses raw commands in the extension itself', async () => {
    // A stand-in relay that skips every relay-side check.
    const server = new WebSocketServer({ host: '127.0.0.1', port: 0 });
    await once(server, 'listening');
    const port = (server.address() as AddressInfo).port;
    const connected = once(server, 'connection');
    const tokens = mintRunTokens(relay.secret, fixtures.origin);
    const started = await bridgeMessage(browser.sw, {
      type: 'startRun',
      relayUrl: `ws://127.0.0.1:${port}`,
      token: tokens.extension,
      tabId,
    });
    expect(started).toEqual({ ok: true });
    const [ext] = (await connected) as [WebSocket];
    const messages: Array<Record<string, unknown>> = [];
    const waiters: Array<() => void> = [];
    ext.on('message', (d) => {
      messages.push(JSON.parse(d.toString()));
      for (const w of waiters.splice(0)) w();
    });
    const next = async (match: (m: Record<string, unknown>) => boolean) => {
      for (;;) {
        const hit = messages.find(match);
        if (hit) return hit;
        await new Promise<void>((r) => waiters.push(r));
      }
    };

    expect(await next((m) => m.type === 'register')).toEqual({
      type: 'register',
      runId: tokens.runId,
      tabId,
      origin: fixtures.origin,
    });
    const ask = (cmd: Record<string, unknown>) => {
      ext.send(JSON.stringify(cmd));
      return next((m) => m.type === 'reply' && m.id === cmd.id);
    };
    const session = `tab-${tabId}`;
    const cookies = await ask({
      id: 'c1',
      type: 'cdp',
      sessionId: session,
      method: 'Network.getAllCookies',
    });
    expect(cookies.error).toBe('method not allowed: Network.getAllCookies');
    expect(cookies.result).toBeUndefined();

    const navigate = await ask({
      id: 'c2',
      type: 'cdp',
      sessionId: session,
      method: 'Page.navigate',
      params: { url: `${fixtures.frameOrigin}/away.html` },
    });
    expect(navigate.error).toMatch(/out of scope/);

    const created = await ask({
      id: 'c3',
      type: 'createTarget',
      url: `${fixtures.frameOrigin}/away.html`,
    });
    expect(created.error).toMatch(/out of scope/);
    const blank = await ask({ id: 'c4', type: 'createTarget' });
    expect(blank.error).toBeDefined();
    const closeTab = await ask({
      id: 'c5',
      type: 'closeTarget',
      targetId: 'x',
    });
    expect(closeTab.error).toBeDefined();
    expect(page.url()).toBe(fixtures.formUrl);

    await bridgeMessage(browser.sw, { type: 'stopRun' });
    server.close();
  });

  it('a fresh agent session sees the cross-origin iframe after a relay restart', async () => {
    const { cdp, runId } = await startRun();
    expect(
      (await agentBrowser(['snapshot', '-i'], cdp, 'before')).success,
    ).toBe(true);

    const { port, secret } = relay;
    await relay.stop();
    relay = await startRelay({ port, secret });
    await until(() =>
      relay.logs.some((l) => l.event === 'registered' && l.runId === runId),
    );

    const snap = await agentBrowser(['snapshot', '-i'], cdp, 'after');
    expect(snap.success).toBe(true);
    expect((snap.data as { snapshot: string }).snapshot).toContain(
      'Iframe "cross"',
    );
    // The extension re-attached, so the iframe's child session was announced
    // afresh to the new relay client.
    expect(
      relay.frames.some(
        (f) =>
          f.includes('"Target.attachedToTarget"') &&
          f.includes(`${fixtures.frameOrigin}/oopif.html`),
      ),
    ).toBe(true);
    await closeAgentSessions();
  });

  it('a page-initiated move to another origin stops the run', async () => {
    const { cdp, runId } = await startRun();
    const agent = await rawAgent(cdp);
    const session = await agent.attach();
    for (const domain of ['Page', 'Runtime', 'Network', 'DOM']) {
      await agent.call(`${domain}.enable`, {}, session);
    }
    const away = `${fixtures.frameOrigin}/away.html`;

    const started = Date.now();
    await page.click('#away');
    expect(await agent.closed).toEqual({ code: 1000, reason: 'run stopped' });
    expect(Date.now() - started).toBeLessThan(5_000);
    await page.waitForURL(away);

    expect(
      relay.logs.some(
        (l) =>
          l.event === 'run-stopped' &&
          l.runId === runId &&
          l.reason === 'left-origin',
      ),
    ).toBe(true);
    expect(await extensionAttached(tabId)).toBe(false);
    // Nothing from (or about) the off-origin document reached the agent.
    expect(agent.frames.filter((f) => f.includes(away))).toEqual([]);

    // Coming back to the origin does not revive the run.
    await page.goto(fixtures.formUrl);
    const again = await agentBrowser(['get', 'title'], cdp, 'returned');
    expect(again.success).toBe(false);
    expect(await extensionAttached(tabId)).toBe(false);
    await closeAgentSessions();
  });

  it('leaving the origin ends the run even when the agent never sent Page.enable', async () => {
    // Record every command the extension sends to chrome.debugger, so the
    // test shows that the bridge itself enabled Page (the agent never does):
    // tabs.onUpdated alone would end the run, so the log is what makes this
    // test sensitive to the bridge skipping Page.enable.
    await browser.sw.evaluate(() => {
      const w = globalThis as unknown as { __cdpLog?: string[] };
      w.__cdpLog = [];
      const dbg = chrome.debugger as unknown as {
        sendCommand: (...a: unknown[]) => unknown;
        __orig?: (...a: unknown[]) => unknown;
      };
      dbg.__orig ??= dbg.sendCommand;
      dbg.sendCommand = (...a: unknown[]) => {
        const d = a[0] as { tabId: number; sessionId?: string };
        w.__cdpLog?.push(`${d.sessionId ? 'child' : 'tab'}:${String(a[1])}`);
        return (dbg.__orig as (...x: unknown[]) => unknown)(...a);
      };
    });
    const { cdp, runId } = await startRun();
    const agent = await rawAgent(cdp);
    const session = await agent.attach();
    // Runtime and DOM only: no Page.enable, no Network.enable.
    for (const domain of ['Runtime', 'DOM']) {
      expect((await agent.call(`${domain}.enable`, {}, session)).error).toBe(
        undefined,
      );
    }
    const log = (await browser.sw.evaluate(
      () => (globalThis as unknown as { __cdpLog: string[] }).__cdpLog,
    )) as string[];
    // Exactly one Page.enable, sent before the agent's first command.
    expect(log.filter((m) => m === 'tab:Page.enable')).toHaveLength(1);
    expect(log.indexOf('tab:Page.enable')).toBeLessThan(
      log.indexOf('tab:Runtime.enable'),
    );
    await browser.sw.evaluate(() => {
      const dbg = chrome.debugger as unknown as {
        sendCommand: unknown;
        __orig?: unknown;
      };
      if (dbg.__orig) dbg.sendCommand = dbg.__orig;
    });
    const away = `${fixtures.frameOrigin}/away.html`;
    const started = Date.now();
    await page.goto(away);
    expect(await agent.closed).toEqual({ code: 1000, reason: 'run stopped' });
    expect(Date.now() - started).toBeLessThan(5_000);
    expect(
      relay.logs.some(
        (l) =>
          l.event === 'run-stopped' &&
          l.runId === runId &&
          l.reason === 'left-origin',
      ),
    ).toBe(true);
    expect(await extensionAttached(tabId)).toBe(false);
    expect(agent.frames.filter((f) => f.includes(away))).toEqual([]);
    await page.goto(fixtures.formUrl);
    await closeAgentSessions();
  });

  it('the bridge enables the Page domain: page events arrive without a Page.enable from the agent', async () => {
    const { cdp } = await startRun();
    const agent = await rawAgent(cdp);
    const session = await agent.attach();
    await agent.call('Runtime.enable', {}, session);
    await page.reload();
    const event = await agent.waitFor(
      (m) => m.method === 'Page.frameNavigated' && m.sessionId === session,
    );
    expect(event.method).toBe('Page.frameNavigated');
    await agent.close();
    await bridgeMessage(browser.sw, { type: 'stopRun' });
  });

  it('Page.disable is refused and the bridge keeps the Page domain on', async () => {
    const { cdp } = await startRun();
    const agent = await rawAgent(cdp);
    const session = await agent.attach();
    await agent.call('Runtime.enable', {}, session);
    const reply = await agent.call('Page.disable', {}, session);
    expect(reply.error?.message).toMatch(/not allowed/);
    expect(reply.result).toBeUndefined();
    // Still on: a reload reports its navigation.
    await page.reload();
    await agent.waitFor(
      (m) => m.method === 'Page.frameNavigated' && m.sessionId === session,
    );
    await agent.close();
    await bridgeMessage(browser.sw, { type: 'stopRun' });
  });

  it('history.pushState and a failed request still reach the agent', async () => {
    const { cdp } = await startRun();
    const agent = await rawAgent(cdp);
    const session = await agent.attach();
    await agent.call('Page.enable', {}, session);
    await agent.call('Network.enable', {}, session);
    await page.evaluate(() => {
      history.pushState({}, '', '/pushed');
      fetch('https://localhost:1/unreachable').catch(() => {});
    });
    await agent.waitFor((m) => m.method === 'Page.navigatedWithinDocument');
    await agent.waitFor((m) => m.method === 'Network.loadingFailed');
    await agent.close();
    await bridgeMessage(browser.sw, { type: 'stopRun' });
    await page.goto(fixtures.formUrl);
  });

  describe('dialogs without AGENT_BROWSER_NO_AUTO_DIALOG', () => {
    const open: Dialog[] = [];
    const keepOpen = (d: Dialog) => {
      open.push(d);
    };
    let savedNoAuto: string | undefined;
    beforeAll(() => {
      savedNoAuto = process.env.AGENT_BROWSER_NO_AUTO_DIALOG;
      Reflect.deleteProperty(process.env, 'AGENT_BROWSER_NO_AUTO_DIALOG');
      // Playwright dismisses dialogs nobody listens for; leave them open.
      page.on('dialog', keepOpen);
    });
    afterAll(async () => {
      page.off('dialog', keepOpen);
      // Settle what Playwright still tracks (agent-browser may have answered).
      await Promise.all(open.splice(0).map((d) => d.dismiss().catch(() => {})));
      if (savedNoAuto !== undefined) {
        process.env.AGENT_BROWSER_NO_AUTO_DIALOG = savedNoAuto;
      }
      await page.goto(fixtures.formUrl);
    });

    it('the agent hears of a dialog with an empty message', async () => {
      const { cdp } = await startRun();
      const agent = await rawAgent(cdp);
      const session = await agent.attach();
      await agent.call('Page.enable', {}, session);
      const opened = agent.waitFor(
        (m) => m.method === 'Page.javascriptDialogOpening',
      );
      await page.evaluate(() => {
        setTimeout(() => alert('dialog-sentinel-9'), 0);
      });
      expect((await opened).params).toEqual({
        type: 'alert',
        hasBrowserHandler: true, // Playwright's page listener handles it
        url: '',
        message: '',
        defaultPrompt: '',
      });
      expect(
        agent.frames.filter((f) => f.includes('dialog-sentinel-9')),
      ).toEqual([]);
      // Let the page move on: close the raw agent first, then the dialog.
      await agent.close();
      await bridgeMessage(browser.sw, { type: 'stopRun' });
      await page.reload();
    });

    it('an alert raised by an out-of-scope same-site frame is auto-accepted', async () => {
      await page.goto(`${fixtures.origin}/samesite.html`);
      const frame = page
        .frames()
        .find((f) => f.url().startsWith(fixtures.sameSiteOrigin)) as Frame;
      await frame.locator('#h').waitFor({ state: 'attached' });
      const { cdp } = await startRun({ allowedFrameOrigins: [] });
      expect(
        (await agentBrowser(['snapshot', '-i'], cdp, 'ss-dialog')).success,
      ).toBe(true);
      await frame.evaluate(() => {
        setTimeout(() => alert('ss-dialog-sentinel'), 50);
      });
      await new Promise((r) => setTimeout(r, 500));
      const next = await agentBrowser(['get', 'title'], cdp, 'ss-dialog');
      expect(next.success, JSON.stringify(next)).toBe(true);
      expect(relay.frames.some((f) => f.includes('ss-dialog-sentinel'))).toBe(
        false,
      );
      await closeAgentSessions();
      await bridgeMessage(browser.sw, { type: 'stopRun' });
      await page.goto(fixtures.formUrl);
    });

    it("a prompt raised by an out-of-scope frame never takes the agent's text, and the tab stays usable", async () => {
      await page.goto(`${fixtures.origin}/samesite.html`);
      const frame = page
        .frames()
        .find((f) => f.url().startsWith(fixtures.sameSiteOrigin)) as Frame;
      await frame.locator('#h').waitFor({ state: 'attached' });
      const { cdp } = await startRun({ allowedFrameOrigins: [] });
      expect(
        (await agentBrowser(['snapshot', '-i'], cdp, 'ss-prompt')).success,
      ).toBe(true);
      await frame.evaluate(() => {
        setTimeout(() => {
          (window as unknown as { __answer: unknown }).__answer =
            prompt('question');
        }, 50);
      });
      await new Promise((r) => setTimeout(r, 500));
      const accept = await agentBrowser(
        ['dialog', 'accept', 'hello'],
        cdp,
        'ss-prompt',
      );
      expect(JSON.stringify(accept)).not.toContain('hello');
      // The bridge dismissed it: the frame got null, not the agent's text.
      expect(
        await frame.evaluate(
          () => (window as unknown as { __answer: unknown }).__answer,
        ),
      ).toBeNull();
      const next = await agentBrowser(['get', 'title'], cdp, 'ss-prompt');
      expect(next.success, JSON.stringify(next)).toBe(true);
      await closeAgentSessions();
      await bridgeMessage(browser.sw, { type: 'stopRun' });
      await page.goto(fixtures.formUrl);
    });

    it('an alert and a beforeunload prompt are auto-accepted and nothing hangs', async () => {
      const { cdp } = await startRun();
      expect(
        (await agentBrowser(['snapshot', '-i'], cdp, 'dialogs')).success,
      ).toBe(true);
      await page.evaluate(() => {
        setTimeout(() => alert('auto-sentinel'), 50);
      });
      await new Promise((r) => setTimeout(r, 500));
      const next = await agentBrowser(['get', 'title'], cdp, 'dialogs');
      expect(next.success, JSON.stringify(next)).toBe(true);

      // beforeunload needs a user gesture on the page.
      await page.evaluate(() => {
        window.onbeforeunload = () => 'leave?';
      });
      await page.click('body');
      await page.evaluate(() => {
        setTimeout(() => {
          location.href = '/page2.html';
        }, 50);
      });
      await new Promise((r) => setTimeout(r, 800));
      const url = await agentBrowser(['get', 'url'], cdp, 'dialogs');
      expect(url.success, JSON.stringify(url)).toBe(true);
      expect(JSON.stringify(url.data)).toContain('page2.html');
      expect(
        relay.frames.some(
          (f) =>
            f.includes('"Page.javascriptDialogOpening"') &&
            f.includes('beforeunload'),
        ),
      ).toBe(true);
      expect(relay.frames.filter((f) => f.includes('auto-sentinel'))).toEqual(
        [],
      );
      await closeAgentSessions();
    });
  });

  it('an unlisted event never reaches the agent', async () => {
    const { cdp } = await startRun();
    const agent = await rawAgent(cdp);
    const session = await agent.attach();
    await agent.call('Runtime.enable', {}, session);
    await agent.call('Page.enable', {}, session);
    await page.evaluate(() => {
      console.log('console-sentinel-7');
      setTimeout(() => {
        throw new Error('exception-sentinel-8');
      }, 0);
    });
    // An allowed event after the page activity proves delivery is working.
    await page.reload();
    await agent.waitFor(
      (m) => m.method === 'Page.frameNavigated' && m.sessionId === session,
    );
    await new Promise((r) => setTimeout(r, 300));
    expect(
      agent.events.filter(
        (m) =>
          m.method === 'Runtime.consoleAPICalled' ||
          m.method === 'Runtime.exceptionThrown',
      ),
    ).toEqual([]);
    expect(
      agent.frames.filter(
        (f) =>
          f.includes('console-sentinel-7') ||
          f.includes('exception-sentinel-8'),
      ),
    ).toEqual([]);
    await agent.close();
    await bridgeMessage(browser.sw, { type: 'stopRun' });
  });

  describe('frames that share the tab session', () => {
    const HIDDEN_LABEL = 'Hidden frame field';
    const HIDDEN_VALUE = 'ss-value-7f3a9c';
    const sameSiteUrl = () => `${fixtures.origin}/samesite.html`;

    /** A CDP command straight through chrome.debugger, past the bridge. */
    const direct = (method: string, params: Record<string, unknown> = {}) =>
      browser.sw.evaluate(
        ({ tab, method, params }) =>
          chrome.debugger.sendCommand({ tabId: tab }, method, params),
        { tab: tabId, method, params },
      ) as Promise<Record<string, unknown>>;

    /** The id of the tab's frame whose URL starts with `prefix`. */
    const frameIdFor = async (prefix: string) => {
      const { frameTree } = (await direct('Page.getFrameTree')) as {
        frameTree: FrameTree;
      };
      const walk = (tree: FrameTree): string | undefined =>
        tree.frame.url.startsWith(prefix)
          ? tree.frame.id
          : (tree.childFrames ?? []).map(walk).find(Boolean);
      const id = walk(frameTree);
      if (!id) throw new Error(`no frame at ${prefix}`);
      return id;
    };

    /** A frame's main-world execution context id, read past the bridge. */
    const mainWorldOf = async (frameId: string) => {
      const owner = await direct('DOM.getFrameOwner', { frameId });
      const described = (await direct('DOM.describeNode', {
        backendNodeId: owner.backendNodeId,
      })) as { node: { contentDocument: { backendNodeId: number } } };
      const resolved = (await direct('DOM.resolveNode', {
        backendNodeId: described.node.contentDocument.backendNodeId,
      })) as { object: { objectId: string } };
      return Number(resolved.object.objectId.split('.')[1]);
    };

    const seenNothingHidden = (texts: string[]) => {
      const all = texts.join('\n');
      expect(all).not.toContain(HIDDEN_VALUE);
      expect(all).not.toContain(HIDDEN_LABEL);
    };

    beforeAll(async () => {
      await bridgeMessage(browser.sw, { type: 'stopRun' });
      await page.goto(sameSiteUrl());
      await page
        .frameLocator('#ss')
        .locator('#h')
        .waitFor({ state: 'attached' });
    });

    afterAll(async () => {
      await closeAgentSessions();
      await bridgeMessage(browser.sw, { type: 'stopRun' });
      await page.goto(fixtures.formUrl);
    });

    it('a same-site cross-origin iframe shares the tab session (no child target)', async () => {
      const targets = (await browser.sw.evaluate(async () =>
        (await chrome.debugger.getTargets()).map((t) => t.url),
      )) as string[];
      expect(targets.some((u) => u.startsWith(fixtures.sameSiteOrigin))).toBe(
        false,
      );
      expect(
        page.frames().some((f) => f.url().startsWith(fixtures.sameSiteOrigin)),
      ).toBe(true);
    });

    it('a command targeting the out-of-scope same-site frame is refused', async () => {
      const { cdp } = await startRun({ allowedFrameOrigins: [] });
      const agent = await rawAgent(cdp);
      const session = await agent.attach();
      await agent.call('Runtime.enable', {}, session);
      await agent.call('DOM.enable', {}, session);
      const ss = await frameIdFor(fixtures.sameSiteOrigin);
      const inner = await frameIdFor(`${fixtures.origin}/inner.html`);
      const ssContext = await mainWorldOf(ss);
      const innerContext = await mainWorldOf(inner);
      const refused = (reply: CdpMessage) => {
        expect(reply.error?.message).toBe(FRAME_OUT_OF_SCOPE);
        expect(reply.result).toBeUndefined();
      };

      // By context id.
      refused(
        await agent.call(
          'Runtime.evaluate',
          {
            expression: 'document.querySelector("#h").value',
            contextId: ssContext,
            returnByValue: true,
          },
          session,
        ),
      );
      const ok = await agent.call(
        'Runtime.evaluate',
        { expression: 'document.title', contextId: innerContext },
        session,
      );
      expect((ok.result?.result as { value: string }).value).toBe(
        'Inner frame',
      );

      // By frame id.
      refused(
        await agent.call(
          'Accessibility.getFullAXTree',
          { frameId: ss },
          session,
        ),
      );
      const innerTree = await agent.call(
        'Accessibility.getFullAXTree',
        { frameId: inner },
        session,
      );
      expect(JSON.stringify(innerTree.result)).toContain('Inner field');
      refused(
        await agent.call(
          'Page.navigate',
          { frameId: ss, url: `${fixtures.origin}/inner.html` },
          session,
        ),
      );

      // By object id and by (guessable) backend node id.
      const field = (await direct('Runtime.evaluate', {
        expression: 'document.querySelector("#h")',
        contextId: ssContext,
      })) as { result: { objectId: string } };
      refused(
        await agent.call(
          'Runtime.callFunctionOn',
          {
            objectId: field.result.objectId,
            functionDeclaration: 'function () { return this.value }',
            returnByValue: true,
          },
          session,
        ),
      );
      const { node } = (await direct('DOM.describeNode', {
        objectId: field.result.objectId,
      })) as { node: { backendNodeId: number } };
      for (const [method, params] of [
        ['DOM.describeNode', { backendNodeId: node.backendNodeId }],
        ['DOM.getBoxModel', { backendNodeId: node.backendNodeId }],
        ['DOM.resolveNode', { backendNodeId: node.backendNodeId }],
        // Wrapping the node in the top frame's world is refused too.
        [
          'DOM.resolveNode',
          {
            backendNodeId: node.backendNodeId,
            executionContextId: await mainWorldOf(inner),
          },
        ],
      ] as const) {
        refused(await agent.call(method, params, session));
      }
      // The iframe element itself is in the top document: describing it
      // (pierced) works but carries neither the frame id nor its document.
      // Its src attribute is the top document's own content.
      const owner = await direct('DOM.getFrameOwner', { frameId: ss });
      const described = await agent.call(
        'DOM.describeNode',
        { backendNodeId: owner.backendNodeId, depth: -1, pierce: true },
        session,
      );
      expect(described.error).toBeUndefined();
      const ownerNode = described.result?.node as {
        localName: string;
        frameId?: string;
        contentDocument?: unknown;
      };
      expect(ownerNode.localName).toBe('iframe');
      expect(ownerNode.frameId).toBeUndefined();
      expect(ownerNode.contentDocument).toBeUndefined();

      // Several handles at once: an in-scope one cannot cover another.
      const innerBody = (await direct('Runtime.evaluate', {
        expression: 'document.body',
        contextId: innerContext,
      })) as { result: { objectId: string } };
      for (const [method, params] of [
        [
          'Runtime.callFunctionOn',
          {
            objectId: innerBody.result.objectId,
            executionContextId: ssContext,
            functionDeclaration: 'function () { return 1 }',
          },
        ],
        [
          'DOM.describeNode',
          {
            objectId: innerBody.result.objectId,
            backendNodeId: node.backendNodeId,
          },
        ],
        [
          'Runtime.evaluate',
          { expression: '1', contextId: innerContext, uniqueContextId: 'x' },
        ],
      ] as const) {
        refused(await agent.call(method, params, session));
      }
      seenNothingHidden(agent.frames);
      await agent.close();
    });

    it('a context of a frame that has since moved to another origin is refused', async () => {
      const { cdp } = await startRun({ allowedFrameOrigins: [] });
      const agent = await rawAgent(cdp);
      const session = await agent.attach();
      await agent.call('Runtime.enable', {}, session);
      try {
        const inner = await frameIdFor(`${fixtures.origin}/inner.html`);
        const oldContext = await mainWorldOf(inner);
        const before = await agent.call(
          'Runtime.evaluate',
          { expression: 'document.title', contextId: oldContext },
          session,
        );
        expect(before.error).toBeUndefined();
        const moved = `${fixtures.sameSiteOrigin}/samesite-frame.html?moved`;
        await page.evaluate((url) => {
          const frame = document.getElementById('inner') as HTMLIFrameElement;
          (frame.contentWindow as Window).location.href = url;
        }, moved);
        await until(() => page.frames().some((f) => f.url() === moved), 5_000);
        const after = await agent.call(
          'Runtime.evaluate',
          { expression: 'document.title', contextId: oldContext },
          session,
        );
        // Refused by the bridge, not answered by Chrome.
        expect(after.error?.message).toBe(FRAME_OUT_OF_SCOPE);
        seenNothingHidden(agent.frames);
      } finally {
        await agent.close();
        await bridgeMessage(browser.sw, { type: 'stopRun' });
        await page.goto(sameSiteUrl());
        await page
          .frameLocator('#ss')
          .locator('#h')
          .waitFor({ state: 'attached' });
      }
    });

    it('a screenshot is refused while the out-of-scope frame is present', async () => {
      for (const [allowed, ok] of [
        [[], false],
        [[fixtures.sameSiteOrigin], true],
      ] as const) {
        const { cdp } = await startRun({ allowedFrameOrigins: [...allowed] });
        const agent = await rawAgent(cdp);
        const session = await agent.attach();
        const shot = await agent.call('Page.captureScreenshot', {}, session);
        if (ok) {
          expect(shot.error).toBeUndefined();
          expect(typeof shot.result?.data).toBe('string');
        } else {
          expect(shot.error?.message).toBe(FRAME_OUT_OF_SCOPE);
          expect(shot.result).toBeUndefined();
        }
        await agent.close();
      }
    });

    it('keys and clicks cannot reach the out-of-scope frame; in-scope input still works', async () => {
      const hidden = page.frameLocator('#ss').locator('#h');
      // Focus inside the out-of-scope frame (as Tab from the page would).
      await hidden.focus();
      const { cdp } = await startRun({ allowedFrameOrigins: [] });
      const agent = await rawAgent(cdp);
      const session = await agent.attach();
      for (const [method, params] of [
        ['Input.insertText', { text: 'typed' }],
        ['Input.dispatchKeyEvent', { type: 'keyDown', key: 'y', text: 'y' }],
      ] as const) {
        const reply = await agent.call(method, params, session);
        expect(reply.error?.message).toBe(FRAME_OUT_OF_SCOPE);
      }
      // A click at a point inside the frame.
      const box = await page.locator('#ss').boundingBox();
      const at = {
        x: Math.round((box?.x ?? 0) + 20),
        y: Math.round((box?.y ?? 0) + 20),
      };
      for (const type of ['mousePressed', 'mouseReleased']) {
        const reply = await agent.call(
          'Input.dispatchMouseEvent',
          { type, ...at, button: 'left', clickCount: 1 },
          session,
        );
        expect(reply.error?.message).toBe(FRAME_OUT_OF_SCOPE);
      }
      await agent.close();

      // Through agent-browser: `press` into the focused frame is refused;
      // typing and clicking in-scope fields on the same page work.
      const pressed = await agentBrowser(['press', 'x'], cdp, 'input');
      expect(pressed.success).toBe(false);
      const snap = await agentBrowser(['snapshot', '-i'], cdp, 'input');
      const top = refFor(snap, 'textbox', 'Top field');
      expect(
        (await agentBrowser(['fill', top, 'Grace'], cdp, 'input')).success,
      ).toBe(true);
      expect((await agentBrowser(['click', top], cdp, 'input')).success).toBe(
        true,
      );
      expect(await page.locator('#top').inputValue()).toBe('Grace');
      expect(await hidden.inputValue()).toBe(HIDDEN_VALUE);
      await closeAgentSessions();
      await page.locator('#top').fill('');
    });

    it('focus is followed through shadow roots: an out-of-scope frame in an open root, a closed root', async () => {
      const shadowSrc = `${fixtures.sameSiteOrigin}/samesite-frame.html?shadow`;
      await page.evaluate((src) => {
        const open = document.createElement('div');
        open.id = 'open-host';
        open.attachShadow({
          mode: 'open',
        }).innerHTML = `<iframe title="shadow frame" src="${src}"></iframe>`;
        const closed = document.createElement('div');
        closed.id = 'closed-host';
        const root = closed.attachShadow({ mode: 'closed' });
        root.innerHTML = '<label>Closed field <input></label>';
        (window as unknown as { __closedInput: Element }).__closedInput =
          root.querySelector('input') as Element;
        document.body.append(open, closed);
      }, shadowSrc);
      const insert = async (allowed: string[]) => {
        const { cdp } = await startRun({ allowedFrameOrigins: allowed });
        const agent = await rawAgent(cdp);
        const session = await agent.attach();
        const reply = await agent.call(
          'Input.insertText',
          { text: 'k' },
          session,
        );
        await agent.close();
        return reply.error?.message;
      };
      try {
        await until(
          () => page.frames().some((f) => f.url() === shadowSrc),
          5_000,
        );
        const shadowFrame = page
          .frames()
          .find((f) => f.url() === shadowSrc) as Frame;
        await shadowFrame.locator('#h').focus();
        expect(await insert([])).toBe(FRAME_OUT_OF_SCOPE);
        expect(await shadowFrame.locator('#h').inputValue()).toBe(HIDDEN_VALUE);

        const focusClosed = () =>
          page.evaluate(() =>
            (
              window as unknown as { __closedInput: HTMLInputElement }
            ).__closedInput.focus(),
          );
        await focusClosed();
        // Its focused element cannot be seen: refused while a frame out of
        // scope is present...
        expect(await insert([])).toBe(FRAME_OUT_OF_SCOPE);
        // ...and typed normally when nothing is out of scope.
        await focusClosed();
        expect(await insert([fixtures.sameSiteOrigin])).toBeUndefined();
        expect(
          await page.evaluate(
            () =>
              (window as unknown as { __closedInput: HTMLInputElement })
                .__closedInput.value,
          ),
        ).toBe('k');
      } finally {
        await page.evaluate(() => {
          document.getElementById('open-host')?.remove();
          document.getElementById('closed-host')?.remove();
        });
      }
    });

    it('with the frame origin allowed, input reaches it and costs no lookups', async () => {
      const hidden = page.frameLocator('#ss').locator('#h');
      // Record what the bridge itself sends to chrome.debugger.
      await browser.sw.evaluate(() => {
        const w = globalThis as unknown as { __cdpLog?: string[] };
        w.__cdpLog = [];
        const dbg = chrome.debugger as unknown as {
          sendCommand: (...a: unknown[]) => unknown;
          __orig?: (...a: unknown[]) => unknown;
        };
        dbg.__orig ??= dbg.sendCommand;
        dbg.sendCommand = (...a: unknown[]) => {
          w.__cdpLog?.push(String(a[1]));
          return (dbg.__orig as (...x: unknown[]) => unknown)(...a);
        };
      });
      try {
        const { cdp } = await startRun({
          allowedFrameOrigins: [fixtures.sameSiteOrigin],
        });
        const snap = await agentBrowser(['snapshot', '-i'], cdp, 'allowed-in');
        const field = refFor(snap, 'textbox', 'Hidden frame field');
        const top = refFor(snap, 'textbox', 'Top field');
        await browser.sw.evaluate(() => {
          (globalThis as unknown as { __cdpLog: string[] }).__cdpLog = [];
        });
        expect(
          (await agentBrowser(['click', top], cdp, 'allowed-in')).success,
        ).toBe(true);
        expect(
          (await agentBrowser(['fill', field, 'zz'], cdp, 'allowed-in'))
            .success,
        ).toBe(true);
        expect(
          (await agentBrowser(['press', 'q'], cdp, 'allowed-in')).success,
        ).toBe(true);
        expect(await hidden.inputValue()).toBe('zzq');
        const log = (await browser.sw.evaluate(
          () => (globalThis as unknown as { __cdpLog: string[] }).__cdpLog,
        )) as string[];
        expect(
          log.filter(
            (m) =>
              m === 'DOM.getNodeForLocation' ||
              m === 'Page.createIsolatedWorld',
          ),
        ).toEqual([]);
        await closeAgentSessions();
      } finally {
        await browser.sw.evaluate(() => {
          const dbg = chrome.debugger as unknown as {
            sendCommand: unknown;
            __orig?: unknown;
          };
          if (dbg.__orig) dbg.sendCommand = dbg.__orig;
        });
        await hidden.fill(HIDDEN_VALUE);
      }
    });

    it('its fields do not appear in a snapshot, nor its URL in the frame tree', async () => {
      const mark = relay.frames.length;
      const { cdp } = await startRun({ allowedFrameOrigins: [] });
      const agent = await rawAgent(cdp);
      const session = await agent.attach();
      const tree = await agent.call('Page.getFrameTree', {}, session);
      const treeText = JSON.stringify(tree.result);
      expect(treeText).toContain(`${fixtures.origin}/inner.html`);
      expect(treeText).toContain('about:srcdoc');
      expect(treeText).not.toContain(fixtures.sameSiteOrigin);
      await agent.close();

      const snap = await agentBrowser(['snapshot', '-i'], cdp, 'samesite');
      expect(snap.success).toBe(true);
      const text = (snap.data as { snapshot: string }).snapshot;
      expect(text).toContain('textbox "Top field"');
      expect(text).toContain('textbox "Srcdoc field"');
      expect(text).toContain('textbox "Inner field"');
      expect(text).toContain('Iframe "samesite frame"');
      seenNothingHidden([text, ...relay.frames.slice(mark)]);
      await closeAgentSessions();
    });

    it('the frame is visible once its origin is allowed', async () => {
      const { cdp } = await startRun({
        allowedFrameOrigins: [fixtures.sameSiteOrigin],
      });
      const snap = await agentBrowser(['snapshot', '-i'], cdp, 'allowed');
      expect(snap.success).toBe(true);
      expect((snap.data as { snapshot: string }).snapshot).toContain(
        HIDDEN_LABEL,
      );
      await closeAgentSessions();
    });

    it('an unknown execution context is refused', async () => {
      const { cdp } = await startRun({ allowedFrameOrigins: [] });
      const agent = await rawAgent(cdp);
      const session = await agent.attach();
      // No Runtime.enable: the bridge looks contexts up itself, so a real
      // frame's context still works...
      const inner = await frameIdFor(`${fixtures.origin}/inner.html`);
      const known = await agent.call(
        'Runtime.evaluate',
        { expression: 'document.title', contextId: await mainWorldOf(inner) },
        session,
      );
      expect(known.error).toBeUndefined();
      // ...and anything it cannot place is refused.
      await agent.call('Runtime.enable', {}, session);
      for (const [method, params] of [
        ['Runtime.evaluate', { expression: '1', contextId: 987_654 }],
        ['Runtime.evaluate', { expression: '1', uniqueContextId: 'nope' }],
        [
          'Runtime.callFunctionOn',
          { objectId: '1.987654.1', functionDeclaration: 'function(){}' },
        ],
        [
          'Runtime.callFunctionOn',
          { objectId: 'not-an-id', functionDeclaration: 'function(){}' },
        ],
        [
          'Runtime.callFunctionOn',
          {
            executionContextId: '3',
            functionDeclaration: 'function(){}',
          },
        ],
        ['Accessibility.getFullAXTree', { frameId: 'NOPE' }],
      ] as const) {
        const reply = await agent.call(method, params, session);
        expect(reply.error?.message).toBe(FRAME_OUT_OF_SCOPE);
      }
      await agent.close();
    });

    it('events for the out-of-scope frame never reach the agent', async () => {
      const { cdp } = await startRun({ allowedFrameOrigins: [] });
      const agent = await rawAgent(cdp);
      const session = await agent.attach();
      for (const domain of ['Page', 'Runtime', 'Network']) {
        await agent.call(`${domain}.enable`, {}, session);
      }
      const mark = agent.frames.length;
      const lateSs = `${fixtures.sameSiteOrigin}/samesite-frame.html?late`;
      const lateInner = `${fixtures.origin}/inner.html?late`;
      // No src attribute, so the parent's DOM never names the URL; the
      // frames are navigated from script instead.
      await page.evaluate(
        ([ss, inner]) => {
          for (const [id, url] of [
            ['late-ss', ss],
            ['late-inner', inner],
          ]) {
            const frame = document.createElement('iframe');
            frame.id = id;
            document.body.append(frame);
            (frame.contentWindow as Window).location.href = url;
          }
        },
        [lateSs, lateInner],
      );
      try {
        await until(() => page.frames().some((f) => f.url() === lateSs), 5_000);
        await (page.frames().find((f) => f.url() === lateSs) as Frame)
          .locator('#h')
          .waitFor({ state: 'attached' });
        // Positive control: the in-scope frame's commit reaches the agent.
        await agent.waitFor(
          (m) =>
            m.method === 'Page.frameNavigated' &&
            (m.params?.frame as { url?: string })?.url === lateInner,
        );
        // A round trip after the out-of-scope frame loaded flushes its events.
        await page.waitForTimeout(300);
        await agent.call('Runtime.evaluate', { expression: '1' }, session);
        const seen = agent.frames.slice(mark);
        expect(seen.filter((f) => f.includes(fixtures.sameSiteOrigin))).toEqual(
          [],
        );
        seenNothingHidden(seen);
      } finally {
        await page.evaluate(() => {
          document.getElementById('late-ss')?.remove();
          document.getElementById('late-inner')?.remove();
        });
        await agent.close();
      }
    });

    it('same-origin srcdoc and same-origin iframes still work', async () => {
      const { cdp } = await startRun({ allowedFrameOrigins: [] });
      const snap = await agentBrowser(['snapshot', '-i'], cdp, 'fill');
      for (const [name, value] of [
        ['Srcdoc field', 'Ada'],
        ['Inner field', 'Byron'],
      ]) {
        const filled = await agentBrowser(
          ['fill', refFor(snap, 'textbox', name), value],
          cdp,
          'fill',
        );
        expect(filled.success).toBe(true);
      }
      const values = await Promise.all(
        [
          ['#srcdoc', '#s'],
          ['#inner', '#i'],
        ].map(([frame, field]) =>
          page.frameLocator(frame).locator(field).inputValue(),
        ),
      );
      expect(values).toEqual(['Ada', 'Byron']);
      await closeAgentSessions();
    });

    it('leaving for a same-site origin still ends the run', async () => {
      const { cdp, runId } = await startRun({ allowedFrameOrigins: [] });
      const agent = await rawAgent(cdp);
      await agent.attach();
      await page.goto(`${fixtures.sameSiteOrigin}/samesite-frame.html`);
      expect(await agent.closed).toEqual({ code: 1000, reason: 'run stopped' });
      expect(
        relay.logs.some(
          (l) =>
            l.event === 'run-stopped' &&
            l.runId === runId &&
            l.reason === 'left-origin',
        ),
      ).toBe(true);
      seenNothingHidden(agent.frames);
      await page.goto(sameSiteUrl());
    });
  });

  it('a same-site frame inside an allowed out-of-process frame is scoped on the child session', async () => {
    await page.goto(fixtures.formUrl);
    const { cdp } = await startRun({
      allowedFrameOrigins: [fixtures.frameOrigin],
    });
    const agent = await rawAgent(cdp);
    const session = await agent.attach();
    const nesting = `${fixtures.frameOrigin}/oopif-nested.html`;
    const nestedOrigin = fixtures.sameSiteOrigin.replace(
      'localhost',
      '127.0.0.1',
    );
    await agent.call(
      'Target.setAutoAttach',
      { autoAttach: true, flatten: true, waitForDebuggerOnStart: false },
      session,
    );
    try {
      await page.evaluate((url) => {
        const frame = document.createElement('iframe');
        frame.id = 'nesting';
        frame.src = url;
        document.body.append(frame);
      }, nesting);
      await until(
        () => page.frames().some((f) => f.url().startsWith(nestedOrigin)),
        5_000,
      );
      const nestedFrame = page
        .frames()
        .find((f) => f.url().startsWith(nestedOrigin));
      await nestedFrame?.locator('#h').waitFor({ state: 'attached' });
      // Find this frame's child session by asking each where it is.
      let child = '';
      await until(async () => {
        const sessions = agent.events
          .filter(
            (m) =>
              m.method === 'Target.attachedToTarget' &&
              (m.params?.targetInfo as { type?: string })?.type === 'iframe',
          )
          .map((m) => m.params?.sessionId as string);
        for (const candidate of sessions) {
          const reply = await agent.call(
            'Runtime.evaluate',
            { expression: 'location.href', returnByValue: true },
            candidate,
          );
          if ((reply.result?.result as { value?: string })?.value === nesting) {
            child = candidate;
          }
        }
        return child !== '';
      }, 5_000);
      await agent.call('Runtime.enable', {}, child);

      // Past the bridge: the nested frame's id, context and field node, all
      // in the child's renderer.
      const directChild = (method: string, params: Record<string, unknown>) =>
        browser.sw.evaluate(
          ({ tab, sessionId, method, params }) =>
            chrome.debugger.sendCommand(
              { tabId: tab, sessionId },
              method,
              params,
            ),
          { tab: tabId, sessionId: child, method, params },
        ) as Promise<Record<string, unknown>>;
      const { frameTree } = (await directChild('Page.getFrameTree', {})) as {
        frameTree: FrameTree;
      };
      const nestedId = frameTree.childFrames?.find((c) =>
        c.frame.url.startsWith(nestedOrigin),
      )?.frame.id as string;
      expect(nestedId).toBeDefined();
      const owner = await directChild('DOM.getFrameOwner', {
        frameId: nestedId,
      });
      const described = (await directChild('DOM.describeNode', {
        backendNodeId: owner.backendNodeId,
      })) as { node: { contentDocument: { backendNodeId: number } } };
      const doc = (await directChild('DOM.resolveNode', {
        backendNodeId: described.node.contentDocument.backendNodeId,
      })) as { object: { objectId: string } };
      const nestedContext = Number(doc.object.objectId.split('.')[1]);
      const field = (await directChild('Runtime.evaluate', {
        expression: 'document.querySelector("#h")',
        contextId: nestedContext,
      })) as { result: { objectId: string } };
      const { node } = (await directChild('DOM.describeNode', {
        objectId: field.result.objectId,
      })) as { node: { backendNodeId: number } };

      // The child's own document still works.
      const own = await agent.call(
        'Runtime.evaluate',
        { expression: 'document.title', returnByValue: true },
        child,
      );
      expect((own.result?.result as { value: string }).value).toBe(
        'Nesting frame',
      );
      for (const [method, params] of [
        [
          'Runtime.evaluate',
          { expression: 'document.title', contextId: nestedContext },
        ],
        ['Accessibility.getFullAXTree', { frameId: nestedId }],
        ['DOM.describeNode', { backendNodeId: node.backendNodeId }],
        [
          'Runtime.callFunctionOn',
          {
            objectId: field.result.objectId,
            functionDeclaration: 'function () { return this.value }',
          },
        ],
      ] as const) {
        const reply = await agent.call(method, params, child);
        expect(reply.error?.message).toBe(FRAME_OUT_OF_SCOPE);
      }
      const tree = await agent.call('Page.getFrameTree', {}, child);
      expect(tree.error).toBeUndefined();
      expect(JSON.stringify(tree.result)).not.toContain(nestedOrigin);
      const all = agent.frames.join('\n');
      expect(all).not.toContain('ss-value-7f3a9c');
      expect(all).not.toContain('Hidden frame field');
      expect(all).not.toContain(nestedOrigin);
    } finally {
      await page.evaluate(() => document.getElementById('nesting')?.remove());
      await agent.close();
      await bridgeMessage(browser.sw, { type: 'stopRun' });
    }
  });

  it('input cannot reach a disallowed cross-site iframe; with its origin allowed it can', async () => {
    await page.goto(fixtures.formUrl);
    const box = await page.locator('#xo').boundingBox();
    const at = {
      x: Math.round((box?.x ?? 0) + 30),
      y: Math.round((box?.y ?? 0) + 20),
    };
    const drive = async (allowed: string[]) => {
      const { cdp } = await startRun({ allowedFrameOrigins: allowed });
      const agent = await rawAgent(cdp);
      const session = await agent.attach();
      await agent.childSession(session);
      const clicks = [];
      for (const type of ['mousePressed', 'mouseReleased']) {
        clicks.push(
          (
            await agent.call(
              'Input.dispatchMouseEvent',
              { type, ...at, button: 'left', clickCount: 1 },
              session,
            )
          ).error?.message,
        );
      }
      await crossFrame(page).locator('#x').focus();
      const typed = (
        await agent.call('Input.insertText', { text: 'typed' }, session)
      ).error?.message;
      await agent.close();
      await bridgeMessage(browser.sw, { type: 'stopRun' });
      return { clicks, typed };
    };
    try {
      expect(await drive([])).toEqual({
        clicks: [FRAME_OUT_OF_SCOPE, FRAME_OUT_OF_SCOPE],
        typed: FRAME_OUT_OF_SCOPE,
      });
      expect(await crossValue(page)).toBe('');
      expect(await drive([fixtures.frameOrigin])).toEqual({
        clicks: [undefined, undefined],
        typed: undefined,
      });
      expect(await crossValue(page)).toBe('typed');
    } finally {
      await crossFrame(page).evaluate(() => {
        (document.querySelector('#x') as HTMLInputElement).value = '';
      });
    }
  });

  it('a child session whose frame navigates cross-origin stops being allowed within the same event loop turn', async () => {
    const { cdp } = await startRun({
      allowedFrameOrigins: [fixtures.frameOrigin],
    });
    const agent = await rawAgent(cdp);
    const session = await agent.attach();
    const { child } = await agent.childSession(session);
    await agent.call('Page.enable', {}, child);
    const before = await agent.call(
      'Runtime.evaluate',
      { expression: 'location.href', returnByValue: true },
      child,
    );
    expect(before.error).toBeUndefined();
    const mark = agent.frames.length;
    // Same site as the frame (127.0.0.1), another origin: Chrome keeps the
    // frame in its process, so the child session survives the move.
    const away = `${fixtures.sameSiteOrigin.replace('localhost', '127.0.0.1')}/elsewhere?child-nav`;
    await crossFrame(page).evaluate((url) => {
      location.href = url;
    }, away);
    await until(() => page.frames().some((f) => f.url() === away), 5_000);
    const after = await agent.call(
      'Runtime.evaluate',
      { expression: 'location.href', returnByValue: true },
      child,
    );
    expect(after.error?.message).toBe(FRAME_OUT_OF_SCOPE);
    // The child's own frameNavigated (which names the new URL) was dropped.
    expect(agent.frames.slice(mark).filter((f) => f.includes(away))).toEqual(
      [],
    );
    expect(
      agent.events
        .slice()
        .filter(
          (e) =>
            e.sessionId === child &&
            e.method === 'Page.frameNavigated' &&
            JSON.stringify(e.params).includes('elsewhere'),
        ),
    ).toEqual([]);
    await agent.close();
    await bridgeMessage(browser.sw, { type: 'stopRun' });
    await page.goto(fixtures.formUrl);
  });

  describe('in-origin page activity keeps the run alive', () => {
    let cdp: string;
    let runId: string;

    beforeAll(async () => {
      ({ cdp, runId } = await startRun());
      expect(
        (await agentBrowser(['snapshot', '-i'], cdp, 'alive')).success,
      ).toBe(true);
    });

    afterAll(async () => {
      await closeAgentSessions();
      if (page.url() !== fixtures.formUrl) await page.goto(fixtures.formUrl);
    });

    /** The run was not stopped and the agent can still run a command. */
    const expectAlive = async (title: string) => {
      const result = await agentBrowser(['get', 'title'], cdp, 'alive');
      expect(result.success, JSON.stringify(result)).toBe(true);
      expect(JSON.stringify(result.data)).toContain(title);
      expect(
        relay.logs.some((l) => l.event === 'run-stopped' && l.runId === runId),
      ).toBe(false);
      expect(await extensionAttached(tabId)).toBe(true);
    };

    it('javascript: links and assignments', async () => {
      await page.click('#js-void');
      await page.click('#js-set');
      await page.waitForFunction(
        () => (window as unknown as { __x?: number }).__x === 1,
      );
      await page.evaluate(() => {
        location.href = 'javascript:window.__y=2';
      });
      await page.waitForFunction(
        () => (window as unknown as { __y?: number }).__y === 2,
      );
      await expectAlive('Bridge Form');
    });

    it('history.pushState, replaceState and a hash change', async () => {
      await page.evaluate(() => {
        history.pushState({}, '', '/form.html?step=2');
        history.replaceState({}, '', '/form.html?step=3');
        location.hash = '#part-2';
      });
      await page.waitForURL(`${fixtures.formUrl}?step=3#part-2`);
      await expectAlive('Bridge Form');
    });

    it('a reload', async () => {
      await page.reload();
      await expectAlive('Bridge Form');
    });

    it('a same-origin link to another page', async () => {
      await page.click('#next');
      await page.waitForURL(`${fixtures.origin}/page2.html`);
      await expectAlive('Second Page');
      await page.goto(fixtures.formUrl);
      await expectAlive('Bridge Form');
    });

    it('a same-origin form POST', async () => {
      // Submitted by page script, as the caseworker's page would; no submit
      // control is clicked.
      await page.evaluate(() =>
        (document.getElementById('post') as HTMLFormElement).submit(),
      );
      await page.waitForURL(`${fixtures.origin}/submitted`);
      await expectAlive('Submitted');
      await page.goto(fixtures.formUrl);
      await expectAlive('Bridge Form');
    });
  });

  it('stopRun closes the agent at once and detaches', async () => {
    const { cdp } = await startRun();
    expect((await agentBrowser(['snapshot', '-i'], cdp, 'stop')).success).toBe(
      true,
    );
    expect(await extensionAttached(tabId)).toBe(true);
    const started = Date.now();
    expect(await bridgeMessage(browser.sw, { type: 'stopRun' })).toEqual({
      ok: true,
    });
    const next = await agentBrowser(['get', 'title'], cdp, 'stop');
    expect(next.success).toBe(false);
    expect(Date.now() - started).toBeLessThan(5_000);
    expect(await extensionAttached(tabId)).toBe(false);
    await closeAgentSessions();
  });

  it('dismissing the debugging infobar ends the run for good', async () => {
    const { cdp, runId } = await startRun();
    expect(
      (await agentBrowser(['snapshot', '-i'], cdp, 'infobar')).success,
    ).toBe(true);
    // Headless Chromium has no infobar to click; deliver the detach event the
    // extension gets when the user cancels debugging.
    await browser.sw.evaluate(
      (id) =>
        (
          globalThis as unknown as {
            bridgeOnDebuggerDetach: (s: { tabId: number }, r: string) => void;
          }
        ).bridgeOnDebuggerDetach({ tabId: id }, 'canceled_by_user'),
      tabId,
    );
    await until(() =>
      relay.logs.some(
        (l) =>
          l.event === 'run-stopped' &&
          l.runId === runId &&
          l.reason === 'debugger-canceled',
      ),
    );
    // The relay cannot re-attach: a fresh agent session is refused.
    const started = Date.now();
    const again = await agentBrowser(['snapshot', '-i'], cdp, 'infobar-2');
    expect(again.success).toBe(false);
    expect(Date.now() - started).toBeLessThan(5_000);
    expect(await extensionAttached(tabId)).toBe(false);
    await closeAgentSessions();
  });

  it('getRunStatus reports the run by id and counts, never the token or URL', async () => {
    const { cdp, runId } = await startRun();
    expect(
      (await agentBrowser(['snapshot', '-i'], cdp, 'status')).success,
    ).toBe(true);
    const status = await bridgeMessage(browser.sw, { type: 'getRunStatus' });
    // Exactly these keys: no token, no relay URL (this build has no Jev).
    // The tab id lets a reopened panel refresh the run's token.
    expect(status).toEqual({
      ok: true,
      active: true,
      runId,
      origin: fixtures.origin,
      tabId: expect.any(Number),
      attached: true,
      needsRegistration: false,
    });
    expect(JSON.stringify(status)).not.toContain(relay.url);
    expect(
      await bridgeMessage(browser.sw, {
        type: 'registrationComplete',
        runId,
      }),
    ).toEqual({ ok: true });
    await bridgeMessage(browser.sw, { type: 'stopRun' });
    expect(await bridgeMessage(browser.sw, { type: 'getRunStatus' })).toEqual({
      ok: true,
      active: false,
    });
    await closeAgentSessions();
  });

  it('two concurrent startRun calls leave one run, the later one', async () => {
    const a = mintRunTokens(relay.secret, fixtures.origin);
    const b = mintRunTokens(relay.secret, fixtures.origin);
    const start = (token: string) =>
      bridgeMessage(browser.sw, {
        type: 'startRun',
        relayUrl: relay.url,
        token,
        tabId,
        allowedFrameOrigins: [fixtures.frameOrigin],
      });
    expect(await Promise.all([start(a.extension), start(b.extension)])).toEqual(
      [{ ok: true }, { ok: true }],
    );
    expect(
      await bridgeMessage(browser.sw, { type: 'getRunStatus' }),
    ).toMatchObject({ active: true, runId: b.runId });
    await until(() =>
      relay.logs.some((l) => l.event === 'registered' && l.runId === b.runId),
    );
    const cdp = cdpUrl(relay, b.agent);
    expect(
      (await agentBrowser(['snapshot', '-i'], cdp, 'double')).success,
    ).toBe(true);
    await bridgeMessage(browser.sw, { type: 'stopRun' });
    await closeAgentSessions();
  });

  it('a detach that is not re-attached within 5 s ends the run (debugger-detached)', async () => {
    const { cdp, runId } = await startRun();
    expect(
      (await agentBrowser(['snapshot', '-i'], cdp, 'detach')).success,
    ).toBe(true);
    await browser.sw.evaluate(
      (id) =>
        (
          globalThis as unknown as {
            bridgeOnDebuggerDetach: (s: { tabId: number }, r: string) => void;
          }
        ).bridgeOnDebuggerDetach({ tabId: id }, 'target_closed'),
      tabId,
    );
    await until(
      () =>
        relay.logs.some(
          (l) =>
            l.event === 'run-stopped' &&
            l.runId === runId &&
            l.reason === 'debugger-detached',
        ),
      10_000,
    );
    expect(await bridgeMessage(browser.sw, { type: 'getRunStatus' })).toEqual({
      ok: true,
      active: false,
    });
    expect(await extensionAttached(tabId)).toBe(false);
    await closeAgentSessions();
  });

  it('closing the tab fails in-flight and later commands quickly', async () => {
    const doomed = await browser.context.newPage();
    const url = `${fixtures.formUrl}?doomed`;
    await doomed.goto(url);
    const { cdp } = await startRun({ tab: await tabIdFor(url) });
    expect(
      (await agentBrowser(['snapshot', '-i'], cdp, 'doomed')).success,
    ).toBe(true);

    const started = Date.now();
    const inFlight = agentBrowser(
      ['eval', 'new Promise((r) => setTimeout(r, 25000))'],
      cdp,
      'doomed',
    );
    await until(() =>
      relay.frames.some((f) => f.includes('setTimeout(r, 25000)')),
    );
    await doomed.close();

    expect((await inFlight).success).toBe(false);
    const following = await agentBrowser(['get', 'title'], cdp, 'doomed');
    expect(following.success).toBe(false);
    expect(Date.now() - started).toBeLessThan(5_000);
  });

  // Review Focus 6. Kept last: Playwright reports no handle for the restarted
  // worker, so `browser.sw` is dead afterwards.
  it('recovers a run after the service worker restarts, with a fresh agent session', async () => {
    await page.goto(fixtures.formUrl);
    tabId = await tabIdFor(fixtures.formUrl);
    const { cdp, runId } = await startRun();

    // The first agent connection fills a field.
    const first = await rawAgent(cdp);
    const session = await first.attach();
    await first.call(
      'Runtime.evaluate',
      { expression: "document.getElementById('first').focus()" },
      session,
    );
    expect(
      (await first.call('Input.insertText', { text: 'Grace' }, session)).error,
    ).toBeUndefined();
    await expect(page.locator('#first').inputValue()).resolves.toBe('Grace');

    // Stop the worker for real, then wake it with a tab event. Only a fresh
    // worker sends `resumed`: it comes from the run rebuilt out of storage.
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

    // The old connection is cut, not the run.
    expect(await first.closed).toEqual({
      code: 1000,
      reason: 'extension restarted',
    });
    expect(
      relay.logs.some((l) => l.event === 'run-closed' && l.runId === runId),
    ).toBe(false);

    // The rebuilt run lost its known values: reads are refused until the
    // panel registers them again (fail closed).
    const refused = await agentBrowser(['snapshot', '-i'], cdp, 'unregistered');
    expect(refused.success).toBe(false);
    expect(JSON.stringify(refused)).toContain('registration required');
    await closeAgentSessions();
    expect(
      await bridgeMessageFromPage(browser.context, {
        type: 'registrationComplete',
        runId,
        count: 0,
      }),
    ).toEqual({ ok: true });

    // A new agent session snapshots and fills a second field.
    const snap = await agentBrowser(['snapshot', '-i'], cdp, 'restarted');
    expect(snap.success).toBe(true);
    const last = refFor(snap, 'textbox', 'Last name');
    expect(
      (await agentBrowser(['fill', last, 'Hopper'], cdp, 'restarted')).success,
    ).toBe(true);
    await expect(page.locator('#last').inputValue()).resolves.toBe('Hopper');
    await expect(page.locator('#first').inputValue()).resolves.toBe('Grace');

    // Restart again: a ref from the snapshot taken before it is not reused.
    // The daemon re-dials on its next command but has dropped its refs.
    const firstRef = refFor(snap, 'textbox', 'First name');
    const count = (event: string) =>
      relay.logs.filter((l) => l.event === event && l.runId === runId).length;
    const restarts = () => count('extension-restarted');
    const detachedBefore = count('extension-detached');
    await devtools.send('ServiceWorker.stopAllWorkers');
    await until(() => count('extension-detached') > detachedBefore);
    const waker2 = await browser.context.newPage();
    await waker2.goto('about:blank');
    await until(() => restarts() === 2);
    const stale = await agentBrowser(
      ['fill', firstRef, 'Stale'],
      cdp,
      'restarted',
    );
    expect(stale.success).toBe(false);
    // The measured agent-browser 0.33.2 wording: the re-dialed daemon has
    // dropped its refs, rather than failing for some other reason.
    expect(stale.error).toMatch(/Unknown ref/);
    await expect(page.locator('#first').inputValue()).resolves.toBe('Grace');
    await waker2.close();

    await closeAgentSessions();
    await devtools.detach();
    await waker.close();
  });
});
