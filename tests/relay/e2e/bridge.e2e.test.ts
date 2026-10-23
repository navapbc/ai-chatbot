import { randomBytes } from 'node:crypto';
import { once } from 'node:events';
import type { AddressInfo } from 'node:net';
import type { Frame, Page } from 'playwright';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import WebSocket, { WebSocketServer } from 'ws';
import {
  type AgentResult,
  agentBrowser,
  bridgeMessage,
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
  await closeAgentSessions();
  await bridgeMessage(browser.sw, { type: 'stopRun' }).catch(() => {});
  await browser?.stop();
  await relay?.stop();
  await fixtures?.stop();
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
      const result = await agentBrowser(['tab', 'new'], cdp);
      expect(result.success).toBe(false);
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
});
