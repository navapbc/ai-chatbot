// Throwaway spike: forwards CDP between a relay WebSocket and chrome.debugger.
importScripts('config.js');
const RELAY = `${self.RELAY_URL}?token=${self.RELAY_TOKEN}`;
const childOwner = new Map(); // child sessionId -> tabId
let ws;

const badge = (text, color) => {
  chrome.action.setBadgeText({ text });
  chrome.action.setBadgeBackgroundColor({ color });
};

const send = (msg) => ws?.readyState === 1 && ws.send(JSON.stringify(msg));

const connect = () => {
  ws = new WebSocket(RELAY);
  ws.onopen = () => {
    console.log('relay connected');
    badge('ON', '#16a34a');
  };
  ws.onmessage = async (e) => {
    const msg = JSON.parse(e.data);
    try {
      const result = await handle(msg);
      send({ kind: 'reply', id: msg.id, result });
    } catch (err) {
      send({ kind: 'reply', id: msg.id, error: String(err?.message ?? err) });
    }
  };
  ws.onclose = () => {
    badge('OFF', '#dc2626');
    setTimeout(connect, 1000);
  };
};

const targetInfo = (t) => ({
  targetId: t.id,
  type: t.type,
  title: t.title,
  url: t.url,
  attached: t.attached,
  canAccessOpener: false,
});

// Only tabs on TAB_ORIGIN's exact origin are ever exposed.
// Fails closed: with no TAB_ORIGIN configured, nothing is in scope.
const inScope = (url) =>
  Boolean(self.TAB_ORIGIN) && new URL(url, 'http://x').origin === self.TAB_ORIGIN;

const pages = async () =>
  (await chrome.debugger.getTargets()).filter(
    (t) => t.type === 'page' && t.tabId && inScope(t.url),
  );

const handle = async (msg) => {
  switch (msg.type) {
    case 'listTargets':
      return { targetInfos: (await pages()).map(targetInfo) };
    case 'attach': {
      const t = (await pages()).find((p) => p.id === msg.targetId);
      if (!t) throw new Error(`no such target ${msg.targetId}`);
      // Always start from a clean attachment so a new relay client sees every
      // child (cross-origin iframe) session announced afresh.
      try { await chrome.debugger.detach({ tabId: t.tabId }); } catch {}
      await chrome.debugger.attach({ tabId: t.tabId }, '1.3');
      return { sessionId: `tab-${t.tabId}`, targetInfo: targetInfo(t) };
    }
    case 'createTarget': {
      if (!inScope(msg.url || 'about:blank')) {
        throw new Error('new tab outside the allowed origin is blocked');
      }
      const tab = await chrome.tabs.create({ url: msg.url || 'about:blank', active: false });
      await new Promise((r) => setTimeout(r, 300));
      const t = (await chrome.debugger.getTargets()).find((x) => x.tabId === tab.id);
      return { targetId: t.id };
    }
    case 'closeTarget': {
      const t = (await chrome.debugger.getTargets()).find((x) => x.id === msg.targetId);
      if (t?.tabId && inScope(t.url)) await chrome.tabs.remove(t.tabId);
      return { success: true };
    }
    case 'cdp': {
      const isTab = msg.sessionId.startsWith('tab-');
      const tabId = isTab ? Number(msg.sessionId.slice(4)) : childOwner.get(msg.sessionId);
      const debuggee = isTab ? { tabId } : { tabId, sessionId: msg.sessionId };
      // Enforce scope on every command against the tab's *current* URL, so a
      // tab that navigated off-origin (link, redirect, Page.navigate) is cut off.
      const cur = (await chrome.debugger.getTargets()).find((t) => t.tabId === tabId);
      if (!cur || !inScope(cur.url)) {
        try { await chrome.debugger.detach({ tabId }); } catch {}
        throw new Error('tab is outside the allowed origin');
      }
      if (msg.method === 'Page.navigate' && !inScope(msg.params?.url ?? '')) {
        throw new Error('navigation outside the allowed origin is blocked');
      }
      return await chrome.debugger.sendCommand(debuggee, msg.method, msg.params ?? {});
    }
    default:
      throw new Error(`unknown type ${msg.type}`);
  }
};

chrome.debugger.onEvent.addListener((source, method, params) => {
  if (method === 'Target.attachedToTarget') childOwner.set(params.sessionId, source.tabId);
  send({
    kind: 'event',
    sessionId: source.sessionId ?? `tab-${source.tabId}`,
    method,
    params,
  });
});

chrome.debugger.onDetach.addListener((source, reason) =>
  send({ kind: 'event', method: 'Target.detachedFromTarget', params: { sessionId: `tab-${source.tabId}`, reason } }),
);

// Belt and braces: drop the debugger the moment an attached tab leaves scope.
chrome.tabs.onUpdated.addListener(async (tabId, info, tab) => {
  if (!info.url || inScope(info.url)) return;
  try { await chrome.debugger.detach({ tabId }); } catch {}
});

connect();
