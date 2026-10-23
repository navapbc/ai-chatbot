// Throwaway spike: presents a browser-level CDP endpoint (/cdp) backed by a
// chrome.debugger extension (/extension).
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { execFile } from 'node:child_process';
import { createRequire } from 'node:module';

const root = path.resolve(import.meta.dirname, '../..');
const pnpm = path.join(root, 'node_modules/.pnpm');
const nodeRequire = createRequire(import.meta.url);
const { WebSocketServer } = (() => {
  try {
    return nodeRequire('ws'); // container: npm-installed
  } catch {
    const wsDir = fs.readdirSync(pnpm).find((d) => d.startsWith('ws@'));
    return nodeRequire(path.join(pnpm, wsDir, 'node_modules/ws'));
  }
})();

const PORT = Number(process.env.PORT ?? 9333);
const INSTANCE = process.env.INSTANCE ?? `i${PORT}`;
const EXT_URL = process.env.EXT_URL ?? 'ws://127.0.0.1:9333/extension';
const REDIS_URL = process.env.REDIS_URL;
// Per-run secret. Both sockets must present it; the extension reads it from a
// generated config file, agent-browser from the printed URL.
const TOKEN = process.env.SPIKE_TOKEN ?? crypto.randomBytes(32).toString('hex');
if (process.env.WRITE_CONFIG !== '0')
fs.writeFileSync(
  path.join(import.meta.dirname, 'extension/config.js'),
  `self.RELAY_TOKEN = '${TOKEN}';\nself.RELAY_URL = '${EXT_URL}';\n`,
);
let ext = null;
let client = null;
let nextId = 1;
const pending = new Map();
const log = (...a) => console.log(new Date().toISOString().slice(11, 23), ...a);


// Optional shared bus so the extension and agent-browser can land on
// different relay instances (Cloud Run has several).
let pub = null;
if (REDIS_URL) {
  const { createClient } = createRequire(import.meta.url)(
    path.join(pnpm, fs.readdirSync(pnpm).find((d) => d.startsWith('redis@')), 'node_modules/redis'),
  );
  pub = createClient({ url: REDIS_URL });
  const sub = pub.duplicate();
  await pub.connect();
  await sub.connect();
  // Whoever holds the extension socket forwards commands to it.
  await sub.subscribe('to-ext', (raw) => {
    if (ext?.readyState === 1) ext.send(raw);
  });
  // Whoever issued the command resolves it; whoever has the client forwards events.
  await sub.subscribe('from-ext', (raw) => handleFromExt(JSON.parse(raw)));
  log('bus connected');
}

const extCall = (msg) =>
  new Promise((resolve, reject) => {
    const id = `${INSTANCE}:${nextId++}`;
    pending.set(id, { resolve, reject });
    const raw = JSON.stringify({ ...msg, id });
    if (pub) return void pub.publish('to-ext', raw);
    if (!ext) return reject(new Error('extension not connected'));
    ext.send(raw);
  });

function handleFromExt(m) {
  if (m.kind === 'reply') {
    const p = pending.get(m.id);
    if (!p) return; // another instance's command
    pending.delete(m.id);
    m.error ? p.reject(new Error(m.error)) : p.resolve(m.result);
  } else if (m.kind === 'event') {
    toClient({ sessionId: m.sessionId, method: m.method, params: m.params });
  }
}

const toClient = (m) => client?.readyState === 1 && client.send(JSON.stringify(m));

const tokenMatches = (t) =>
  crypto.timingSafeEqual(
    crypto.createHash('sha256').update(t ?? '').digest(),
    crypto.createHash('sha256').update(TOKEN).digest(),
  );

const hostOk = (req) => {
  const host = (req.headers.host ?? '').replace(/:\d+$/, '');
  const suffix = process.env.ALLOWED_HOST_SUFFIX;
  return (
    host === '127.0.0.1' ||
    host === 'localhost' ||
    Boolean(suffix && host.endsWith(suffix))
  );
};

const server = http.createServer((req, res) => {
  if (!hostOk(req)) {
    res.statusCode = 421;
    return res.end();
  }
  if (req.url === '/form.html') {
    res.setHeader('content-type', 'text/html');
    return res.end(fs.readFileSync(path.join(import.meta.dirname, 'form.html')));
  }
  if (req.url === '/oopif.html') {
    res.setHeader('content-type', 'text/html');
    return res.end(fs.readFileSync(path.join(import.meta.dirname, 'oopif.html')));
  }
  if (req.url === '/json/version') {
    res.setHeader('content-type', 'application/json');
    return res.end(JSON.stringify({
      Browser: 'Chrome/relay', 'Protocol-Version': '1.3',
      webSocketDebuggerUrl: `ws://127.0.0.1:${PORT}/cdp`, // token is supplied out-of-band
    }));
  }
  const u = new URL(req.url, 'http://x');
  if (u.pathname === '/ping' || u.pathname === '/bench') {
    if (!tokenMatches(u.searchParams.get('token'))) {
      res.statusCode = 401;
      return res.end();
    }
    if (!ext) {
      res.statusCode = 503;
      return res.end('extension not connected\n');
    }
    (async () => {
      if (u.pathname === '/ping') {
        const rtts = [];
        for (let i = 0; i < 15; i++) {
          const t = performance.now();
          await extCall({ type: 'listTargets' });
          rtts.push(Math.round(performance.now() - t));
        }
        const sorted = [...rtts].sort((a, b) => a - b);
        res.setHeader('content-type', 'application/json');
        return res.end(JSON.stringify({ rtts, median: sorted[7], min: sorted[0], max: sorted.at(-1) }) + '\n');
      }
      execFile(
        'node',
        [path.join(import.meta.dirname, 'bench.mjs'), `ws://127.0.0.1:${PORT}/cdp?token=${TOKEN}`, 'cloud'],
        { timeout: 280000, env: { ...process.env, SPIKE_TOKEN: undefined } },
        (err, stdout, stderr) => {
          res.setHeader('content-type', 'text/plain');
          res.end(`${stdout}${err ? `\nERR ${err.message}\n${stderr.slice(-800)}` : ''}\n`);
        },
      );
    })().catch((e) => { res.statusCode = 500; res.end(String(e.message) + '\n'); });
    return;
  }
  res.statusCode = 404;
  res.end();
});

const wss = new WebSocketServer({ noServer: true });
server.on('upgrade', (req, socket, head) => {
  if (!hostOk(req)) {
    socket.destroy();
    return;
  }
  const url = new URL(req.url, 'http://localhost');
  const origin = req.headers.origin;
  const tokenOk = crypto.timingSafeEqual(
    Buffer.from(crypto.createHash('sha256').update(url.searchParams.get('token') ?? '').digest()),
    Buffer.from(crypto.createHash('sha256').update(TOKEN).digest()),
  );
  // Browsers always send Origin on WebSocket handshakes, so a page cannot
  // reach /cdp; only the extension may carry one, and only to /extension.
  const originOk =
    url.pathname === '/extension'
      ? Boolean(origin?.startsWith('chrome-extension://'))
      : url.pathname === '/cdp' && !origin;
  if (!tokenOk || !originOk) {
    log('rejected upgrade', url.pathname, { origin, tokenOk });
    socket.destroy();
    return;
  }
  wss.handleUpgrade(req, socket, head, (ws) => {
    if (url.pathname === '/extension') onExtension(ws);
    else onClient(ws);
  });
});

const onExtension = (ws) => {
  ext = ws;
  log('extension connected');
  ws.on('close', () => { if (ext === ws) ext = null; log('extension closed'); });
  ws.on('message', (data) => {
    if (pub) return void pub.publish('from-ext', data.toString());
    handleFromExt(JSON.parse(data));
  });
};

const attached = new Set();
const browserLevel = async (method, params) => {
  switch (method) {
    case 'Browser.getVersion':
      return { protocolVersion: '1.3', product: 'Chrome/relay', revision: '0', userAgent: 'relay', jsVersion: '0' };
    case 'Target.getTargets':
      return await extCall({ type: 'listTargets' });
    case 'Target.attachToTarget': {
      const r = await extCall({ type: 'attach', targetId: params.targetId });
      attached.add(r.sessionId);
      setTimeout(() => toClient({
        method: 'Target.attachedToTarget',
        params: { sessionId: r.sessionId, targetInfo: { ...r.targetInfo, attached: true }, waitingForDebugger: false },
      }), 0);
      return { sessionId: r.sessionId };
    }
    case 'Target.createTarget':
      return await extCall({ type: 'createTarget', url: params.url });
    case 'Target.closeTarget':
      return await extCall({ type: 'closeTarget', targetId: params.targetId });
    case 'Target.getTargetInfo': {
      const { targetInfos } = await extCall({ type: 'listTargets' });
      return { targetInfo: targetInfos.find((t) => t.targetId === params?.targetId) ?? targetInfos[0] };
    }
    default:
      log('STUB browser-level', method);
      return {};
  }
};

const onClient = (ws) => {
  client = ws;
  log('cdp client connected');
  ws.on('close', () => log('cdp client closed'));
  ws.on('message', async (data) => {
    const m = JSON.parse(data);
    const reply = (o) => ws.send(JSON.stringify({ id: m.id, ...(m.sessionId ? { sessionId: m.sessionId } : {}), ...o }));
    log('<-', m.sessionId ? `[${m.sessionId}]` : '[browser]', m.method);
    try {
      const result = m.sessionId
        ? await extCall({ type: 'cdp', sessionId: m.sessionId, method: m.method, params: m.params })
        : await browserLevel(m.method, m.params);
      reply({ result: result ?? {} });
    } catch (err) {
      log('!!', m.method, err.message);
      reply({ error: { code: -32000, message: err.message } });
    }
  });
};

server.listen(PORT, process.env.HOST ?? '127.0.0.1', () =>
  log(`relay on ${process.env.HOST ?? '127.0.0.1'}:${PORT}`),
);
