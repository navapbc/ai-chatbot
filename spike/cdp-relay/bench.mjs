// Throwaway spike: times agent-browser commands through the relay.
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
const run = promisify(execFile);
const [cdp, label] = process.argv.slice(2);
const bin = process.env.AB_BIN ?? new URL('../../node_modules/.bin/agent-browser', import.meta.url).pathname;
const session = `bench-${label}-${Date.now()}`;
const call = async (...args) => {
  const t = performance.now();
  const { stdout } = await run(bin, ['--session', session, '--cdp', cdp, '--json', ...args], { maxBuffer: 1 << 26 });
  const j = JSON.parse(stdout);
  return { ms: performance.now() - t, ok: Array.isArray(j) ? j.every((x) => !x.error) : j.success, j };
};
const med = (a) => [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)];
const out = {};
const tabs = await call('tab');
await call('tab', tabs.j.data.tabs[0].tabId);
const snaps = [];
let refs;
for (let i = 0; i < 3; i++) {
  const r = await call('snapshot', '-i');
  snaps.push(r.ms);
  refs = r.j.data.refs;
}
const ref = (name) => '@' + Object.entries(refs).find(([, v]) => v.name === name)[0];
const first = ref('First name');
const last = ref('Last name');
out.snapshot_ms_median = Math.round(med(snaps));
const gets = [];
for (let i = 0; i < 5; i++) gets.push((await call('get', 'url')).ms);
out.get_url_ms_median = Math.round(med(gets));
let t = performance.now();
let allOk = true;
for (let i = 0; i < 10; i++) allOk &&= (await call('fill', first, `name${i}`)).ok;
out.ten_sequential_fills_ms = Math.round(performance.now() - t);
t = performance.now();
const b = await call('batch', '--bail', ...Array.from({ length: 10 }, (_, i) => `fill ${last} "name${i}"`));
out.ten_fills_one_batch_ms = Math.round(performance.now() - t);
out.all_ok = allOk && b.ok;
const v = await call('eval', 'document.getElementById("last").value');
out.final_value_ok = JSON.stringify(v.j).includes('name9');
console.log(label.padEnd(14), JSON.stringify(out));
