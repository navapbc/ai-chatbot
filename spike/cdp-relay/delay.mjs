// Throwaway spike: TCP proxy that delays every chunk by DELAY_MS in each
// direction, standing in for the internet between a caseworker and the server.
import net from 'node:net';
const [listen, target, delay] = [Number(process.env.LISTEN), Number(process.env.TARGET), Number(process.env.DELAY_MS)];
net.createServer((c) => {
  const u = net.connect(target, '127.0.0.1');
  // Preserve order: each chunk is released at (arrival + delay).
  const pipe = (from, to) => from.on('data', (d) => setTimeout(() => to.writable && to.write(d), delay));
  pipe(c, u); pipe(u, c);
  c.on('close', () => u.destroy()); u.on('close', () => c.destroy());
  c.on('error', () => {}); u.on('error', () => {});
}).listen(listen, '127.0.0.1', () => console.log(`delay proxy :${listen} -> :${target} (+${delay}ms each way)`));
