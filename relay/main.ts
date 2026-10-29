import { createLocalBus, createRedisBus } from './bus';
import { checkRelayEnv } from './env';
import { createRelayServer } from './server';

const list = (value: string | undefined): string[] =>
  (value ?? '')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);

async function main() {
  // The multi-instance guard and its known gaps live in checkRelayEnv (env.ts).
  const check = checkRelayEnv(process.env);
  if (!check.ok) {
    console.error(check.message);
    process.exit(1);
  }
  const secret = process.env.RELAY_TOKEN_SECRET as string;

  const redisUrl = process.env.REDIS_URL;
  // createRedisBus keeps the URL out of its errors.
  const bus = redisUrl ? await createRedisBus(redisUrl) : createLocalBus();

  const server = createRelayServer({
    secret,
    allowedHostSuffixes: list(process.env.RELAY_ALLOWED_HOST_SUFFIXES),
    allowedHosts: list(process.env.RELAY_ALLOWED_HOSTS),
    extensionId: process.env.RELAY_EXTENSION_ID || undefined,
    bus,
    log: (line) => console.log(JSON.stringify(line)),
    recordMethods: process.env.RELAY_RECORD_METHODS === '1',
  });

  const port = Number(process.env.PORT ?? 8080);
  server.listen(port, '0.0.0.0', () => {
    console.log(JSON.stringify({ event: 'listening', severity: 'INFO', port }));
  });

  const shutdown = () => {
    server.close();
    bus.close().finally(() => process.exit(0));
  };
  process.on('SIGTERM', shutdown);
  process.on('SIGINT', shutdown);
}

main().catch(() => {
  console.error('relay: failed to start');
  process.exit(1);
});
