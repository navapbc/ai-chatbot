/**
 * Bootstrap the Workflow (Eve durability) schema, under an advisory lock.
 *
 * Eve's durable sessions live in Postgres via `@workflow/world-postgres` (see
 * `agent/agent.ts`). That schema has to exist before any instance calls
 * `world.start()`.
 *
 * Why the lock: the package's own setup bootstraps graphile-worker's schema
 * precisely because `installSchema`'s `CREATE SCHEMA IF NOT EXISTS` is not
 * race-safe — concurrent callers fail with
 * `duplicate key value violates unique constraint "pg_namespace_nspname_index"`.
 * Cloud Run runs this service at `min_instance_count = 2`, so the first
 * revision against a fresh database starts two containers at once and both run
 * this script. `pg_advisory_lock` serializes them: the first does the work, the
 * second waits and then finds every migration already applied.
 *
 * This is NOT the same exposure as `lib/db/migrate`. Drizzle's migrator is
 * guarded by a migrations table and tolerates concurrency; the graphile-worker
 * DDL path explicitly does not.
 *
 * Idempotent — safe to run on every container start, which is where the
 * Dockerfile calls it (alongside the existing app migration).
 */

import { Pool } from 'pg';

// Arbitrary but fixed: advisory locks are keyed by a signed 64-bit integer, and
// any other process taking this same key is by definition also bootstrapping
// this schema. This is ASCII "workflow", which fits in a signed bigint.
const ADVISORY_LOCK_KEY = '8606223218533756791'; // ASCII "workflow", 0x776f726b666c6f77

// Generous enough for a real first-deploy migration run, short enough to fail
// before Cloud Run's startup probe gives up on the instance.
const LOCK_TIMEOUT_MS = 120_000;

function resolveConnectionString(): string {
  // Same precedence @workflow/world-postgres uses internally, so the lock is
  // always taken against the database the bootstrap will actually migrate.
  const url =
    process.env.WORKFLOW_POSTGRES_URL ?? process.env.DATABASE_URL ?? null;
  if (url === null) {
    throw new Error(
      'bootstrap-workflow-db: neither WORKFLOW_POSTGRES_URL nor DATABASE_URL is set',
    );
  }
  return url;
}

async function main(): Promise<void> {
  const connectionString = resolveConnectionString();
  // Dedicated single connection: an advisory lock is held by the session that
  // took it, so it must not be handed back to a pool mid-bootstrap.
  const pool = new Pool({ connectionString, max: 1 });
  const client = await pool.connect();

  try {
    // Bounded rather than an indefinite block. The container opens no port
    // until this finishes, and with min_instance_count = 2 the second instance
    // waits behind the first one's bootstrap. If that ever outlasts Cloud Run's
    // startup deadline the instance is killed with no explanation — a loud
    // failure here is far easier to diagnose than a restart loop.
    await client.query(`SET lock_timeout = '${LOCK_TIMEOUT_MS}ms'`);

    console.log('[workflow-db] waiting for advisory lock…');
    try {
      await client.query('SELECT pg_advisory_lock($1)', [ADVISORY_LOCK_KEY]);
    } catch (error: unknown) {
      // 55P03 = lock_not_available
      if ((error as { code?: string }).code === '55P03') {
        throw new Error(
          `bootstrap-workflow-db: timed out after ${LOCK_TIMEOUT_MS}ms waiting for the workflow schema lock. Another instance is probably still bootstrapping; if this repeats, check for a stale session holding the advisory lock.`,
        );
      }
      throw error;
    }
    console.log('[workflow-db] lock acquired, running schema setup');

    // Imported lazily so the lock is already held before the package opens its
    // own pool and starts issuing DDL.
    const { setupDatabase } = await import('@workflow/world-postgres/cli');
    await setupDatabase();

    console.log('[workflow-db] schema ready');
  } finally {
    // Releasing explicitly rather than relying on disconnect keeps the window
    // short if the pool lingers during shutdown.
    await client
      .query('SELECT pg_advisory_unlock($1)', [ADVISORY_LOCK_KEY])
      .catch(() => {});
    client.release();
    await pool.end();
  }
}

main().catch((error: unknown) => {
  console.error('[workflow-db] bootstrap failed:', error);
  process.exit(1);
});
