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
    console.log('[workflow-db] waiting for advisory lock…');
    await client.query('SELECT pg_advisory_lock($1)', [ADVISORY_LOCK_KEY]);
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
