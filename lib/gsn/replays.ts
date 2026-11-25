import postgres from 'postgres';
const db = postgres(process.env.DATABASE_URL ?? '', { max: 2 });
export async function claimRequest(
  id: string,
): Promise<'claimed' | 'duplicate' | 'limited'> {
  return db.begin(async (tx) => {
    // Serialize claims across instances, bounding cost even with public demo logins.
    await tx`SELECT pg_advisory_xact_lock(7426137)`;
    const existing =
      await tx`SELECT id FROM gsn_bridge_requests WHERE id = ${id}`;
    if (existing.length) return 'duplicate';
    const [counts] =
      await tx`SELECT count(*) FILTER (WHERE created_at >= date_trunc('day', now() AT TIME ZONE 'UTC') AT TIME ZONE 'UTC') AS daily,
      count(*) FILTER (WHERE finished_at IS NULL AND created_at > now() - interval '4 minutes') AS active
      FROM gsn_bridge_requests WHERE created_at > now() - interval '1 day'`;
    if (Number(counts?.daily) >= 100 || Number(counts?.active) >= 4)
      return 'limited';
    await tx`INSERT INTO gsn_bridge_requests (id) VALUES (${id})`;
    return 'claimed';
  });
}
export async function finishRequest(id: string): Promise<void> {
  await db`UPDATE gsn_bridge_requests SET finished_at = now() WHERE id = ${id}`;
}
