/**
 * Upsert every record in lib/data/participants.ts into the Participant table.
 *
 * Run: dotenv -e .env.local -- tsx scripts/seed-participants.ts
 *
 * Imports no server-only modules; it opens its own connection.
 */

import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';
import {
  buildSearchText,
  summarize,
} from '../lib/data/participant-source';
import { PARTICIPANTS } from '../lib/data/participants';
import { participant } from '../lib/db/schema';
import { sql } from 'drizzle-orm';

const run = async () => {
  if (!process.env.DATABASE_URL) {
    throw new Error('DATABASE_URL is not defined');
  }
  const connection = postgres(process.env.DATABASE_URL, { max: 1 });
  const db = drizzle(connection);

  for (const record of PARTICIPANTS) {
    const { name, county } = summarize(record);
    const values = {
      recordId: record.record_id,
      displayName: name,
      county,
      searchText: buildSearchText(record),
      record,
    };
    await db
      .insert(participant)
      .values(values)
      .onConflictDoUpdate({
        target: participant.recordId,
        set: {
          displayName: values.displayName,
          county: values.county,
          searchText: values.searchText,
          record: sql`excluded."record"`,
        },
      });
  }
  console.log(`Seeded ${PARTICIPANTS.length} participants`);
  await connection.end();
};

run().catch((err) => {
  console.error('Seeding participants failed');
  console.error(err instanceof Error ? err.message : 'unknown error');
  process.exit(1);
});
