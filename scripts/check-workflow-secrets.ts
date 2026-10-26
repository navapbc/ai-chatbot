/**
 * Check that Eve's durable workflow store holds no given string.
 *
 * Usage: pnpm tsx scripts/check-workflow-secrets.ts <string>...
 *
 * Reads WORKFLOW_POSTGRES_URL, scans every text, json, jsonb and bytea column
 * of the `workflow` schema (decompressing zstd frames inside bytea values,
 * which is how Eve stores event and run payloads), and prints, per table and
 * column, how many rows contain each string. Output is counts only: never row
 * contents, and never the strings themselves (they are labelled needle#1,
 * needle#2, ... in argument order). Exit code 1 when any string is found.
 */

import zlib from 'node:zlib';
import { Client } from 'pg';

const ZSTD_MAGIC = Buffer.from([0x28, 0xb5, 0x2f, 0xfd]);
const MAX_DEPTH = 4;

/**
 * The buffer itself plus every zstd frame decompressed from it, recursively.
 * A frame is located by its magic number wherever it sits, so a frame after a
 * header (Eve prefixes values with `zstd(`) is found. A frame that does not
 * decompress is skipped.
 */
export function expandFrames(buffer: Buffer, depth = 0): Buffer[] {
  const out = [buffer];
  if (depth >= MAX_DEPTH) return out;
  let at = buffer.indexOf(ZSTD_MAGIC);
  while (at !== -1) {
    try {
      const frame = zlib.zstdDecompressSync(buffer.subarray(at));
      out.push(...expandFrames(frame, depth + 1));
    } catch {
      // not a valid frame: keep scanning past it
    }
    at = buffer.indexOf(ZSTD_MAGIC, at + ZSTD_MAGIC.length);
  }
  return out;
}

/**
 * For one value, 1 for each needle found in the raw bytes or in any zstd
 * frame inside it, else 0. Every needle appears in the result.
 */
export function scanBuffer(
  buffer: Buffer,
  needles: string[],
): Map<string, number> {
  const views = expandFrames(buffer);
  const found = new Map<string, number>();
  for (const needle of needles) {
    found.set(needle, views.some((view) => view.includes(needle)) ? 1 : 0);
  }
  return found;
}

const TEXT_TYPES = ['text', 'character varying', 'json', 'jsonb', 'bytea'];

async function main(): Promise<number> {
  const needles = process.argv.slice(2).filter((arg) => arg.length > 0);
  if (needles.length === 0) {
    console.error('usage: tsx scripts/check-workflow-secrets.ts <string>...');
    return 2;
  }
  const url = process.env.WORKFLOW_POSTGRES_URL;
  if (!url) {
    console.error('WORKFLOW_POSTGRES_URL is not set');
    return 2;
  }

  const client = new Client({ connectionString: url });
  await client.connect();
  try {
    const columns = (
      await client.query<{ t: string; col: string; typ: string }>(
        `select table_name t, column_name col, data_type typ
           from information_schema.columns
          where table_schema = 'workflow' and data_type = any($1)
          order by table_name, ordinal_position`,
        [TEXT_TYPES],
      )
    ).rows;

    let rowsScanned = 0;
    let framesDecompressed = 0;
    let totalFound = 0;
    for (const { t, col, typ } of columns) {
      const rows = (
        await client.query<{ v: unknown }>(
          `select "${col}" v from workflow."${t}" where "${col}" is not null`,
        )
      ).rows;
      const counts = needles.map(() => 0);
      for (const { v } of rows) {
        const buffer =
          typ === 'bytea'
            ? (v as Buffer)
            : Buffer.from(typeof v === 'string' ? v : JSON.stringify(v));
        rowsScanned++;
        framesDecompressed += expandFrames(buffer).length - 1;
        const found = scanBuffer(buffer, needles);
        needles.forEach((needle, i) => {
          counts[i] += found.get(needle) ?? 0;
        });
      }
      counts.forEach((count, i) => {
        if (count === 0) return;
        totalFound += count;
        console.log(
          `workflow.${t}.${col} (${typ}): needle#${i + 1} in ${count} row(s)`,
        );
      });
    }
    console.log(
      `scanned ${columns.length} columns, ${rowsScanned} values, ${framesDecompressed} zstd frames decompressed`,
    );
    console.log(
      totalFound === 0
        ? `none of ${needles.length} needle(s) found`
        : `FOUND: ${totalFound} row match(es) across ${needles.length} needle(s)`,
    );
    return totalFound === 0 ? 0 : 1;
  } finally {
    await client.end();
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().then(
    (code) => process.exit(code),
    (error) => {
      console.error(
        `scan failed: ${error instanceof Error ? error.message : 'unknown error'}`,
      );
      process.exit(2);
    },
  );
}
