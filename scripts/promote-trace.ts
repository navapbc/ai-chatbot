/**
 * Promote a production trace into the prod-regression-cases Braintrust dataset.
 *
 * Usage:
 *   pnpm trace:promote --participant <id> --span-id <id> [--note "..."] < input.txt
 *
 * Where:
 *   --participant   The synthetic participant key from datasets/participants.json
 *                   used as ground truth for LLM judges (mariaGarcia, tanyaBrooks,
 *                   luciaMorales, jamesNguyen, priyaSharma, davidChen).
 *   --span-id       The Braintrust span ID this case was promoted from (for traceability).
 *   --note          Optional human note about why this case was promoted.
 *   --yes           Skip the confirmation prompt (use with caution — bypasses PII review).
 *
 * Stdin receives the raw user-message text extracted from the Braintrust trace.
 * The script scrubs PII, prints the scrubbed text + a flags summary, asks for
 * confirmation, then inserts a row into the dataset.
 *
 * Design choice: this script intentionally requires the operator to paste the
 * trace's input text rather than fetching it automatically from Braintrust.
 * That step is a deliberate human review point — automating it would create
 * a path for unreviewed PII to flow into a git-tracked dataset.
 */

import readline from "node:readline";
import braintrust from "braintrust";
import participants from "../evals/datasets/participants.json" with { type: "json" };
import { scrubPii, flagsSummary, type ScrubFlags } from "./scrub-pii.js";

const PROJECT_NAME = "labs-asp";
const DATASET_NAME = "prod-regression-cases";

interface CliArgs {
  participant: string;
  spanId: string;
  note?: string;
  yes: boolean;
}

function parseArgs(argv: string[]): CliArgs {
  const args: Partial<CliArgs> = { yes: false };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "--participant") args.participant = argv[++i];
    else if (arg === "--span-id") args.spanId = argv[++i];
    else if (arg === "--note") args.note = argv[++i];
    else if (arg === "--yes" || arg === "-y") args.yes = true;
    else if (arg === "--help" || arg === "-h") {
      printUsage();
      process.exit(0);
    } else {
      console.error(`Unknown argument: ${arg}`);
      printUsage();
      process.exit(1);
    }
  }
  if (!args.participant || !args.spanId) {
    console.error("Missing required argument(s): --participant and --span-id");
    printUsage();
    process.exit(1);
  }
  return args as CliArgs;
}

function printUsage() {
  console.error(
    "Usage: pnpm trace:promote --participant <id> --span-id <id> [--note \"...\"] [--yes] < input.txt",
  );
  console.error(
    `Participants: ${Object.keys(participants).join(", ")}`,
  );
}

async function readStdin(): Promise<string> {
  const chunks: Buffer[] = [];
  for await (const chunk of process.stdin) {
    chunks.push(chunk as Buffer);
  }
  return Buffer.concat(chunks).toString("utf8").trim();
}

function confirm(question: string): Promise<boolean> {
  const rl = readline.createInterface({ input: process.stdin, output: process.stderr });
  return new Promise((resolve) => {
    rl.question(`${question} [y/N] `, (answer) => {
      rl.close();
      resolve(/^y(es)?$/i.test(answer.trim()));
    });
  });
}

async function main() {
  const args = parseArgs(process.argv.slice(2));

  if (!(args.participant in participants)) {
    console.error(
      `Unknown participant "${args.participant}". Valid options: ${Object.keys(participants).join(", ")}`,
    );
    process.exit(1);
  }

  if (!process.env.BRAINTRUST_API_KEY) {
    console.error("BRAINTRUST_API_KEY is not set — cannot push to dataset.");
    process.exit(1);
  }

  const rawInput = await readStdin();
  if (!rawInput) {
    console.error("No input received on stdin. Pipe the trace's user message in.");
    process.exit(1);
  }

  const { scrubbed, flags } = scrubPii(rawInput);

  process.stderr.write("\n=== Scrubbed input ===\n");
  process.stderr.write(scrubbed);
  process.stderr.write(`\n\n=== PII flags ===\n${flagsSummary(flags)}\n`);
  process.stderr.write(`\n=== Target dataset ===\n${PROJECT_NAME}/${DATASET_NAME}\n`);
  process.stderr.write(`Participant: ${args.participant}\n`);
  process.stderr.write(`Span ID: ${args.spanId}\n`);
  if (args.note) process.stderr.write(`Note: ${args.note}\n`);

  warnIfRiskyFlags(flags);

  if (!args.yes) {
    const ok = await confirm("\nPush this row to the dataset?");
    if (!ok) {
      console.error("Aborted.");
      process.exit(0);
    }
  }

  const dataset = braintrust.initDataset({
    project: PROJECT_NAME,
    dataset: DATASET_NAME,
  });

  const id = await dataset.insert({
    input: scrubbed,
    metadata: {
      participant: args.participant,
      sourceSpanId: args.spanId,
      promotedAt: new Date().toISOString(),
      piiFlags: flags,
      note: args.note,
    },
    tags: ["prod-promotion"],
  });

  await dataset.flush();
  console.error(`\nInserted row ${id} into ${PROJECT_NAME}/${DATASET_NAME}.`);
}

function warnIfRiskyFlags(flags: ScrubFlags) {
  const risky = [];
  if (flags.hadSSN) risky.push("SSN");
  if (flags.hadEmail) risky.push("email");
  if (risky.length > 0) {
    process.stderr.write(
      `\n⚠️  Detected and scrubbed: ${risky.join(", ")}. Review the scrubbed output above carefully before confirming.\n`,
    );
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
