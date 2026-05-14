/**
 * Regex-based PII scrubber for production traces.
 *
 * Used by scripts/promote-trace.ts before a production session is pushed
 * into the prod-regression-cases Braintrust dataset. Errs on the side of
 * over-redacting — false positives are cheap (a few extra [REDACTED]
 * tags), false negatives leak real participant data into a dataset
 * that lives in git and Braintrust.
 *
 * What this handles
 *   - SSN (XXX-XX-XXXX, X X X-X X-X X X X with stray whitespace)
 *   - Email addresses
 *   - US phone numbers ((XXX) XXX-XXXX, XXX-XXX-XXXX, +1 XXX...)
 *   - US ZIP codes (5 or 9 digit)
 *   - Apricot numeric record IDs (>=4 digit numbers near "record" or "participant")
 *
 * What this does NOT handle
 *   - Names — too many false positives with regex; the script asks
 *     the operator to eyeball + manually swap before saving
 *   - Free-form addresses — street names are arbitrary; partial city/state
 *     redaction is left to the operator
 *
 * If `flags.hadName` would be set by a future NER step, the promote-trace
 * script should block on operator review.
 */

export interface ScrubFlags {
  hadSSN: boolean;
  hadEmail: boolean;
  hadPhone: boolean;
  hadZip: boolean;
  hadRecordId: boolean;
}

export interface ScrubResult {
  scrubbed: string;
  flags: ScrubFlags;
}

const PATTERNS = {
  // SSN: 3-2-4 digits with optional space or dash separators
  ssn: /\b\d{3}[\s-]?\d{2}[\s-]?\d{4}\b/g,
  // Email: standard RFC-ish — intentionally permissive
  email: /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b/g,
  // US phone: optional country code, optional parens, common separators
  phone: /(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b/g,
  // ZIP: 5 digits or ZIP+4
  zip: /\b\d{5}(?:-\d{4})?\b/g,
  // Apricot record IDs: 4+ digit numbers appearing after "record"/"participant"/"id"
  recordId: /\b(record(?:\s+id)?|participant(?:\s+id)?|id)\s*[:#=]?\s*(\d{4,})\b/gi,
};

/**
 * Scrub PII from a free-form string (typically a user message extracted from
 * a Braintrust span).
 *
 * Order matters: emails are matched before phones because @-domains can
 * contain digits that the phone regex would otherwise eat.
 */
export function scrubPii(input: string): ScrubResult {
  const flags: ScrubFlags = {
    hadSSN: false,
    hadEmail: false,
    hadPhone: false,
    hadZip: false,
    hadRecordId: false,
  };

  let scrubbed = input;

  if (PATTERNS.email.test(scrubbed)) {
    flags.hadEmail = true;
    scrubbed = scrubbed.replace(PATTERNS.email, "[REDACTED_EMAIL]");
  }

  if (PATTERNS.ssn.test(scrubbed)) {
    flags.hadSSN = true;
    scrubbed = scrubbed.replace(PATTERNS.ssn, "[REDACTED_SSN]");
  }

  if (PATTERNS.phone.test(scrubbed)) {
    flags.hadPhone = true;
    scrubbed = scrubbed.replace(PATTERNS.phone, "[REDACTED_PHONE]");
  }

  if (PATTERNS.recordId.test(scrubbed)) {
    flags.hadRecordId = true;
    scrubbed = scrubbed.replace(
      PATTERNS.recordId,
      (_, label) => `${label} [REDACTED_RECORD_ID]`,
    );
  }

  // ZIP last — it would otherwise match the 5-digit chunks inside SSNs and phones.
  if (PATTERNS.zip.test(scrubbed)) {
    flags.hadZip = true;
    scrubbed = scrubbed.replace(PATTERNS.zip, "[REDACTED_ZIP]");
  }

  return { scrubbed, flags };
}

/**
 * Human-readable summary of which PII categories were present.
 * Used by promote-trace.ts to show the operator what was scrubbed.
 */
export function flagsSummary(flags: ScrubFlags): string {
  const present = Object.entries(flags)
    .filter(([, v]) => v)
    .map(([k]) => k.replace(/^had/, ""));
  return present.length === 0 ? "(no PII patterns matched)" : present.join(", ");
}

/**
 * CLI entry: `pnpm scrub:check <<< "some input string"` or
 * `pnpm scrub:check < trace.txt`. Reads stdin, writes scrubbed output to stdout
 * and a summary to stderr.
 */
async function main() {
  const chunks: Buffer[] = [];
  for await (const chunk of process.stdin) {
    chunks.push(chunk as Buffer);
  }
  const input = Buffer.concat(chunks).toString("utf8");
  const { scrubbed, flags } = scrubPii(input);
  process.stdout.write(scrubbed);
  process.stderr.write(`\n--- scrubbed: ${flagsSummary(flags)} ---\n`);
}

if (process.argv[1] && /scrub-pii\.ts$/.test(process.argv[1])) {
  main().catch((err) => {
    console.error(err);
    process.exit(1);
  });
}
