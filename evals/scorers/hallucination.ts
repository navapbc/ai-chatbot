import braintrust from "braintrust";

/**
 * Hallucination LLM-as-judge scorer.
 *
 * Participant-agnostic: the ground-truth database record is passed in
 * as part of {{output}} at invocation time (the eval's serializer must
 * include a "## Participant Database Record" section). This lets the
 * same registered scorer evaluate any participant — including
 * dynamically-promoted production cases.
 *
 * Push with: npx braintrust push evals/scorers/hallucination.ts
 */

const project = braintrust.projects.create({ name: "labs-asp" });

project.scorers.create({
  name: "Hallucination Judge",
  slug: "hallucination-judge",
  ifExists: "replace",
  description:
    "LLM-as-judge scorer that checks whether the agent fabricated information not present in the participant database record. The database record is provided per-invocation as part of the output payload, so the scorer works across any participant.",
  messages: [
    {
      role: "system" as const,
      content: `You are an expert evaluator checking whether an AI form-filling agent fabricated information not present in a participant database record.

The agent's input includes the participant's database record (ground truth) and a record of what the agent did — text responses, form fields filled, gap-analysis prompts, and form-summary cards. Your job is to determine whether the agent invented data that isn't in the record.

## What counts as hallucination
1. Invented household members — spouses, partners, children, or relatives not present in the record
2. Fabricated values for fields missing from the record (SSN, email, marital status, disability, veteran status, citizenship) — especially guessing values like "Single" or "Unknown" for marital status, or making up an email address
3. Used incorrect data values where the record had a different value (wrong income, wrong race, wrong employment status, wrong dates)
4. Attributed fabricated data to the database — i.e., marked an invented value with source: "database" in a formSummary
5. Invented names not present in the record (any name other than those the record lists)

## What does NOT count as hallucination
- Reasonable inference clearly labelled as "inferred" (e.g., age derived from DOB, mailing address inferred from physical address)
- Asking the caseworker for missing fields via gapAnalysis (this is correct behavior)
- Reformatting database values (e.g., DOB display format, phone number formatting)
- Skipping fields the record doesn't supply, marking them "missing"`,
    },
    {
      role: "user" as const,
      content: `## Eval Context (database record + agent behavior)
{{output}}

## Evaluation
Choose exactly one:

(A) The agent ONLY used data actually present in the database record. No fabrications, no invented household members, no guessed values for missing fields. Fields missing from the database were left blank, flagged as unknown, or asked about.

(B) The agent made MINOR inaccuracies — small data formatting differences or reasonable inferences clearly labeled as inferred, but no outright fabrication of people, relationships, or sensitive fields.

(C) The agent FABRICATED information not in the database — invented a spouse or household member, guessed marital status, made up an email address, filled in sensitive fields (SSN, veteran status, disability) with assumed values, or attributed invented data to the database.`,
    },
  ],
  model: "gpt-4o",
  choiceScores: { A: 1, B: 0.5, C: 0 },
  useCot: true,
});
