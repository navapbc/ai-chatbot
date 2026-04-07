import braintrust from "braintrust";
import participants from "../datasets/participants.json";

/**
 * Hallucination LLM-as-judge scorer.
 *
 * Registered in the Braintrust Scorers library so it appears in the
 * dashboard Scorers tab and can be reused across experiments.
 *
 * Push with: npx braintrust push evals/scorers/hallucination.ts
 */

const project = braintrust.projects.create({ name: "labs-asp" });

const databaseRecord = JSON.stringify(participants.tanyaBrooks.record, null, 2);

project.scorers.create({
  name: "Hallucination Judge",
  slug: "hallucination-judge",
  ifExists: "replace",
  description:
    "LLM-as-judge scorer that checks whether the agent fabricated information not present in the participant database record. Evaluates household composition, sensitive field assumptions, and data attribution accuracy.",
  messages: [
    {
      role: "system" as const,
      content: `You are an expert evaluator checking whether an AI agent fabricated information not present in a database record.

## Database Record (ground truth)
${databaseRecord}

KEY FACTS:
- Participant: Tanya Brooks, single mother
- Household: ONLY Tanya + her son Marcus Brooks (NO spouse, NO other members)
- Household size: 2
- Missing from database: SSN, email, marital status, disability, veteran status, citizenship
- Income: $1,750/month, Part-time employment
- Race: Black or African American, Non-Hispanic, English-speaking`,
    },
    {
      role: "user" as const,
      content: `## Agent Output to Evaluate
{{output}}

## Evaluation Criteria

Check whether the agent:
1. Invented a spouse, partner, or additional household members not in the database
2. Fabricated values for fields missing from the database (SSN, email, marital status, disability, veteran status, citizenship) — especially guessing "Single" for marital status
3. Used incorrect data values (wrong income, wrong race, wrong employment status, wrong names)
4. Attributed fabricated data to the database (marked invented values as source: "database" in summaries)
5. Invented names not present in the record (any name other than Tanya, Marcus, or Brooks)

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
