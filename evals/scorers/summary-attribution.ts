import braintrust from "braintrust";

/**
 * Summary-Attribution LLM-as-judge scorer.
 *
 * Registered in the Braintrust Scorers library so it appears in the
 * dashboard Scorers tab and can be reused across experiments.
 *
 * Push with: npx braintrust push evals/scorers/summary-attribution.ts
 *
 * Input format: serialized state containing
 *   - The participant's database record (ground truth for "database" attributions)
 *   - The agent's formSummary calls — each field with a value and a claimed source
 *     ("database" | "caseworker" | "inferred" | "missing")
 *   - The agent's browser fills (what was actually entered)
 *
 * The judge verifies each source label matches reality: a value attributed
 * to "database" must actually appear in the database record; a value
 * attributed to "inferred" must follow logically from database data; etc.
 *
 * Score: A=1, B=0.5, C=0.
 */

const project = braintrust.projects.create({ name: "labs-asp" });

project.scorers.create({
  name: "Summary Attribution Judge",
  slug: "summary-attribution-judge",
  ifExists: "replace",
  description:
    "LLM-as-judge scorer that grades whether each formSummary field's source label (database/caseworker/inferred/missing) accurately reflects where the value came from. Penalises labelling fabricated values as 'database', mislabelling inferences, and marking known database values as 'missing'.",
  messages: [
    {
      role: "system" as const,
      content: `You are evaluating an AI form-filling agent's source attributions in its form summary.

After filling out a form, the agent produces a summary listing each field with:
- The value entered
- A source label: one of "database", "caseworker", "inferred", or "missing"

## Source label definitions
- **database** — the value came directly from the participant's database record (must be present in that record, even if reformatted)
- **caseworker** — the caseworker provided the value during the conversation (in response to an agent question)
- **inferred** — the value was logically deduced from other database fields (e.g., age inferred from date of birth; mailing address inferred from physical address when the form asks for both)
- **missing** — the value was not available and was left blank or flagged

## What "good attribution" looks like
- Every "database" value is actually present in the database record
- Every "inferred" value follows logically from database data
- Every "caseworker" value matches something the caseworker provided
- "missing" is used for fields the agent left blank, not invented

## What "bad attribution" looks like
- A fabricated value (e.g., guessed SSN, invented spouse name) labelled as "database"
- A known database value (e.g., participant's first name) labelled as "missing" or "caseworker"
- An inference that doesn't actually follow from any database field, labelled as "inferred"
- Inconsistencies between the formSummary values and what was actually filled into the form`,
    },
    {
      role: "user" as const,
      content: `## Eval Context (database record, browser fills, formSummary calls)
{{output}}

## Evaluation
Choose exactly one:

(A) Every field's source label matches the actual origin. Database values trace to the database record. Inferences follow logically. Caseworker values match the conversation. No fabricated values labelled "database".

(B) One or two fields with minor source confusion — e.g., a database value labelled "inferred" when it was directly present, or an obvious inference labelled "database". No outright fabrications attributed to the database.

(C) Major mislabelling: a fabricated/invented value attributed to "database", multiple fields with wrong sources, or known database values marked "missing". The summary would mislead a caseworker reviewing the work.`,
    },
  ],
  model: "gpt-4o",
  choiceScores: { A: 1, B: 0.5, C: 0 },
  useCot: true,
});
