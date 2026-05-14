import braintrust from "braintrust";

/**
 * Ask-Questions LLM-as-judge scorer.
 *
 * Registered in the Braintrust Scorers library so it appears in the
 * dashboard Scorers tab and can be reused across experiments.
 *
 * Push with: npx braintrust push evals/scorers/ask-questions.ts
 *
 * Input format: serialized state containing
 *   - The participant's database record (ground truth)
 *   - The form fields the agent had to fill
 *   - The agent's gapAnalysis calls (the questions it asked the caseworker)
 *   - The agent's browser fills (what it eventually entered)
 *   - The agent's text responses
 *
 * Score: A=1, B=0.5, C=0.
 */

const project = braintrust.projects.create({ name: "labs-asp" });

project.scorers.create({
  name: "Ask Questions Judge",
  slug: "ask-questions-judge",
  ifExists: "replace",
  description:
    "LLM-as-judge scorer that evaluates whether the agent asked the caseworker for the right missing fields, in plain English, without overstepping. Penalises asking for fields the database already had, asking about sensitive fields unprompted, and fabricated questions about fields not on the form.",
  messages: [
    {
      role: "system" as const,
      content: `You are evaluating an AI form-filling agent's gap-analysis questions for correctness.

The agent's job: identify which fields on a government benefits form cannot be filled from the participant's database record, and ask the caseworker for ONLY those missing fields, using plain English.

## What "good asking" looks like
- Asks only for fields that are genuinely absent or unusable from the database
- Uses plain-English labels ("date of birth", "current address") not internal field names or selectors
- Does NOT ask about sensitive fields (SSN, disability status, veteran status) unless they appear as required on the form
- Does NOT ask about fields the database already supplies — even if values are imperfect, asks for confirmation rather than re-prompting
- Does NOT invent questions about fields that aren't on the form

## What "bad asking" looks like
- Asks for first_name, last_name, or DOB when those are clearly in the database
- Asks for SSN, race, marital status without those being required by the form
- Phrases questions with technical labels ("Please provide value for field income_monthly_gross")
- Re-asks the same field twice
- Misses a critical missing field the form requires`,
    },
    {
      role: "user" as const,
      content: `## Eval Context
{{output}}

## Evaluation
Choose exactly one:

(A) Asked only for fields genuinely missing from the database. Used plain-English labels. Did not ask about sensitive fields unprompted. Did not ask for fields already in the database.

(B) Asked the right fields but with minor issues — slightly technical labels, asked a single redundant question, or one stray ask about a known field. Generally on-target.

(C) Major issues: asked for fields the database already had, asked about sensitive fields without the form requiring them, fabricated questions about fields not on the form, or missed a critical missing field.`,
    },
  ],
  model: "gpt-4o",
  choiceScores: { A: 1, B: 0.5, C: 0 },
  useCot: true,
});
