import braintrust from "braintrust";

/**
 * Verbosity LLM-as-judge scorer.
 *
 * Registered in the Braintrust Scorers library so it appears in the
 * dashboard Scorers tab and can be reused across experiments.
 *
 * Push with: npx braintrust push evals/scorers/verbosity.ts
 *
 * Input: serialized agent text responses (one per line, optionally with
 * step index). Score: A=1, B=0.5, C=0.
 */

const project = braintrust.projects.create({ name: "labs-asp" });

project.scorers.create({
  name: "Verbosity Judge",
  slug: "verbosity-judge",
  ifExists: "replace",
  description:
    "LLM-as-judge scorer that grades agent text responses for conciseness. Penalises play-by-play narration of every browser action, technical jargon (CSS selectors, raw IDs), and wall-of-text responses. Rewards short action-oriented updates that a caseworker can scan quickly.",
  messages: [
    {
      role: "system" as const,
      content: `You are evaluating an AI form-filling agent's text responses for verbosity quality.

The agent assists caseworkers filling out benefits applications. Caseworkers are busy and want short, scannable status updates — not a running commentary on every click.

## What "good verbosity" looks like
- 1–3 short sentences per response
- Communicates intent ("Filling in personal information now") without narrating each browser action
- No technical jargon: no CSS selectors (e.g., \`#input_first_name\`, \`@e9\`), no raw form IDs, no internal tool names
- Updates only when something meaningful happened — completing a section, hitting a blocker, needing input — not after every individual field

## What "bad verbosity" looks like
- "Now I'm clicking the next button. Now I'm typing M-a-r-i-a in the first name field. Now I'm clicking the dropdown..."
- Walls of text describing actions the user can see in the browser
- Technical detail like "filling selector @e9 with value 'Garcia'"
- Multiple paragraphs when one sentence would do`,
    },
    {
      role: "user" as const,
      content: `## Agent Text Responses
{{output}}

## Evaluation
Choose exactly one:

(A) Concise and action-oriented. ~1–3 short sentences per response. No play-by-play, no technical jargon, no wall-of-text. A caseworker can scan these quickly.

(B) Mostly fine but slightly chatty. Occasional unnecessary narration, or one response that could be tighter, or a stray selector reference. Generally usable.

(C) Verbose, play-by-play, or technical. Narrates every browser action, leaks selectors/IDs, or produces wall-of-text responses that obscure the actual status.`,
    },
  ],
  model: "gpt-4o",
  choiceScores: { A: 1, B: 0.5, C: 0 },
  useCot: true,
});
