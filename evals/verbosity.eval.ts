import { Eval, initFunction } from "braintrust";
import { generateText, stepCountIs, type ModelMessage } from "ai";
import { getWebAutomationSystemPrompt } from "@/lib/ai/prompts/web-automation";
import participants from "./datasets/participants.json";
import snapshots from "./datasets/snapshots.json";
import testCaseData from "./datasets/test-cases.json";
import { openai } from "@ai-sdk/openai";
import {
  createBaseStubTools,
  browserOk,
  collectIndexedTextResponses,
  type BaseRunState,
} from "./helpers";

/**
 * Verbosity Eval
 *
 * Tests that the agent communicates concisely — providing brief, relevant
 * updates at meaningful decision points without narrating every click or
 * overwhelming the caseworker with text.
 */

// ── Mock data ────────────────────────────────────────────────────────────

const mockParticipant = participants.priyaSharma;

// ── Page snapshots (5-page form to give the agent many opportunities to be verbose) ──

const snapshotPage1 = snapshots.verbosity.pages.page1;
const snapshotPage2 = snapshots.verbosity.pages.page2;
const snapshotPage3 = snapshots.verbosity.pages.page3;
const snapshotPage4 = snapshots.verbosity.pages.page4;
const snapshotPage5Review = snapshots.verbosity.pages.review;

// ── Stateful tools ──────────────────────────────────────────────────────

interface RunState extends BaseRunState {
  currentPage: number;
  browserActions: Array<{ action: string; selector?: string; [key: string]: unknown }>;
  /** Every text response the agent produces, with the step index */
  textResponses: Array<{ stepIndex: number; text: string }>;
  /** Total number of tool-call steps */
  totalSteps: number;
}

function createStubTools(state: RunState) {
  return createBaseStubTools(state, {
    getApricotRecord: async () => mockParticipant,

    browser: async (input: Record<string, unknown>) => {
      state.browserActions.push(input as RunState["browserActions"][0]);

      if (input.action === "snapshot") {
        switch (state.currentPage) {
          case 1: return browserOk(snapshotPage1);
          case 2: return browserOk(snapshotPage2);
          case 3: return browserOk(snapshotPage3);
          case 4: return browserOk(snapshotPage4);
          default: return browserOk(snapshotPage5Review);
        }
      }

      if (input.action === "click" && input.selector) {
        if (/@e1|@e16|@e26|@e34|next|begin|continue/i.test(input.selector as string)) {
          state.currentPage++;
        }
      }

      if (input.action === "navigate") {
        state.currentPage = 1;
      }

      if (input.action === "inputvalue") {
        return browserOk("value-ok");
      }

      return browserOk();
    },
  });
}

// ── Test cases ──────────────────────────────────────────────────────────

const testCases = testCaseData.verbosity.cases;

// ── Scoring helpers ──────────────────────────────────────────────────────

// Technical jargon patterns a caseworker should never see
const TECHNICAL_JARGON = /\bref=|@e\d+|selector|css|dom\b|aria-|<select>|<input|\.modal|\.select2|getElementById|querySelector|xpath|locator/i;

// Patterns indicating the agent is narrating individual clicks
const ACTION_NARRATION = /(?:i(?:'ll| will|'m going to)?\s*(?:now\s+)?(?:click|type|fill|select|enter|input|navigate|press|check|uncheck|hover))/i;

/**
 * Average characters per text response.
 * Good: 50–300 chars. Too verbose: >500 avg. Too terse: <20 avg.
 */
function avgResponseLength(state: RunState): number {
  if (state.textResponses.length === 0) return 0;
  const total = state.textResponses.reduce((sum, r) => sum + r.text.length, 0);
  return total / state.textResponses.length;
}

/**
 * Ratio of steps that include text responses.
 * The agent should NOT produce text on most tool-call-only steps.
 * Ideal: text on ~20-50% of steps (key milestones only).
 */
function textResponseRatio(state: RunState): number {
  if (state.totalSteps === 0) return 0;
  return state.textResponses.length / state.totalSteps;
}

/**
 * Score: responses are concise (not walls of text).
 * Penalizes if average response length > 600 chars or if any single
 * response exceeds 1200 chars.
 */
function responsesAreConcise(state: RunState): boolean {
  const avg = avgResponseLength(state);
  const hasWall = state.textResponses.some((r) => r.text.length > 1200);
  return avg <= 600 && !hasWall;
}

/**
 * Score: agent doesn't narrate every individual action.
 * Counts how many text responses describe clicking/typing/filling.
 * Allowing up to 3 is generous — more than that is excessive narration.
 */
function doesNotNarrateEveryAction(state: RunState): boolean {
  const narrations = state.textResponses.filter((r) => {
    // Count how many action narration phrases appear in this response
    const matches = r.text.match(new RegExp(ACTION_NARRATION.source, "gi"));
    return matches && matches.length >= 2; // Multiple narrations in one response
  });
  return narrations.length <= 2;
}

/**
 * Score: agent doesn't expose technical jargon to the caseworker.
 */
function noTechnicalJargon(state: RunState): boolean {
  return !state.textResponses.some((r) => TECHNICAL_JARGON.test(r.text));
}

/**
 * Score: agent provides at least some text updates (not completely silent).
 * Should have at least 2 meaningful text responses across the whole flow.
 */
function providesUpdates(state: RunState): boolean {
  const meaningful = state.textResponses.filter((r) => r.text.trim().length > 15);
  return meaningful.length >= 2;
}

/**
 * Score: agent doesn't produce text on more than 60% of steps.
 * Most steps should be silent tool calls; text only at milestones.
 */
function textIsInfrequent(state: RunState): boolean {
  const ratio = textResponseRatio(state);
  return ratio <= 0.6;
}

/**
 * Score: no individual response has more than 5 sentences about
 * what was just done. Brief summaries are fine; play-by-play is not.
 */
function noPlayByPlay(state: RunState): boolean {
  for (const r of state.textResponses) {
    // Split on sentence-ending punctuation
    const sentences = r.text.split(/[.!?]+/).filter((s) => s.trim().length > 10);
    // If a response has 8+ sentences AND most mention browser actions, it's play-by-play
    if (sentences.length >= 8) {
      const actionSentences = sentences.filter((s) => ACTION_NARRATION.test(s));
      if (actionSentences.length >= 5) return false;
    }
  }
  return true;
}

// ── LLM-as-judge scorer (registered in Braintrust Scorers tab) ──────────

const verbosityJudge = initFunction({
  projectName: "labs-asp",
  slug: "verbosity-judge",
});

function serializeForVerbosityJudge(state: RunState): string {
  if (state.textResponses.length === 0) {
    return "(agent produced no text responses)";
  }
  return state.textResponses
    .map((r) => `[step ${r.stepIndex}] ${r.text.trim()}`)
    .join("\n\n");
}

// ── Eval ────────────────────────────────────────────────────────────────

const model = openai('gpt-5-mini');

Eval("labs-asp", {
  experimentName: "Verbosity",
  data: () =>
    testCases.map((tc) => ({
      input: tc.input,
      expected: tc.name,
      metadata: { maxSteps: tc.maxSteps },
    })),

  task: async (input: string, { metadata }) => {
    const state: RunState = {
      currentPage: 0,
      toolCallLog: [],
      browserActions: [],
      textResponses: [],
      totalSteps: 0,
    };

    const tools = createStubTools(state);
    const messages: ModelMessage[] = [{ role: "user", content: input }];

    const result = await generateText({
      model,
      system: getWebAutomationSystemPrompt(),
      messages,
      tools,
      stopWhen: stepCountIs((metadata as { maxSteps: number }).maxSteps),
    });

    state.totalSteps = result.steps.length;
    state.textResponses = collectIndexedTextResponses(result.steps);

    return state;
  },

  scores: [
    ({ output }) => ({
      name: "responses_are_concise",
      score: responsesAreConcise(output as RunState) ? 1 : 0,
    }),
    ({ output }) => ({
      name: "does_not_narrate_every_action",
      score: doesNotNarrateEveryAction(output as RunState) ? 1 : 0,
    }),
    ({ output }) => ({
      name: "no_technical_jargon",
      score: noTechnicalJargon(output as RunState) ? 1 : 0,
    }),
    ({ output }) => ({
      name: "provides_updates",
      score: providesUpdates(output as RunState) ? 1 : 0,
    }),
    ({ output }) => ({
      name: "text_is_infrequent",
      score: textIsInfrequent(output as RunState) ? 1 : 0,
    }),
    ({ output }) => ({
      name: "no_play_by_play",
      score: noPlayByPlay(output as RunState) ? 1 : 0,
    }),
    async ({ output }) => {
      const serialized = serializeForVerbosityJudge(output as RunState);
      const result = (await verbosityJudge({ output: serialized })) as {
        score?: number | null;
        metadata?: Record<string, unknown>;
      };
      return {
        name: "verbosity_judge",
        score: result.score ?? 0,
        metadata: result.metadata,
      };
    },
  ],
});
