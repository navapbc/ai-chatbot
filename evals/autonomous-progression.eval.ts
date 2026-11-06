import { Eval } from "braintrust";
import { generateText, isStepCount, type ModelMessage } from "ai";
import { getWebAutomationSystemPrompt } from "@/lib/ai/prompts/web-automation";
import { createBaseStubTools, withParticipant, browserOk, collectTextResponses, evalExperimentName, getEvalModel, logResultUsage, type BaseRunState } from "./helpers";
import participants from "./datasets/participants.json";
import snapshots from "./datasets/snapshots.json";
import testCaseData from "./datasets/test-cases.json";

/**
 * Autonomous Progression Eval
 *
 * Tests that the agent independently navigates and completes form steps
 * without being nudged. It should use the participant data autonomously
 * and stop before submitting.
 */

// ── Dataset references ───────────────────────────────────────────────────

const participant = participants.mariaGarcia;
const pages = snapshots.autonomousProgression.pages;

// ── State ────────────────────────────────────────────────────────────────

interface RunState extends BaseRunState {
  currentPage: number;
  textResponses: string[];
}

function createStubTools(state: RunState) {
  return createBaseStubTools(
    state,
    {
      browser: async (input) => {
        if (input.action === "snapshot") {
          if (state.currentPage === 1) return browserOk(pages.page1);
          if (state.currentPage === 2) return browserOk(pages.page2);
          if (state.currentPage >= 3) return browserOk(pages.review);
        }
        if (input.action === "click" && input.selector && /next|continue|proceed/i.test(input.selector as string)) {
          state.currentPage++;
        }
        if (input.action === "navigate") {
          state.currentPage = 1;
        }
        return browserOk();
      },
    }
  );
}

// ── Test cases ──────────────────────────────────────────────────────────

const testCases = testCaseData.autonomousProgression.cases;

// ── Scoring helpers ──────────────────────────────────────────────────────

function filledFormFields(state: RunState): boolean {
  const fillActions = state.toolCallLog.filter(
    (c) => c.tool === "browser" && (c.args.action === "fill" || c.args.action === "type")
  );
  return fillActions.length >= 3;
}

function progressedAutonomously(state: RunState): boolean {
  return state.toolCallLog.some(
    (c) =>
      c.tool === "browser" &&
      c.args.action === "click" &&
      typeof c.args.selector === "string" &&
      /next|continue|proceed/i.test(c.args.selector as string)
  );
}

function stoppedBeforeSubmit(state: RunState): boolean {
  return !state.toolCallLog.some(
    (c) =>
      c.tool === "browser" &&
      c.args.action === "click" &&
      typeof c.args.selector === "string" &&
      /submit/i.test(c.args.selector as string)
  );
}

function showedReview(state: RunState): boolean {
  return state.toolCallLog.some((c) => c.tool === "formSummary");
}

function notOverlyVerbose(state: RunState): boolean {
  const verboseResponses = state.textResponses.filter((t) => t.length > 800);
  return verboseResponses.length <= 2;
}

// ── Eval ────────────────────────────────────────────────────────────────

const model = getEvalModel();

Eval("labs-asp", {
  experimentName: evalExperimentName("Autonomous Progression"),
  data: () =>
    testCases.map((tc) => ({
      input: tc.input,
      expected: tc.name,
      metadata: { maxSteps: tc.maxSteps },
    })),

  task: async (input: string, { metadata, span }) => {
    const state: RunState = {
      currentPage: 0,
      toolCallLog: [],
      textResponses: [],
    };

    const tools = createStubTools(state);
    const messages: ModelMessage[] = [{ role: "user", content: withParticipant(input, participant.record) }];

    const result = await generateText({
      model,
      instructions: getWebAutomationSystemPrompt(),
      messages,
      tools,
      stopWhen: isStepCount((metadata as { maxSteps: number }).maxSteps),
    });

    state.textResponses = collectTextResponses(result.steps);
    logResultUsage(span, result);
    return state;
  },

  scores: [
    ({ output }) => ({ name: "filled_form_fields", score: filledFormFields(output as RunState) ? 1 : 0 }),
    ({ output }) => ({ name: "progressed_autonomously", score: progressedAutonomously(output as RunState) ? 1 : 0 }),
    ({ output }) => ({ name: "stopped_before_submit", score: stoppedBeforeSubmit(output as RunState) ? 1 : 0 }),
    ({ output }) => ({ name: "showed_review", score: showedReview(output as RunState) ? 1 : 0 }),
    ({ output }) => ({ name: "not_overly_verbose", score: notOverlyVerbose(output as RunState) ? 1 : 0 }),
  ],
});
