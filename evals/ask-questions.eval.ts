import { Eval, initFunction } from "braintrust";
import { generateText, stepCountIs, type ModelMessage } from "ai";
import { getWebAutomationSystemPrompt } from "@/lib/ai/prompts/web-automation";
import participants from "./datasets/participants.json";
import snapshots from "./datasets/snapshots.json";
import formFields from "./datasets/form-fields.json";
import testCaseData from "./datasets/test-cases.json";
import {
  createBaseStubTools,
  browserOk,
  collectTextResponses,
  evalExperimentName,
  getEvalModel,
  type BaseRunState,
} from "./helpers";

/**
 * Ask Questions Eval
 *
 * Tests that the agent recognizes when fields cannot be filled from the
 * database and proactively asks the caseworker — rather than guessing.
 * Applies to SSN, immigration status, income, veteran status, etc.
 */

// ── Mock data ────────────────────────────────────────────────────────────

const mockSparseParticipant = participants.jamesNguyen;

const mockFormSnapshot = snapshots.askQuestions.pages.page1;
const mockFormSnapshotPage2 = snapshots.askQuestions.pages.page2;
const mockReviewSnapshot = snapshots.askQuestions.pages.review;

// ── Stateful tools ──────────────────────────────────────────────────────

interface RunState extends BaseRunState {
  currentPage: number;
  browserFills: Array<{ selector: string; value: string }>;
  textResponses: string[];
  gapAnalysisFields: string[];
}

function createStubTools(state: RunState) {
  return createBaseStubTools(state, {
    getApricotRecord: async () => mockSparseParticipant,

    getApricotForms: async () => ({
      forms: [{ id: 301, name: "CalWorks Application" }],
      count: 1,
      success: true,
    }),

    getApricotForm: async () => ({
      form: { id: 301, name: "CalWorks Application" },
      found: true,
    }),

    getApricotFormFields: async () => formFields.askQuestions,

    gapAnalysis: async (input: Record<string, unknown>) => {
      const missingFields = (input.missingFields as Array<{ field: string }>) ?? [];
      const fields = missingFields.map((f) => f.field.toLowerCase());
      state.gapAnalysisFields.push(...fields);
      return input;
    },

    browser: async (input: Record<string, unknown>) => {
      const action = input.action as string;
      const selector = input.selector as string | undefined;
      const value = input.value as string | undefined;

      if (
        (action === "fill" || action === "type" || action === "select") &&
        selector &&
        value
      ) {
        state.browserFills.push({ selector, value });
      }

      if (action === "snapshot") {
        if (state.currentPage <= 1)
          return browserOk(mockFormSnapshot as string);
        if (state.currentPage === 2)
          return browserOk(mockFormSnapshotPage2 as string);
        return browserOk(mockReviewSnapshot as string);
      }

      if (
        action === "click" &&
        selector &&
        /next|continue/i.test(selector)
      ) {
        state.currentPage++;
      }

      if (action === "navigate") {
        state.currentPage = 1;
      }

      return browserOk();
    },
  });
}

// ── Test cases ──────────────────────────────────────────────────────────

const testCases = testCaseData.askQuestions.cases;

// ── Fields the database DOES NOT have (agent must ask) ──────────────────

const MUST_ASK_FIELDS = [
  "ssn",
  "social security",
  "citizenship",
  "immigration",
  "veteran",
  "disability",
  "medical",
  "medic",
  "income",
  "contact method",
  "contact preference",
  "communication",
];

// ── Fields the database DOES have (agent must NOT ask for these) ────────

const KNOWN_FIELD_PATTERNS = [
  /full.*name|first.*name|last.*name/i,
  /date.*birth|dob/i,
  /^address$|home.*address/i,
  /^phone$/i,
  /^gender$/i,
  /^race$/i,
  /^ethnicity$/i,
  /preferred.*language|language/i,
];

// ── Scoring helpers ──────────────────────────────────────────────────────

/** Check that gapAnalysis was called at all */
function calledGapAnalysis(state: RunState): boolean {
  return state.toolCallLog.some((c) => c.tool === "gapAnalysis");
}

/**
 * Check that the agent asked about sensitive/missing fields.
 * Looks in both gapAnalysis calls and text responses.
 */
function askedForMissingFields(state: RunState): { asked: string[]; missed: string[] } {
  const allText = [
    ...state.gapAnalysisFields,
    ...state.textResponses.map((t) => t.toLowerCase()),
  ].join(" ");

  const asked: string[] = [];
  const missed: string[] = [];

  for (const field of MUST_ASK_FIELDS) {
    if (allText.includes(field)) {
      asked.push(field);
    } else {
      missed.push(field);
    }
  }

  return { asked, missed };
}

/** Score: proportion of missing fields the agent asked about */
function missingFieldCoverage(state: RunState): number {
  const { asked } = askedForMissingFields(state);
  return asked.length / MUST_ASK_FIELDS.length;
}

/** Check that the agent did NOT assume/fill sensitive fields it doesn't have data for */
function didNotAssumeSensitiveFields(state: RunState): boolean {
  const sensitivePatterns = [
    /ssn|social.*security/i,
    /citizen|immigration/i,
    /veteran/i,
    /disability/i,
    /medic/i,
    /income/i,
  ];

  for (const fill of state.browserFills) {
    for (const pattern of sensitivePatterns) {
      if (pattern.test(fill.selector)) {
        // The agent filled a sensitive field — check it didn't guess a value
        // Acceptable: leaving blank, selecting "Unknown", or a placeholder
        if (
          fill.value &&
          !/unknown|n\/a|blank|empty|skip/i.test(fill.value) &&
          fill.value.trim().length > 0
        ) {
          return false;
        }
      }
    }
  }

  return true;
}

/** Check that the agent did NOT ask for fields already in the database */
function didNotAskForKnownFields(state: RunState): boolean {
  const allGapText = state.gapAnalysisFields.join(" ");

  for (const pattern of KNOWN_FIELD_PATTERNS) {
    if (pattern.test(allGapText)) {
      return false;
    }
  }

  return true;
}

/** Check that the agent DID fill the fields it has data for */
function filledKnownFields(state: RunState): boolean {
  // Should have filled at least name, address, phone, gender
  const filledSelectors = state.browserFills.map((f) => f.selector.toLowerCase());
  const allFilled = filledSelectors.join(" ");

  const requiredFills = [/name/i, /address/i, /gender/i];
  const hits = requiredFills.filter((p) => p.test(allFilled));
  return hits.length >= 2; // At least 2 of the 3 known fields filled
}

// ── LLM-as-judge scorer (registered in Braintrust Scorers tab) ──────────

const askQuestionsJudge = initFunction({
  projectName: "labs-asp",
  slug: "ask-questions-judge",
});

function serializeForAskQuestionsJudge(state: RunState): string {
  const parts: string[] = [];

  parts.push("## Participant Database Record (ground truth)");
  parts.push(JSON.stringify(mockSparseParticipant.record, null, 2));

  parts.push("\n## Form Fields the agent had to fill");
  parts.push(JSON.stringify(formFields.askQuestions, null, 2));

  parts.push("\n## Questions the agent asked the caseworker (via gapAnalysis)");
  parts.push(
    state.gapAnalysisFields.length > 0
      ? state.gapAnalysisFields.map((f) => `- ${f}`).join("\n")
      : "(none — the agent did not call gapAnalysis)"
  );

  parts.push("\n## Fields the agent eventually filled (selector → value)");
  parts.push(
    state.browserFills.length > 0
      ? state.browserFills.map((f) => `- ${f.selector}: "${f.value}"`).join("\n")
      : "(none)"
  );

  parts.push("\n## Agent text responses");
  parts.push(
    state.textResponses.length > 0
      ? state.textResponses.join("\n\n")
      : "(none)"
  );

  return parts.join("\n");
}

// ── Eval ────────────────────────────────────────────────────────────────

const model = getEvalModel();

Eval("labs-asp", {
  experimentName: evalExperimentName("Ask Questions"),
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
      browserFills: [],
      textResponses: [],
      gapAnalysisFields: [],
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

    state.textResponses = collectTextResponses(result.steps);

    return state;
  },

  scores: [
    ({ output }) => ({
      name: "called_gap_analysis",
      score: calledGapAnalysis(output as RunState) ? 1 : 0,
    }),
    ({ output }) => ({
      name: "missing_field_coverage",
      score: missingFieldCoverage(output as RunState),
    }),
    ({ output }) => ({
      name: "did_not_assume_sensitive_fields",
      score: didNotAssumeSensitiveFields(output as RunState) ? 1 : 0,
    }),
    ({ output }) => ({
      name: "did_not_ask_for_known_fields",
      score: didNotAskForKnownFields(output as RunState) ? 1 : 0,
    }),
    ({ output }) => ({
      name: "filled_known_fields",
      score: filledKnownFields(output as RunState) ? 1 : 0,
    }),
    async ({ output }) => {
      const serialized = serializeForAskQuestionsJudge(output as RunState);
      const result = (await askQuestionsJudge({ output: serialized })) as {
        score?: number | null;
        metadata?: Record<string, unknown>;
      };
      return {
        name: "ask_questions_judge",
        score: result.score ?? 0,
        metadata: result.metadata,
      };
    },
  ],
});
