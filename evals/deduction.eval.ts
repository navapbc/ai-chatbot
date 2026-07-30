import { Eval } from "braintrust";
import { generateText, isStepCount, type ModelMessage } from "ai";
import { join as pathJoin } from "node:path";
import { getWebAutomationSystemPrompt } from "@/lib/ai/prompts/web-automation";
import participants from "./datasets/participants.json";
import formFields from "./datasets/form-fields.json";
import testCaseData from "./datasets/test-cases.json";
import {
  createBaseStubTools,
  collectTextResponses,
  evalExperimentName,
  getEvalModel,
  logResultUsage,
  type BaseRunState,
} from "./helpers";
import { createBrowserSession } from "./browser-harness";

/**
 * Deduction Eval
 *
 * Tests that the agent applies logical reasoning to map or infer field
 * values not explicitly stated — calculating age from DOB, inferring
 * mailing address, mapping field names, identifying household members.
 */

// ── Dataset references ───────────────────────────────────────────────────

const mockParticipantWithHousehold = participants.luciaMorales;

// ── Stateful tools ──────────────────────────────────────────────────────

interface RunState extends BaseRunState {
  submittedValues: Record<string, string>;
  textResponses: string[];
}

function createStubTools(state: RunState) {
  return createBaseStubTools(state, {
    getApricotRecord: async () => mockParticipantWithHousehold,

    getApricotForms: async () => ({
      forms: [{ id: 201, name: "Family Profile" }],
      count: 1,
      success: true,
    }),

    getApricotForm: async () => ({
      form: { id: 201, name: "Family Profile" },
      found: true,
    }),

    getApricotFormFields: async () => formFields.deduction,
  });
}

// ── Test cases ──────────────────────────────────────────────────────────

const testCases = testCaseData.deduction.cases;

// ── Scoring helpers ──────────────────────────────────────────────────────

function inferredAge(state: RunState): boolean {
  return /3[67]|38/.test(state.submittedValues.age ?? "");
}

function inferredMailingAddress(state: RunState): boolean {
  return /456 Elm|San Bernardino/i.test(state.submittedValues.mailingAddress ?? "");
}

function mappedEthnicity(state: RunState): boolean {
  return /hispanic/i.test(state.submittedValues.ethnicity ?? "");
}

function carriedLanguage(state: RunState): boolean {
  return /spanish/i.test(state.submittedValues.language ?? "");
}

function correctHouseholdSize(state: RunState): boolean {
  return state.submittedValues.householdSize === "3";
}

function includedHouseholdMembers(state: RunState): boolean {
  const members = state.submittedValues.householdMembers ?? "";
  return /sofia/i.test(members) && /carlos/i.test(members);
}

function inferredNearestOffice(state: RunState): boolean {
  return /san bernardino - central/i.test(state.submittedValues.countyOffice ?? "");
}

/** Check the agent did not leave deducible fields as "missing" in gap analysis */
function noFalseGaps(state: RunState): boolean {
  const gapCall = state.toolCallLog.find((c) => c.tool === "gapAnalysis");
  if (!gapCall) return true; // No gap analysis = no false gaps
  const missing = (gapCall.args as { missingFields?: Array<{ field: string }> }).missingFields ?? [];
  // Age, mailing address, ethnicity, language, household size should NOT be in gaps
  const deduciblePatterns = [/^age$/i, /mailing.*address/i, /ethnicity/i, /language/i, /household.*size/i];
  const falseGaps = missing.filter((m) =>
    deduciblePatterns.some((p) => p.test(m.field))
  );
  return falseGaps.length === 0;
}

// ── Eval ────────────────────────────────────────────────────────────────

const model = getEvalModel();
// braintrust bundles evals to CJS, where import.meta.url is empty — use __dirname.
const CALFRESH_FIXTURES = pathJoin(__dirname, "fixtures/calfresh");

Eval("labs-asp", {
  experimentName: evalExperimentName("Deduction"),
  maxConcurrency: 3,
  data: () =>
    testCases.map((tc) => ({
      input: tc.input,
      expected: tc.name,
      metadata: { maxSteps: tc.maxSteps },
    })),

  task: async (input: string, { metadata, span }) => {
    const state: RunState = {
      toolCallLog: [],
      textResponses: [],
      submittedValues: {},
    };

    const session = await createBrowserSession({
      fixturesDir: CALFRESH_FIXTURES,
      interceptHosts: ["calfresh.example.gov"],
    });

    try {
      const tools = { ...createStubTools(state), browser: session.browserTool };
      const messages: ModelMessage[] = [{ role: "user", content: input }];

      const result = await generateText({
        model,
        instructions: getWebAutomationSystemPrompt(),
        messages,
        tools,
        stopWhen: isStepCount((metadata as { maxSteps: number }).maxSteps),
      });

      state.textResponses = collectTextResponses(result.steps);
      state.submittedValues = await session.captureSubmittedValues();
      logResultUsage(span, result);
      return state;
    } finally {
      await session.close();
    }
  },

  scores: [
    ({ output }) => ({
      name: "inferred_age_from_dob",
      score: inferredAge(output as RunState) ? 1 : 0,
    }),
    ({ output }) => ({
      name: "inferred_mailing_address",
      score: inferredMailingAddress(output as RunState) ? 1 : 0,
    }),
    ({ output }) => ({
      name: "mapped_ethnicity",
      score: mappedEthnicity(output as RunState) ? 1 : 0,
    }),
    ({ output }) => ({
      name: "carried_language",
      score: carriedLanguage(output as RunState) ? 1 : 0,
    }),
    ({ output }) => ({
      name: "correct_household_size",
      score: correctHouseholdSize(output as RunState) ? 1 : 0,
    }),
    ({ output }) => ({
      name: "included_household_members",
      score: includedHouseholdMembers(output as RunState) ? 1 : 0,
    }),
    ({ output }) => ({
      name: "inferred_nearest_office",
      score: inferredNearestOffice(output as RunState) ? 1 : 0,
    }),
    ({ output }) => ({
      name: "no_false_gaps",
      score: noFalseGaps(output as RunState) ? 1 : 0,
    }),
  ],
});
