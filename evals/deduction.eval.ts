import { Eval } from "braintrust";
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
  logResultUsage,
  type BaseRunState,
} from "./helpers";

/**
 * Deduction Eval
 *
 * Tests that the agent applies logical reasoning to map or infer field
 * values not explicitly stated — calculating age from DOB, inferring
 * mailing address, mapping field names, identifying household members.
 */

// ── Dataset references ───────────────────────────────────────────────────

const mockParticipantWithHousehold = participants.luciaMorales;
const mockFormSnapshot = snapshots.deduction.pages.form;
const mockReviewSnapshot = snapshots.deduction.pages.review;

// ── Stateful tools ──────────────────────────────────────────────────────

interface RunState extends BaseRunState {
  currentPage: number;
  browserFills: Array<{ selector: string; value: string }>;
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

    browser: async (input: Record<string, unknown>) => {
      const action = input.action as string;
      const selector = (input.selector as string) ?? "";
      const value = (input.value as string) ?? "";

      // Track fill/type/select actions for scoring
      if (
        (action === "fill" || action === "type" || action === "select") &&
        selector &&
        value
      ) {
        state.browserFills.push({ selector, value });
      }

      if (action === "snapshot") {
        if (state.currentPage <= 1)
          return browserOk(mockFormSnapshot);
        return browserOk(mockReviewSnapshot);
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

const testCases = testCaseData.deduction.cases;

// ── Scoring helpers ──────────────────────────────────────────────────────

/** Check that the agent computed age from DOB (should be 37 as of 2026-04-07) */
function inferredAge(state: RunState): boolean {
  const ageFill = state.browserFills.find(
    (f) => /age/i.test(f.selector) && /3[67]|38/.test(f.value)
  );
  // Also check formSummary for age
  const summaryCall = state.toolCallLog.find((c) => c.tool === "formSummary");
  if (summaryCall) {
    const fields = (summaryCall.args as { fields?: Array<{ field: string; value?: string }> }).fields;
    if (fields?.some((f) => /age/i.test(f.field) && /3[67]|38/.test(f.value ?? ""))) {
      return true;
    }
  }
  return !!ageFill;
}

/** Check that mailing address was carried over from the database address */
function inferredMailingAddress(state: RunState): boolean {
  const addressFill = state.browserFills.find(
    (f) =>
      /mail|address/i.test(f.selector) &&
      /456 Elm|San Bernardino/i.test(f.value)
  );
  return !!addressFill;
}

/** Check that ethnicity was correctly mapped (Hispanic or Latino → Hispanic) */
function mappedEthnicity(state: RunState): boolean {
  const ethnicityFill = state.browserFills.find(
    (f) => /ethnicity/i.test(f.selector) && /hispanic/i.test(f.value)
  );
  return !!ethnicityFill;
}

/** Check that preferred language was carried over */
function carriedLanguage(state: RunState): boolean {
  const langFill = state.browserFills.find(
    (f) => /language/i.test(f.selector) && /spanish/i.test(f.value)
  );
  return !!langFill;
}

/** Check that household size was correctly set to 3 */
function correctHouseholdSize(state: RunState): boolean {
  const sizeFill = state.browserFills.find(
    (f) => /household|people/i.test(f.selector) && f.value === "3"
  );
  return !!sizeFill;
}

/** Check that household members were included (Sofia and Carlos) */
function includedHouseholdMembers(state: RunState): boolean {
  const membersFill = state.browserFills.find(
    (f) =>
      /member|household|names/i.test(f.selector) &&
      /sofia/i.test(f.value) &&
      /carlos/i.test(f.value)
  );
  // Also check text responses and formSummary
  if (membersFill) return true;
  const summaryCall = state.toolCallLog.find((c) => c.tool === "formSummary");
  if (summaryCall) {
    const fields = (summaryCall.args as { fields?: Array<{ field: string; value?: string }> }).fields;
    if (
      fields?.some(
        (f) =>
          /member|household/i.test(f.field) &&
          /sofia/i.test(f.value ?? "") &&
          /carlos/i.test(f.value ?? "")
      )
    ) {
      return true;
    }
  }
  return false;
}

/** Check that nearest office was inferred from address (San Bernardino - Central) */
function inferredNearestOffice(state: RunState): boolean {
  const officeFill = state.browserFills.find(
    (f) =>
      /office|county/i.test(f.selector) &&
      /san bernardino/i.test(f.value)
  );
  return !!officeFill;
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

Eval("labs-asp", {
  experimentName: evalExperimentName("Deduction"),
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
      browserFills: [],
      textResponses: [],
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

    logResultUsage(span, result);
    return state;
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
