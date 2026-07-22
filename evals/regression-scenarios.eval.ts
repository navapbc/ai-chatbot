import { Eval } from "braintrust";
import { generateText, isStepCount, type ModelMessage } from "ai";
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
 * Regression Scenarios Eval
 *
 * Cross-walked from cwilkes-npbc/AI-Evaluations regression suite (TC1/TC2/TC3).
 * Covers gap behaviors that the per-category suites don't exercise because they
 * require form-specific fields (WIC auth checkbox, IHSS applying-for-self radio,
 * gender→sex mapping, mother-eligibility ask, blind-from-special-needs flag).
 *
 * Each test case targets one Rosa/Carolina + WIC/IHSS combination. Scorers
 * return `null` for scenarios they don't apply to, so Braintrust skips them
 * in the per-experiment aggregate.
 */

// ── Participant lookup ──────────────────────────────────────────────────

const PARTICIPANTS_BY_ID = {
  339688: participants.rosaFlores,
  339702: participants.carolinaDelgado,
} as const;

function lookupParticipant(recordId: number) {
  const match = PARTICIPANTS_BY_ID[recordId as keyof typeof PARTICIPANTS_BY_ID];
  return match ?? { record: null, found: false };
}

// ── Form routing ────────────────────────────────────────────────────────

type FormType = "wic" | "ihss";

function detectFormFromUrl(url: string | undefined): FormType | null {
  if (!url) return null;
  if (/wic|ruhealth/i.test(url)) return "wic";
  if (/ihss/i.test(url)) return "ihss";
  return null;
}

const FORM_IDS: Record<FormType, number> = { wic: 501, ihss: 502 };
const FORM_NAMES: Record<FormType, string> = {
  wic: "WIC Application",
  ihss: "IHSS Application",
};

function formByFormId(formId: number): FormType | null {
  if (formId === FORM_IDS.wic) return "wic";
  if (formId === FORM_IDS.ihss) return "ihss";
  return null;
}

// ── State ───────────────────────────────────────────────────────────────

interface RunState extends BaseRunState {
  scenario: string;
  participantId: number | null;
  currentForm: FormType | null;
  currentPage: number;
  browserFills: Array<{ selector: string; value: string }>;
  browserChecks: Array<{ selector: string }>;
  textResponses: string[];
  gapAnalysisFields: string[];
  formSummaryFields: Array<{ field: string; value?: string; source: string }>;
}

function createStubTools(state: RunState) {
  return createBaseStubTools(state, {
    getApricotRecord: async ({ recordId }) => {
      state.participantId = recordId;
      return lookupParticipant(recordId);
    },

    getApricotForms: async () => ({
      forms: [
        { id: FORM_IDS.wic, name: FORM_NAMES.wic },
        { id: FORM_IDS.ihss, name: FORM_NAMES.ihss },
      ],
      count: 2,
      success: true,
    }),

    getApricotForm: async ({ formId }) => {
      const form = formByFormId(formId);
      if (!form) return { form: null, found: false };
      return { form: { id: formId, name: FORM_NAMES[form] }, found: true };
    },

    getApricotFormFields: async ({ formId }) => {
      const form = formByFormId(formId);
      if (!form) return { fields: [], count: 0, success: false };
      return formFields[form];
    },

    gapAnalysis: async (input) => {
      const missingFields = (input.missingFields as Array<{ field: string }>) ?? [];
      state.gapAnalysisFields.push(...missingFields.map((f) => f.field.toLowerCase()));
      return input;
    },

    formSummary: async (input) => {
      const fields = (input.fields as Array<{ field: string; value?: string; source: string }>) ?? [];
      state.formSummaryFields.push(...fields);
      return input;
    },

    browser: async (input) => {
      const action = input.action as string;
      const selector = (input.selector as string | undefined) ?? "";
      const value = (input.value as string | undefined) ?? "";
      const url = input.url as string | undefined;

      if (action === "navigate") {
        const form = detectFormFromUrl(url);
        if (form) state.currentForm = form;
        state.currentPage = 1;
      }

      if (action === "snapshot") {
        if (!state.currentForm) return browserOk("[no page loaded]");
        const pages = snapshots[state.currentForm].pages;
        if (state.currentPage >= 2) return browserOk(pages.review);
        return browserOk(pages.form);
      }

      if ((action === "fill" || action === "type" || action === "select") && selector) {
        state.browserFills.push({ selector, value });
      }

      if (action === "check" && selector) {
        state.browserChecks.push({ selector });
      }

      if (action === "click" && selector) {
        // Capture clicks on checkbox-shaped or radio-shaped selectors so the
        // scorers can detect "agent checked the WIC auth box" regardless of
        // whether the agent uses `check` or `click`.
        state.browserChecks.push({ selector });
        if (/next|continue|review/i.test(selector)) state.currentPage++;
      }

      return browserOk();
    },
  });
}

// ── Scoring helpers ─────────────────────────────────────────────────────

function evidenceText(state: RunState): string {
  const fills = state.browserFills.map((f) => `${f.selector}=${f.value}`).join(" ");
  const checks = state.browserChecks.map((c) => c.selector).join(" ");
  const summary = state.formSummaryFields
    .map((f) => `${f.field}=${f.value ?? ""}[${f.source}]`)
    .join(" ");
  const text = state.textResponses.join(" ");
  const gaps = state.gapAnalysisFields.join(" ");
  return [fills, checks, summary, text, gaps].join(" ").toLowerCase();
}

function hasCheckOrFill(state: RunState, pattern: RegExp, valuePattern?: RegExp): boolean {
  for (const c of state.browserChecks) if (pattern.test(c.selector)) return true;
  for (const f of state.browserFills) {
    if (pattern.test(f.selector)) {
      if (!valuePattern || valuePattern.test(f.value)) return true;
    }
  }
  for (const f of state.formSummaryFields) {
    if (pattern.test(f.field)) {
      if (!valuePattern || valuePattern.test(f.value ?? "")) return true;
    }
  }
  return false;
}

// ── Scorers (return null when not applicable to the scenario) ───────────

function checkedWicAuth(state: RunState): number | null {
  if (state.scenario !== "wic_auth") return null;
  return hasCheckOrFill(state, /authoriz|authorize_appointments|wic.*appoint/i) ? 1 : 0;
}

function selectedApplyingForSelf(state: RunState): number | null {
  if (state.scenario !== "applying_for_self_and_gender_to_sex") return null;
  // Pass if agent selected "Yes" on the applying-for-self field, via fill, click, or summary.
  const yesValuePattern = /yes|true/i;
  if (hasCheckOrFill(state, /applying.*for.*self|applying_for_self/i, yesValuePattern)) return 1;
  // Some agents may click the "yes" option directly without setting value
  for (const c of state.browserChecks) {
    if (/applying.*for.*self.*yes|applying_for_self.*yes/i.test(c.selector)) return 1;
  }
  return 0;
}

function mappedGenderToSex(state: RunState): number | null {
  if (state.scenario !== "applying_for_self_and_gender_to_sex") return null;
  // Rosa's DB gender is Female → agent should set sex=Female on IHSS.
  const femalePattern = /female/i;
  if (hasCheckOrFill(state, /^sex$|sex_|"sex"/i, femalePattern)) return 1;
  for (const c of state.browserChecks) {
    if (/sex.*female|sex_female/i.test(c.selector)) return 1;
  }
  return 0;
}

function askedMotherEligibility(state: RunState): number | null {
  if (state.scenario !== "mother_eligibility") return null;
  // Carolina has no child 0-5 — agent should ask about pregnancy / child age.
  const text = evidenceText(state);
  const eligibilityKeywords = /pregnant|pregnancy|postpartum|breastfeed|child.*age|0.?5|how old|do you have a child|are you a mother/i;
  return eligibilityKeywords.test(text) ? 1 : 0;
}

function selectedBlindFromFlag(state: RunState): number | null {
  if (state.scenario !== "blind_from_flag") return null;
  // Carolina has specialNeeds=Yes, specialNeedsNotes=Blind → agent should check
  // the "Applicant is blind" box without being told.
  if (hasCheckOrFill(state, /applicant_blind|blind|visually.*impair/i)) return 1;
  for (const c of state.browserChecks) {
    if (/applicant_blind|blind|visually.*impair/i.test(c.selector)) return 1;
  }
  return 0;
}

// ── Eval ────────────────────────────────────────────────────────────────

const model = getEvalModel();
const testCases = testCaseData.regressionScenarios.cases;

Eval("labs-asp", {
  experimentName: evalExperimentName("Regression Scenarios"),
  data: () =>
    testCases.map((tc) => ({
      input: tc.input,
      expected: tc.name,
      metadata: {
        maxSteps: tc.maxSteps,
        scenario: tc.scenario,
        participantId: tc.participantId,
        form: tc.form,
      },
    })),

  task: async (input: string, { metadata, span }) => {
    const meta = metadata as {
      maxSteps: number;
      scenario: string;
      participantId: number;
      form: FormType;
    };

    const state: RunState = {
      scenario: meta.scenario,
      participantId: null,
      currentForm: null,
      currentPage: 0,
      toolCallLog: [],
      browserFills: [],
      browserChecks: [],
      textResponses: [],
      gapAnalysisFields: [],
      formSummaryFields: [],
    };

    const tools = createStubTools(state);
    const messages: ModelMessage[] = [{ role: "user", content: input }];

    const result = await generateText({
      model,
      instructions: getWebAutomationSystemPrompt(),
      messages,
      tools,
      stopWhen: isStepCount(meta.maxSteps),
    });

    state.textResponses = collectTextResponses(result.steps);

    logResultUsage(span, result);
    return state;
  },

  scores: [
    ({ output }) => ({ name: "checked_wic_auth", score: checkedWicAuth(output as RunState) }),
    ({ output }) => ({ name: "selected_applying_for_self", score: selectedApplyingForSelf(output as RunState) }),
    ({ output }) => ({ name: "mapped_gender_to_sex", score: mappedGenderToSex(output as RunState) }),
    ({ output }) => ({ name: "asked_mother_eligibility", score: askedMotherEligibility(output as RunState) }),
    ({ output }) => ({ name: "selected_blind_from_flag", score: selectedBlindFromFlag(output as RunState) }),
  ],
});
