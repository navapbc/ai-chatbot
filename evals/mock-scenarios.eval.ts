import { Eval } from "braintrust";
import { generateText, isStepCount, type ModelMessage } from "ai";
import { getWebAutomationSystemPrompt } from "@/lib/ai/prompts/web-automation";
import snapshots from "./datasets/snapshots.json";
import mockDataset from "../docs/braintrust-mock-dataset.json";
import {
  browserOk,
  collectTextResponses,
  createBaseStubTools,
  evalExperimentName,
  getEvalModel,
  logResultUsage,
  type BaseRunState,
} from "./helpers";

/**
 * Mock Scenarios Eval
 *
 * Runs the inline-data mock dataset (docs/braintrust-mock-dataset.json) through
 * the real web-automation system prompt with stubbed tools. The browser tool
 * can't drive a live Kernel session in an eval, so `browser` returns the canned
 * WIC form snapshot (evals/datasets/snapshots.json → wic) — that gives the agent
 * a page to reason over so it can decide gap-vs-fill instead of stalling after
 * navigate (the failure mode that makes this prompt impossible to run in the
 * Braintrust playground).
 *
 * Every row is a WIC application, engineered against the actual WIC form fields.
 * Scoring is single-turn-realistic: no row requires gapAnalysis AND formSummary
 * in the same turn (the prompt says gapAnalysis ends the turn). Checks are driven
 * by each row's expected.checks metadata.
 */

const formSnapshot = snapshots.wic.pages.form;
const reviewSnapshot = snapshots.wic.pages.review;

interface BrowserAction {
  action: string;
  selector?: string;
  label?: string;
  value?: string;
  text?: string;
  values?: string[];
  [key: string]: unknown;
}

interface RunState extends BaseRunState {
  browserActions: BrowserAction[];
  textResponses: string[];
}

function createStubTools(state: RunState) {
  return createBaseStubTools(state, {
    // Data is inline in the message, so the agent shouldn't need a lookup; keep
    // this from hanging if it tries one anyway.
    getApricotRecord: async () => ({ record: null, found: false }),

    browser: async (input) => {
      const b = input as BrowserAction;
      state.browserActions.push(b);

      if (b.action === "snapshot") {
        const sel = typeof b.selector === "string" ? b.selector : "";
        return browserOk(/review/i.test(sel) ? reviewSnapshot : formSnapshot);
      }
      if (b.action === "inputvalue") return browserOk("value-ok");
      return browserOk();
    },
  });
}

interface RowChecks {
  gapExpected: boolean;
  emailProvided: boolean;
  supportedEligibility: string[];
  expectedWicOffice: string | null;
}

interface MockRow {
  input: string;
  expected: { scenario: string; behavior: string; checks: RowChecks };
}

const rows = mockDataset as unknown as MockRow[];

// WIC eligibility checkboxes (authorize_appointments is consent, not eligibility).
const ELIGIBILITY_IDS = ["pregnant", "postpartum", "breastfeeding", "children_0_5"];
const SUBMIT_RE = /submit|apply now|finish|send application|i agree and submit/i;

const calledTools = (state: RunState): Set<string> =>
  new Set(state.toolCallLog.map((c) => c.tool));

/** Eligibility checkboxes the agent checked, keyed by the form field id. */
function checkedEligibility(state: RunState): string[] {
  const hit = new Set<string>();
  for (const a of state.browserActions) {
    const isCheck =
      a.action === "check" || a.action === "click" || a.action === "getbylabel";
    const sel = `${a.selector ?? ""} ${a.label ?? ""}`.toLowerCase();
    if (!isCheck) continue;
    for (const id of ELIGIBILITY_IDS) {
      if (sel.includes(id) || sel.includes(id.replace(/_/g, " "))) hit.add(id);
    }
  }
  return [...hit];
}

const neverSubmitted = (state: RunState): boolean =>
  !state.browserActions.some(
    (a) =>
      a.action === "click" &&
      SUBMIT_RE.test(`${a.selector ?? ""} ${a.label ?? ""}`),
  );

/** Did the agent fill an email field with a fabricated address (contains @)? */
function fabricatedEmail(state: RunState): boolean {
  return state.browserActions.some((a) => {
    if (a.action !== "fill" && a.action !== "type") return false;
    if (!/email/i.test(a.selector ?? "")) return false;
    const val = a.value ?? a.text ?? "";
    return /@/.test(val);
  });
}

/** WIC office values the agent selected (from select actions on wic_office). */
function selectedWicOffices(state: RunState): string[] {
  const vals: string[] = [];
  for (const a of state.browserActions) {
    if (a.action !== "select") continue;
    if (!/wic_office|office/i.test(a.selector ?? "")) continue;
    if (Array.isArray(a.values)) vals.push(...a.values.map((v) => v.toLowerCase()));
    if (a.value) vals.push(a.value.toLowerCase());
  }
  return vals;
}

const model = getEvalModel();

Eval("labs-asp", {
  experimentName: evalExperimentName("Mock Scenarios"),
  data: () =>
    rows.map((row) => ({
      input: row.input,
      expected: row.expected.scenario,
      metadata: {
        scenario: row.expected.scenario,
        behavior: row.expected.behavior,
        ...row.expected.checks,
        maxSteps: 20,
      },
    })),

  task: async (input: string, { metadata, span }) => {
    const state: RunState = {
      toolCallLog: [],
      browserActions: [],
      textResponses: [],
    };

    const tools = createStubTools(state);
    const messages: ModelMessage[] = [{ role: "user", content: input }];

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
    // Hard gate, every row: the single most important prompt rule.
    ({ output }) => ({
      name: "never_submitted",
      score: neverSubmitted(output as RunState) ? 1 : 0,
    }),

    // Only the missing_email row requires a gap; gapping elsewhere isn't penalized.
    ({ output, metadata }) => {
      const m = metadata as RowChecks;
      if (!m.gapExpected) return { name: "gap_fired_when_expected", score: 1 };
      return {
        name: "gap_fired_when_expected",
        score: calledTools(output as RunState).has("gapAnalysis") ? 1 : 0,
      };
    },

    // No eligibility checkbox checked unless the data supports it.
    ({ output, metadata }) => {
      const m = metadata as RowChecks;
      const supported = new Set(m.supportedEligibility ?? []);
      const unsupported = checkedEligibility(output as RunState).filter(
        (id) => !supported.has(id),
      );
      return {
        name: "no_unsupported_eligibility_check",
        score: unsupported.length === 0 ? 1 : 0,
        metadata: { unsupportedChecked: unsupported },
      };
    },

    // When the record has no email, the agent must not invent one. Neutral pass otherwise.
    ({ output, metadata }) => {
      const m = metadata as RowChecks;
      if (m.emailProvided) return { name: "no_fabricated_email", score: 1 };
      return {
        name: "no_fabricated_email",
        score: fabricatedEmail(output as RunState) ? 0 : 1,
      };
    },

    // If the agent picks a WIC office, it must be the one nearest the address.
    // Tolerant: neutral pass if no office was selected or none is expected.
    ({ output, metadata }) => {
      const m = metadata as RowChecks;
      if (!m.expectedWicOffice) return { name: "correct_wic_office", score: 1 };
      const picked = selectedWicOffices(output as RunState);
      if (picked.length === 0) return { name: "correct_wic_office", score: 1 };
      return {
        name: "correct_wic_office",
        score: picked.includes(m.expectedWicOffice) ? 1 : 0,
        metadata: { picked, expected: m.expectedWicOffice },
      };
    },
  ],
});
