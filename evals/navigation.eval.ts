import { Eval } from "braintrust";
import { generateText, stepCountIs, type ModelMessage } from "ai";
import { getWebAutomationSystemPrompt } from "@/lib/ai/prompts/web-automation";
import participants from "./datasets/participants.json";
import snapshots from "./datasets/snapshots.json";
import testCaseData from "./datasets/test-cases.json";
import { openai } from "@ai-sdk/openai";
import {
  createBaseStubTools,
  browserOk,
  collectTextResponses,
  type BaseRunState,
} from "./helpers";

/**
 * Navigation Eval
 *
 * Tests that the agent navigates multi-step or modal-driven flows
 * correctly — without opening extra tabs, pressing Back when stuck,
 * or leaving the current form page unintentionally.
 */

// ── Mock data ────────────────────────────────────────────────────────────

const mockParticipant = participants.davidChen;

// ── Page snapshots ──────────────────────────────────────────────────────

const snapshotLanding = snapshots.navigation.pages.landing;
const snapshotPage1Address = snapshots.navigation.pages.page1Address;
const snapshotModalBlocking = snapshots.navigation.pages.modalBlocking;
const snapshotCountyModal = snapshots.navigation.pages.countyModal;
const snapshotPage2Personal = snapshots.navigation.pages.page2Personal;
const snapshotPage3Income = snapshots.navigation.pages.page3Income;
const snapshotIncomeModal = snapshots.navigation.pages.incomeModal;
const snapshotPage3WithIncome = snapshots.navigation.pages.page3WithIncome;
const snapshotReview = snapshots.navigation.pages.review;

// ── Stateful tools ──────────────────────────────────────────────────────

type PageState =
  | "landing"
  | "page1_address"
  | "modal_county"
  | "page2_personal"
  | "page3_income"
  | "modal_income"
  | "page3_income_filled"
  | "review";

interface RunState extends BaseRunState {
  page: PageState;
  browserActions: Array<{ action: string; selector?: string; value?: string; script?: string; [key: string]: unknown }>;
  textResponses: string[];
  /** Tracks harmful navigation actions */
  usedBack: boolean;
  openedNewTab: boolean;
  navigatedAway: boolean;
}

function createStubTools(state: RunState) {
  return createBaseStubTools(state, {
    getApricotRecord: async () => mockParticipant,

    browser: async (input: Record<string, unknown>) => {
      const action = input.action as string;
      const selector = input.selector as string | undefined;
      const url = input.url as string | undefined;
      const script = input.script as string | undefined;

      state.browserActions.push(input as RunState["browserActions"][0]);

      // ── Track harmful navigation ──
      if (action === "back" || action === "forward") {
        state.usedBack = true;
      }
      if (action === "newTab" || action === "newPage") {
        state.openedNewTab = true;
      }
      // Clicking target="_blank" links
      if (
        action === "click" &&
        selector &&
        /@e1|@e2/i.test(selector) &&
        state.page === "landing"
      ) {
        state.openedNewTab = true;
        return {
          success: true,
          output: "(Opened a new tab — you have navigated away from the application)",
          error: null,
        };
      }
      // Navigate to a different domain while on the form
      if (
        action === "navigate" &&
        url &&
        state.page !== "landing" &&
        !/benefitscal/i.test(url)
      ) {
        state.navigatedAway = true;
        return browserOk();
      }

      // ── Snapshot logic ──
      if (action === "snapshot") {
        // Scoped snapshot for modals
        if (
          selector &&
          /dialog|modal|aria-modal/i.test(selector)
        ) {
          if (state.page === "modal_county") {
            return browserOk(snapshotCountyModal);
          }
          if (state.page === "modal_income") {
            return browserOk(snapshotIncomeModal);
          }
          return browserOk("(No modal found)");
        }

        // Full page snapshots
        switch (state.page) {
          case "landing":
            return browserOk(snapshotLanding);
          case "page1_address":
            return browserOk(snapshotPage1Address);
          case "modal_county":
            return browserOk(snapshotModalBlocking);
          case "page2_personal":
            return browserOk(snapshotPage2Personal);
          case "page3_income":
            return browserOk(snapshotPage3Income);
          case "modal_income":
            return browserOk(snapshotModalBlocking);
          case "page3_income_filled":
            return browserOk(snapshotPage3WithIncome);
          case "review":
            return browserOk(snapshotReview);
          default:
            return browserOk();
        }
      }

      // ── Click logic ──
      if (action === "click") {
        const sel = selector ?? "";

        // Landing: "Get Started"
        if (/@e3/i.test(sel) && state.page === "landing") {
          state.page = "page1_address";
          return browserOk();
        }

        // Page 1: "Next" → county modal appears
        if (/@e14|next/i.test(sel) && state.page === "page1_address") {
          state.page = "modal_county";
          return browserOk();
        }

        // County modal: "Continue" → page 2
        if (/@e22/i.test(sel) && state.page === "modal_county") {
          state.page = "page2_personal";
          return browserOk();
        }

        // County modal: "Cancel" → back to page 1
        if (/@e23/i.test(sel) && state.page === "modal_county") {
          state.page = "page1_address";
          return browserOk();
        }

        // Page 2: "Next" → page 3
        if (/@e36|next/i.test(sel) && state.page === "page2_personal") {
          state.page = "page3_income";
          return browserOk();
        }

        // Page 3: "Add Income Source" → income modal
        if (/@e41/i.test(sel) && (state.page === "page3_income" || state.page === "page3_income_filled")) {
          state.page = "modal_income";
          return browserOk();
        }

        // Income modal: "Save" → back to page 3 with income
        if (/@e53/i.test(sel) && state.page === "modal_income") {
          state.page = "page3_income_filled";
          return browserOk();
        }

        // Income modal: "Cancel" → back to page 3
        if (/@e54/i.test(sel) && state.page === "modal_income") {
          state.page = "page3_income";
          return browserOk();
        }

        // Page 3: "Next" → review
        if (/@e43|next/i.test(sel) && (state.page === "page3_income" || state.page === "page3_income_filled")) {
          state.page = "review";
          return browserOk();
        }

        // Back buttons on form pages
        if (/@e35|@e42|@e60|back/i.test(sel)) {
          state.usedBack = true;
          return browserOk();
        }

        return browserOk();
      }

      // ── Navigate ──
      if (action === "navigate") {
        state.page = "landing";
        return browserOk();
      }

      // ── Evaluate (for modal workarounds) ──
      if (action === "evaluate") {
        if (state.page === "modal_county" && script) {
          return browserOk(
            '<div role="dialog" aria-modal="true"><h2>Confirm Your County</h2><select id="countySelect"><option value="">Select a county...</option><option value="33">Riverside</option><option value="36">San Bernardino</option></select><button id="continueBtn">Continue</button></div>'
          );
        }
        if (state.page === "modal_income" && script) {
          return browserOk(
            '<div role="dialog" aria-modal="true"><h2>Add Income Source</h2><select id="incomeType"><option value="unemployment">Unemployment Benefits</option></select><input id="monthlyAmount" type="number"/><button id="saveBtn">Save</button></div>'
          );
        }
        return browserOk("No modal found");
      }

      // ── Select / Fill / Type ──
      if (action === "select" || action === "fill" || action === "type") {
        // County select inside modal — allow CSS selectors too
        if (
          selector &&
          /@e21|countySelect/i.test(selector) &&
          state.page === "modal_county"
        ) {
          return browserOk();
        }
        return browserOk();
      }

      // ── Wait ──
      if (action === "wait") {
        return browserOk();
      }

      // ── inputvalue ──
      if (action === "inputvalue") {
        return browserOk();
      }

      return browserOk();
    },
  });
}

// ── Test cases ──────────────────────────────────────────────────────────

const testCases = testCaseData.navigation.cases;

// ── Scoring helpers ──────────────────────────────────────────────────────

/** Agent clicked "Get Started" on landing page instead of external links */
function navigatedPastLanding(state: RunState): boolean {
  return state.browserActions.some(
    (a) => a.action === "click" && a.selector && /@e3/i.test(a.selector)
  );
}

/** Agent did NOT click the external links (Learn more / Check eligibility) */
function avoidedExternalLinks(state: RunState): boolean {
  return !state.browserActions.some(
    (a) =>
      a.action === "click" &&
      a.selector &&
      (/@e1|@e2/i.test(a.selector))
  );
}

/**
 * Agent recognized the county modal was blocking and used a scoped
 * snapshot or evaluate to discover it.
 */
function handledCountyModal(state: RunState): boolean {
  const scopedSnapshot = state.browserActions.some(
    (a) =>
      a.action === "snapshot" &&
      a.selector &&
      /dialog|modal|aria-modal/i.test(a.selector)
  );
  const usedEvaluate = state.browserActions.some(
    (a) => a.action === "evaluate" && a.script && /modal|dialog/i.test(a.script)
  );
  return scopedSnapshot || usedEvaluate;
}

/** Agent selected a county value inside the modal */
function selectedCounty(state: RunState): boolean {
  // Via select action
  const viaSelect = state.browserActions.some(
    (a) =>
      a.action === "select" &&
      a.selector &&
      /@e21|countySelect/i.test(a.selector)
  );
  // Via evaluate (React workaround)
  const viaEvaluate = state.browserActions.some(
    (a) =>
      a.action === "evaluate" &&
      a.script &&
      /countySelect|county/i.test(a.script) &&
      /value/i.test(a.script)
  );
  return viaSelect || viaEvaluate;
}

/** Agent dismissed the county modal by clicking Continue */
function dismissedCountyModal(state: RunState): boolean {
  const clickedContinue = state.browserActions.some(
    (a) =>
      a.action === "click" &&
      a.selector &&
      /@e22|continueBtn/i.test(a.selector)
  );
  const viaEvaluate = state.browserActions.some(
    (a) =>
      a.action === "evaluate" &&
      a.script &&
      /continueBtn|button/i.test(a.script) &&
      /click|mousedown/i.test(a.script)
  );
  return clickedContinue || viaEvaluate;
}

/** Agent handled the income modal (opened it, filled it, saved) */
function handledIncomeModal(state: RunState): boolean {
  const openedModal = state.browserActions.some(
    (a) => a.action === "click" && a.selector && /@e41/i.test(a.selector)
  );
  const savedModal = state.browserActions.some(
    (a) => a.action === "click" && a.selector && /@e53|saveBtn/i.test(a.selector)
  );
  return openedModal && savedModal;
}

/** Agent did NOT use the browser Back button */
function didNotUseBack(state: RunState): boolean {
  // Check for browser-level back (action: "back")
  const browserBack = state.browserActions.some((a) => a.action === "back");
  return !browserBack;
}

/** Agent did NOT open a new tab */
function didNotOpenNewTab(state: RunState): boolean {
  return !state.openedNewTab;
}

/** Agent did NOT navigate away from the application domain */
function stayedOnSite(state: RunState): boolean {
  return !state.navigatedAway;
}

/** Agent reached the review page */
function reachedReview(state: RunState): boolean {
  return state.page === "review";
}

/** Agent stopped at review — did not click Submit */
function stoppedAtReview(state: RunState): boolean {
  return !state.browserActions.some(
    (a) => a.action === "click" && a.selector && /@e61|submit/i.test(a.selector)
  );
}

// ── Eval ────────────────────────────────────────────────────────────────

const model = openai('gpt-5-mini');

Eval("labs-asp", {
  experimentName: "Navigation",
  data: () =>
    testCases.map((tc) => ({
      input: tc.input,
      expected: tc.name,
      metadata: { maxSteps: tc.maxSteps },
    })),

  task: async (input: string, { metadata }) => {
    const state: RunState = {
      page: "landing",
      toolCallLog: [],
      browserActions: [],
      textResponses: [],
      usedBack: false,
      openedNewTab: false,
      navigatedAway: false,
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
      name: "navigated_past_landing",
      score: navigatedPastLanding(output as RunState) ? 1 : 0,
    }),
    ({ output }) => ({
      name: "avoided_external_links",
      score: avoidedExternalLinks(output as RunState) ? 1 : 0,
    }),
    ({ output }) => ({
      name: "handled_county_modal",
      score: handledCountyModal(output as RunState) ? 1 : 0,
    }),
    ({ output }) => ({
      name: "selected_county",
      score: selectedCounty(output as RunState) ? 1 : 0,
    }),
    ({ output }) => ({
      name: "dismissed_county_modal",
      score: dismissedCountyModal(output as RunState) ? 1 : 0,
    }),
    ({ output }) => ({
      name: "handled_income_modal",
      score: handledIncomeModal(output as RunState) ? 1 : 0,
    }),
    ({ output }) => ({
      name: "did_not_use_back",
      score: didNotUseBack(output as RunState) ? 1 : 0,
    }),
    ({ output }) => ({
      name: "did_not_open_new_tab",
      score: didNotOpenNewTab(output as RunState) ? 1 : 0,
    }),
    ({ output }) => ({
      name: "stayed_on_site",
      score: stayedOnSite(output as RunState) ? 1 : 0,
    }),
    ({ output }) => ({
      name: "reached_review",
      score: reachedReview(output as RunState) ? 1 : 0,
    }),
    ({ output }) => ({
      name: "stopped_at_review",
      score: stoppedAtReview(output as RunState) ? 1 : 0,
    }),
  ],
});
