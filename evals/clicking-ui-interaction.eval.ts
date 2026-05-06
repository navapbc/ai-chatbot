import { Eval } from "braintrust";
import { generateText, stepCountIs, type ModelMessage } from "ai";
import { getWebAutomationSystemPrompt } from "@/lib/ai/prompts/web-automation";
import participants from "./datasets/participants.json";
import snapshots from "./datasets/snapshots.json";
import testCaseData from "./datasets/test-cases.json";
import { openai } from "@ai-sdk/openai";
import { createBaseStubTools, browserOk, collectTextResponses, type BaseRunState } from "./helpers";

/**
 * Clicking / UI Interaction Eval
 *
 * Tests that the agent successfully interacts with technically challenging
 * UI elements: date pickers, masked phone inputs, collapsible sections,
 * and dropdown menus.
 */

// ── Mock data ────────────────────────────────────────────────────────────

const mockParticipant = participants.anaReyes;

// ── Page snapshots simulating tricky UI elements ────────────────────────

const snapshotPage1 = snapshots.clickingUiInteraction.pages.page1;
const snapshotAfterExpandCollapsible = snapshots.clickingUiInteraction.pages.page1Expanded;
const snapshotSelect2Open = snapshots.clickingUiInteraction.pages.select2Open;
const snapshotPage2 = snapshots.clickingUiInteraction.pages.page2;
const snapshotReview = snapshots.clickingUiInteraction.pages.review;

// ── Stateful tools ──────────────────────────────────────────────────────

interface BrowserAction {
  action: string;
  selector?: string;
  value?: string;
  text?: string;
  clear?: boolean;
  url?: string;
}

interface RunState extends BaseRunState {
  currentPage: number;
  collapsibleExpanded: boolean;
  select2Open: boolean;
  browserActions: BrowserAction[];
  textResponses: string[];
}

function createStubTools(state: RunState) {
  return createBaseStubTools(state, {
    getApricotRecord: async () => mockParticipant,

    browser: async (input: Record<string, unknown>) => {
      const action = input.action as string;
      const selector = input.selector as string | undefined;
      const value = input.value as string | undefined;
      const text = input.text as string | undefined;
      const clear = input.clear as boolean | undefined;
      const url = input.url as string | undefined;

      state.browserActions.push({ action, selector, value, text, clear, url });

      // ── Snapshot logic ──
      if (action === "snapshot") {
        if (state.select2Open) {
          state.select2Open = false;
          return browserOk(snapshotSelect2Open);
        }
        if (state.currentPage === 1 && state.collapsibleExpanded) {
          return browserOk(snapshotAfterExpandCollapsible);
        }
        if (state.currentPage <= 1) {
          return browserOk(snapshotPage1);
        }
        if (state.currentPage === 2) {
          return browserOk(snapshotPage2);
        }
        return browserOk(snapshotReview);
      }

      // ── Click logic ──
      if (action === "click") {
        // Expand collapsible section
        if (selector && /@e9/i.test(selector)) {
          state.collapsibleExpanded = true;
          return browserOk();
        }
        // Open Select2 dropdown for language
        if (
          selector &&
          (/@e7|@e8/i.test(selector) || /select2|language/i.test(selector))
        ) {
          state.select2Open = true;
          return browserOk();
        }
        // Select a Select2 option
        if (selector && /@e3[0-6]/i.test(selector)) {
          state.select2Open = false;
          return browserOk();
        }
        // Next button
        if (selector && /@e10|@e47|next|continue/i.test(selector)) {
          state.currentPage++;
          return browserOk();
        }
        return browserOk();
      }

      // ── inputvalue verification ──
      if (action === "inputvalue") {
        if (selector && /@e3|dobInput/i.test(selector)) {
          return browserOk("06/18/1992");
        }
        if (selector && /@e4|phoneInput/i.test(selector)) {
          return browserOk("(909) 555-0377");
        }
        if (selector && /@e5|ssnInput/i.test(selector)) {
          return browserOk("***-**-8899");
        }
        if (selector && /@e21|emergencyPhone/i.test(selector)) {
          return browserOk("(909) 555-0411");
        }
        if (selector && /@e42|stateTxt/i.test(selector)) {
          return browserOk("CA");
        }
        return browserOk();
      }

      // ── select on native dropdown ──
      if (action === "select") {
        return browserOk();
      }

      // ── fill/type ──
      if (action === "fill" || action === "type") {
        return browserOk();
      }

      // ── wait ──
      if (action === "wait") {
        return browserOk();
      }

      // ── navigate ──
      if (action === "navigate") {
        state.currentPage = 1;
        return browserOk();
      }

      return browserOk();
    },
  });
}

// ── Test cases ──────────────────────────────────────────────────────────

const testCases = testCaseData.clickingUiInteraction.cases;

// ── Scoring helpers ──────────────────────────────────────────────────────

/**
 * Date field: Agent should use `type` (not `fill`) for masked date input,
 * and the value should be digits-only or properly formatted.
 */
function handledDateField(state: RunState): boolean {
  const dateActions = state.browserActions.filter(
    (a) =>
      (a.action === "type" || a.action === "fill") &&
      a.selector &&
      /@e3|dobInput/i.test(a.selector)
  );
  // Must have used `type` at least once (not just `fill`)
  const usedType = dateActions.some((a) => a.action === "type");
  // Value should contain date digits
  const hasDateValue = dateActions.some(
    (a) => (a.text ?? a.value ?? "").replace(/\D/g, "").length >= 6
  );
  return usedType && hasDateValue;
}

/**
 * Phone field: Agent should use `type` (not `fill`) for masked phone input,
 * with digits-only value respecting maxlength=10.
 */
function handledPhoneField(state: RunState): boolean {
  const phoneActions = state.browserActions.filter(
    (a) =>
      (a.action === "type" || a.action === "fill") &&
      a.selector &&
      /@e4|phoneInput/i.test(a.selector)
  );
  const usedType = phoneActions.some((a) => a.action === "type");
  const hasPhoneValue = phoneActions.some((a) => {
    const digits = (a.text ?? a.value ?? "").replace(/\D/g, "");
    return digits.length >= 7 && digits.length <= 10;
  });
  return usedType && hasPhoneValue;
}

/**
 * Native dropdown: Agent should use `select` action for the gender dropdown.
 */
function handledNativeDropdown(state: RunState): boolean {
  const selectActions = state.browserActions.filter(
    (a) =>
      a.action === "select" &&
      a.selector &&
      /@e6|genderDrpDwn/i.test(a.selector)
  );
  return selectActions.length > 0;
}

/**
 * Custom dropdown (Select2): Agent should click to open, then select an
 * option from the rendered list — NOT use the `select` action.
 */
function handledSelect2Dropdown(state: RunState): boolean {
  // Must have clicked the Select2 trigger (@e7 or @e8)
  const clickedTrigger = state.browserActions.some(
    (a) =>
      a.action === "click" &&
      a.selector &&
      (/@e7|@e8/i.test(a.selector) || /select2|language/i.test(a.selector))
  );
  // Must have clicked an option from the Select2 list (@e31-@e36)
  const clickedOption = state.browserActions.some(
    (a) =>
      a.action === "click" &&
      a.selector &&
      /@e3[0-6]/i.test(a.selector)
  );
  return clickedTrigger && clickedOption;
}

/**
 * Collapsible section: Agent should have clicked to expand the collapsed
 * "Additional Information" section.
 */
function expandedCollapsibleSection(state: RunState): boolean {
  return state.browserActions.some(
    (a) =>
      a.action === "click" &&
      a.selector &&
      (/@e9/i.test(a.selector) || /additional|collaps/i.test(a.selector))
  );
}

/**
 * Verification: Agent should use `inputvalue` to verify masked fields
 * after typing into them.
 */
function verifiedMaskedFields(state: RunState): boolean {
  const verifications = state.browserActions.filter(
    (a) =>
      a.action === "inputvalue" &&
      a.selector &&
      /@e3|@e4|@e5|dobInput|phoneInput|ssnInput/i.test(a.selector)
  );
  // Should verify at least 1 masked field
  return verifications.length >= 1;
}

/**
 * Filled fields in the expanded collapsible section (emergency contact, email, ethnicity).
 */
function filledCollapsibleFields(state: RunState): boolean {
  const collapsibleFills = state.browserActions.filter(
    (a) =>
      (a.action === "fill" || a.action === "type" || a.action === "select") &&
      a.selector &&
      /@e2[0-3]|emergencyName|emergencyPhone|emailTxt|ethnicityDrpDwn/i.test(a.selector)
  );
  // Should have filled at least 2 fields inside the collapsible
  return collapsibleFills.length >= 2;
}

// ── Eval ────────────────────────────────────────────────────────────────

const model = openai('gpt-5-mini');

Eval("labs-asp", {
  experimentName: "Clicking / UI Interaction",
  data: () =>
    testCases.map((tc) => ({
      input: tc.input,
      expected: tc.name,
      metadata: { maxSteps: tc.maxSteps },
    })),

  task: async (input: string, { metadata }) => {
    const state: RunState = {
      currentPage: 0,
      collapsibleExpanded: false,
      select2Open: false,
      toolCallLog: [],
      browserActions: [],
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

    return state;
  },

  scores: [
    ({ output }) => ({
      name: "handled_date_field",
      score: handledDateField(output as RunState) ? 1 : 0,
    }),
    ({ output }) => ({
      name: "handled_phone_field",
      score: handledPhoneField(output as RunState) ? 1 : 0,
    }),
    ({ output }) => ({
      name: "handled_native_dropdown",
      score: handledNativeDropdown(output as RunState) ? 1 : 0,
    }),
    ({ output }) => ({
      name: "handled_select2_dropdown",
      score: handledSelect2Dropdown(output as RunState) ? 1 : 0,
    }),
    ({ output }) => ({
      name: "expanded_collapsible_section",
      score: expandedCollapsibleSection(output as RunState) ? 1 : 0,
    }),
    ({ output }) => ({
      name: "verified_masked_fields",
      score: verifiedMaskedFields(output as RunState) ? 1 : 0,
    }),
    ({ output }) => ({
      name: "filled_collapsible_fields",
      score: filledCollapsibleFields(output as RunState) ? 1 : 0,
    }),
  ],
});
