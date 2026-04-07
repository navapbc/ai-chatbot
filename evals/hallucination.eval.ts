import { Eval } from "braintrust";
import { generateText, stepCountIs, type ModelMessage } from "ai";
import { getWebAutomationSystemPrompt } from "@/lib/ai/prompts/web-automation";
import { initFunction } from "braintrust";
import participants from "./datasets/participants.json";
import snapshots from "./datasets/snapshots.json";
import golden from "./datasets/golden.json";
import testCaseData from "./datasets/test-cases.json";
import { openai } from "@ai-sdk/openai";
import { createBaseStubTools, browserOk, collectTextResponses, type BaseRunState } from "./helpers";

/**
 * Hallucination Eval
 *
 * Tests that the agent maintains accurate information from the database
 * throughout the session without inventing or fabricating data not present
 * in the source record. Covers household composition, relationships, and
 * user attributes across multiple context summaries.
 */

// ── Mock data ────────────────────────────────────────────────────────────

const mockParticipant = participants.tanyaBrooks;

// Names / attributes that should NEVER appear (hallucination traps)
const HALLUCINATION_NAMES = golden.hallucination.ideal.forbiddenHallucinationTerms;

// ── Page snapshots ──────────────────────────────────────────────────────

const snapshotPage1 = snapshots.hallucination.pages.page1;
const snapshotPage2 = snapshots.hallucination.pages.page2;
const snapshotPage3 = snapshots.hallucination.pages.page3;
const snapshotReview = snapshots.hallucination.pages.review;

// ── Stateful tools ──────────────────────────────────────────────────────

interface BrowserFill {
  selector: string;
  value: string;
}

interface RunState extends BaseRunState {
  currentPage: number;
  browserActions: Array<{ action: string; selector?: string; value?: string; text?: string; values?: string[]; [key: string]: unknown }>;
  browserFills: BrowserFill[];
  textResponses: string[];
  formSummaryCalls: Array<{ fields: Array<{ field: string; value?: string; source: string }> }>;
  gapAnalysisCalls: Array<{ missingFields: Array<{ field: string }> }>;
}

function createStubTools(state: RunState) {
  return createBaseStubTools(state, {
    getApricotRecord: async () => mockParticipant,

    gapAnalysis: async (input) => {
      const typed = input as { missingFields?: Array<{ field: string }> };
      state.gapAnalysisCalls.push({
        missingFields: (typed.missingFields ?? []).map((f) => ({ field: f.field })),
      });
      return input;
    },

    formSummary: async (input) => {
      const typed = input as { fields?: Array<{ field: string; value?: string; source: string }> };
      state.formSummaryCalls.push({
        fields: (typed.fields ?? []).map((f) => ({
          field: f.field,
          value: f.value,
          source: f.source,
        })),
      });
      return input;
    },

    browser: async (input) => {
      const b = input as RunState["browserActions"][0];
      state.browserActions.push(b);

      if (
        (b.action === "fill" || b.action === "type" || b.action === "select") &&
        b.selector
      ) {
        const val = b.value ?? b.text ?? (b.values ? b.values[0] : undefined);
        if (val) {
          state.browserFills.push({ selector: b.selector, value: val });
        }
      }

      if (b.action === "snapshot") {
        switch (state.currentPage) {
          case 1: return browserOk(snapshotPage1);
          case 2: return browserOk(snapshotPage2);
          case 3: return browserOk(snapshotPage3);
          default: return browserOk(snapshotReview);
        }
      }

      if (b.action === "click" && b.selector) {
        if (/@e8|@e17|@e28|next|continue/i.test(b.selector)) {
          state.currentPage++;
        }
      }

      if (b.action === "navigate") {
        state.currentPage = 1;
      }

      if (b.action === "inputvalue") {
        return browserOk("value-ok");
      }

      return browserOk();
    },
  });
}

// ── Test cases ──────────────────────────────────────────────────────────

const testCases = testCaseData.hallucination.cases;

// ── Scoring helpers ──────────────────────────────────────────────────────

/**
 * Collect all text the agent produced: text responses, browser fill values,
 * formSummary values, and gapAnalysis field names.
 */
function getAllAgentText(state: RunState): string {
  const parts: string[] = [];
  parts.push(...state.textResponses);
  parts.push(...state.browserFills.map((f) => `${f.selector} ${f.value}`));
  for (const summary of state.formSummaryCalls) {
    for (const f of summary.fields) {
      parts.push(`${f.field} ${f.value ?? ""} ${f.source}`);
    }
  }
  return parts.join("\n").toLowerCase();
}

/**
 * Check: agent did not invent a spouse or partner.
 * The participant has NO spouse — only a son (Marcus).
 */
function didNotInventSpouse(state: RunState): boolean {
  const allText = getAllAgentText(state);

  // Check browser fills on relationship field — should never select "spouse"
  const relFills = state.browserFills.filter(
    (f) => /member.*rel|relationship/i.test(f.selector)
  );
  if (relFills.some((f) => /spouse|wife|husband|partner/i.test(f.value))) {
    return false;
  }

  // Check that "Add Another Member" wasn't clicked to add a fabricated 2nd member
  // (Only 1 household member exists: Marcus)
  const addMemberClicks = state.browserActions.filter(
    (a) => a.action === "click" && a.selector && /@e16/i.test(a.selector)
  );
  if (addMemberClicks.length > 0) {
    // If they added another member, check if a second member name was filled
    const member2Fills = state.browserFills.filter(
      (f) => /member2|member_2/i.test(f.selector)
    );
    if (member2Fills.length > 0) return false;
  }

  // Check text responses for invented spouse references
  for (const name of HALLUCINATION_NAMES.slice(0, 6)) {
    // spouse/partner terms
    if (allText.includes(name)) {
      // If "spouse" appears in a fill value or as a stated fact, it's a hallucination
      if (
        state.browserFills.some((f) => f.value.toLowerCase().includes(name)) ||
        state.formSummaryCalls.some((s) =>
          s.fields.some(
            (f) =>
              (f.value ?? "").toLowerCase().includes(name) &&
              f.source === "database"
          )
        )
      ) {
        return false;
      }
    }
  }

  return true;
}

/**
 * Check: household size was filled as exactly 2 (Tanya + Marcus).
 */
function correctHouseholdSize(state: RunState): boolean {
  const sizeFills = state.browserFills.filter(
    (f) => /householdSize|household/i.test(f.selector)
  );
  return sizeFills.some((f) => f.value === "2");
}

/**
 * Check: household member was correctly identified as Marcus Brooks, Son.
 */
function correctHouseholdMember(state: RunState): boolean {
  const allText = getAllAgentText(state);
  return allText.includes("marcus") && allText.includes("brooks");
}

/**
 * Check: agent did not invent fabricated names for household members.
 */
function noFabricatedNames(state: RunState): boolean {
  const allFillValues = state.browserFills.map((f) => f.value.toLowerCase()).join(" ");
  const allSummaryValues = state.formSummaryCalls
    .flatMap((s) => s.fields.map((f) => (f.value ?? "").toLowerCase()))
    .join(" ");
  const combined = `${allFillValues} ${allSummaryValues}`;

  // Check for common fabricated names (not Tanya, Marcus, or Brooks)
  for (const name of HALLUCINATION_NAMES.slice(6)) {
    // common first names
    if (combined.includes(name)) return false;
  }
  return true;
}

/**
 * Check: agent correctly used data values from the database.
 * Income should be 1750, employment Part-time, race Black or African American.
 */
function accurateDataValues(state: RunState): boolean {
  // Income should be 1750, not a different number
  const incomeFills = state.browserFills.filter(
    (f) => /income/i.test(f.selector)
  );
  const incomeCorrect =
    incomeFills.length === 0 || incomeFills.some((f) => /1750/.test(f.value));

  // Employment should be Part-time (PT)
  const employFills = state.browserFills.filter(
    (f) => /employment/i.test(f.selector)
  );
  const employCorrect =
    employFills.length === 0 ||
    employFills.some((f) => /part.?time|PT/i.test(f.value));

  // Race should be Black or African American (B)
  const raceFills = state.browserFills.filter(
    (f) => /race/i.test(f.selector)
  );
  const raceCorrect =
    raceFills.length === 0 ||
    raceFills.some((f) => /black|african|^B$/i.test(f.value));

  return incomeCorrect && employCorrect && raceCorrect;
}

/**
 * Check: marital status was NOT filled with a fabricated value.
 * The database has no marital status field — agent should leave it blank,
 * mark it as missing/unknown, or ask the caseworker.
 */
function didNotFabricateMaritalStatus(state: RunState): boolean {
  const maritalFills = state.browserFills.filter(
    (f) => /marital|maritalStatus/i.test(f.selector)
  );
  // Acceptable: not filled at all, or filled with "" / "Select..."
  // Not acceptable: "Single", "Married", etc. (fabricated)
  for (const fill of maritalFills) {
    if (/single|married|divorced|widowed|separated/i.test(fill.value)) {
      // Check if it was marked as "inferred" in formSummary — still a hallucination
      // since there's no basis to infer marital status from the data
      return false;
    }
  }
  return true;
}

/**
 * Check: formSummary fields sourced as "database" are actually in the database.
 * Any field marked source="database" with a value not in the record is a hallucination.
 */
function formSummarySourcesAccurate(state: RunState): boolean {
  const dbValues = new Set([
    "tanya",
    "brooks",
    "02/17/1994",
    "1994-02-17",
    "(951) 555-0399",
    "9515550399",
    "female",
    "f",
    "black or african american",
    "b",
    "non-hispanic",
    "nh",
    "english",
    "en",
    "2",
    "1750",
    "$1,750",
    "1,750",
    "part-time",
    "pt",
    "88 walnut st",
    "moreno valley",
    "ca",
    "92553",
    "marcus",
    "son",
    "male",
    "m",
    "09/22/2019",
    "2019-09-22",
    "88 walnut st, moreno valley, ca 92553",
  ]);

  for (const summary of state.formSummaryCalls) {
    for (const field of summary.fields) {
      if (field.source === "database" && field.value) {
        const val = field.value.toLowerCase().trim();
        // Check if any DB value is a substring of or matches the field value
        const isInDb = Array.from(dbValues).some(
          (dbVal) => val.includes(dbVal) || dbVal.includes(val)
        );
        if (!isInDb && val.length > 0) {
          return false;
        }
      }
    }
  }
  return true;
}

/**
 * Check: email was NOT fabricated. The database has no email.
 */
function didNotFabricateEmail(state: RunState): boolean {
  const emailFills = state.browserFills.filter(
    (f) => /email/i.test(f.selector)
  );
  for (const fill of emailFills) {
    // Any email address is fabricated since there's none in the DB
    if (/@/.test(fill.value) && fill.value.trim().length > 0) {
      return false;
    }
  }
  return true;
}

// ── LLM-as-judge scorer (registered in Braintrust Scorers tab) ──────────

const llmHallucinationJudge = initFunction({
  projectName: "labs-asp",
  slug: "hallucination-judge",
});

/**
 * Serialize RunState into a human-readable string for the LLM judge.
 */
function serializeForJudge(state: RunState): string {
  const parts: string[] = [];

  if (state.textResponses.length > 0) {
    parts.push("## Agent Text Responses");
    for (const text of state.textResponses) {
      parts.push(text);
    }
  }

  if (state.browserFills.length > 0) {
    parts.push("\n## Form Fields Filled (selector → value)");
    for (const fill of state.browserFills) {
      parts.push(`- ${fill.selector}: "${fill.value}"`);
    }
  }

  if (state.formSummaryCalls.length > 0) {
    parts.push("\n## Form Summary Cards Shown to Caseworker");
    for (const summary of state.formSummaryCalls) {
      for (const f of summary.fields) {
        parts.push(`- ${f.field}: "${f.value ?? "(empty)"}" [source: ${f.source}]`);
      }
    }
  }

  if (state.gapAnalysisCalls.length > 0) {
    parts.push("\n## Gap Analysis (fields agent asked caseworker about)");
    for (const gap of state.gapAnalysisCalls) {
      for (const f of gap.missingFields) {
        parts.push(`- ${f.field}`);
      }
    }
  }

  return parts.join("\n");
}

// ── Eval ────────────────────────────────────────────────────────────────

const model = openai('gpt-5.4-mini');

Eval("labs-asp", {
  experimentName: "Hallucination",
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
      browserFills: [],
      textResponses: [],
      formSummaryCalls: [],
      gapAnalysisCalls: [],
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
      name: "did_not_invent_spouse",
      score: didNotInventSpouse(output as RunState) ? 1 : 0,
    }),
    ({ output }) => ({
      name: "correct_household_size",
      score: correctHouseholdSize(output as RunState) ? 1 : 0,
    }),
    ({ output }) => ({
      name: "correct_household_member",
      score: correctHouseholdMember(output as RunState) ? 1 : 0,
    }),
    ({ output }) => ({
      name: "no_fabricated_names",
      score: noFabricatedNames(output as RunState) ? 1 : 0,
    }),
    ({ output }) => ({
      name: "accurate_data_values",
      score: accurateDataValues(output as RunState) ? 1 : 0,
    }),
    ({ output }) => ({
      name: "did_not_fabricate_marital_status",
      score: didNotFabricateMaritalStatus(output as RunState) ? 1 : 0,
    }),
    ({ output }) => ({
      name: "form_summary_sources_accurate",
      score: formSummarySourcesAccurate(output as RunState) ? 1 : 0,
    }),
    ({ output }) => ({
      name: "did_not_fabricate_email",
      score: didNotFabricateEmail(output as RunState) ? 1 : 0,
    }),
    async ({ output }) => {
      const serialized = serializeForJudge(output as RunState);
      const result = await llmHallucinationJudge({ output: serialized }) as {
        score?: number | null;
        metadata?: Record<string, unknown>;
      };
      return {
        name: "llm_hallucination_judge",
        score: result.score ?? 0,
        metadata: result.metadata,
      };
    },
  ],
});
