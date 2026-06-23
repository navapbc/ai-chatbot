import { Eval } from "braintrust";
import { generateText, stepCountIs, type ModelMessage } from "ai";
import { getWebAutomationSystemPrompt } from "@/lib/ai/prompts/web-automation";
import participants from "./datasets/participants.json";
import testCaseData from "./datasets/test-cases.json";
import {
  createBaseStubTools,
  browserOk,
  collectTextResponses,
  evalExperimentName,
  getEvalModel,
  emptyUsage,
  addUsage,
  logUsageAndCost,
  type BaseRunState,
} from "./helpers";

/**
 * Session Carryover Eval
 *
 * Multi-turn rubric from cwilkes-npbc/AI-Evaluations (steps 9, 10, 25, 38, 46, 52).
 * The agent must persist the participant identity across sequential user
 * messages — no re-asking "which user" when the next message says "now fill
 * out X", and Q&A about the participant ("How old is Rosa?") answered from
 * existing context rather than re-fetching or asking the caseworker.
 */

// ── Participant lookup ──────────────────────────────────────────────────

const PARTICIPANTS_BY_ID = {
  339688: participants.rosaFlores,
  339702: participants.carolinaDelgado,
} as const;

// ── State ───────────────────────────────────────────────────────────────

interface TurnRecord {
  userMessage: string;
  text: string[];
  toolCallsBefore: number;
  toolCallsAfter: number;
}

interface RunState extends BaseRunState {
  scenario: string;
  expectedParticipantId: number;
  expectedAnswer: string | null;
  turns: TurnRecord[];
}

const FORM_REGISTRY: Record<number, string> = {
  501: "WIC Application",
  502: "IHSS Application",
  503: "BenefitsCal Application",
};

function createStubTools(state: RunState) {
  return createBaseStubTools(state, {
    getApricotRecord: async ({ recordId }) => {
      const match = PARTICIPANTS_BY_ID[recordId as keyof typeof PARTICIPANTS_BY_ID];
      return match ?? { record: null, found: false };
    },

    getApricotForms: async () => ({
      forms: Object.entries(FORM_REGISTRY).map(([id, name]) => ({ id: Number(id), name })),
      count: Object.keys(FORM_REGISTRY).length,
      success: true,
    }),

    getApricotForm: async ({ formId }) => {
      const name = FORM_REGISTRY[formId];
      if (!name) return { form: null, found: false };
      return { form: { id: formId, name }, found: true };
    },

    getApricotFormFields: async () => ({
      fields: [
        { id: 1, label: "First Name", type: "text" },
        { id: 2, label: "Last Name", type: "text" },
        { id: 3, label: "Date of Birth", type: "date" },
        { id: 4, label: "Address", type: "text" },
        { id: 5, label: "Phone", type: "text" },
      ],
      count: 5,
      success: true,
    }),

    browser: async (input) => {
      const action = input.action as string;
      if (action === "snapshot") {
        return browserOk("[generic application page — fields collapsed; this eval only scores cross-turn behavior]");
      }
      return browserOk();
    },
  });
}

// ── Scoring helpers ─────────────────────────────────────────────────────

const RE_ASK_FOR_USER =
  /which\s+(user|client|participant|record)|who\s+(is|are)\s+(the|we|this|that)|what(?:'s|\s+is)\s+(?:the|their)\s+(?:name|first\s+name|last\s+name)|whose\s+(record|profile|file)|do\s+you\s+have\s+a\s+(record|client)|please\s+provide.*(name|id|record)/i;

function followingTurns(state: RunState): TurnRecord[] {
  return state.turns.slice(1);
}

function followingTurnText(state: RunState): string {
  return followingTurns(state).flatMap((t) => t.text).join(" ").toLowerCase();
}

function finalTurnText(state: RunState): string {
  if (state.turns.length === 0) return "";
  return state.turns[state.turns.length - 1].text.join(" ");
}

function lookedUpOtherParticipant(state: RunState): boolean {
  const calls = state.toolCallLog.filter((c) => c.tool === "getApricotRecord");
  return calls.some((c) => {
    const id = (c.args as { recordId?: number }).recordId;
    return typeof id === "number" && id !== state.expectedParticipantId;
  });
}

function sameUserAcrossTurns(state: RunState): number | null {
  if (state.scenario !== "same_user_carryover") return null;
  if (lookedUpOtherParticipant(state)) return 0;
  if (RE_ASK_FOR_USER.test(followingTurnText(state))) return 0;
  return 1;
}

function didNotReaskForUser(state: RunState): number | null {
  if (state.turns.length < 2) return null;
  return RE_ASK_FOR_USER.test(followingTurnText(state)) ? 0 : 1;
}

function answeredAge(state: RunState): number | null {
  if (state.scenario !== "qa_age") return null;
  const expected = state.expectedAnswer ?? "37";
  const pattern = new RegExp(`\\b${expected}\\b`);
  return pattern.test(finalTurnText(state)) ? 1 : 0;
}

function answeredLastName(state: RunState): number | null {
  if (state.scenario !== "qa_last_name") return null;
  const expected = (state.expectedAnswer ?? "Flores").toLowerCase();
  return finalTurnText(state).toLowerCase().includes(expected) ? 1 : 0;
}

// ── Eval ────────────────────────────────────────────────────────────────

const model = getEvalModel();
const testCases = testCaseData.sessionCarryover.cases;

Eval("labs-asp", {
  experimentName: evalExperimentName("Session Carryover"),
  data: () =>
    testCases.map((tc) => ({
      input: tc.turns[0],
      expected: tc.name,
      metadata: {
        scenario: tc.scenario,
        participantId: tc.participantId,
        turns: tc.turns,
        maxStepsPerTurn: tc.maxStepsPerTurn,
        expectedAnswer: (tc as { expectedAnswer?: string }).expectedAnswer ?? null,
      },
    })),

  task: async (_input, { metadata, span }) => {
    const meta = metadata as {
      scenario: string;
      participantId: number;
      turns: string[];
      maxStepsPerTurn: number;
      expectedAnswer: string | null;
    };

    const state: RunState = {
      scenario: meta.scenario,
      expectedParticipantId: meta.participantId,
      expectedAnswer: meta.expectedAnswer,
      toolCallLog: [],
      turns: [],
    };

    const tools = createStubTools(state);
    const messages: ModelMessage[] = [];
    const usage = emptyUsage();

    for (const userMsg of meta.turns) {
      const toolCallsBefore = state.toolCallLog.length;
      messages.push({ role: "user", content: userMsg });

      const result = await generateText({
        model,
        system: getWebAutomationSystemPrompt(),
        messages,
        tools,
        stopWhen: stepCountIs(meta.maxStepsPerTurn),
      });
      addUsage(usage, result);

      messages.push(...result.response.messages);

      state.turns.push({
        userMessage: userMsg,
        text: collectTextResponses(result.steps),
        toolCallsBefore,
        toolCallsAfter: state.toolCallLog.length,
      });
    }

    logUsageAndCost(span, usage);
    return state;
  },

  scores: [
    ({ output }) => ({ name: "same_user_across_turns", score: sameUserAcrossTurns(output as RunState) }),
    ({ output }) => ({ name: "did_not_reask_for_user", score: didNotReaskForUser(output as RunState) }),
    ({ output }) => ({ name: "answered_age", score: answeredAge(output as RunState) }),
    ({ output }) => ({ name: "answered_last_name", score: answeredLastName(output as RunState) }),
  ],
});
