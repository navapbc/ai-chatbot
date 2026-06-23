import { Eval, initDataset, initFunction } from "braintrust";
import { generateText, stepCountIs, type ModelMessage } from "ai";
import { getWebAutomationSystemPrompt } from "@/lib/ai/prompts/web-automation";
import participants from "./datasets/participants.json";
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
 * Prod Regression Eval
 *
 * Reads cases from the prod-regression-cases Braintrust Dataset — rows
 * promoted from real production traces via scripts/promote-trace.ts.
 * Each row carries metadata.participant referencing a synthetic
 * participant in datasets/participants.json that the LLM judges use as
 * ground truth.
 *
 * Unlike the hand-crafted suites, this one has no scenario-specific
 * snapshots or heuristic scorers — it relies on the 4 registered
 * LLM-as-judge scorers (which are participant-agnostic now that
 * hallucination-judge has been parametrized). The bet is that those
 * four judges, applied to real user inputs, catch the long tail of
 * regressions that hand-written cases can't anticipate.
 *
 * The dataset will be empty until the first trace is promoted; that's
 * fine — Braintrust will simply log a zero-row experiment.
 */

type ParticipantKey = keyof typeof participants;
type ParticipantRecord = typeof participants[ParticipantKey];

const DEFAULT_PARTICIPANT: ParticipantKey = "mariaGarcia";

interface RunState extends BaseRunState {
  browserFills: Array<{ selector: string; value: string }>;
  textResponses: string[];
  formSummaryCalls: Array<{
    fields: Array<{ field: string; value?: string; source: string }>;
  }>;
  gapAnalysisCalls: Array<{ missingFields: Array<{ field: string }> }>;
  participant: ParticipantRecord;
}

function createStubTools(state: RunState) {
  return createBaseStubTools(state, {
    getApricotRecord: async () => state.participant,

    gapAnalysis: async (input) => {
      const typed = input as { missingFields?: Array<{ field: string }> };
      state.gapAnalysisCalls.push({
        missingFields: (typed.missingFields ?? []).map((f) => ({ field: f.field })),
      });
      return input;
    },

    formSummary: async (input) => {
      const typed = input as {
        fields?: Array<{ field: string; value?: string; source: string }>;
      };
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
      const action = (input as { action?: string }).action ?? "";
      const selector = (input as { selector?: string }).selector ?? "";
      const value =
        (input as { value?: string }).value ??
        (input as { text?: string }).text ??
        "";
      if ((action === "fill" || action === "type" || action === "select") && selector && value) {
        state.browserFills.push({ selector, value });
      }
      return browserOk();
    },
  });
}

// ── Registered LLM-as-judge scorers ─────────────────────────────────────

const hallucinationJudge = initFunction({
  projectName: "labs-asp",
  slug: "hallucination-judge",
});
const summaryAttributionJudge = initFunction({
  projectName: "labs-asp",
  slug: "summary-attribution-judge",
});
const verbosityJudge = initFunction({
  projectName: "labs-asp",
  slug: "verbosity-judge",
});
const askQuestionsJudge = initFunction({
  projectName: "labs-asp",
  slug: "ask-questions-judge",
});

function serializeFull(state: RunState): string {
  const parts: string[] = [];

  parts.push("## Participant Database Record (ground truth)");
  parts.push(JSON.stringify(state.participant.record, null, 2));

  if (state.textResponses.length > 0) {
    parts.push("\n## Agent Text Responses");
    for (const text of state.textResponses) parts.push(text);
  }

  if (state.browserFills.length > 0) {
    parts.push("\n## Form Fields Filled (selector → value)");
    for (const f of state.browserFills) {
      parts.push(`- ${f.selector}: "${f.value}"`);
    }
  }

  if (state.formSummaryCalls.length > 0) {
    parts.push("\n## Form Summary Cards");
    for (const summary of state.formSummaryCalls) {
      for (const f of summary.fields) {
        parts.push(`- ${f.field}: "${f.value ?? "(empty)"}" [source: ${f.source}]`);
      }
    }
  }

  if (state.gapAnalysisCalls.length > 0) {
    parts.push("\n## Gap Analysis (fields agent asked caseworker about)");
    for (const gap of state.gapAnalysisCalls) {
      for (const f of gap.missingFields) parts.push(`- ${f.field}`);
    }
  }

  return parts.join("\n");
}

function serializeForVerbosity(state: RunState): string {
  return state.textResponses.length === 0
    ? "(agent produced no text responses)"
    : state.textResponses.join("\n\n");
}

// ── Eval ────────────────────────────────────────────────────────────────

const model = getEvalModel();

function resolveParticipant(meta: unknown): ParticipantRecord {
  const id = (meta as { participant?: string } | undefined)?.participant;
  if (id && id in participants) {
    return (participants as Record<string, ParticipantRecord>)[id];
  }
  return participants[DEFAULT_PARTICIPANT];
}

interface DatasetRow {
  input: string;
  metadata: { participant?: string } & Record<string, unknown>;
}

async function loadDatasetRows(): Promise<DatasetRow[]> {
  const ds = initDataset({
    project: "labs-asp",
    dataset: "prod-regression-cases",
  });
  const rows: DatasetRow[] = [];
  for await (const row of ds) {
    const input = typeof row.input === "string" ? row.input : JSON.stringify(row.input);
    const metadata = (row.metadata as DatasetRow["metadata"]) ?? {};
    rows.push({ input, metadata });
  }
  return rows;
}

Eval("labs-asp", {
  experimentName: evalExperimentName("Prod Regression"),
  data: loadDatasetRows,

  task: async (input: string, hooks) => {
    const participant = resolveParticipant(hooks.metadata);

    const state: RunState = {
      toolCallLog: [],
      browserFills: [],
      textResponses: [],
      formSummaryCalls: [],
      gapAnalysisCalls: [],
      participant,
    };

    const tools = createStubTools(state);
    const messages: ModelMessage[] = [{ role: "user", content: input }];

    const result = await generateText({
      model,
      system: getWebAutomationSystemPrompt(),
      messages,
      tools,
      stopWhen: stepCountIs(30),
    });

    state.textResponses = collectTextResponses(result.steps);

    logResultUsage(hooks.span, result);
    return state;
  },

  scores: [
    async ({ output }) => {
      const serialized = serializeFull(output as RunState);
      const result = (await hallucinationJudge({ output: serialized })) as {
        score?: number | null;
        metadata?: Record<string, unknown>;
      };
      return {
        name: "hallucination_judge",
        score: result.score ?? 0,
        metadata: result.metadata,
      };
    },
    async ({ output }) => {
      const serialized = serializeFull(output as RunState);
      const result = (await summaryAttributionJudge({ output: serialized })) as {
        score?: number | null;
        metadata?: Record<string, unknown>;
      };
      return {
        name: "summary_attribution_judge",
        score: result.score ?? 0,
        metadata: result.metadata,
      };
    },
    async ({ output }) => {
      const serialized = serializeFull(output as RunState);
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
    async ({ output }) => {
      const serialized = serializeForVerbosity(output as RunState);
      const result = (await verbosityJudge({ output: serialized })) as {
        score?: number | null;
        metadata?: Record<string, unknown>;
      };
      return {
        name: "verbosity_judge",
        score: result.score ?? 0,
        metadata: result.metadata,
      };
    },
  ],
});
