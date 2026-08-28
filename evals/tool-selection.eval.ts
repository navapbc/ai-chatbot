import { Eval } from "braintrust";
import { generateText, isStepCount, tool, type ModelMessage } from "ai";
import { z } from "zod";
import { getWebAutomationSystemPrompt } from "@/lib/ai/prompts/web-automation";
import testCaseData from "./datasets/test-cases.json";
import { evalExperimentName, getEvalModel, logResultUsage } from "./helpers";

/**
 * Stub tool definitions — identical schemas to production but with no-op
 * execute functions. We only care about which tool the model *selects*,
 * not what it returns.
 */
const stubTools = {
  getApricotRecord: tool({
    description:
      "Get a participant/client record from Apricot360 by record ID. Use this to fetch participant data for form filling. Returns field values with human-readable labels resolved from the form definition.",
    inputSchema: z.object({
      recordId: z.number().describe("The unique record ID of the participant"),
    }),
    execute: async () => ({ record: null, found: false }),
  }),

  getApricotForms: tool({
    description:
      "Fetch forms from Apricot360 with optional pagination and filtering.",
    inputSchema: z.object({
      pageSize: z.number().optional(),
      pageNumber: z.number().optional(),
      sort: z.string().optional(),
      filters: z.record(z.string(), z.string()).optional(),
    }),
    execute: async () => ({ forms: [], count: 0, success: true }),
  }),

  getApricotForm: tool({
    description: "Get a specific form from Apricot360 by form ID.",
    inputSchema: z.object({
      formId: z.number().describe("The unique ID of the form in Apricot360"),
    }),
    execute: async () => ({ form: null, found: false }),
  }),

  getApricotFormFields: tool({
    description:
      "Get all fields for a specific form from Apricot360. Returns field definitions including labels, types, options, and validation requirements.",
    inputSchema: z.object({
      formId: z.number().describe("The unique ID of the form in Apricot360"),
    }),
    execute: async () => ({ fields: [], count: 0, success: true }),
  }),

  testApricotAuth: tool({
    description:
      "Test authentication with Apricot360 API. Use this to verify API credentials are working.",
    inputSchema: z.object({}),
    execute: async () => ({ success: true, message: "Auth OK" }),
  }),

  gapAnalysis: tool({
    description:
      "Display a gap analysis card showing ONLY the missing fields the caseworker needs to provide.",
    inputSchema: z.object({
      formName: z.string().optional(),
      missingFields: z.array(
        z.object({
          field: z.string(),
          options: z.array(z.string()).optional(),
          inputType: z
            .enum(["text", "select", "date", "boolean", "textarea"])
            .optional(),
        })
      ),
    }),
    execute: async (input) => input,
  }),

  formSummary: tool({
    description:
      "Display a form summary card showing what was filled in and where each value came from.",
    inputSchema: z.object({
      formName: z.string().optional(),
      fields: z.array(
        z.object({
          field: z.string(),
          value: z.string().optional(),
          source: z.enum(["database", "caseworker", "inferred", "missing"]),
        })
      ),
    }),
    execute: async (input) => input,
  }),

  actionLabel: tool({
    description:
      "Label the upcoming group of browser actions with a human-readable title.",
    inputSchema: z.object({
      category: z.enum([
        "fill",
        "navigate",
        "interact",
        "read",
        "search",
        "misc",
      ]),
    }),
    execute: async (input) => input,
  }),

  browser: tool({
    description: `Execute browser automation commands on a remote Kernel browser. Commands include navigate, snapshot, click, fill, type, select, press, hover, check, uncheck, screenshot, etc.`,
    inputSchema: z.object({
      action: z.string(),
      selector: z.string().optional(),
      value: z.string().optional(),
      url: z.string().optional(),
    }),
    execute: async () => ({ success: true, output: "", error: null }),
  }),

  readReference: tool({
    description:
      'Load a reference document. Use the path the system prompt instructs you to load (e.g. "field-patterns.md", "custom-dropdowns.md", "form-submission.md", "browser-commands.md").',
    inputSchema: z.object({
      path: z.string(),
    }),
    execute: async () => ({ content: "" }),
  }),
};

// ── Test data ──────────────────────────────────────────────────────────

const testCases = testCaseData.toolSelection.cases;

// ── Eval ────────────────────────────────────────────────────────────────

const model = getEvalModel();

Eval("labs-asp", {
  experimentName: evalExperimentName("Tool Selection"),
  data: () =>
    testCases.map((tc) => ({
      input: tc.input,
      expected: tc.expected,
    })),

  task: async (input: string, { span }) => {
    const messages: ModelMessage[] = [{ role: "user", content: input }];

    const result = await generateText({
      model,
      instructions: getWebAutomationSystemPrompt(),
      messages,
      tools: stubTools,
      stopWhen: isStepCount(1),
    });

    // Extract the tool names the model chose to call
    const toolCalls = result.steps.flatMap((step) =>
      step.toolCalls.map((tc) => tc.toolName)
    );

    logResultUsage(span, result);
    return toolCalls;
  },

  scores: [
    ({ output, expected }) => {
      // The first tool called should be one of the expected tools
      const score =
        output && output.length > 0 && expected?.includes(output[0]) ? 1 : 0;
      return { name: "first_tool_correct", score };
    },
    ({ output, expected }) => {
      if (!output || output.length === 0 || !expected || expected.length === 0)
        return { name: "all_expected_tools_called", score: 0 };
      const called = new Set(output);
      const hits = expected.filter((t) => called.has(t)).length;
      return { name: "all_expected_tools_called", score: hits / expected.length };
    },
    ({ output, expected }) => {
      if (!output || output.length === 0)
        return { name: "no_hallucinated_tools", score: 1 };
      // Allow actionLabel and readReference as bonus tools (always acceptable)
      const allowList = new Set([...(expected ?? []), "actionLabel", "readReference"]);
      const unexpected = output.filter((t) => !allowList.has(t));
      return { name: "no_hallucinated_tools", score: unexpected.length === 0 ? 1 : 0 };
    },
  ],
});
