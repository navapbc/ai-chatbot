import { tool, type Tool } from "ai";
import { z } from "zod";

/**
 * Shared stub tool builder for evals.
 *
 * Every eval needs the same 11 Apricot / browser / skill tools with
 * identical descriptions and input schemas. Only the execute callbacks
 * differ — they return different mock data and track different state.
 *
 * Pass an `overrides` object to customise individual tool execute fns.
 * Any tool not overridden gets a no-op default.
 */

// ── Shared types ────────────────────────────────────────────────────────

export interface ToolCallEntry {
  tool: string;
  args: Record<string, unknown>;
}

/** Minimal state every eval tracks */
export interface BaseRunState {
  toolCallLog: ToolCallEntry[];
}

// ── Tool schema constants ───────────────────────────────────────────────

const TOOL_DESCRIPTIONS = {
  getApricotRecord:
    "Get a participant/client record from Apricot360 by record ID. Use this to fetch participant data for form filling. Returns field values with human-readable labels resolved from the form definition.",
  getApricotForms:
    "Fetch forms from Apricot360 with optional pagination and filtering.",
  getApricotForm: "Get a specific form from Apricot360 by form ID.",
  getApricotFormFields:
    "Get all fields for a specific form from Apricot360. Returns field definitions including labels, types, options, and validation requirements.",
  testApricotAuth:
    "Test authentication with Apricot360 API. Use this to verify API credentials are working.",
  updateApricotRecord:
    "Update a participant record in Apricot360. This modifies the database record.",
  gapAnalysis:
    "Display a gap analysis card showing ONLY the missing fields the caseworker needs to provide.",
  formSummary:
    "Display a form summary card showing what was filled in and where each value came from.",
  actionLabel:
    "Label the upcoming group of browser actions with a human-readable title.",
  browser:
    "Execute browser automation commands on a remote Kernel browser. Commands include navigate, snapshot, click, fill, type, select, press, hover, check, uncheck, screenshot, inputvalue, wait, evaluate, etc.",
  loadSkill:
    'Load a skill to get specialized instructions. Available skills: "agent-browser", "caseworker-communication".',
  readSkillFile: "Read a reference file from a skill directory.",
} as const;

// ── Tool execute callback types ─────────────────────────────────────────

type ExecuteFn<T = Record<string, unknown>> = (input: T) => Promise<unknown>;

export interface ToolOverrides {
  getApricotRecord?: ExecuteFn<{ recordId: number }>;
  getApricotForms?: ExecuteFn<{
    pageSize?: number;
    pageNumber?: number;
    sort?: string;
    filters?: Record<string, string>;
  }>;
  getApricotForm?: ExecuteFn<{ formId: number }>;
  getApricotFormFields?: ExecuteFn<{ formId: number }>;
  testApricotAuth?: ExecuteFn<Record<string, never>>;
  updateApricotRecord?: ExecuteFn<{
    recordId: number;
    fields: Record<string, string>;
  }>;
  gapAnalysis?: ExecuteFn;
  formSummary?: ExecuteFn;
  actionLabel?: ExecuteFn;
  browser?: ExecuteFn;
  loadSkill?: ExecuteFn<{ name: string }>;
  readSkillFile?: ExecuteFn<{ path: string }>;
}

// ── Default no-ops ──────────────────────────────────────────────────────

function logAndReturn<S extends BaseRunState>(
  state: S,
  toolName: string,
  input: unknown,
  result: unknown
) {
  state.toolCallLog.push({ tool: toolName, args: input as Record<string, unknown> });
  return result;
}

// ── Builder ─────────────────────────────────────────────────────────────

/**
 * Create a full set of stub tools for an eval.
 *
 * @param state  - The eval's RunState object (must extend BaseRunState).
 *                 Every tool call is automatically pushed to `state.toolCallLog`.
 * @param overrides - Per-tool execute callbacks. Unset tools use sensible defaults.
 * @param options.includeUpdateTool - Include the updateApricotRecord trap tool (default false).
 */
export function createBaseStubTools<S extends BaseRunState>(
  state: S,
  overrides: ToolOverrides = {},
  options: { includeUpdateTool?: boolean } = {}
) {
  const wrap = <T>(name: string, fn: ExecuteFn<T>): ExecuteFn<T> => {
    return async (input: T) => {
      const result = await fn(input);
      return logAndReturn(state, name, input, result);
    };
  };

  const tools: Record<string, Tool> = {
    getApricotRecord: tool({
      description: TOOL_DESCRIPTIONS.getApricotRecord,
      inputSchema: z.object({
        recordId: z.number().describe("The unique record ID of the participant"),
      }),
      execute: wrap("getApricotRecord", overrides.getApricotRecord ?? (async () => ({ record: null, found: false }))),
    }),

    getApricotForms: tool({
      description: TOOL_DESCRIPTIONS.getApricotForms,
      inputSchema: z.object({
        pageSize: z.number().optional(),
        pageNumber: z.number().optional(),
        sort: z.string().optional(),
        filters: z.record(z.string()).optional(),
      }),
      execute: wrap("getApricotForms", overrides.getApricotForms ?? (async () => ({ forms: [], count: 0, success: true }))),
    }),

    getApricotForm: tool({
      description: TOOL_DESCRIPTIONS.getApricotForm,
      inputSchema: z.object({
        formId: z.number().describe("The unique ID of the form in Apricot360"),
      }),
      execute: wrap("getApricotForm", overrides.getApricotForm ?? (async () => ({ form: null, found: false }))),
    }),

    getApricotFormFields: tool({
      description: TOOL_DESCRIPTIONS.getApricotFormFields,
      inputSchema: z.object({
        formId: z.number().describe("The unique ID of the form in Apricot360"),
      }),
      execute: wrap("getApricotFormFields", overrides.getApricotFormFields ?? (async () => ({ fields: [], count: 0, success: true }))),
    }),

    testApricotAuth: tool({
      description: TOOL_DESCRIPTIONS.testApricotAuth,
      inputSchema: z.object({}),
      execute: wrap("testApricotAuth", overrides.testApricotAuth ?? (async () => ({ success: true, message: "Auth OK" }))),
    }),

    gapAnalysis: tool({
      description: TOOL_DESCRIPTIONS.gapAnalysis,
      inputSchema: z.object({
        formName: z.string().optional(),
        missingFields: z.array(
          z.object({
            field: z.string(),
            options: z.array(z.string()).optional(),
            inputType: z.enum(["text", "select", "date", "boolean", "textarea"]).optional(),
          })
        ),
      }),
      execute: wrap("gapAnalysis", overrides.gapAnalysis ?? (async (input) => input)),
    }),

    formSummary: tool({
      description: TOOL_DESCRIPTIONS.formSummary,
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
      execute: wrap("formSummary", overrides.formSummary ?? (async (input) => input)),
    }),

    actionLabel: tool({
      description: TOOL_DESCRIPTIONS.actionLabel,
      inputSchema: z.object({
        category: z.enum(["fill", "navigate", "interact", "read", "search", "misc"]),
      }),
      execute: wrap("actionLabel", overrides.actionLabel ?? (async (input) => input)),
    }),

    browser: tool({
      description: TOOL_DESCRIPTIONS.browser,
      inputSchema: z.object({
        action: z.string(),
        selector: z.string().optional(),
        value: z.string().optional(),
        values: z.array(z.string()).optional(),
        text: z.string().optional(),
        clear: z.boolean().optional(),
        url: z.string().optional(),
        interactive: z.boolean().optional(),
        timeout: z.number().optional(),
        key: z.string().optional(),
        script: z.string().optional(),
        label: z.string().optional(),
        subaction: z.string().optional(),
      }),
      execute: wrap("browser", overrides.browser ?? (async () => ({ success: true, output: "", error: null }))),
    }),

    loadSkill: tool({
      description: TOOL_DESCRIPTIONS.loadSkill,
      inputSchema: z.object({ name: z.string() }),
      execute: wrap("loadSkill", overrides.loadSkill ?? (async () => ({ content: "Skill loaded.", skillDirectory: "" }))),
    }),

    readSkillFile: tool({
      description: TOOL_DESCRIPTIONS.readSkillFile,
      inputSchema: z.object({ path: z.string() }),
      execute: wrap("readSkillFile", overrides.readSkillFile ?? (async () => ({ content: "" }))),
    }),
  };

  if (options.includeUpdateTool) {
    tools.updateApricotRecord = tool({
      description: TOOL_DESCRIPTIONS.updateApricotRecord,
      inputSchema: z.object({
        recordId: z.number(),
        fields: z.record(z.string()),
      }),
      execute: wrap(
        "updateApricotRecord",
        overrides.updateApricotRecord ?? (async () => ({ success: false, error: "Read-only access" }))
      ),
    });
  }

  return tools;
}

// ── Shared scoring utilities ────────────────────────────────────────────

/** Browser success response shorthand */
export const browserOk = (output = "") => ({ success: true, output, error: null });

/** Collect text responses from generateText result steps */
export function collectTextResponses(steps: Array<{ text?: string }>): string[] {
  return steps
    .filter((s) => s.text && s.text.trim().length > 0)
    .map((s) => s.text!);
}

/** Collect text responses with step index */
export function collectIndexedTextResponses(
  steps: Array<{ text?: string }>
): Array<{ stepIndex: number; text: string }> {
  const result: Array<{ stepIndex: number; text: string }> = [];
  for (let i = 0; i < steps.length; i++) {
    if (steps[i].text && steps[i].text!.trim().length > 0) {
      result.push({ stepIndex: i, text: steps[i].text! });
    }
  }
  return result;
}
