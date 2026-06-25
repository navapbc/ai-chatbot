# Braintrust How-To Guide

A task-oriented guide for working with [Braintrust](https://www.braintrust.dev/docs) in this
repository. Braintrust is the LLM evals and observability platform we use to (1) score the
web-automation agent against fixed scenarios offline and (2) collect production traces in the same
project so regressions are visible side-by-side.

This guide covers the common operations. For the full inventory of suites, scorers, and datasets,
see [`evals/README.md`](../evals/README.md). For canonical API details, follow the doc links at the
bottom — claims here are grounded in those pages and the code in this repo, not memory.

## Prerequisites

Install dependencies with `pnpm install`. The relevant packages are already pinned in
`package.json`:

| Package | Role |
|---------|------|
| `braintrust` | `Eval()` SDK + `braintrust eval` / `braintrust push` CLI |
| `autoevals` | Prebuilt scorers (`Factuality`, `ExactMatch`, LLM-as-judge classifiers) |
| `@braintrust/otel` | `BraintrustExporter` for production tracing |
| `@vercel/otel` | `registerOTel` wiring in `instrumentation.ts` |

Set these in `.env.local` (loaded by `pnpm eval` via `dotenv-cli`):

| Var | Required for | Notes |
|-----|--------------|-------|
| `BRAINTRUST_API_KEY` | Every eval run and all tracing | braintrust.dev → Settings → API keys |
| `BRAINTRUST_PARENT` | Eval runs / tracing | Defaults to `project_name:labs-asp`. Format is `project_name:<name>` |
| `OPENAI_API_KEY` | `gpt-*` models + the `gpt-4o` LLM judges | |
| `ANTHROPIC_API_KEY` | `claude-*` models | |
| `GOOGLE_GENERATIVE_AI_API_KEY` | `gemini-*` models | |

Without `BRAINTRUST_API_KEY`, `instrumentation.ts` returns early and no traces are exported.

## Running Evals

The eval CLI auto-discovers `*.eval.ts` files. The repo wraps it in pnpm scripts:

```bash
pnpm eval                 # run every suite in evals/ against the default model (gpt-5-mini)
pnpm eval:tool-selection  # run a single suite
pnpm eval:ci              # same as `pnpm eval` but without the dotenv wrapper (CI shell already has env)
```

Under the hood `pnpm eval` runs `dotenv -e .env.local -- braintrust eval evals`. You can call the
CLI directly for finer control — the upstream `braintrust eval` command supports:

- `braintrust eval <dir>` — discover and run eval files under a directory
- `braintrust eval --watch` — re-run on file changes during development
- `braintrust eval --filter <name>` — run a single evaluator by name

Pick the model under test with `EVAL_MODEL` (resolved by `getEvalModel()` in
[`evals/helpers.ts`](../evals/helpers.ts)):

```bash
EVAL_MODEL=claude-opus-4-7 pnpm eval
```

Recognized prefixes: `gpt-*` / `o1*` / `o3*` (OpenAI), `claude-*` (Anthropic), `gemini-*` (Google).
Each model lands in a separate experiment because `evalExperimentName()` suffixes the experiment
name with the active model id.

After a run finishes, the CLI prints a link to the experiment in the Braintrust dashboard.

## Writing a New Eval

An eval is a call to `Eval(projectName, options)` from the `braintrust` package. The shape, per the
[Eval SDK docs](https://www.braintrust.dev/docs/start/eval-sdk):

```typescript
import { Eval } from "braintrust";
import { evalExperimentName, getEvalModel } from "./helpers";

Eval("labs-asp", {
  experimentName: evalExperimentName("My Suite"),

  // data: the test cases. Each item has `input` and (optionally) `expected`.
  data: () => testCases.map((tc) => ({ input: tc.input, expected: tc.expected })),

  // task: receives one input, returns the model's output.
  task: async (input) => {
    const result = await generateText({ model: getEvalModel(), /* ... */ });
    return /* the output your scorers will inspect */;
  },

  // scores: array of scorer functions. Each returns { name, score } with score in [0, 1].
  scores: [
    ({ output, expected }) => ({
      name: "first_tool_correct",
      score: expected?.includes(output[0]) ? 1 : 0,
    }),
  ],
});
```

[`evals/tool-selection.eval.ts`](../evals/tool-selection.eval.ts) is the simplest worked example.
The step-by-step for adding a suite (participant data, snapshots, test cases, stub tools) is in
[`evals/README.md` → Adding a new eval](../evals/README.md#adding-a-new-eval).

Key conventions in this repo:

- Always wrap the experiment name in `evalExperimentName(...)` so matrix runs don't collide.
- Always get the model from `getEvalModel()` instead of hardcoding a provider.
- Build stub tools with `createBaseStubTools(state, overrides)` from `helpers.ts` rather than
  redefining the whole tool surface.

### Scorers

A scorer is any function returning `{ name, score }` where `score` is in `[0, 1]`. Two kinds are used here:

1. **Heuristic scorers** — inline functions, usually `0` or `1` (a few return fractions for
   set-intersection checks). They live directly in the `scores: [...]` array.
2. **LLM-as-judge scorers** — a model grades the output against a rubric. These are registered
   prompts pushed to Braintrust and invoked by slug. The four registered judges live in
   [`evals/scorers/`](../evals/scorers/).

You can also import prebuilt scorers from `autoevals` (e.g. `Factuality`, `ExactMatch`) and drop
them straight into `scores`.

### Registered LLM-as-Judge Scorers

A registered scorer is a prompt stored in Braintrust so it appears under the project's Scorers tab
and can be invoked by `slug`. After editing a rubric in `evals/scorers/<name>.ts`, push it:

```bash
npx braintrust push evals/scorers/<name>.ts
```

The eval file then invokes it through `initFunction({ projectName: "labs-asp", slug: "<slug>" })`.
A rubric change does **not** take effect until you push — the dashboard prompt is what actually
runs at evaluation time. Use `ifExists: "replace"` in the scorer definition so re-pushing
overwrites instead of failing.

## Datasets

Suites can read rows from a Braintrust **Dataset** instead of a local JSON file by passing
`initDataset(...)` as `data`. The dataset must exist in the dashboard before the eval runs; if it's
missing or empty, Braintrust logs a zero-row experiment (not an error).

All current suites read from local JSON fixtures under `evals/datasets/`. Do not back an eval with
real production sessions — the suites are built on synthetic participants, and there is no
sanctioned path for real user data to enter a dataset.

## Tracing

AI SDK spans flow to the same `labs-asp` project as the eval experiments. The wiring is in
[`instrumentation.ts`](../instrumentation.ts) at the repo root:

```typescript
import { registerOTel } from "@vercel/otel";
import { BraintrustExporter } from "@braintrust/otel";

export function register() {
  if (!process.env.BRAINTRUST_API_KEY) return;

  registerOTel({
    serviceName: "labs-asp-chat",
    traceExporter: new BraintrustExporter({ filterAISpans: true }),
  });
}
```

`filterAISpans: true` forwards only LLM-related spans, dropping application-level noise. The
destination project is read from `BRAINTRUST_PARENT` (format `project_name:labs-asp`). For spans to
be produced at all, the `streamText` call in `app/(chat)/api/chat/route.ts` enables
`experimental_telemetry`.

`BRAINTRUST_API_KEY` is injected only in non-production environments — `terraform/cloud_run.tf` sets
it for `dev` and `preview` but never for `prod`. Production sessions therefore never reach
Braintrust; `register()` early-returns without the key.

## Viewing Results

In the Braintrust dashboard, open the `labs-asp` project:

- **Experiments** — each eval run is an experiment; every suite shows the mean of each scorer across
  its rows. A regression is typically a multi-point drop on a scorer between two adjacent
  experiments. Matrix runs appear as separate experiments suffixed with their model id.
- **Logs / Traces** — dev/preview sessions exported via OTEL (prod is never exported).
- **Scorers** — the registered LLM-as-judge prompts pushed with `braintrust push`.

## Common Gotchas

- **`pnpm eval` fails with a dotenv error** — `.env.local` is missing. Use `pnpm eval:ci` if env
  vars are already in your shell.
- **Rubric edited but score didn't change** — registered scorers need `npx braintrust push` to take
  effect; the dashboard prompt is what runs.
- **No traces appearing** — confirm `BRAINTRUST_API_KEY` is set (otherwise `instrumentation.ts`
  returns early) and that `BRAINTRUST_PARENT` points at the intended project. The key is only set in
  `dev`/`preview`, so prod intentionally produces no traces.

## Canonical Documentation

- Eval SDK quickstart — https://www.braintrust.dev/docs/start/eval-sdk
- Writing evals — https://www.braintrust.dev/docs/guides/evals/write
- Datasets — https://www.braintrust.dev/docs/guides/datasets
- OpenTelemetry integration — https://www.braintrust.dev/docs/integrations/sdk-integrations/opentelemetry
- CLI reference — https://www.braintrust.dev/docs/reference/cli
- `autoevals` library — https://www.braintrust.dev/docs/reference/autoevals
