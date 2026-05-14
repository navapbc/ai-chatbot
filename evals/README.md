# Evals

Offline evaluation suite for the web-automation chat agent. Runs the agent against fixed scenarios with mocked tool surfaces, scores its behavior, and uploads results to [Braintrust](https://www.braintrust.dev) for tracking.

The goal: catch regressions when prompts, tools, or models change — before they reach production.

## Quick start

```bash
# Run all 8 suites against the default model (gpt-5-mini)
pnpm eval

# Run a single suite
pnpm eval:tool-selection

# Run against a different model (any prefix from EVAL_MODEL table below)
EVAL_MODEL=claude-opus-4-7 pnpm eval

# Run in CI mode (skips the dotenv wrapper)
pnpm eval:ci
```

`pnpm eval` uses `dotenv-cli` to load `.env.local`. `pnpm eval:ci` expects env vars to already be in the shell — used by the GitHub Actions matrix.

## The 8 suites

| Suite | What it measures | Heuristic scorers | LLM-as-judge |
|-------|------------------|-------------------|--------------|
| **prod-regression** | Real production sessions promoted via `scripts/promote-trace.ts`. Self-growing — rows come from the `prod-regression-cases` Braintrust dataset, scored by the four registered LLM judges. | — | **hallucination-judge**, **summary-attribution-judge**, **ask-questions-judge**, **verbosity-judge** |
| **tool-selection** | Agent picks the right tool for each user request (one-step) | first-tool-correct, all-expected-tools-called, no-hallucinated-tools | — |
| **autonomous-progression** | Agent moves through a multi-page form without being nudged | used-database-data, filled-form-fields, progressed-autonomously, stopped-before-submit, did-not-modify-database, showed-review, not-overly-verbose | — |
| **navigation** | Agent handles modals, county pickers, income popups; doesn't open new tabs or hit Back | navigated-past-landing, avoided-external-links, handled-county-modal, handled-income-modal, did-not-use-back, did-not-open-new-tab, stayed-on-site, reached-review, stopped-at-review | — |
| **clicking-ui-interaction** | Agent handles date masks, phone formats, native + select2 dropdowns, collapsible sections | handled-date-field, handled-phone-field, handled-native-dropdown, handled-select2-dropdown, expanded-collapsible-section, verified-masked-fields, filled-collapsible-fields | — |
| **ask-questions** | Agent asks for genuinely missing fields, in plain English, without overstepping into sensitive territory | called-gap-analysis, missing-field-coverage, did-not-assume-sensitive-fields, did-not-ask-for-known-fields, filled-known-fields | **ask-questions-judge** |
| **deduction** | Agent infers age from DOB, mailing from physical, ethnicity mapping, language carry-over, household composition | inferred-age-from-dob, inferred-mailing-address, mapped-ethnicity, carried-language, correct-household-size, included-household-members, inferred-nearest-office, no-false-gaps | — |
| **hallucination** | Agent doesn't invent household members, fabricate SSNs/emails, or attribute invented data to "database" | did-not-invent-spouse, correct-household-size, correct-household-member, no-fabricated-names, accurate-data-values, did-not-fabricate-marital-status, form-summary-sources-accurate, did-not-fabricate-email | **hallucination-judge**, **summary-attribution-judge** |
| **verbosity** | Agent communicates concisely, no play-by-play, no technical jargon, no wall-of-text | responses-are-concise, does-not-narrate-every-action, no-technical-jargon, provides-updates, text-is-infrequent, no-play-by-play | **verbosity-judge** |

## Scoring

Every scorer returns a value in `[0, 1]`.

- **Heuristic scorers** — typically `0` (fail) or `1` (pass). A few return fractions for set-intersection style checks (e.g., `all_expected_tools_called` returns hits/expected).
- **LLM-as-judge scorers** — choice scoring `A=1.0 / B=0.5 / C=0.0` with chain-of-thought. The judge picks one letter; the score is the mapped value. See each registered scorer file in `scorers/` for the rubric.

In the Braintrust dashboard, each experiment shows the mean across rows for every scorer. A regression is typically a multi-point drop on one or more scorers between two adjacent experiments.

## Cross-model matrix

`EVAL_MODEL` controls which model runs the agent under test. Recognised prefixes:

| Prefix | Provider | Required API key |
|--------|----------|------------------|
| `gpt-*` / `o1*` / `o3*` | `@ai-sdk/openai` | `OPENAI_API_KEY` |
| `claude-*` | `@ai-sdk/anthropic` (direct) | `ANTHROPIC_API_KEY` |
| `gemini-*` | `@ai-sdk/google` | `GOOGLE_GENERATIVE_AI_API_KEY` |

Default is `gpt-5-mini`. CI runs a 3-leg matrix in `.github/workflows/evals.yml` over `gpt-5-mini` / `claude-opus-4-7` / `gemini-2.5-pro`. Each leg uploads to a distinct Braintrust experiment (the model id is suffixed via `evalExperimentName()` in `helpers.ts`).

Production uses `claude-opus-4-7` via Vertex AI (see `lib/ai/providers.ts:17`). The CI matrix uses **direct Anthropic API** instead of Vertex for simpler secret management. Model behavior is identical between routes — only auth and rate-limit ceilings differ.

## Registered scorers

LLM-as-judge scorers live in `scorers/` and are registered in the Braintrust dashboard so they appear under the project's Scorers tab and can be invoked by `slug`.

Currently registered:

| File | Slug | Used in |
|------|------|---------|
| `scorers/hallucination.ts` | `hallucination-judge` | `hallucination.eval.ts` |
| `scorers/summary-attribution.ts` | `summary-attribution-judge` | `hallucination.eval.ts` |
| `scorers/ask-questions.ts` | `ask-questions-judge` | `ask-questions.eval.ts` |
| `scorers/verbosity.ts` | `verbosity-judge` | `verbosity.eval.ts` |

All four use `gpt-4o` + chain-of-thought + `ifExists: "replace"` so re-pushing overwrites the rubric.

### Push a scorer to Braintrust

After editing a rubric, push it once so the Scorers tab and `initFunction()` calls see the update:

```bash
npx braintrust push evals/scorers/<name>.ts
```

The `Eval()` calls invoke registered scorers via `initFunction({ projectName: "labs-asp", slug: "<slug>" })`. The eval file serializes the agent's `RunState` into a single string and passes it as `{ output: serialized }` — the registered prompt template fills `{{output}}` at evaluation time.

## Datasets

| File | Contents |
|------|----------|
| `datasets/participants.json` | Synthetic participant database records — names, addresses, household composition, income, ethnicity, language. Each suite picks a participant whose data shape matches the scenario (sparse for ask-questions, household for hallucination/deduction, etc). |
| `datasets/snapshots.json` | Browser DOM snapshots for each page of each mock form. The stub `browser` tool returns these when the agent calls `action: "snapshot"`. Snapshots are keyed by suite → page name. |
| `datasets/form-fields.json` | Form-field definitions returned by the stubbed `getApricotFormFields` tool. |
| `datasets/test-cases.json` | Per-suite test case lists — input prompts, expected outputs (for tool-selection), max step counts. |
| `datasets/golden.json` | Forbidden-term lists and ideal-behavior references for the hallucination suite. |

When you change a participant's data, also update any heuristic scorer that hardcodes that participant's expected values. The `hallucination-judge` registered scorer also has Tanya Brooks's record baked into its system prompt — that's a known limitation (see the audit notes in commit `453311e`).

## Production tracing

`instrumentation.ts` at the repo root registers a `BraintrustExporter` from `@braintrust/otel` when `BRAINTRUST_API_KEY` is set. The chat route's `streamText` call has `experimental_telemetry` enabled, so production AI SDK spans flow to the same `labs-asp` Braintrust project as the eval experiments. This means:

- Eval experiments and production sessions sit side-by-side in the dashboard
- You can compare a regressed eval score against the production traces that produced it
- Problematic sessions can be **promoted into a self-growing regression suite** via `scripts/promote-trace.ts` (see below)

## Promoting a production trace to a regression test

When a real production session exposes a regression you want to lock in, push it into the `prod-regression-cases` Braintrust dataset. The `prod-regression.eval.ts` suite picks up new rows automatically on the next eval run.

### One-time setup

1. In the Braintrust dashboard, create a **Dataset** named `prod-regression-cases` under the `labs-asp` project.
2. Decide which synthetic participants from `datasets/participants.json` you'll use as ground truth proxies for production scenarios. Common picks:
   - `mariaGarcia` — complete record, household of 3
   - `tanyaBrooks` — sparse record (single mother, missing SSN/email/marital)
   - `luciaMorales` — full record with children + linked family
   - `jamesNguyen` — sparse, used by ask-questions
   - `priyaSharma` — used by verbosity
   - `davidChen` — used by navigation

### Per-promotion workflow

1. Find the problematic trace in the Braintrust dashboard.
2. Copy the user's first message (the agent's input).
3. Choose a synthetic participant whose record shape resembles the production scenario.
4. Run:
   ```bash
   echo "<copied user message>" | pnpm trace:promote \
     --participant mariaGarcia \
     --span-id <braintrust-span-id> \
     --note "Why this case matters"
   ```
5. The script scrubs PII (SSN, email, phone, ZIP, Apricot record IDs), prints the scrubbed version + flags, and asks for confirmation before pushing.

The dataset row carries `metadata.participant` pointing at your chosen synthetic participant. `prod-regression.eval.ts` reads that metadata, looks up the participant, and uses it as ground truth for the (now participant-agnostic) `hallucination-judge` and `summary-attribution-judge`.

### Why is this a manual paste rather than auto-fetch?

The script intentionally doesn't fetch from Braintrust's API. The operator-paste step is a **deliberate PII review checkpoint** — automating it would create a path for unreviewed real-user data to flow into a git-tracked dataset. The PII scrubber catches the obvious patterns (`scripts/scrub-pii.ts`) but free-form names and addresses are hard to regex out, so the human eyeball is the safety net.

### Test the scrubber without pushing

```bash
echo "Look up record 4521 for Maria Garcia (maria.garcia@email.com, 951-555-0142)" \
  | pnpm scrub:check
```

## Adding a new eval

1. Pick a participant in `datasets/participants.json` whose shape fits your scenario (or add a new one).
2. Add a page snapshot to `datasets/snapshots.json` under a new suite key.
3. Add test cases to `datasets/test-cases.json` under a new suite key.
4. Create `<name>.eval.ts` modeled on the closest existing suite. Extend `BaseRunState` from `helpers.ts` with whatever new tracking fields your scorers need.
5. Use `createBaseStubTools(state, { ... })` from `helpers.ts` — only override the tool callbacks that need custom behavior.
6. Call `getEvalModel()` for the agent model and `evalExperimentName("<Pretty Name>")` for the Braintrust experiment name.
7. Add heuristic scorers inline as `({ output }) => ({ name, score })`. For semantic checks too fuzzy for regex, register a new LLM-as-judge in `scorers/` and invoke it via `initFunction`.

## Adding a new registered scorer

1. Create `scorers/<name>.ts` modelled on `scorers/verbosity.ts`.
2. Keep the system prompt generic — pass any ground-truth data (e.g., the participant record) as part of `{{output}}` at invocation time rather than baking it into the prompt at registration time. (`hallucination-judge` is hardcoded to Tanya Brooks for historical reasons; new scorers should be participant-agnostic.)
3. Use `ifExists: "replace"` so re-pushing overwrites instead of failing.
4. Push it: `npx braintrust push evals/scorers/<name>.ts`.
5. Add `initFunction({ projectName: "labs-asp", slug: "<your-slug>" })` to the relevant `*.eval.ts` and call it from an async scorer entry.

## Environment

Local development (`pnpm eval` reads `.env.local`):

| Var | Required for | Notes |
|-----|--------------|-------|
| `BRAINTRUST_API_KEY` | All eval runs | Get from braintrust.dev → Settings → API keys |
| `BRAINTRUST_PARENT` | All eval runs | Defaults to `project_name:labs-asp` — only change if you want to point at a different Braintrust project |
| `EVAL_MODEL` | Optional | Defaults to `gpt-5-mini`. See prefix table above. |
| `OPENAI_API_KEY` | `gpt-*` / `o1*` / `o3*` models, all `gpt-4o` LLM judges | |
| `ANTHROPIC_API_KEY` | `claude-*` models | |
| `GOOGLE_GENERATIVE_AI_API_KEY` | `gemini-*` models | |

CI uses the same names as GitHub Actions secrets. The workflow soft-skips a matrix leg when its provider key is missing.

## Common gotchas

- **Missing `.env.local`**: `pnpm eval` will fail with a dotenv error. Use `pnpm eval:ci` instead if env vars are already in your shell.
- **Scorer rubric changed but score didn't move**: registered scorers need `npx braintrust push` to take effect. The eval file picks up the slug, but the dashboard prompt is what runs.
- **Experiments colliding in the dashboard**: every suite's `experimentName` is wrapped by `evalExperimentName()` which suffixes the active `EVAL_MODEL`. If multiple PRs are running the same matrix entry, Braintrust auto-namespaces by timestamp + git context — but very fast back-to-back runs can group as one experiment.
- **All 4 LLM judges are participant-agnostic**: each takes the participant record as part of the `{{output}}` payload at invocation time. Re-pushing them is required after any rubric change (`npx braintrust push evals/scorers/<name>.ts`).
- **`prod-regression-cases` dataset must exist before running prod-regression.eval.ts**: the eval reads from `initDataset({ project: "labs-asp", dataset: "prod-regression-cases" })`. If the dataset doesn't exist or is empty, Braintrust logs a zero-row experiment — not an error, but nothing to evaluate either.
