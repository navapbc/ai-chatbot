# Convert `web-automation.ts` Prompt into the Eve Agent — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the `lib/ai/prompts/web-automation.ts` system prompt into idiomatic Eve structure under `agent/` — always-on `instructions.md`, a runtime-date dynamic instruction, two on-demand skills, representative example tools, three subagents, and a sandbox — as a demonstrative reference conversion.

**Architecture:** Additive-only. The prompt CONTENT moves verbatim into Eve slots; tools/subagents/sandbox are representative, well-commented examples (real zod schemas + placeholder `execute`), not full logic ports. `lib/` is untouched and still powers the live app. Eve discovers `instructions.md`, `instructions/`, `skills/`, `tools/`, `subagents/`, and `sandbox.ts` by directory convention — there are no `tools`/`skills`/`subagents`/`sandbox` keys in `agent.ts`.

**Tech Stack:** Eve `0.27.0` (beta) · Vercel AI Gateway (`anthropic/claude-sonnet-4.6`) · zod 4.4.3 · TypeScript · pnpm · Node 24.

## Global Constraints

- **Node 24 for every command.** The shell defaults to Node v20.15.0 and shell state does NOT persist between Bash calls, so prefix EVERY node/npx/pnpm command with `export PATH="$HOME/.nvm/versions/node/v24.18.0/bin:$PATH"` and verify `node -v` shows v24.
- **AI Gateway key** is in `.env.local` as `AI_GATEWAY_API_KEY=...` (gitignored — never print/commit). `npx eve dev` does NOT auto-load `.env.local`; export the key into the environment before starting it.
- **pnpm only.** Never commit `node_modules`, `package-lock.json`, or `.eve/` (transient runtime).
- **Additive only — touch ONLY `agent/**`** (plus committing this plan's own doc edits). Do NOT modify anything under `lib/`, `app/`, `vitest.config.mjs`, `package.json`, or the existing spike files `agent/tools/read_reference.ts` (annotation comment only) and `agent/tools/update_working_memory.ts`. The live app keeps using `lib/ai/prompts/*`.
- **Port prompt text faithfully.** Move the existing prose verbatim from the named source files/line ranges. Do not rewrite, summarize, or change its meaning. The one rule that MUST survive in the always-on prompt is the Forbidden Actions "NEVER click the final submit button" safety block.
- **Confirmed Eve APIs (against installed `eve@0.27.0` + `node_modules/eve/docs`):** `agent/instructions.md` (root, always-on) and an `agent/instructions/` directory coexist (root file first, then directory entries alphabetically). `defineInstructions` and `defineDynamic` both import from `eve/instructions`. A `.ts` instructions module runs ONCE at build time, so a runtime value (today's date) MUST come from a `defineDynamic` resolver (event `session.started`), NOT a bare `defineInstructions`. Instructions/skills `defineDynamic` take NO `fallback` (that is model-only; adding it is a build error). Packaged skills (`agent/skills/<name>/SKILL.md`) MUST carry `description:` frontmatter. Subagents live at `agent/subagents/<id>/agent.ts` and their `defineAgent` MUST include a `description` (the parent routes on it). `defineAgent` accepts only: `model`, `compaction` (`{ model?, thresholdPercent?, modelContextWindowTokens? }`), `description`, `build`, `experimental`, `limits`, `modelOptions`, `modelContextWindowTokens`, `outputSchema`, `reasoning` — NO directory-slot keys. `defineSandbox` imports from `eve/sandbox`; omit `backend` to use `defaultBackend()`. `defineTool` imports from `eve/tools` and zod `inputSchema` works (zod4 installed). Verify anything not listed here against `node_modules/eve/docs/*.mdx` at author time rather than guessing.
- Model strings are dot-versioned gateway slugs (`anthropic/claude-sonnet-4.6`).
- After each task, boot `npx eve dev` (own server, default port ~2000; use another port if occupied) and confirm it starts WITHOUT registration/schema errors — that boot is the integration gate for this demonstrative work.

---

### Task 1: Always-on instructions + runtime-date dynamic instruction

**Files:**
- Modify (replace spike stub): `agent/instructions.md`
- Create: `agent/instructions/date.ts`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: the agent's always-on system prompt (root file) plus a `session.started` dynamic instruction that appends today's date.

- [ ] **Step 1: Write the always-on `agent/instructions.md`**

Replace the spike stub entirely. Assemble it by copying these exact sections VERBATIM from the source files (do not reword):
- From `lib/ai/prompts/web-automation.ts`: the mission sentence (line 12), the entire `## IMPORTANT — Applicant Identity` section (lines 14–24), `## Core Approach` (lines 26–30), `## Step Management` (lines 32–39), and `## Action Labeling` (lines 55–56).
- From `lib/ai/prompts/application-protocol.ts`: the entire `## Communication Rules` section including the `### Language` subsection (lines 102–128).
- From `lib/ai/prompts/browser-and-forms.ts`: the entire `## Forbidden Actions` section (lines 223–230).

Add a short top note that on-demand procedures live in skills the model loads with `load_skill` (browser mechanics → the `browser-automation` skill; benefits-application protocol → the `benefits-application` skill), and that detailed database verification, requirements research, and final form review are delegated to subagents. Keep this file to identity + standing rules + safety; do NOT paste the browser mechanics or the full application protocol here (those are Task 2 skills).

- [ ] **Step 2: Write the runtime-date dynamic instruction**

Create `agent/instructions/date.ts` (converts `getCurrentDateString()` from `web-automation.ts:4-9`):
```ts
import { defineDynamic, defineInstructions } from 'eve/instructions';

// Runtime date must resolve per session, not at build time — a plain
// defineInstructions module is captured once at build. defineDynamic's
// session.started resolver runs at session start, so the date is fresh.
export default defineDynamic({
  events: {
    'session.started': () => {
      const now = new Date();
      const formatted = now.toLocaleDateString('en-US', {
        weekday: 'long',
        year: 'numeric',
        month: 'long',
        day: 'numeric',
      });
      const iso = now.toISOString().split('T')[0];
      return defineInstructions({
        markdown: `Today's date is ${formatted} (${iso}). Use this date for any age calculations, "today's date" fields, or date-relative logic.`,
      });
    },
  },
});
```

- [ ] **Step 3: Boot-verify Eve loads both instruction sources**

Run (Node 24, key exported):
```bash
export PATH="$HOME/.nvm/versions/node/v24.18.0/bin:$PATH"
export AI_GATEWAY_API_KEY=$(grep '^AI_GATEWAY_API_KEY=' .env.local | cut -d= -f2-)
npx eve dev --no-ui --port 2010 &
sleep 8
curl -sS -X POST "http://127.0.0.1:2010/eve/v1/session" -H 'content-type: application/json' -d '{"message":"What is today'\''s date, and in one sentence what is your single most important rule about submitting forms?"}' -D - -o /dev/null | grep -i x-eve-session-id
# then read the stream for that session id and confirm the reply states today's date AND "never submit"
kill %1 2>/dev/null
```
Expected: server boots with no registration error; the model's reply states today's actual date (proving the dynamic instruction ran) and the never-submit rule (proving the always-on prompt is in context). Record the session reply. If `eve dev` errors on the `instructions/` directory or the dynamic shape, consult `node_modules/eve/docs/instructions.mdx` + `guides/dynamic-capabilities.md` and adjust to the installed API.

- [ ] **Step 4: Commit**

```bash
git add agent/instructions.md agent/instructions/date.ts
git commit -m "feat(eve-agent): always-on instructions + runtime-date dynamic instruction

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: On-demand skills (browser-automation + benefits-application)

**Files:**
- Create: `agent/skills/browser-automation/SKILL.md`
- Create: `agent/skills/browser-automation/field-patterns.md`
- Create: `agent/skills/browser-automation/custom-dropdowns.md`
- Create: `agent/skills/browser-automation/browser-commands.md`
- Create: `agent/skills/benefits-application/SKILL.md`

**Interfaces:**
- Consumes: the booting agent from Task 1.
- Produces: two `load_skill`-loadable skills; the browser skill packages the three reference files as siblings.

- [ ] **Step 1: Create the `browser-automation` skill**

Create `agent/skills/browser-automation/SKILL.md` with required frontmatter, then the body copied VERBATIM from `lib/ai/prompts/browser-and-forms.ts` lines 1–221 (everything from `## Browser Automation` through the end of `## Form Submission Protocol`) AND the `## Resuming After Interruption` section from `web-automation.ts:49-53` AND `## Parameter Types` (`browser-and-forms.ts:232-238`). Do NOT include the `## Forbidden Actions` section (that went to always-on instructions in Task 1). Update the "Reference Files" / "Field Type Patterns" / "Custom Dropdowns" mentions to say the sibling files are loaded on demand (see Step 3) rather than via a `readReference` tool call.
```md
---
description: Use for any browser interaction while filling a web form — snapshotting, selectors, masked fields, native/custom dropdowns, multi-page forms, modal handling, error recovery, and the submission gate.
---

<verbatim body from browser-and-forms.ts:1-221 + resuming section + parameter types, with readReference mentions rephrased to "load the sibling reference file">
```

- [ ] **Step 2: Add the three reference files as skill siblings**

Copy each file's contents VERBATIM into the skill package (they are already skill-shaped procedures):
```bash
cp lib/ai/prompts/references/field-patterns.md   agent/skills/browser-automation/field-patterns.md
cp lib/ai/prompts/references/custom-dropdowns.md  agent/skills/browser-automation/custom-dropdowns.md
cp lib/ai/prompts/references/browser-commands.md  agent/skills/browser-automation/browser-commands.md
```
In `SKILL.md`, reference them as sibling paths (e.g. "see `field-patterns.md` in this skill for exact JSON examples"). Note in a short comment/line that Eve materializes these under `$HOME/.agents/skills/browser-automation/` and reaching them at runtime uses `ctx.getSkill('browser-automation').file('field-patterns.md')` — which is why the agent has a sandbox (Task 5).

- [ ] **Step 3: Create the `benefits-application` skill**

Create `agent/skills/benefits-application/SKILL.md` with frontmatter, body copied VERBATIM from `lib/ai/prompts/application-protocol.ts`: `## Benefits Applications` (lines 1–3), `## Autofilled Field Detection` (31–33), `## Filling Fields` (35–43), `## No vs Unknown Distinction` (45–49), `## Autonomous Progression` (72–89), `## Review Screen (REQUIRED)` (91–100), `## Gap Analysis Protocol` (130–149), and `## Form Completion Summary` (151–182). Do NOT include `## Database Retrieval & Verification`, `## Data Provenance`, or `## Field Mapping & Inference Rules` — those go to the `database_verification` subagent (Task 4).
```md
---
description: Use when filling a benefits application — gap-analysis-first workflow, autofill detection, field-filling rules, no-vs-unknown handling, autonomous page progression, the required review screen, and the form-completion summary.
---

<verbatim sections listed above>
```

- [ ] **Step 4: Boot-verify skills are discovered and loadable**

Run `npx eve dev` (as in Task 1 Step 3). Confirm boot has no skill-parse error. Then POST a message that should trigger a skill load, e.g. `{"message":"I'm about to fill a form and a dropdown isn't responding to select — what should I do?"}` and confirm from the stream that the model calls `load_skill` for `browser-automation`. Record the evidence. If frontmatter/packaging errors, check `node_modules/eve/docs/skills.mdx`.

- [ ] **Step 5: Commit**

```bash
git add agent/skills/
git commit -m "feat(eve-agent): browser-automation + benefits-application skills (readReference -> skills)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Representative example tools

**Files:**
- Create: `agent/tools/action_label.ts`, `agent/tools/gap_analysis.ts`, `agent/tools/form_summary.ts`, `agent/tools/check_submit_gate.ts`, `agent/tools/browser.ts`
- Modify (annotation comment only): `agent/tools/read_reference.ts`

**Interfaces:**
- Consumes: the booting agent from Tasks 1–2.
- Produces: five example `defineTool`s the main agent exposes, each with a real zod `inputSchema` and a placeholder `execute` that returns a shaped result, commented with what the production version does and where its real logic lives in `lib/`.

- [ ] **Step 1: `action_label.ts`**

```ts
import { defineTool } from 'eve/tools';
import { z } from 'zod';

// Example tool. Production logic: lib/ai/tools/action-label.ts.
// Call once before each logical group of browser actions.
export default defineTool({
  description:
    'Label the next group of related browser actions. Call once with the best-fit category before a batch of actions.',
  inputSchema: z.object({
    category: z.enum(['fill', 'navigate', 'interact', 'read', 'search', 'misc']),
  }),
  async execute({ category }) {
    // Demonstrative stub: production emits a UI action label.
    return { labeled: category };
  },
});
```

- [ ] **Step 2: `gap_analysis.ts`**

```ts
import { defineTool } from 'eve/tools';
import { z } from 'zod';

// Example tool. Production logic: lib/ai/tools/gap-analysis.ts (renders an
// interactive card client-side). Per the benefits-application skill, calling
// this ENDS the turn — the agent must stop and wait for the caseworker.
export default defineTool({
  description:
    'Render the gap-analysis card listing required form fields with no traceable data. Calling this ends your turn.',
  inputSchema: z.object({
    formName: z.string(),
    clientName: z.string().optional(),
    missingFields: z.array(
      z.object({
        field: z.string(),
        options: z.array(z.string()).optional(),
        inputType: z.enum(['select', 'radio', 'checkbox', 'text']).optional(),
        multiSelect: z.boolean().optional(),
        required: z.boolean().optional(),
        note: z.string().optional(),
      }),
    ),
  }),
  async execute({ formName, missingFields }) {
    // Demonstrative stub: production renders the interactive gap card.
    return { rendered: true, formName, missingCount: missingFields.length };
  },
});
```

- [ ] **Step 3: `form_summary.ts`**

```ts
import { defineTool } from 'eve/tools';
import { z } from 'zod';

// Example tool. Production logic: lib/ai/tools/form-summary.ts (interactive
// review card). Called instead of writing a text summary.
export default defineTool({
  description:
    'Render the form-completion summary card. Call instead of writing a text summary of filled fields.',
  inputSchema: z.object({
    clientName: z.string().optional(),
    fields: z.array(
      z.object({
        field: z.string(),
        value: z.string().optional(),
        source: z.enum(['database', 'caseworker', 'inferred', 'missing']),
        inputType: z.enum(['select', 'radio', 'checkbox', 'text']).optional(),
        options: z.array(z.string()).optional(),
        required: z.boolean().optional(),
      }),
    ),
  }),
  async execute({ fields }) {
    return { rendered: true, fieldCount: fields.length };
  },
});
```

- [ ] **Step 4: `check_submit_gate.ts`**

```ts
import { defineTool } from 'eve/tools';
import { z } from 'zod';

// Example tool. Production logic: lib/ai/tools/check-submit-gate.ts. Probes a
// Turnstile page and force-enables a stuck-disabled submit button so the
// caseworker can take over. It does NOT click submit.
export default defineTool({
  description:
    'On a page with a Cloudflare Turnstile widget where the submit button is stuck disabled, probe and force-enable it. Does not click submit. Do not call on pages without Turnstile.',
  inputSchema: z.object({
    reason: z.string().describe('Why the submit button appears stuck-disabled'),
  }),
  async execute({ reason }) {
    return { enabled: true, reason };
  },
});
```

- [ ] **Step 5: `browser.ts` (thin example)**

```ts
import { defineTool } from 'eve/tools';
import { z } from 'zod';

// Example tool ONLY. Production browser automation runs against Kernel.sh via
// lib/kernel/browser.ts + lib/ai/tools/browser.ts, and re-architecting it for
// Eve's durable execution is migration sub-project 3 (see docs/eve-spike-findings.md
// "Browser session sketch"). Eve tools run in the app runtime (not the sandbox),
// so a real port would call Kernel.sh here, re-resolving the session by its
// stable id each call. This stub only shows the command shape.
export default defineTool({
  description:
    'Send a structured browser command (navigate, snapshot, click, fill, type, select, check, evaluate, wait). Snapshot before interacting.',
  inputSchema: z.object({
    action: z.enum([
      'navigate', 'snapshot', 'click', 'fill', 'type', 'select', 'check',
      'evaluate', 'press', 'wait', 'inputvalue', 'back', 'reload',
    ]),
    url: z.string().optional(),
    selector: z.string().optional(),
    value: z.string().optional(),
    text: z.string().optional(),
  }),
  async execute(input) {
    // Demonstrative stub: production dispatches to Kernel.sh.
    return { ok: true, action: input.action };
  },
});
```

- [ ] **Step 6: Annotate `read_reference.ts` as superseded**

Add a comment at the top of `agent/tools/read_reference.ts` (change nothing else — the tested logic stays intact):
```ts
// NOTE: Superseded by Eve skills. The three reference docs now ship as sibling
// files of the `browser-automation` skill and load on demand via `load_skill`
// (Eve's native progressive-disclosure mechanism), so this tool is retained only
// as the spike's proof tool. New reference material should be a skill, not this.
```

- [ ] **Step 7: Boot-verify the tools register**

Run `npx eve dev`; confirm boot lists/accepts the new tools with no zod-schema serialization error (the zod4 migration is what makes `z.object()` schemas work here). POST `{"message":"Label that you are about to fill some fields."}` and confirm the model can call `action_label`. Record evidence.

- [ ] **Step 8: Commit**

```bash
git add agent/tools/
git commit -m "feat(eve-agent): example tools (action_label, gap_analysis, form_summary, check_submit_gate, browser)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Three subagents

**Files:**
- Create `agent/subagents/database_verification/`: `agent.ts`, `instructions.md`, `tools/get_apricot_record.ts`, `tools/get_apricot_form_fields.ts`
- Create `agent/subagents/requirements_research/`: `agent.ts`, `instructions.md`, `tools/web_search.ts`
- Create `agent/subagents/form_review/`: `agent.ts`, `instructions.md`, `tools/form_summary.ts`

**Interfaces:**
- Consumes: the booting agent from Tasks 1–3.
- Produces: three declared subagents, each surfaced to the root as a delegation tool named after its directory. Each `agent.ts` MUST export a `description` (the root routes on it). Subagents inherit NOTHING from root — each carries its own instructions + tools.

- [ ] **Step 1: `database_verification` subagent config**

`agent/subagents/database_verification/agent.ts`:
```ts
import { defineAgent } from 'eve';

export default defineAgent({
  description:
    'Retrieve a participant\'s Apricot records and resolve every field_NNNN to its confirmed label before any value is trusted. Returns source-tagged, verified data. Delegate here before reasoning about participant data.',
  model: 'anthropic/claude-sonnet-4.6',
});
```
`agent/subagents/database_verification/instructions.md`: copy VERBATIM from `lib/ai/prompts/application-protocol.ts` the sections `## Database Retrieval & Verification` (lines 5–29, including the example), `## Data Provenance (No Fabrication)` (lines 51–63), and `## Field Mapping & Inference Rules` (lines 65–70). Add a one-line lead: "You are the database-verification specialist. Return confirmed, source-tagged participant data to the parent; never guess a field's meaning from its numeric ID."

- [ ] **Step 2: `database_verification` tools (examples)**

`agent/subagents/database_verification/tools/get_apricot_record.ts`:
```ts
import { defineTool } from 'eve/tools';
import { z } from 'zod';

// Example tool. Production logic: lib/ai/tools/apricot/.
export default defineTool({
  description: 'Fetch a participant record (and linked records) from Apricot by participant ID.',
  inputSchema: z.object({ participantId: z.string() }),
  async execute({ participantId }) {
    // Demonstrative stub: production calls the Apricot API (lib/apricot-api.ts).
    return { participantId, fields: {}, note: 'stub — see lib/ai/tools/apricot' };
  },
});
```
`agent/subagents/database_verification/tools/get_apricot_form_fields.ts`:
```ts
import { defineTool } from 'eve/tools';
import { z } from 'zod';

// Example tool. Production logic: lib/ai/tools/apricot/. Resolves field_NNNN -> label.
export default defineTool({
  description: 'Resolve the field_NNNN -> label map for an Apricot form, so raw field IDs can be trusted.',
  inputSchema: z.object({ formId: z.string() }),
  async execute({ formId }) {
    return { formId, labels: {}, note: 'stub — see lib/ai/tools/apricot' };
  },
});
```

- [ ] **Step 2b: `requirements_research` subagent**

`agent/subagents/requirements_research/agent.ts`:
```ts
import { defineAgent } from 'eve';

export default defineAgent({
  description:
    'Research a benefits program\'s application up front and enumerate ALL fields it will require across every page, so gap analysis is complete before form-filling starts. Returns a field checklist.',
  model: 'anthropic/claude-sonnet-4.6',
});
```
`agent/subagents/requirements_research/instructions.md`: lead line "You are the requirements-research specialist. Given a program and locale, return the complete list of fields the whole application will require." then copy VERBATIM the `## Web Search Protocol` (`web-automation.ts:42-47`) and the research step from the Gap Analysis Protocol (`application-protocol.ts:134` — the "Research the application requirements upfront" bullet).
`agent/subagents/requirements_research/tools/web_search.ts`:
```ts
import { defineTool } from 'eve/tools';
import { z } from 'zod';

// Example tool. Production would use the app's web-search integration.
export default defineTool({
  description: 'Search the web for a benefits program application and its required fields.',
  inputSchema: z.object({ query: z.string() }),
  async execute({ query }) {
    return { query, results: [], note: 'stub — wire to production web search' };
  },
});
```

- [ ] **Step 2c: `form_review` subagent**

`agent/subagents/form_review/agent.ts`:
```ts
import { defineAgent } from 'eve';

export default defineAgent({
  description:
    'Walk the application\'s review/summary screen at the end of filling and produce the structured, source-tagged formSummary field list for the caseworker to review before submission.',
  model: 'anthropic/claude-sonnet-4.6',
});
```
`agent/subagents/form_review/instructions.md`: lead line "You are the form-review specialist. Produce the ordered, source-tagged field list for the review card. Never click submit." then copy VERBATIM `## Review Screen (REQUIRED)` (`application-protocol.ts:91-100`) and `## Form Completion Summary` (`application-protocol.ts:151-182`).
`agent/subagents/form_review/tools/form_summary.ts`: same body as the top-level `agent/tools/form_summary.ts` from Task 3 Step 3 (the subagent owns its own copy — subagents inherit no tools). Copy that file's contents verbatim into this path.

- [ ] **Step 3: Boot-verify subagents register and the root can delegate**

Run `npx eve dev`; confirm boot reports three subagents with no "missing description" error. POST `{"message":"Before I reason about participant 12345's data, get their verified records."}` and confirm from the stream that the root calls the `database_verification` delegation tool. Record evidence. If the subagent directory layout errors, check `node_modules/eve/docs/subagents.mdx`.

- [ ] **Step 4: Commit**

```bash
git add agent/subagents/
git commit -m "feat(eve-agent): database_verification, requirements_research, form_review subagents

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Sandbox example + agent compaction config

**Files:**
- Create: `agent/sandbox.ts`
- Modify: `agent/agent.ts`

**Interfaces:**
- Consumes: the full structure from Tasks 1–4.
- Produces: a `defineSandbox` example env and a root `agent.ts` with compaction configured.

- [ ] **Step 1: `agent/sandbox.ts`**

```ts
import { defineSandbox } from 'eve/sandbox';

// Representative sandbox. Omitting `backend` uses defaultBackend() — Vercel
// Sandbox on Vercel, else Docker/microsandbox/just-bash locally.
//
// Two relevant uses for THIS agent:
//  1. The browser-automation skill's sibling reference files
//     ($HOME/.agents/skills/browser-automation/*) are read through the sandbox
//     at runtime via ctx.getSkill(...).file(...).
//  2. Browser automation itself runs today via Kernel.sh as an app-runtime tool
//     (Eve tools run in the app runtime, NOT the sandbox). A future Eve-native
//     port could instead run headless Chromium inside this sandbox — see
//     docs/eve-spike-findings.md "Browser session sketch" (sub-project 3).
export default defineSandbox({
  async onSession({ use }) {
    // Per-session setup would go here (network policy, credentials). Kept
    // minimal for the demonstrative conversion.
    await use();
  },
});
```

- [ ] **Step 2: Update `agent/agent.ts` with compaction config**

```ts
import { defineAgent } from 'eve';

// Model resolves through Vercel AI Gateway (AI_GATEWAY_API_KEY locally; OIDC on
// Vercel). Eve manages context compaction internally (there is no prepareStep
// hook) — configure it here rather than porting lib/ai/context-compression.ts.
// See docs/eve-spike-findings.md Q2.
export default defineAgent({
  model: 'anthropic/claude-sonnet-4.6',
  compaction: {
    // Compact when context passes this fraction of the window (default 0.9).
    thresholdPercent: 0.75,
  },
});
```
Note: `compaction.recentWindowSize` is NOT authorable (hard-coded to 10 in eve@0.27.0 — see findings Q2); do not attempt to set it. Verify the `compaction` key shape against `node_modules/eve/docs/agent-config.md` before relying on it; if `thresholdPercent` is rejected, fall back to `{}` / omit and note it.

- [ ] **Step 3: Boot-verify the full agent**

Run `npx eve dev`; confirm clean boot with sandbox + compaction present and no error. Record the boot banner.

- [ ] **Step 4: Commit**

```bash
git add agent/sandbox.ts agent/agent.ts
git commit -m "feat(eve-agent): sandbox example + internal compaction config

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: End-to-end verification and structure doc

**Files:**
- Create: `agent/README.md`

**Interfaces:**
- Consumes: the full converted agent.
- Produces: a short map of the converted structure + a green end-to-end check.

- [ ] **Step 1: Full boot + capability check**

Run `npx eve dev` (Node 24, key exported). In one session, POST a message that exercises multiple pieces, e.g. `{"message":"I need to apply for CalFresh for participant 12345. What is today, and what's your plan?"}`. From the stream, confirm: (a) the reply states today's date (dynamic instruction), (b) the model either loads a skill or delegates to a subagent, and (c) no runtime error. Record the evidence. This is the demonstrative deliverable's proof.

- [ ] **Step 2: Confirm the spike test still passes and the change is additive**

```bash
export PATH="$HOME/.nvm/versions/node/v24.18.0/bin:$PATH"
pnpm exec vitest run -c vitest.config.node.mjs tests/agent/read-reference.test.ts
git status --porcelain
```
Expected: 4/4 pass; `git status` shows only `agent/**` additions/modifications (and committed docs) — nothing under `lib/`, `app/`, or `package.json`.

- [ ] **Step 3: Write `agent/README.md`**

A short map (for the next engineer): the `agent/` tree, what each slot came from in `lib/ai/prompts/*`, that tools/subagents are demonstrative stubs (real logic lives in `lib/`), that `readReference` is superseded by skills, and a pointer to `docs/eve-spike-findings.md` and this plan. One paragraph per section; no marketing language.

- [ ] **Step 4: Commit**

```bash
git add agent/README.md
git commit -m "docs(eve-agent): structure map for the converted web-automation agent

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review notes

- **Spec coverage:** instructions.md always-on core → Task 1; dynamic date → Task 1; browser-automation + benefits-application skills (incl. 3 reference sibling files) → Task 2; example tools + readReference-superseded annotation → Task 3; three subagents with own instructions+tools → Task 4; sandbox + compaction → Task 5; readReference→skills story surfaced in Tasks 2/3/6; additive-only + lib untouched → Global Constraints + Task 6 Step 2; demonstrative-stub honesty → comments in every tool.
- **Content-placement consistency:** `## Forbidden Actions` is in Task 1 (always-on) and explicitly excluded from the Task 2 browser skill; `## Review Screen` + `## Form Completion Summary` appear in BOTH the `benefits-application` skill (Task 2) and the `form_review` subagent instructions (Task 4) — intentional, because a declared subagent inherits no skills and needs its own copy (per subagents.mdx "duplicate anything the child needs"). `form_summary` tool exists at top-level (Task 3) and duplicated under `form_review` (Task 4) for the same isolation reason.
- **Beta-verify points (not placeholders):** the `instructions/` dir + dynamic shape, skill frontmatter/packaging, subagent directory layout, and the `compaction` key shape are each confirmed against `node_modules/eve/docs/*` and re-checked at author time; every task's boot step is the catch-net.
- **Type/name consistency:** tool file slugs are snake_case (`action_label`, `gap_analysis`, `form_summary`, `check_submit_gate`, `get_apricot_record`, `get_apricot_form_fields`, `web_search`); subagent dirs are `database_verification`, `requirements_research`, `form_review`; model slug `anthropic/claude-sonnet-4.6` throughout; imports: `eve/instructions` (defineInstructions, defineDynamic), `eve/tools` (defineTool), `eve/sandbox` (defineSandbox), `eve` (defineAgent).
