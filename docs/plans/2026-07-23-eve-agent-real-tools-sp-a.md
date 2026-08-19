# SP-A — Make the Eve Agent Functionally Real Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `npx eve dev` drive a real Kernel.sh browser end-to-end and archive Apricot (retargeting the agent's data model to caseworker + inference), so the Eve agent is functionally real standalone.

**Architecture:** Replace the demonstrative stub `browser` and `check_submit_gate` tools with real ones that reuse the existing `@onkernel/sdk` + `agent-browser` path (`getOrCreateBrowser` + `executeCommand`) via a new additive shared helper `lib/kernel/eve-browser.ts`. Eve tools run in the app runtime, so `KERNEL_API_KEY`/`DATABASE_URL` are available and the browser session is resolved per call from `ctx.session.id`. `lib/` legacy code is untouched (the live app keeps building). Apricot is removed from `agent/` only.

**Tech Stack:** Eve `0.27.0` · `@onkernel/sdk` + `agent-browser` (Kernel.sh) · zod 4.4.3 · TypeScript · pnpm · Node 24.

## Global Constraints

- **Node 24 for every command.** Shell defaults to v20.15.0; state does NOT persist between Bash calls. Prefix EVERY node/npx/pnpm command with `export PATH="$HOME/.nvm/versions/node/v24.18.0/bin:$PATH"`; verify `node -v`=v24.
- **Secrets** live in `.env.local` (gitignored — never print/commit): `AI_GATEWAY_API_KEY`, `KERNEL_API_KEY`, `DATABASE_URL`. `npx eve dev` does NOT auto-load `.env.local`; export the needed vars first, e.g. `set -a; . ./.env.local; set +a` (or export individually).
- **pnpm only.** Never commit `node_modules`, `package-lock.json`, or `.eve/`.
- **Additive to `lib/`; changes to `agent/` only otherwise.** Create `lib/kernel/eve-browser.ts` (new); do NOT modify `lib/kernel/browser.ts`, `lib/ai/tools/*`, `app/`, `route.ts`, or `package.json`. The legacy build must still compile. Do NOT touch `lib/apricot-api.ts` / `lib/ai/tools/apricot/*` (those are removed in the SP-C cutover, not here).
- **Apricot is archived within `agent/` only.** Delete `agent/subagents/database_verification/`; retarget prose in `agent/instructions.md` and `agent/skills/benefits-application/SKILL.md` to caseworker + inference. Keep the anti-fabrication provenance framework, just without Apricot.
- **Never click submit.** `check_submit_gate` force-enables but never clicks; the never-submit rule in `agent/instructions.md` stays intact.
- Eve tool `execute(input, ctx)` — `ctx.session.id` is the stable session id; `ctx.session.auth.current` is `SessionAuthContext | null`. Confirm the principal field on `SessionAuthContext` at implementation (`grep -rEn "interface SessionAuthContext" node_modules/eve/dist`); use a `'eve-local'` fallback for standalone.
- Boot check each task: `npx eve dev` starts without registration/schema errors.

---

### Task 1: Real browser tool via Kernel (the crux — verify stateful multi-call first)

**Files:**
- Create: `lib/kernel/eve-browser.ts`
- Modify (replace stub): `agent/tools/browser.ts`

**Interfaces:**
- Consumes: `getOrCreateBrowser(sessionId, userId)` and the `BrowserSession.browserManager` from `lib/kernel/browser.ts`; `executeCommand` from `agent-browser/dist/actions.js`; Eve `ToolContext` (`ctx.session.id`, `ctx.session.auth`).
- Produces: `browserIdentity(ctx): { sessionId: string; userId: string }` and `runBrowserCommand(ctx, params: Record<string, unknown>): Promise<{ success: boolean; data?: unknown; error?: string }>` (the `executeCommand` response), for reuse by Task 2.

- [ ] **Step 1: Confirm the `SessionAuthContext` principal field**

Run:
```bash
grep -rEn "interface SessionAuthContext|SessionAuthContext = " node_modules/eve/dist/src 2>/dev/null | head
grep -rEn "subject|principal|userId|id\??:" node_modules/eve/dist/src/channel/types.d.ts 2>/dev/null | head
```
Note the field that identifies the caller (e.g. `subject`, `id`). Use it in Step 2's `browserIdentity`; if none/uncertain, the `'eve-local'` fallback covers standalone.

- [ ] **Step 2: Write the shared Kernel helper**

Create `lib/kernel/eve-browser.ts`:
```ts
import { nanoid } from 'nanoid';
import { executeCommand } from 'agent-browser/dist/actions.js';
import type { Command } from 'agent-browser/dist/types.js';
import { getOrCreateBrowser } from '@/lib/kernel/browser';
import type { ToolContext } from 'eve/tools';

const COMMAND_TIMEOUT_MS = 120_000; // 2 minutes — Kernel commands can hang.

// Per-session serialization: Playwright's page is not concurrency-safe, so if
// Eve ever dispatches two browser commands for the same session concurrently we
// queue them. Keyed by the Eve session id; lives for the eve-dev process.
const sessionQueues = new Map<string, Promise<unknown>>();
function withSessionQueue<T>(sessionId: string, fn: () => Promise<T>): Promise<T> {
  const prev = sessionQueues.get(sessionId) ?? Promise.resolve();
  const next = prev.then(fn, fn);
  sessionQueues.set(sessionId, next.then(() => {}, () => {}));
  return next;
}

// Stable browser-session identity from Eve's session context. Re-resolved on
// EVERY call (no module-held Playwright handle passed around), which is the
// durable-safe pattern from the spike's browser sketch. getOrCreateBrowser's
// own in-memory cache reuses the live BrowserManager within the eve-dev process.
export function browserIdentity(ctx: ToolContext): { sessionId: string; userId: string } {
  const sessionId = ctx.session.id;
  // Standalone `eve dev` has no channel auth. getOrCreateBrowser requires a
  // non-empty userId for cache-key isolation, so fall back to a constant.
  // Replace `.subject` with the confirmed SessionAuthContext principal field.
  const userId =
    (ctx.session.auth.current as { subject?: string } | null)?.subject ??
    'eve-local';
  return { sessionId, userId };
}

export async function runBrowserCommand(
  ctx: ToolContext,
  params: Record<string, unknown>,
): Promise<{ success: boolean; data?: unknown; error?: string }> {
  const { sessionId, userId } = browserIdentity(ctx);
  return withSessionQueue(sessionId, async () => {
    const session = await getOrCreateBrowser(sessionId, userId);
    const command = { id: nanoid(), ...params } as Command;
    return Promise.race([
      executeCommand(command, session.browserManager),
      new Promise<never>((_, reject) =>
        setTimeout(
          () => reject(new Error('Command timed out after 2 minutes')),
          COMMAND_TIMEOUT_MS,
        ),
      ),
    ]);
  });
}
```
Note: `ctx.session.id` (an Eve id like `wrun_...`) is used directly as the Kernel session key. `getOrCreateBrowser` also derives a replay/`chatId` from `${chatId}-${userId}`; an Eve id won't match that suffix, so replay archival is simply skipped — exactly SP-A's intent (replay/mapping is SP-C).

- [ ] **Step 3: Replace the stub browser tool**

Overwrite `agent/tools/browser.ts` (keep the full command reference in the description; keep the complete legacy input schema so every action works):
```ts
import { defineTool } from 'eve/tools';
import { z } from 'zod';
import { runBrowserCommand } from '@/lib/kernel/eve-browser';

export default defineTool({
  description: `Execute browser automation commands on a remote Kernel browser.

Send a structured command with an "action" field and action-specific parameters. See the browser-automation skill for snapshot discipline, selector strategy, and workflow rules. Snapshot before interacting; re-snapshot after every DOM change. NEVER navigate away from the target application domain and NEVER click the final submit button.

Actions: navigate, snapshot (optional selector / interactive), click, fill, type (clear?), select (values[]), getbylabel (subaction), press (key), hover, check, uncheck, scrollintoview, wait (selector or timeout), waitforloadstate (state), gettext, inputvalue, url, title, scroll (direction/amount), screenshot, back, forward, evaluate (script — reading only), tab_list/tab_switch/tab_new/tab_close, dialog (response), frame/mainframe.`,
  inputSchema: z.object({
    action: z.string(),
    selector: z.string().optional(),
    value: z.string().optional(),
    text: z.string().optional(),
    url: z.string().optional(),
    key: z.string().optional(),
    label: z.string().optional(),
    subaction: z.string().optional(),
    script: z.string().optional(),
    values: z.array(z.string()).optional(),
    timeout: z.number().optional(),
    amount: z.number().optional(),
    delay: z.number().optional(),
    interactive: z.boolean().optional(),
    clear: z.boolean().optional(),
    direction: z.string().optional(),
    state: z.string().optional(),
    index: z.number().optional(),
    response: z.string().optional(),
    promptText: z.string().optional(),
  }),
  async execute(params, ctx) {
    try {
      const response = await runBrowserCommand(ctx, params);
      if (response.success) {
        const output =
          typeof response.data === 'string'
            ? response.data
            : JSON.stringify(response.data);
        return { success: true, output, error: null };
      }
      return { success: false, output: null, error: response.error ?? 'command failed' };
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : String(error);
      return { success: false, output: null, error: message };
    }
  },
});
```

- [ ] **Step 4: Boot-verify + THE CRUX (multi-call ref survival)**

Start the server (Node 24, env loaded):
```bash
export PATH="$HOME/.nvm/versions/node/v24.18.0/bin:$PATH"
set -a; . ./.env.local; set +a
npx eve dev --no-ui --port 2010
```
In one Eve session, send a message that forces a real multi-step browser flow, e.g.:
`{"message":"Navigate to https://example.com, snapshot the page, then tell me the exact text of the top heading."}`
Read the session stream. Expected: the `browser` tool executes `navigate` then `snapshot`, the snapshot returns REAL page content (refs like `@e1` and the "Example Domain" heading), and the model reports the heading text. Then, in the SAME session, send a follow-up that must use a ref from the prior snapshot (e.g. "click the 'More information' link") and confirm the click resolves the ref — **this proves the BrowserManager (and its ref map) survived across tool calls** in the eve-dev process.

**If refs do NOT survive across calls** (the follow-up ref errors as unknown/stale even though the snapshot just returned it), STOP and report BLOCKED: the durable-execution model is re-instantiating state between tool calls, and the browser tool needs a different design (e.g. attach-by-Kernel-session-id without relying on the in-memory `sessions` cache, or persisting the ref map). This is the spec's primary gating risk — do not build Task 2+ on top of a broken assumption.

Kill the server when done. Record the full evidence (both turns) in the report.

- [ ] **Step 5: Commit**

```bash
git add lib/kernel/eve-browser.ts agent/tools/browser.ts
git commit -m "feat(eve-agent): real Kernel browser tool (durable-safe per-call session resolution)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Real check_submit_gate

**Files:**
- Modify (replace stub): `agent/tools/check_submit_gate.ts`

**Interfaces:**
- Consumes: `runBrowserCommand(ctx, params)` from Task 1's `lib/kernel/eve-browser.ts`.
- Produces: a `check_submit_gate` tool returning `{ success, state, action, error }`.

- [ ] **Step 1: Port the real logic**

Overwrite `agent/tools/check_submit_gate.ts`, copying the `PROBE_SCRIPT` and `FORCE_ENABLE_SCRIPT` constants VERBATIM from `lib/ai/tools/check-submit-gate.ts` (do not alter the DOM logic), and driving them through `runBrowserCommand`:
```ts
import { defineTool } from 'eve/tools';
import { z } from 'zod';
import { runBrowserCommand } from '@/lib/kernel/eve-browser';

// PROBE_SCRIPT and FORCE_ENABLE_SCRIPT copied verbatim from
// lib/ai/tools/check-submit-gate.ts — do not modify the DOM logic.
const PROBE_SCRIPT = `<copy verbatim from lib/ai/tools/check-submit-gate.ts>`;
const FORCE_ENABLE_SCRIPT = (selector: string, callbackName: string | null) =>
  `<copy verbatim from lib/ai/tools/check-submit-gate.ts>`;

export default defineTool({
  description:
    'On a page with a Cloudflare Turnstile widget where the submit button is stuck disabled, probe the DOM and (if forceEnable) force-enable the button so the caseworker can take control and submit. Never clicks submit. Do not call on pages without a Turnstile widget.',
  inputSchema: z.object({
    forceEnable: z.boolean().default(true),
  }),
  async execute({ forceEnable }, ctx) {
    try {
      const probe = await runBrowserCommand(ctx, { action: 'evaluate', script: PROBE_SCRIPT });
      if (!probe.success) return { success: false, error: probe.error ?? 'probe failed', state: null, action: null };
      if (!probe.data) return { success: false, error: 'probe returned no data', state: null, action: null };
      const state = typeof probe.data === 'string' ? JSON.parse(probe.data) : probe.data;

      if (!forceEnable || !state.submit?.found || state.submit?.disabled !== true) {
        return { success: true, state, action: null };
      }
      const enable = await runBrowserCommand(ctx, {
        action: 'evaluate',
        script: FORCE_ENABLE_SCRIPT(state.submit.selector, state.turnstile.callbackName),
      });
      const action = enable.success
        ? (typeof enable.data === 'string' ? JSON.parse(enable.data) : enable.data)
        : { error: enable.error ?? 'force-enable failed' };
      return { success: true, state, action };
    } catch (error: unknown) {
      return { success: false, error: error instanceof Error ? error.message : String(error), state: null, action: null };
    }
  },
});
```

- [ ] **Step 2: Boot-verify**

`npx eve dev` (as Task 1 Step 4). Confirm clean boot and the tool registers. A full Turnstile-page test needs such a page, so at minimum confirm: on `https://example.com`, a turn instructed to "check the submit gate" calls `check_submit_gate`, the probe `evaluate` runs against the real page, and it returns `state` with `submit.found: false` (no submit on example.com) and does NOT force-enable. Record the evidence.

- [ ] **Step 3: Commit**

```bash
git add agent/tools/check_submit_gate.ts
git commit -m "feat(eve-agent): real check_submit_gate (Turnstile probe + force-enable via Kernel)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: De-stub the card tools

**Files:**
- Modify: `agent/tools/action_label.ts`, `agent/tools/gap_analysis.ts`, `agent/tools/form_summary.ts`

**Interfaces:**
- Consumes: nothing new.
- Produces: the same three tools, comments updated to reflect their real standalone behavior (return validated structured data; card rendering is SP-B).

- [ ] **Step 1: Update comments only (behavior already correct)**

These tools already validate their zod input and return structured data — that IS their real standalone behavior; only the "Demonstrative stub" framing is wrong now. In each of the three files, replace the top "Example tool. Production logic: ..." comment with a note like:
```ts
// Returns validated structured data for the <name> card. The interactive card
// RENDER is wired to the chat UI in SP-B; standalone this tool's job is to
// validate + surface the data, which it does here.
```
Do NOT change the schemas or the `execute` return shape. For `action_label`, keep it as the lightweight real signal it is (returns `{ labeled: category }`).

- [ ] **Step 2: Boot-verify**

`npx eve dev`; confirm clean boot and that `action_label` / `gap_analysis` / `form_summary` still register. A turn asking the agent to "label that you're filling fields" should still fire `action_label`.

- [ ] **Step 3: Commit**

```bash
git add agent/tools/action_label.ts agent/tools/gap_analysis.ts agent/tools/form_summary.ts
git commit -m "refactor(eve-agent): de-stub card tools (real standalone behavior; render in SP-B)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Archive Apricot; retarget data model to caseworker + inference

**Files:**
- Delete: `agent/subagents/database_verification/` (agent.ts, instructions.md, tools/get_apricot_record.ts, tools/get_apricot_form_fields.ts)
- Modify: `agent/instructions.md`, `agent/skills/benefits-application/SKILL.md`

**Interfaces:**
- Consumes: nothing.
- Produces: an Apricot-free agent whose remaining subagents are `requirements_research` and `form_review`, and whose data-provenance rule is sourced to caseworker + inference + missing.

- [ ] **Step 1: Delete the database_verification subagent**

```bash
git rm -r agent/subagents/database_verification
```

- [ ] **Step 2: Retarget `agent/instructions.md`**

Two edits (leave everything else, including the never-submit block, unchanged):
1. In the Applicant Identity section, the "Applicant's age unknown" line references confirming DOB via `getApricotFormFields`. Replace that clause so age/DOB comes from the caseworker: change "Check the database for date of birth (confirm the field via `getApricotFormFields` — see Data Provenance). If still unknown, clarify with the caseworker before choosing an option." to "Ask the caseworker for the date of birth. If still unknown, clarify with the caseworker before choosing an option."
2. In the top "on-demand procedures / delegation" note, the subagent list must read only `requirements_research` and `form_review` (remove any mention of database verification / Apricot).

- [ ] **Step 3: Retarget `agent/skills/benefits-application/SKILL.md`**

Edits (preserve all non-Apricot procedure text):
1. **Gap Analysis Protocol** — anywhere it says data must be "traceable to a confirmed Apricot field or a caseworker message," drop the Apricot half: data is traceable to a caseworker message or a valid inference. Remove the sentence about a "`field_NNNN` value whose label you have NOT verified via `getApricotFormFields`."
2. **Form Completion Summary** — the `source` enum becomes `caseworker | inferred | missing` (delete the `database` bullet entirely). Remove the phrase "only valid if you've confirmed the field label via `getApricotFormFields`" and any other `getApricotFormFields` / Apricot mention.
3. **Add a retargeted Data Provenance section** near the top of the skill (adapted from the deleted subagent's version, Apricot removed):
```md
## Data Provenance (No Fabrication)

Every value you fill into a form, exclude from a gap analysis, or mark as filled in `formSummary` MUST trace to ONE of these sources:

1. **Caseworker message this session** — an explicit value the caseworker typed in this conversation. (Mark `source: "caseworker"`.)
2. **Inference from a caseworker message** — a value you reasoned from what the caseworker provided (e.g., "lives alone — no household members mentioned"). (Mark `source: "inferred"`.)

If a value does not trace to one of these, it does not exist. Do not type it into the form and mark the field missing (`source: "missing"`, no value). This applies to every field. **Shape is not identity**: a 9-digit number is not an SSN, a date in the right range is not a DOB, "this is probably what it would be" is fabrication. Before every gap-analysis, form-fill, and `formSummary` call, name each value's source — which caseworker message, or which inference from one. If you cannot name one, the field is missing.
```
4. Confirm no other Apricot/`getApricotRecord`/`getApricotFormFields`/"database source" references remain in the skill.

- [ ] **Step 4: Boot-verify no Apricot remains**

```bash
export PATH="$HOME/.nvm/versions/node/v24.18.0/bin:$PATH"
grep -rni "apricot\|getApricotFormFields\|getApricotRecord\|database_verification" agent/ && echo "!! apricot refs remain (fix)" || echo "OK: no apricot refs"
set -a; . ./.env.local; set +a
npx eve dev --no-ui --port 2010   # confirm clean boot, now 2 subagents; kill after
```
Expected: the grep prints "OK: no apricot refs"; the server boots with `requirements_research` + `form_review` only, no missing-subagent error. (The `data-*`/`source` enum note in the benefits skill now lists caseworker/inferred/missing.) Record evidence.

- [ ] **Step 5: Commit**

```bash
git add -A agent/
git commit -m "feat(eve-agent): archive Apricot; retarget data model to caseworker + inference

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: End-to-end verification

**Files:**
- Modify: `agent/README.md` (update to reflect real tools + Apricot removal)

**Interfaces:**
- Consumes: the whole SP-A agent.
- Produces: a green end-to-end check and an accurate README.

- [ ] **Step 1: Full real-browser turn**

`npx eve dev` (Node 24, env loaded). One session: `{"message":"Go to https://example.com and read me the main heading and the first paragraph."}`. Confirm the `browser` tool did a real navigate + snapshot and the reply quotes the real page text. Record evidence. This is SP-A's headline proof (real automation, no Apricot).

- [ ] **Step 2: Regression + additive checks**

```bash
export PATH="$HOME/.nvm/versions/node/v24.18.0/bin:$PATH"
pnpm exec vitest run -c vitest.config.node.mjs tests/agent/read-reference.test.ts   # expect 4/4
git diff --stat $(git merge-base HEAD main)..HEAD -- app package.json                # expect empty (SP-A touched neither)
git diff --stat HEAD~5..HEAD -- lib/kernel/browser.ts lib/ai/tools                    # expect empty (legacy untouched; only lib/kernel/eve-browser.ts is new)
```
Confirm: 4/4; `app/` + `package.json` untouched by SP-A; legacy `lib/kernel/browser.ts` and `lib/ai/tools/*` unmodified (only the new `lib/kernel/eve-browser.ts` was added).

- [ ] **Step 3: Update `agent/README.md`**

Update the tool section: `browser` and `check_submit_gate` are now REAL (Kernel.sh via `lib/kernel/eve-browser.ts`, session resolved from `ctx.session.id`); `gap_analysis`/`form_summary` return data (render in SP-B); the `database_verification` subagent and all Apricot references are removed and the data model is caseworker + inference. Remove now-stale Apricot rows from the tool-name map. Keep it accurate to what shipped.

- [ ] **Step 4: Commit**

```bash
git add agent/README.md
git commit -m "docs(eve-agent): README — real browser/submit-gate, Apricot removed (SP-A)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review notes

- **Spec coverage:** real Kernel browser → Task 1; durable-safe per-call session resolution + the stateful-crux mini-spike → Task 1 Step 4 (gating BLOCKED path included); real check_submit_gate → Task 2; card tools return data → Task 3; archive Apricot + retarget provenance to caseworker+inference (relocated into benefits skill) → Task 4; standalone validation + additive-to-lib + spike test green → Task 5; non-goals (no UI render, no replay/DB-mapping build, no legacy removal, no web-search port, no Vercel) respected — no task touches `route.ts`/`lib/apricot-api.ts`/`app/`/`package.json` or builds card rendering.
- **Grounded vs verify-at-impl:** the browser/submit-gate reuse (`getOrCreateBrowser` + `executeCommand`), `ctx.session.id`/`ctx.session.auth`, and `defineTool(execute(input, ctx))` are confirmed against the installed package and legacy code. The one verify-at-impl item is the exact `SessionAuthContext` principal field (Task 1 Step 1) — covered by the `'eve-local'` fallback regardless.
- **Type/name consistency:** `browserIdentity(ctx)` and `runBrowserCommand(ctx, params)` defined in Task 1 and consumed verbatim by Task 2; tool file slugs stay snake_case; the `source` enum is `caseworker | inferred | missing` consistently in Task 4; model/env constraints match SP-A spec.
- **Primary risk front-loaded:** if Task 1's ref-survival crux fails, execution stops before Tasks 2–5 build on a broken browser assumption — exactly as the spec requires.
