# Eve Agent — Structure Map

This directory is a demonstrative Eve conversion of the caseworker-facing web-automation
system prompt that otherwise lives in `lib/ai/prompts/`. It exists to prove out Eve's
concepts (always-on instructions + dynamic instructions, skills, tools, subagents,
sandbox, compaction) against real prompt content from this repo — not to replace the
production Next.js agent loop in `app/(chat)/api/chat/route.ts`. The only file added
outside `agent/` across this whole conversion is `lib/kernel/eve-browser.ts`; no
*existing* file under `lib/` or `app/` was modified — see "Additive-only" below.

## Tree

```
agent/
  agent.ts                                  top-level defineAgent (model + compaction)
  channels/eve.ts                           eve dev/TUI + Vercel channel wiring
  instructions.md                           always-on core instructions
  instructions/date.ts                      dynamic per-session date instruction
  sandbox.ts                                representative sandbox definition
  skills/
    benefits-application/SKILL.md           benefits-application protocol
    browser-automation/
      SKILL.md                              browser mechanics + modal handling
      field-patterns.md                     sibling reference (loaded on demand)
      custom-dropdowns.md                   sibling reference (loaded on demand)
      browser-commands.md                   sibling reference (loaded on demand)
  tools/
    action_label.ts                         real: returns the label as data
    browser.ts                              real: drives a Kernel.sh browser session
    check_submit_gate.ts                    real: Turnstile probe + force-enable via Kernel
    form_summary.ts                         real: validates + returns card data
    gap_analysis.ts                         real: validates + returns card data
    read_reference.ts                       spike proof tool, superseded by skills
    update_working_memory.ts                defineState compaction prototype
  subagents/
    requirements_research/                  agent.ts + instructions.md + 1 tool
    form_review/                            agent.ts + instructions.md + 1 tool
```

The external-record-verification subagent and its two lookup tools have been
archived out of this agent (Task 4 of the SP-A plan), and the data model is
retargeted to caseworker messages + inference only. See
`docs/plans/2026-07-23-web-automation-prompt-to-eve.md` and the SDD task
brief/report under `.superpowers/sdd/` for the before/after.

## Instructions and Dynamic Date

`agent/instructions.md` is the always-on core. It carries identity, Applicant Identity,
Core Approach, Step Management, and Action Labeling from the top level of
`lib/ai/prompts/web-automation.ts`, but not that file's entire top level: Web Search
Protocol moved to the `requirements_research` subagent and Resuming After Interruption
moved to the `browser-automation` skill, since both are situational rather than
always-needed. It also carries two sections promoted from elsewhere for the same
reason — always-needed, not situational: Communication Rules (originally in
`lib/ai/prompts/application-protocol.ts`) and Forbidden Actions (originally in
`lib/ai/prompts/browser-and-forms.ts`). `agent/instructions/date.ts` replaces
`getCurrentDateString()` from `web-automation.ts` with a `defineDynamic` resolver on
`session.started`, so the date is computed fresh per session rather than baked in at
build time.

## Skills

`agent/skills/browser-automation/` carries the bulk of `lib/ai/prompts/browser-and-forms.ts`
(Core Workflow, Ref Format, Snapshot Modes, Selector Rules, Masked Fields, Field Type
Patterns, Custom Dropdowns, Multi-Page Forms, Dynamic/Conditional Fields, Modal Handling,
Error Recovery, Form Submission Protocol, Parameter Types) plus Resuming After
Interruption from `web-automation.ts`, minus Forbidden Actions (moved to always-on
instructions, see above). Its three sibling reference files
(`field-patterns.md`, `custom-dropdowns.md`, `browser-commands.md`) are verbatim copies
of `lib/ai/prompts/references/*.md`, loaded on demand through Eve's skill mechanism
rather than a tool call.

`agent/skills/benefits-application/` carries most of `lib/ai/prompts/application-protocol.ts`
(Autofilled Field Detection, Filling Fields, No vs Unknown Distinction, Autonomous
Progression, Review Screen, Gap Analysis Protocol, Form Completion Summary). Data
Provenance now lives directly in this skill as **Data Provenance (No Fabrication)**,
sourced to caseworker messages + inference only.
Review Screen and Form Completion Summary are intentionally duplicated in the `form_review`
subagent — a declared subagent inherits no skills, so anything it needs must be copied into
its own `instructions.md`.

## Tools

The five tools under `agent/tools/` are no longer stubs — SP-A made them functionally
real. `browser` and `check_submit_gate` drive an actual remote browser on Kernel.sh
through `lib/kernel/eve-browser.ts`, a new module that reimplements the minimal slice of
the production `getOrCreateBrowser` needed for a
working browser tool (create-or-reuse a Kernel session + `BrowserManager`, cached by
session, re-resolved on every call from `ctx.session.id` — see the file-level comment in
`lib/kernel/eve-browser.ts` for why it doesn't import `getOrCreateBrowser` directly and
what it drops relative to it: Kernel replay/GCS archival and the `SessionMapping` DB
upsert, both deferred to a later sub-project). `action_label`, `form_summary`, and
`gap_analysis` validate their input and return real structured data (`{ labeled: ... }`,
`{ rendered: true, fieldCount }`, `{ rendered: true, formName, missingCount }`) — the
interactive card *render* stays deferred to SP-B, but the data path is no longer a stub.
None of these five replace the production tools, which stay wired into
`app/(chat)/api/chat/route.ts` as before; `agent/tools/read_reference.ts` is the original spike proof tool (reads
`lib/ai/prompts/references/*.md` off disk) and is superseded by the
`browser-automation` skill's sibling reference files — it is retained only as a record
of the pre-skills approach, not as the recommended pattern for new reference material.
`agent/tools/update_working_memory.ts` is not derived from any prompt file; it is a
Task 5 prototype using `defineState` from `eve/context` to persist participant/form
data outside the model's message history so it survives Eve's internal compaction —
see `docs/eve-spike-findings.md` Q2.

## Subagents

Two declared subagents each get their own `agent.ts` (description + model) and
`instructions.md`, since a subagent inherits none of the top-level instructions or
skills and must carry everything it needs. `requirements_research` carries Web Search
Protocol from `web-automation.ts` plus its own `web_search` tool. `form_review` carries
Review Screen and Form Completion Summary from `application-protocol.ts` (duplicated with
the `benefits-application` skill, for the reason given above) plus its own `form_summary`
tool. A third subagent that previously carried Data Provenance and Field Mapping &
Inference Rules plus a pair of external-record-lookup tools was deleted in Task 4 when
that integration was archived; see the Tree section above and the SDD task report for
details.

`requirements_research/instructions.md` also carries, verbatim, step 1 of the Gap
Analysis Protocol ("Research the application requirements upfront…") from
`application-protocol.ts` — the same sentence that opens the `benefits-application`
skill's Gap Analysis Protocol. This is a second intended duplication alongside the
`form_summary`/Review-Screen one above: that step is squarely this subagent's job, so
it is copied in rather than referenced across agents.

## Sandbox, Compaction, and Channel

`agent/agent.ts` is the top-level `defineAgent`, configuring the model and Eve's
built-in compaction (`thresholdPercent`) instead of porting
`lib/ai/context-compression.ts` or a `prepareStep` model-switch hook — Eve manages
context internally and exposes no such hook. `agent/sandbox.ts` is a representative
`defineSandbox` showing where per-session setup would go and how a skill's sibling
reference files are reached at runtime via `ctx.getSkill(...).file(...)`; it is not
tied to any specific prompt file. `agent/channels/eve.ts` predates Tasks 1–5 — it is
the minimal scaffold from the original spike (`eve dev` / TUI / Vercel OIDC wiring),
not something converted from the prompt.

## Additive-only

No *existing* file under `lib/`, `app/`, or `package.json` was modified across this
whole conversion — the only change any of these three paths saw is one new file, added
by SP-A: `lib/kernel/eve-browser.ts`. `git diff --stat d2bbaf0..HEAD -- app package.json`
is empty; `git diff --stat d2bbaf0..HEAD -- lib` shows only that one addition. The
converted agent otherwise lives entirely under `agent/`, alongside the unmodified
production code it mirrors — see `lib/kernel/eve-browser.ts`'s own file-level comment
for why it's a new file instead of a direct import of `lib/kernel/browser.ts`'s
`getOrCreateBrowser`.

## Cross-Reference Note

Because prompt sections were split verbatim across `instructions.md`, the two skills,
and the subagents, a few in-text references now point across files instead of
to a section in the same file. The source files were not reworded to fix this — the
point of the conversion was verbatim fidelity to the original prompt text — so use
this table to find the referenced section. (Task 4 removed the third subagent that
used to be the target of the Data Provenance cross-references below; Data Provenance
now lives directly in `skills/benefits-application/SKILL.md`, so those two rows below
resolve locally rather than across files.)

| Reference text | Found in | Actually lives in |
|---|---|---|
| "See **Data Provenance** above" | `skills/benefits-application/SKILL.md` (Gap Analysis Protocol) | same file — **Data Provenance (No Fabrication)**, near the top |
| "see the **Data Provenance (No Fabrication)** section in the `benefits-application` skill" | `subagents/form_review/instructions.md` (Form Completion Summary) | `skills/benefits-application/SKILL.md` — Data Provenance (No Fabrication) |
| "follow Web Search Protocol normally" | `skills/browser-automation/SKILL.md` (Resuming After Interruption) | `subagents/requirements_research/instructions.md` — Web Search Protocol |
| "follow the Modal Handling section above" | `instructions.md` (Forbidden Actions, `evaluate` restrictions) | `skills/browser-automation/SKILL.md` — Modal Handling |

### Tool names

The ported prose (`instructions.md`, both skills, both subagents'
`instructions.md`) keeps the original camelCase tool names from
`lib/ai/tools/` verbatim — `gapAnalysis`, `formSummary`,
`checkSubmitGate`, `actionLabel` — and `readReference`
is also mentioned by name in one retained comment. Verbatim fidelity was the
point of the conversion (see the note above the reference table), so these
were not reworded. But Eve does not read a `name` field out of `defineTool` —
none of the tool files in `agent/tools/`, `agent/subagents/*/tools/` declare
one — it registers each tool under its snake_case **file slug**. A reader
following the prose literally would be pointed at a name that does not
resolve. Map:

| Prose name (camelCase) | Registered as (file slug) | File |
|---|---|---|
| `gapAnalysis` | `gap_analysis` | `agent/tools/gap_analysis.ts` |
| `formSummary` | `form_summary` | `agent/tools/form_summary.ts` (root) / `agent/subagents/form_review/tools/form_summary.ts` (subagent) |
| `checkSubmitGate` | `check_submit_gate` | `agent/tools/check_submit_gate.ts` |
| `actionLabel` | `action_label` | `agent/tools/action_label.ts` |
| `readReference` | `read_reference` | `agent/tools/read_reference.ts` |

Separately: `benefits-application/SKILL.md`'s Gap Analysis Protocol step 1
and `requirements_research/instructions.md`'s Web Search Protocol both say
"web search," but there is no root-level `web_search` tool — it lives only
on the `requirements_research` subagent (`agent/subagents/requirements_research/tools/web_search.ts`).
Prose written before the subagent split points at a tool that has since
moved out of the top-level tool surface.

### Sanctioned rewording: readReference → skills

One part of the ported prose was deliberately reworded, not kept verbatim,
and that is the exception to the "prose ported verbatim" rule stated above.
The `browser-automation` skill's **Field Type Patterns**, **Custom
Dropdowns**, and **Reference Files** sections replace the original
`readReference({ path: … })` tool-call mentions with "load the sibling
reference file `<name>.md`" — reflecting that these three reference files
are now reached through Eve's skill mechanism (`ctx.getSkill(...).file(...)`,
see Sandbox below) rather than the superseded `read_reference` tool. This is
the documented readReference → skills supersession (see "Tools" above), not
a verbatim-fidelity violation.

## Using Eve from the app UI (SP-B)

SP-B wires this Eve agent into the Next.js chat UI as an alternative transport,
behind a flag, alongside the existing production loop
(`app/(chat)/api/chat/route.ts`) — it does not replace it. The new pieces are
`app/(chat)/api/eve-chat/route.ts` (an authenticated adapter route),
`lib/ai/eve/eve-client.ts` (HTTP client for Eve's session API),
`lib/ai/eve/stream-adapter.ts` (Eve NDJSON → AI SDK `UIMessageStream` chunks,
plus the tool-name map documented above), and
`lib/ai/eve/session-continuity.ts` (an in-memory map from `userId:chatId` to
the Eve session id + continuation token).

### Running it

Two servers, side by side:

```bash
# Terminal 1 — the Eve agent server (Node 24, secrets loaded)
export PATH="$HOME/.nvm/versions/node/v24.18.0/bin:$PATH"
set -a; . ./.env.local; set +a
npx eve dev --no-ui --port 2000

# Terminal 2 — the Next app, pointed at that Eve server
EVE_SERVER_URL=http://127.0.0.1:2000 pnpm dev
```

`EVE_SERVER_URL` defaults to `http://127.0.0.1:2000` if unset (see
`lib/ai/eve/eve-client.ts`), so it only needs to be set explicitly when Eve is
running on a different port. The Next side needs no Eve import and no Node 24
— it only does HTTP + AI SDK stream translation.

### Enabling the flag

The chat transport is chosen by the `useEveAgent` feature flag
(`lib/feature-flags.ts`), default OFF. Either:

- Dev flag menu — the "Flags" button (flask icon) in the chat header, dev/preview
  only — toggle "Use Eve agent", or
- `localStorage['ff:useEveAgent'] = 'true'` in the browser console.

**Reload the page after toggling.** `components/chat.tsx` reads the flag once,
at first render (`isFeatureEnabled('useEveAgent')`, not the reactive
`useFeatureFlag` hook), to pick the transport's `api` URL. Toggling the flag
without reloading leaves the current chat instance on its original transport —
this is the single most likely point of tester confusion.

### Manual end-to-end checklist

The full browser round-trip (real Kernel session, interactive cards, a login
session, follow-up continuity) requires a human driving a browser and is not
covered by the automated test suite. Run this checklist manually after the
two servers above are up:

1. Sign in (or continue as guest, if enabled) and open a chat.
2. Confirm the flag is ON and the page has been reloaded since toggling it.
3. Send a simple task, e.g. "Go to example.com and read me the main heading."
   Confirm: text streams token-by-token, and the network tab shows the POST
   going to `/api/eve-chat` (not `/api/chat`).
4. Confirm the `browser` tool actually runs — a real Kernel session, not a
   stub — and the reply quotes real page content back.
5. Send a task that reaches gap analysis (e.g. ask it to start a benefits
   application with some fields missing). Confirm the `gap_analysis` tool
   call renders as an interactive card (`tool-gapAnalysis` in
   `components/message.tsx`), not a raw tool-call block.
6. Fill the card and submit. Confirm the follow-up turn continues the *same*
   Eve session rather than starting a new one. There is no visible session id
   in the UI to check directly — continuity is tracked server-side, in memory,
   keyed by `userId:chatId` (`lib/ai/eve/session-continuity.ts`); the
   observable proof is behavioral: the agent's next reply reflects the
   values you just filled in (e.g. references them or proceeds past that gap)
   without you having to repeat them.
7. Continue to form completion. Confirm `form_summary` renders as a card
   (`tool-formSummary`), not raw tool-call output.
8. Toggle the flag OFF, reload, send another message. Confirm the network tab
   now shows `/api/chat` (the legacy route) and behavior is unchanged from
   before SP-B.
9. Stop the Eve server (`npx eve dev` process) and, with the flag ON, send a
   message. Confirm the UI surfaces a clean error rather than hanging or
   crashing the page — a request is routed to the adapter, which cannot reach
   `EVE_SERVER_URL` and returns `offline:chat`
   (machine-verified separately via `curl`; see the SP-B task 6 report).

### What is and isn't covered here

- Continuity is **in-memory only** — it is lost on a server restart and does
  not survive across multiple server instances. Postgres-backed continuity is
  SP-C.
- Chat history/persistence through this path (writing Eve turns to the
  `message` table, resumable streams) is **not implemented** — SP-C.
- The legacy `/api/chat` route is untouched and remains the default; removing
  it is **not** part of this or a future sub-project's stated scope here — it
  stays as the flag-OFF path until a decision is made to retire it.

## Model selection (dev/eval)

The dev model picker in the chat header (the same one that drives the legacy
`/api/chat` route's `customProvider`, see `lib/ai/providers.ts`) also drives
which model the Eve agent runs on, when the `useEveAgent` flag is on. The path:

1. **Picker → `modelOverride`.** `components/chat.tsx` sends the currently
   selected model id as `modelOverride` on the request body, but only in
   non-production environments (`!isProductionEnvironment`) — this is the
   same conditional send the legacy route already relies on; nothing new was
   added here.
2. **`modelOverride` → gateway slug.** `app/(chat)/api/eve-chat/route.ts` (the
   adapter) passes `body.modelOverride` through
   `toGatewaySlug` (`lib/ai/eve/model-map.ts`), which maps the picker's dev
   ids (`claude-opus-4-8`, `claude-opus-4-7`, `claude-sonnet-4-6`,
   `claude-haiku-4-5`, `gpt-5.4`, `gpt-5.4-pro`, `gpt-5.4-mini`,
   `gpt-5.4-nano`) to dot-versioned AI Gateway slugs (e.g.
   `anthropic/claude-opus-4.8`). Ids with no entry (`chat-model`,
   `chat-model-reasoning`, unknown/empty) map to `undefined`.
3. **Slug → `x-eve-model` header.** Only on session *create*
   (`createEveSession` in `lib/ai/eve/eve-client.ts`) — never on continue —
   the adapter sends the resolved slug as the `x-eve-model` request header.
   If `toGatewaySlug` returned `undefined`, no header is sent at all.
4. **Header → auth attribute.** `agent/channels/eve.ts`'s `modelAttributeAuth`
   `AuthFn` reads `x-eve-model` off the request, but only accepts it on a
   loopback request (`isLoopbackRequest`, the same trust boundary as Eve's own
   `localDev()`). It returns the value as auth attribute `eveModel`; every
   other channel/auth path is unchanged. The header value is treated as
   untrusted input — it only ever gets looked up as a known gateway slug
   downstream, never used as a credential.
5. **Attribute → resolved model.** `agent/agent.ts`'s `defineDynamic({
   fallback: 'anthropic/claude-sonnet-4.6', events: { 'session.started': ... } })`
   reads `ctx.session.auth.initiator?.attributes?.eveModel` (falling back to
   `ctx.session.auth.current?.attributes?.eveModel`) on `session.started` and
   uses it as the session's model if present.

**Session-scoped, not per-turn.** The header is only read on session create,
so a model change from the picker takes effect on the **next new chat**, not
mid-conversation — an existing Eve session keeps whatever model it started
with.

**Fallback.** Any of these break the chain back to the `anthropic/claude-sonnet-4.6`
fallback: production (adapter never receives `modelOverride`), an unmapped
picker id (`toGatewaySlug` returns `undefined`, no header sent), or a
non-loopback request (channel auth attribute never set).

**Note on gateway access tiers.** Selecting a model here only changes *which
model Eve asks the AI Gateway to run* — it does not change what the
underlying Vercel account is provisioned for. Some models (observed:
`anthropic/claude-haiku-4.5`, `anthropic/claude-opus-4.7`) return a 403
(`Free tier users do not have access to this model`) on a free-tier gateway
account. That is a billing/tier restriction downstream of selection, not a
failure of the selection mechanism — the gateway's own routing debug info
echoes back the exact requested slug (`originalModelId`) before rejecting it,
confirming the header's model was the one actually attempted.

### Automated verification

- `pnpm exec vitest run -c vitest.config.node.mjs tests/agent/eve-model-map.test.ts`
  covers `toGatewaySlug`'s id→slug mapping and its `undefined` fallthrough for
  unmapped/empty/missing ids.
- The header→attribute→resolver wiring (steps 4–5 above) was proven directly
  against a running `eve dev` server with `curl`: a request with no
  `x-eve-model` header resolves and completes on
  `dynamic:anthropic/claude-sonnet-4.6`; a request with
  `x-eve-model: anthropic/claude-sonnet-4.6` also completes; a request with
  `x-eve-model: openai/gpt-5.4-mini` completes without error, with a
  token/cost usage signature (far fewer input tokens, different cache
  behavior) that differs from the sonnet-4.6 runs — consistent with, though
  not conclusive proof of, a different model actually running. The
  unambiguous proof is two other mapped-model requests
  (`anthropic/claude-haiku-4.5`, `anthropic/claude-opus-4.7`): both are
  rejected by the gateway itself (403, free-tier restriction), but the
  rejection's own routing metadata echoes back `originalModelId` equal to
  the *requested* model, not the sonnet-4.6 fallback — proving the header's
  value reached the gateway as the selected model even though the
  completion itself was billing-blocked. See
  `.superpowers/sdd/task-4-report.md` for the full transcripts.

### Manual end-to-end checklist (picker → Eve, in the browser)

Driving the *authenticated* Next.js route through a real login and the dev
picker UI needs a human at a browser — it isn't reliably automatable end to
end. With both servers up (see "Running it" above):

1. In the browser, non-production build: open the dev flag menu and confirm
   `useEveAgent` is ON (reload after toggling — see "Enabling the flag"
   above).
2. Open the model picker and select a mapped model your AI Gateway account
   can actually complete (`claude-sonnet-4.6` is the safe default; some
   others may 403 on a free-tier account — see the note above).
3. Start a **new** chat (not a continuation of an existing one — the header
   is only read on session create) and send a message.
4. Confirm the turn completes and, in the Eve dev-server log/output, the
   session's model metadata matches the picked model (not the
   `anthropic/claude-sonnet-4.6` fallback), unless you picked sonnet-4.6
   itself.
5. Change the picker to a different mapped model, start another **new**
   chat, and confirm that chat resolves to the newly picked model.
6. Reset the picker to the unmapped/base option (or leave it at first load),
   start a new chat, and confirm it falls back to
   `anthropic/claude-sonnet-4.6`.
7. Confirm an **existing** chat's model does not change mid-conversation
   after changing the picker — the change should only be visible on the
   next *new* chat.

## Further Reading

- `docs/eve-spike-findings.md` — the underlying spike's findings on Eve's session
  API, context/compaction model, and durable state, with live `curl` proof of each.
- `docs/plans/2026-07-23-web-automation-prompt-to-eve.md` — the task-by-task plan
  this directory was built from.
- `docs/specs/2026-07-23-eve-ui-wiring-sp-b-design.md` — the SP-B design spec.
- `.superpowers/sdd/task-6-report.md` — the SP-B task 6 verification report
  (automated check output, what was machine-verified vs. left manual).
