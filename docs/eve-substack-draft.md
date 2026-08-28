# Substack Draft: What Happened When We Put Vercel's Eve Under a Real Agent

Status: draft template — sections are scaffolding plus the filled-in content from
`feat/eve-integration-v2`. Each section opens with a bracketed note describing what
belongs there, so this doubles as a reusable structure for the next write-up. Delete
the bracketed notes before publishing.

Sources for every claim below: [docs/eve-spike-findings.md](docs/eve-spike-findings.md),
the five design docs in `docs/specs/`, the five plans in `docs/plans/`, and the
`feat/eve-integration-v2` diff (71 files, ~8.1k insertions).

---

## 1. The Hook

> [What belongs here: one paragraph naming the concrete system, the concrete
> framework, and the honest verdict. No suspense, no product voice. The reader
> should know by the end of it whether this post is a recommendation or a warning.]

We run a form-filling agent that drives a real browser through public-benefits
applications: it reads participant data out of Apricot, opens a remote Chrome on
Kernel.sh, fills a multi-step government form, and stops short of submitting
anything. It was built on the Vercel AI SDK with a hand-rolled agent loop —
`streamText` with `stopWhen: [stepCountIs(500)]`, a bespoke context compressor, a
900-line composed system prompt, and a `readReference` tool for on-demand docs.

We spent a sub-project asking whether Vercel's Eve — their filesystem-first
framework for durable agents — should replace that loop. The answer we landed on
was "go, with eyes open." Eve deleted a meaningful amount of code we had written
by hand. It also broke our package manager, forced a major dependency bump, and
crashed on the import graph of our own browser module. Both halves are the story.

---

## 2. What Eve Actually Is (And What `eve init` Actually Does)

> [What belongs here: correct the reader's likely mental model before analyzing
> anything. Most people will assume a library you import. State the topology
> instead, because the topology is what drives half the findings later.]

Eve is not a library you import into your route handler. `npx eve init .` does not
touch your `dev` script and does not add an `app/eve/...` route. It scaffolds a
standalone `agent/` directory — `agent.ts`, `instructions.md`, `channels/`,
`tools/`, `skills/`, `subagents/` — served by **its own CLI dev server on its own
port** (default 2000). `/eve/v1/session` and `/eve/v1/session/:id/stream` are
routes on *that* process, not on your Next.js app.

That single fact shapes everything downstream: how the UI talks to it, how
sessions are identified, where the browser lives, and what "deploy this" means.

The shape of the agent in our repo:

```
agent/
  agent.ts            # defineAgent: model + compaction config
  instructions.md     # always-on rules (71 lines)
  instructions/       # defineDynamic resolvers (e.g. today's date)
  skills/             # on-demand context, progressively disclosed
  tools/              # defineTool modules
  subagents/          # form_review, requirements_research
  instrumentation.ts  # Braintrust OTel exporter
```

Versions in play: `eve@0.27.0`, `@vercel/connect@0.4.0`, and a hard
`"engines": { "node": "24.x" }` that `eve init` adds for you.

---

## 3. What Eve Replaced

> [What belongs here: the strongest, most concrete part of the post. For each
> hand-written subsystem, name the file, name the Eve primitive that replaced it,
> and say what was gained *structurally* — not "it's cleaner," but what class of
> bug became impossible.]

### Context compaction: a whole module became four lines of config

We had `lib/ai/context-compression.ts`: a 75%-of-200K threshold, a Haiku
summarization pass with a hand-written prompt, a pinned working-memory message
re-prepended on every call, and a `prepareStep` hook to run it.

Eve has its own compaction engine — `shouldCompact` / `compactMessages` in
`node_modules/eve/dist/src/harness/compaction.js` — that escalates through
capping oversized tool results, summarizing the older region, degrading the
recent tail to text-only, and shrinking the window. It reads the model's context
window from AI Gateway metadata rather than a hard-coded constant. Our entire
threshold-and-trigger layer collapsed to:

```ts
export default defineAgent({
  compaction: { thresholdPercent: 0.75 },
});
```

There is no authorable hook here — compaction is internal, and
`compaction.requested` / `compaction.completed` are observe-only events. The
recommendation that fell out of the spike was blunt: **delete the compressor, do
not port it.**

### Working memory: from a hand-maintained invariant to a structural guarantee

The interesting part is the piece Eve *doesn't* do. Eve's compaction produces free
text; it has no structured-extraction step. So the participant/form data that must
survive verbatim moved into `defineState`:

```ts
export const workingMemoryState = defineState<Record<string, unknown>>(
  'labs-asp.working-memory',
  () => ({}),
);
```

Two things make this better than what it replaced, and they're worth separating.
First, extraction is now continuous — an always-available `update_working_memory`
tool the agent calls as it learns things — instead of a point-in-time pass fired
at a token threshold, so it can no longer silently miss data because compaction
landed at an awkward moment. Second, and more important: `DurableSession` keeps
`state` and `history` as **disjoint top-level fields**, and `compactMessages` only
ever touches `history`. Re-injecting state into the system prompt each turn via
`defineDynamic` + `defineInstructions` means it is compaction-immune *by
construction*. The old design relied on `compress()` remembering to re-prepend the
working-memory message on every call — a correct-by-convention invariant that a
future edit could quietly break. That class of bug is now unreachable.

### On-demand context: skills replaced our `readReference` tool

We had built a tool whose entire job was "read a markdown file out of
`lib/ai/prompts/references/` when you need it." Eve's skills mechanism is that,
natively, with progressive disclosure built in. The ~984 lines of composed prompt
and reference markdown in `lib/ai/prompts/` became a 71-line always-on
`agent/instructions.md` plus `agent/skills/browser-automation/` and
`agent/skills/benefits-application/`.

The split matters more than the line count: safety-critical rules ("never submit,"
forbidden actions, plain-language communication) stay in always-on instructions,
while browser mechanics and field-type patterns load only when the agent reaches
for them.

### The UI: an adapter, not a rewrite

Eve's NDJSON events map onto AI SDK v7 `UIMessage` parts almost mechanically, so
the chat UI never learned Eve exists. `lib/ai/eve/stream-adapter.ts` is a pure
translator:

| Eve event | AI SDK chunk |
|---|---|
| `message.appended` (`messageDelta`) | `text-delta` |
| `message.completed` | `text-end` |
| `actions.requested` → `actions[].callId` | `tool-input-available` (`toolCallId`) |
| `action.result` → `result.callId` | `tool-output-available` |
| `step.completed.data.usage` | `data-token-usage` (transient) |
| `step.started` | `start-step` |

`callId` correlates calls to results one-to-one, so no bookkeeping was needed. The
translator imports nothing from `eve` and touches no server-only code, which makes
it plain-object unit-testable — `tests/agent/eve-stream-adapter.test.ts` covers it
without a running agent.

---

## 4. What Improved (And Why We Can't Put a Number On It Yet)

> [What belongs here: the effectiveness claims — stated as capability and
> architecture, explicitly labeled as not-yet-measured. Resist the urge to imply
> benchmarks you don't have. Then say plainly why you don't have them; that
> sentence buys credibility for everything above it.]

**Stated honestly up front: none of the following is a measurement.** We have no
A/B latency numbers, no task-success rate comparison, no cost delta between the
old loop and the Eve path. What follows are architecture and capability claims,
each verifiable by reading the diff, and none of them a benchmark.

What got better:

- **Less code we own.** The compaction module, its `prepareStep` wiring, and the
  `readReference` tool all have native replacements. Eve's version also tracks
  each model's real context window instead of a hard-coded 200K constant.
- **One class of bug became impossible**, per the `state`/`history` split above.
- **Subagents are directories, not plumbing.** `agent/subagents/form_review/` is an
  `agent.ts`, an `instructions.md`, and a `tools/` folder. No dispatch code.
- **Observability was ~55 lines.** `agent/instrumentation.ts` uses
  `defineInstrumentation` to push Eve's agent spans to Braintrust via
  `@vercel/otel`, with a `customFilter` that keeps our Kernel browser tracer's
  spans alongside the AI spans. The agent loop runs in Eve's process, so Next's
  root `instrumentation.ts` never sees these — this file is the only path off the
  machine.
- **Model selection is a dynamic resolver.** `defineDynamic` on the `model` field
  reads a per-session auth attribute, so the dev model picker overrides the model
  per session and production ignores the header entirely. The fallback currently
  sits at `openai/gpt-5.4-mini`, moved there in a commit titled "shift to free
  model" — worth confirming the motive before framing that as a cost result.

Why there are no numbers: our evals workflow is green because it *skips*, not
because it passes. Until that's fixed, any effectiveness claim we made would be
architecture dressed up as data. That's the next thing to build, not something to
paper over in a blog post.

---

## 5. Where It Fought Back

> [What belongs here: the specific failures, each with its mechanism. Vague
> friction ("rough edges," "some setup pain") teaches nobody. Every item should
> name the file or the error and say what the workaround cost.]

**1. `eve init` broke every `pnpm` command in the repo.** It generated a
`pnpm-workspace.yaml` with `minimumReleaseAgeExclude`, `allowBuilds`, and a
`packageExtensions` shim — but omitted the required `packages:` field. Under our
pinned pnpm, *every* invocation, including `pnpm --version`, failed with
`ERROR packages field missing or empty`. One-line fix (`packages: ["."]`), but
until you know that, your repo is bricked in a way that looks unrelated to Eve.

**2. `defineTool`'s documented zod path crashed the server at boot.** Passing a
`z.object({...})` as `inputSchema` — exactly what eve.dev shows — crashed
`eve dev` before any request was served: `Cannot read properties of undefined
(reading 'input')`. Root cause: Eve's tool registration treats any schema carrying
a `~standard` key as its own extended `StandardJSONSchemaV1` and calls
`~standard.jsonSchema.input()`, which exists only in the zod v4 line Eve bundles
internally, not in the zod v3 our repo pinned. The escape hatch is a raw
JSON-Schema `inputSchema`. The real fix was a **repo-wide zod v3 → v4.4.3 major
bump** to unblock idiomatic tool authoring. That is a large dependency decision
handed to you by a framework's schema-serialization detail.

**3. `import 'server-only'` anywhere in a tool's import graph crashes Eve's
bundler.** Eve statically loads every `agent/tools/*.ts` import graph at boot to
read schemas. Next's webpack resolves `server-only` to a no-op via the
`react-server` export condition; Eve's rolldown-based bundler doesn't set that
condition, so the package's throwing default export runs at boot. Our
`lib/kernel/browser.ts` transitively imports `lib/db/queries.ts`, which is
`server-only`.

Both obvious workarounds failed. A dynamic `import()` crashed identically (Eve
follows dynamic imports into the same graph). `NODE_OPTIONS='--conditions=react-server'`
fixed *that* crash and then corrupted Eve's own generated module map
(`Export 'moduleMap' is not defined in module`). So `lib/kernel/eve-browser.ts`
reimplements the minimal slice of `getOrCreateBrowser` we needed — and **drops
Kernel replay recording and the `SessionMapping` DB upsert**, because those are
precisely the code paths that can't be pulled into the bundle. That's not a
stylistic compromise; it's shipped functionality deferred by a bundler condition.

**4. Compaction quality is a hard-coded constant.** Eve's compaction prompt lives
in `compaction-prompt.js` with no `defineAgent({ compaction: { prompt } })`
override. Our hand-tuned summarizer extracted named categories — SESSION STATE /
COMPLETED FIELDS / PENDING FIELDS / CASEWORKER INPUTS / GAP ANALYSIS / GAP ANSWERS
/ KEY DECISIONS. Under Eve those degrade to a generic handoff summary for anything
not explicitly promoted into `defineState`. Moving the critical data into state
mitigates the *data* half of this; the *transcript-fidelity* half is a real,
accepted regression.

**5. Durable execution breaks single-process assumptions — as the common case, not
the rare one.** `lib/kernel/browser.ts` keys live browsers in an in-memory `Map`,
and `lib/ai/tools/browser.ts` serializes commands per session with a process-local
promise-chain mutex (Playwright's `page` is not concurrency-safe). Eve sessions are
durable workflow steps designed to outlast crashes and redeploys, so a cache miss
stops being rare — and two calls landing on two processes can each call
`kernel.browsers.create()` against the same profile. The mutex, being a `Map` in
one process's memory, doesn't serialize across them at all. The fix is a
cross-process lock (Redis is already in the stack for resumable streaming) — new
infrastructure, not a port.

**6. Two session-identity namespaces that don't derive from each other.** Eve's
session id is `wrun_01KY5...`; our Kernel cache key is `${chatId}-${userId}`. Neither
is computable from the other, so the bridge has to be explicit — `chatId` and
`userId` threaded through every browser tool call.

**7. Durable streams replay from zero, and the bug looks like something else.**
`GET /eve/v1/session/:id/stream` with no `startIndex` replays from event 0. On a
follow-up turn that means re-reading the *previous* turn, hitting its
`session.waiting`, and stopping before any new event arrives — the agent's work
after a gap-analysis reply simply never reaches the UI. `session-continuity.ts` now
carries an explicit `streamIndex` cursor, advanced only at turn boundaries so a
mid-turn disconnect replays that turn instead of skipping it.

**8. Event ordering has a trap.** `turn.completed` fires *before* `session.waiting`
and carries no `continuationToken`. Break your read loop on `turn.completed` and
you lose the token for the next turn. Also: a tool-then-answer turn is two model
steps, so `step.completed` arrives *mid-turn* — don't treat it as the end of
anything.

**9. Loose ends still in the branch.** `session-continuity.ts` and
`live-view-store.ts` are in-memory single-process maps with comments saying so
(Postgres-backed replacements are scoped to a later sub-project). The
`ai@^7.0.26` peer-dependency gap against our pinned `ai@7.0.19` is an unresolved
warning, not a confirmed-safe mismatch. A `[eve-chat-debug]` `console.log` is still
sitting in the shipped route. Eve doesn't read `.env.local` on its own, which is
why `pnpm eve:dev` is `dotenv -e .env.local -- eve dev`. And authoring
`instrumentation.ts` silently uninstalls Eve's local trace writer, so `eve trace ls`
stops recording — a reasonable trade, but not one we chose knowingly the first time.

---

## 6. What We Still Haven't Verified

> [What belongs here: a short, unhedged list of what the work does *not* establish.
> This is the section that makes the post trustworthy. Keep it separate from the
> setbacks — "broken" and "untested" are different claims and readers conflate
> them.]

- **The Vercel deploy.** Every finding above was verified locally against a running
  `eve dev`. The Vercel-preview half was descoped: no project provisioned, and no
  non-prod Postgres branch for a build that runs Drizzle migrations. Eve running as
  its own server process is a materially different deployment shape from "routes
  inside the Next app," and nothing here confirms it behaves the same under
  Vercel's runtime — function timeouts, cold starts, and whether Eve's server model
  suits serverless at all are all open.
- **Error and abort event shapes.** Our NDJSON capture only exercised the happy
  path: one user turn, one tool call, one answer. Eve's error/cancellation events
  are unverified, and the adapter needs them before production.
- **Surviving a real compaction event.** The `defineState` persistence proof is
  two turns plus structural evidence (`state` and `history` are disjoint;
  `compactMessages` only reads `history`). Driving a session past 75% of a 200K
  window to watch it happen for real is still owed.
- **Any comparative measurement at all.** See section 4.

---

## 7. What This Unlocks Next

> [What belongs here: forward-looking work that is already scoped, not aspiration.
> Each item should be something a reader could pick up. End on the one decision
> that's still genuinely open.]

- **Finish the tool migration.** With zod v4 in, `defineTool({ inputSchema:
  z.object(...) })` works as documented. Tools run in the app runtime with full
  `process.env`, so Apricot calls and DB queries port largely unchanged — the work
  is mechanically rewriting each `lib/ai/tools/*.ts` factory as a `defineTool`
  module, not rewriting tool internals.
- **Re-architect the browser session for durable execution.** A Redis-backed
  cross-process lock keyed on the same stable `${chatId}-${userId}`; treat the
  in-memory cache as a warm-path optimization only; never treat a `BrowserManager`
  or Playwright `page` as durable — only the strings Kernel returns
  (`cdp_ws_url`, `session_id`, `profileName`) are reconstructable.
- **Move continuity and live-view state into Postgres**, so a restart mid-conversation
  stops silently starting a fresh Eve session.
- **Make evals actually run**, then come back and write the section 4 we couldn't
  write this time — with numbers.
- **Resolve the deployment topology.** This is the genuinely open question. Cutover
  means standing up a second service and pointing the adapter route at it, not an
  in-place swap inside the Next process. Where that process runs, how the
  Vercel-hosted app reaches it, and what the Node 24 pin implies for the host are
  all still design decisions.

---

## 8. The Takeaway

> [What belongs here: the one-paragraph verdict a reader would repeat to a
> colleague. Earn it with the specifics above rather than restating them.]

Eve's strongest claim isn't that it saves you code, though it did. It's that
several of the invariants we were maintaining by hand became structural — working
memory can't be dropped from context because it was never a message; compaction
can't be forgotten because we don't call it. The cost is that the framework's
internals become your constraints: its bundler decides which of your modules a
tool may import, its schema serializer picked your zod major version, and its
durable-execution model invalidated a single-process assumption that had been
quietly load-bearing for our entire browser layer. Adopt it for the structural
guarantees, and budget real time for the day its assumptions and yours disagree.

---

## Appendix: Reusing This Structure

> [Keep this section in the repo copy, cut it from the published post.]

The section order is the reusable part: **topology before analysis** (section 2),
**replacements with named mechanisms** (3), **capability claims explicitly separated
from measurement** (4), **failures with their mechanism, not their vibe** (5),
**unverified kept distinct from broken** (6), **scoped next steps ending on the open
decision** (7). Sections 4 and 6 are the two most people skip and the two that make
the difference between a post that reads as evaluation and one that reads as
advocacy.
