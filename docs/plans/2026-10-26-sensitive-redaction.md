# Sensitive-Value Redaction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the extension path a hard guarantee that sensitive values never reach the model, the server, Eve state or traces, and close the plan 3 hardening items from the tracker.

**Architecture:** Redaction runs in the extension service worker on every CDP response and event, through pure modules under `lib/relay/` (bundled into the worker like `policy.ts`, so vitest can test them without a browser). The Eve `browser` tool refuses value-returning commands before a token is minted, using a deny-by-default command allowlist. Participant records are stripped before the opening message is built, and the Eve process exports only an allowlisted set of span attributes on relay runs.

**Tech Stack:** TypeScript, esbuild (extension bundle), vitest (node project, Playwright Chromium e2e), `@typesafe-ai/sdk` (Jev), OpenTelemetry (`@vercel/otel`, `@braintrust/otel` 0.2.0), Eve 0.27.

**Spec:** `docs/specs/2026-10-23-browser-extension-design.md` (sections Sensitive Fields, Sensitive Value Redaction, Security Requirements 6 and 7, Open Items 3 and 5). Tracker: `docs/browser-extension-todo.md` "Plan 3" and "Plan 3 And 4". Earlier plans: `docs/plans/2026-10-23-browser-bridge.md`, `docs/plans/2026-10-26-agent-loop.md`.

## Global Constraints

- Sensitive values never reach the model, the server, Eve state, or traces (spec requirement 6). Redaction runs in the extension worker, never in the relay (the relay is server-side).
- Classification fails toward "sensitive". A field the classifier cannot place stays masked when any signal says sensitive; the mask is the fixed string `[masked]` for a filled field and the empty string for an empty one, so the agent can still tell filled from empty.
- Known sensitive values live in worker memory only, on the run, are never put in `chrome.storage`, and are cleared in `stopRun`.
- The `eval` command and `get` of `value`, `text`, `html`, `attr`, `styles` are refused at the Eve tool layer on relay runs; refusal happens before `fetchAgentToken`, so no token is minted for a refused command.
- Refusal is deny-by-default: an unknown command, a leading global flag, `batch`, or `find ... text` is refused.
- A screenshot whose masking cannot be applied is refused, and the page is restored in every outcome.
- No new extension permission, no new dependency (ask before adding any), no Terraform change, no flag enabled.
- The legacy `lib/ai/prompts/` tree and the legacy chat route are not changed.
- Use Node 24 (`export PATH=/Users/kaylynvannorstrand/.nvm/versions/node/v24.20.0/bin:$PATH`), `pnpm`, `pnpm exec biome lint <files>` (never `pnpm lint`). Node tests: `pnpm exec vitest run --config vitest.config.node.mjs <paths>`. E2E: `pnpm extension:build && pnpm test:relay e2e`. Never read `.env*` files.
- Participant records contain no SSN or bank field today (`lib/data/participants.ts`); strip rules are written against field names so a future field fails toward stripped.

## Decisions Proposed For The Owner

Each is implemented as written and recorded under "Rulings To Confirm" when plan 3 lands; the cost of being wrong is small and listed.

1. **Sensitive categories (Open Item 3):** SSN and ITIN, tax ID, bank account and routing numbers, credit and debit card numbers and security codes, passwords, PINs and security answers, one-time codes, driver license and state ID numbers, passport numbers, immigration (A-) numbers, and benefit case or ID numbers (`calworks_id`). Date of birth, address, phone and email are NOT sensitive (the agent needs them to fill forms). Cost if wrong: add or remove a regex in one file.
2. **Runtime Jev classification from the worker is not wired in this plan.** The worker has no authenticated server route until plan 4. The deterministic classifier (field type, autocomplete, label) runs in the worker; the Jev question is written, versioned and regression-scored here, and its scored dataset is also used to hold the deterministic classifier to the same labels. Cost: the "semantic" second opinion arrives with plan 4.
3. **`calworks_id` is stripped from the record** for relay runs, so the agent asks the caseworker (`provided` or `skip`). Cost: one extra question per run until plan 4's panel types it.
4. **Traces:** on relay runs the Eve process exports an attribute allowlist (names, durations, token counts, model, tool names, error flags), dropping everything else. Cost: less debuggability for extension runs.

## Review Focus

1. A sensitive value split across or repeated in AX node fields (`name`, `description`, `valuetext`, a label node next to the input): every field of a node classified sensitive is masked, not only `value`.
2. A sensitive field the AX tree does not mark (`type=password` with an innocuous label such as "Code", a label in other words such as "Social Sec #", the signal only in `autocomplete`, `name` or `placeholder`): still masked.
3. A known value echoed in other text (review page, title, URL fragment, console, dialog text) in digits-only, hyphenated, spaced or case-changed form: scrubbed from every response and event string.
4. Argv tricks around the tool-layer refusal: `batch`, a leading global flag such as `--cdp`, `find ... text`, `wait --fn`, `eval -b`, upper-case `EVAL`, `get` with an unknown sub-command: all refused.
5. A screenshot while masking throws, the page navigates mid-capture, or a cross-origin frame cannot be reached: refused or restored, never captured unmasked, never left masked.

---

### Task 1: Sensitive field classifier

**Files:**
- Create: `lib/relay/sensitive-fields.ts`, `evals/datasets/sensitive-fields.json`
- Test: `tests/relay/sensitive-fields.test.ts`

**Interfaces:**
- Produces:
  ```ts
  export const MASK = '[masked]';
  export interface FieldDescriptor { role?: string; inputType?: string; autocomplete?: string; name?: string; label?: string; placeholder?: string; description?: string }
  export function isSensitiveField(field: FieldDescriptor): boolean;
  ```
- The dataset is an array of `{ id: string; field: FieldDescriptor; sensitive: boolean; why: string }` (at least 40 rows covering decision 1, near misses such as "Social worker name", "Account holder name", "Card on file: no", and non-English-looking abbreviations such as "SSN", "SS#", "Social Sec #").

- [ ] **Step 1: Write the failing test.** Load the dataset; assert `isSensitiveField` returns the row's `sensitive` for every row. Add named tests: `password type is sensitive whatever the label`, `autocomplete cc-number current-password one-time-code is sensitive`, `an ordinary text field is not sensitive`, `a missing descriptor is not sensitive but password type alone is`.
- [ ] **Step 2: Run** `pnpm exec vitest run --config vitest.config.node.mjs tests/relay/sensitive-fields.test.ts`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement** `isSensitiveField` as a pure function: sensitive when `inputType === 'password'`, when `autocomplete` carries a sensitive token, or when any of `label`, `name`, `placeholder`, `description` matches the category patterns for decision 1 (patterns normalized to lower case, punctuation and spaces collapsed). Near-miss rows must pass, so patterns anchor on whole words.
- [ ] **Step 4: Run** the test. Expected: PASS.
- [ ] **Step 5: Commit** `feat(relay): deterministic sensitive field classifier`.

### Task 2: Jev classifier question and its regression dataset

**Files:**
- Modify: `lib/jev/questions.ts`, `lib/jev/client.ts`, `evals/jev-questions.eval.ts`
- Test: `tests/agent/jev-sensitive-question.test.ts`

**Interfaces:**
- Consumes: `FieldDescriptor`, the dataset from Task 1.
- Produces: `sensitiveFieldQuestions(field: FieldDescriptor)` returning `{ state, questions }` in the style of `gapFieldQuestions`: state is the descriptor only (never a record or a value), one `noul` question `sensitive` worded for decision 1; `JevFeature` gains `'sensitive-field'` and `ALL_FEATURES` includes it. `sensitiveFieldIsSensitive(outcome: JevOutcome<...>): boolean` returns `true` when the outcome is `{ ok: false }` (fail toward sensitive).

- [ ] **Step 1: Write the failing test.** `sensitiveFieldQuestions` state contains no key other than the descriptor keys; `sensitiveFieldIsSensitive` is true for `{ ok: false, reason }`, true for a noul at or above the existing `NOUL_THRESHOLD`, false below; the feature key parses from `JEV_FEATURES=sensitive-field` and `all`.
- [ ] **Step 2: Run** `pnpm exec vitest run --config vitest.config.node.mjs tests/agent/jev-sensitive-question.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement** the factory and the feature key; extend `evals/jev-questions.eval.ts` with a scorer over `evals/datasets/sensitive-fields.json` (`noul_direction_correct`, same threshold and repeat handling as the gap rows). The eval skips with null scores when `TYPESAFE_API_KEY` or the feature is absent, as today.
- [ ] **Step 4: Run** the test and `pnpm exec vitest run --config vitest.config.node.mjs tests/agent`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(jev): sensitive field question and regression rows`.

### Task 3: Response redactor

**Files:**
- Create: `lib/relay/redactor.ts`
- Test: `tests/relay/redactor.test.ts`

**Interfaces:**
- Consumes: `isSensitiveField`, `MASK`.
- Produces:
  ```ts
  export interface Redactor {
    addKnownValue(value: string): void;
    clear(): void;
    redactResult(method: string, result: unknown, describe?: (backendNodeId: number) => Promise<FieldDescriptor | undefined>): Promise<unknown>;
    redactEvent(method: string, params: unknown): unknown;
  }
  export function createRedactor(opts?: { maxKnownValues?: number }): Redactor;   // default 32
  export function valueVariants(value: string): string[];
  ```
- `valueVariants('123-45-6789')` includes `123456789`, `123 45 6789`, and case variants; values shorter than 4 characters are ignored by `addKnownValue` (they would scrub ordinary text).

- [ ] **Step 1: Write the failing tests**, with fixture AX trees as plain objects: `masks the value of a node whose descriptor is sensitive`, `masks name, description and valuetext of a sensitive node, not only value`, `leaves an ordinary node untouched`, `empty sensitive field stays empty`, `a password-type node with an innocuous label is masked using describe()`, `an unknown node (describe returns undefined) with a sensitive label is masked`, `scrubs a known value and its digits-only, hyphenated and spaced variants from any string in any result or event, deeply`, `known values are cleared by clear()`, `redactResult never throws on a malformed result and returns a fully masked or empty structure instead` (fail closed).
- [ ] **Step 2: Run** `pnpm exec vitest run --config vitest.config.node.mjs tests/relay/redactor.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.** `Accessibility.getFullAXTree` results: per node build a `FieldDescriptor` from role, name, description, AX properties (`autocomplete`), and, when AX does not expose the input type, from `describe(backendDOMNodeId)` (the bridge supplies it with `DOM.describeNode`; the implementer measures which properties AX exposes for `type=password` and states it in the report). Every other method and every event: deep string scrub of known values only. Any error inside a redaction path returns a safe empty value, never the input.
- [ ] **Step 4: Run** the test. Expected: PASS.
- [ ] **Step 5: Commit** `feat(relay): response redactor for sensitive values`.

### Task 4: Wire the redactor into the bridge

**Files:**
- Modify: `extension/src/bridge.ts`, `extension/src/service-worker.ts`, `lib/relay/protocol.ts` (no wire change expected; only if a type is needed)
- Create: `tests/relay/fixtures/sensitive.html`
- Test: `tests/relay/e2e/redaction.e2e.test.ts`

**Interfaces:**
- Consumes: `createRedactor`.
- Produces: `Run` holds a `redactor`; `forwardCdp` passes every tab-session and child-session result through `redactResult` before returning (supplying `describe` backed by `DOM.describeNode`); `onDebuggerEvent` passes events through `redactEvent` after `filterEvent`; `stopRun` calls `redactor.clear()`. `handleMessage` accepts `{ type: 'addKnownValue', value: string }` only from this extension (`sender.id === chrome.runtime.id`, no external senders) and registers the value on the current run; a restart clears it (values are never persisted).

- [ ] **Step 1: Write the failing e2e test** using the `rawAgent` helper and `relay.frames`. Fixture `sensitive.html` places sensitive-looking values in: an `input type=password` with a neutral label, an SSN-labelled text input, a `placeholder`, an `aria-label`, an `aria-description`, visible confirmation text, and an iframe field. Tests: `no sensitive string appears in any frame the relay saw after snapshot`, `a value registered with addKnownValue is scrubbed from a snapshot, a console event and a dialog event`, `known values are gone after the run stops and the worker restarts`, `an ordinary field keeps its value`, `the agent can tell a filled sensitive field from an empty one`.
- [ ] **Step 2: Run** `pnpm extension:build && pnpm test:relay e2e/redaction`. Expected: FAIL.
- [ ] **Step 3: Implement** the wiring at the two transform points named above; keep `chrome.debugger` attach logic unchanged.
- [ ] **Step 4: Run** the same command plus `pnpm test:relay` (full). Expected: PASS.
- [ ] **Step 5: Commit** `feat(extension): redact sensitive values in responses and events`.

### Task 5: Screenshot masking

**Files:**
- Create: `lib/relay/screenshot-mask.ts`
- Modify: `extension/src/bridge.ts`
- Test: `tests/relay/screenshot-mask.test.ts`, `tests/relay/e2e/redaction.e2e.test.ts` (extend)

**Interfaces:**
- Produces: `withScreenshotMask<T>(sessions: readonly MaskSession[], capture: () => Promise<T>): Promise<T>` where `MaskSession = { apply(): Promise<void>; restore(): Promise<void> }`; it calls `apply` on every session, runs `capture`, and always calls `restore` (including on throw); if any `apply` rejects it restores the sessions already applied and rejects with `ScreenshotMaskError` without capturing.
- Bridge: for `Page.captureScreenshot` on the tab session, build a `MaskSession` for the tab and for each attached child session; `apply` runs `Runtime.evaluate` through `chrome.debugger` (not through the agent allowlist) with a function that finds inputs, textareas and contenteditable elements, classifies them with the same patterns (the classifier source is bundled into the injected function), and sets `-webkit-text-security: disc !important` on sensitive ones, recording the previous inline value for `restore`.

- [ ] **Step 1: Write the failing tests.** Unit: `applies before capture and restores after`, `restores when capture throws`, `restores the already-masked sessions and never captures when a later apply fails`, `a restore failure is reported and does not hide the capture error`. E2E: `a screenshot of the sensitive fixture succeeds and the field's inline style is unchanged afterward`, `a screenshot is refused when the mask cannot be applied (a child session whose evaluate is forced to fail)`, `a navigation between apply and capture still leaves no masked style behind`.
- [ ] **Step 2: Run** `pnpm exec vitest run --config vitest.config.node.mjs tests/relay/screenshot-mask.test.ts` then the e2e file. Expected: FAIL.
- [ ] **Step 3: Implement** the sequencer and the bridge branch. State in the report how the mask was verified visually (decode the captured PNG region or compare against an unmasked capture in the test) and, if that is not feasible in CI, say so; the sequencer tests are the CI guarantee.
- [ ] **Step 4: Run** the unit file and `pnpm extension:build && pnpm test:relay e2e/redaction`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(extension): mask sensitive fields before screenshots`.

### Task 6: Tool-layer command policy

**Files:**
- Create: `lib/relay/command-policy.ts`
- Modify: `lib/relay/eve-relay-browser.ts`
- Test: `tests/relay/command-policy.test.ts`, `tests/relay/eve-relay-browser.test.ts` (extend)

**Interfaces:**
- Produces:
  ```ts
  export const COMMAND_REFUSED = 'This command is not available in extension runs. Verify values from the snapshot, which masks sensitive fields.';
  export function relayCommandRefusal(command: readonly string[]): string | null;   // null when allowed
  ```
- Allowlist (deny by default): `open`, `snapshot` (flags `-i -c -d -s -u` only), `click`, `fill`, `type`, `press`, `select`, `check`, `uncheck`, `focus`, `hover`, `scroll`, `wait` (a ref, a number, `--text`, `--url`, `--load`; never `--fn`), `tab` (bare or `list`), `back`, `forward`, `reload`, `screenshot` (without `--annotate`), `get` with `url`, `title` or `count` only, `is` with `visible`, `enabled` or `checked`, `find` only with an action verb that is not `text`, and `dialog`. The implementer checks every entry against the measured command list in `tests/relay/e2e/allowlist.e2e.test.ts` and the extension-run skill, and reports any command the skill uses that is not allowed.
- `runRelayCommand` calls `relayCommandRefusal` first and returns `{ success: false, error: COMMAND_REFUSED, liveViewUrl: null }` without calling `fetchAgentToken`; `'text'` leaves `RETRY_SAFE.get`.

- [ ] **Step 1: Write the failing tests.** Policy table with each Review Focus 4 case: `eval`, `EVAL`, `eval -b`, `get value|text|html|attr|styles`, `get` with no sub-command, `batch` in string and `--json` forms, `--cdp` as the first element, `find role textbox text`, `wait --fn`, `storage local`, `console`, `clipboard read`, `read`, `snapshot --json-evil`, an empty command; allowed rows for every allowlist entry. Tool tests: a refused command never calls `fetchAgentToken` or the CLI; `get text` is no longer retried after a connection loss.
- [ ] **Step 2: Run** `pnpm exec vitest run --config vitest.config.node.mjs tests/relay/command-policy.test.ts tests/relay/eve-relay-browser.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement** the allowlist as data (a map from command to allowed argument shapes), not a chain of conditionals.
- [ ] **Step 4: Run** the same command plus `pnpm exec vitest run --config vitest.config.node.mjs tests/agent`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(relay): refuse value-returning commands in extension runs`.

### Task 7: Agent guidance

**Files:**
- Modify: `agent/skills/extension-run/SKILL.md`, `agent/instructions.md` (only if a rule contradicts the new section), `lib/relay/run-prompt.ts` (only if the marker sentence needs one clause)
- Test: `tests/agent/skills-guidance.test.ts` (extend)

**Interfaces:**
- Consumes: `COMMAND_REFUSED`.
- Produces: a section in the extension-run skill: `eval` and value `get` are refused; verify a filled value from the snapshot (sensitive fields show `[masked]` when filled and empty when not); do not try to read a sensitive field, retry a refused command, or route around it with `find`, `batch` or `wait --fn`; sensitive values are never typed by the agent.

- [ ] **Step 1: Write the failing tests**: the skill contains the strings `[masked]`, `eval`, `verify`, `snapshot`, and the sentence that sensitive values are never typed by the agent; the extension-run override text still names `browser-automation` and `benefits-application`; every pre-existing Forbidden Actions rule in `agent/instructions.md` is still present.
- [ ] **Step 2: Run** `pnpm exec vitest run --config vitest.config.node.mjs tests/agent/skills-guidance.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement** the skill section in the tone of the existing skill; do not edit the legacy prompt tree.
- [ ] **Step 4: Run** the test and `pnpm exec vitest run --config vitest.config.node.mjs tests/agent`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(agent): guidance for masked verification`.

### Task 8: Strip sensitive fields from the opening message

**Files:**
- Create: `lib/data/sensitive-record.ts`
- Modify: `app/(chat)/api/eve-chat/route.ts`
- Test: `tests/agent/sensitive-record.test.ts`, `tests/agent/eve-chat-route.test.ts` (extend)

**Interfaces:**
- Consumes: `findParticipantSpan(text)` from `lib/jev/participant.ts`, `isSensitiveField`.
- Produces:
  ```ts
  export const WITHHELD = '[withheld: ask the caseworker]';
  export const SENSITIVE_RECORD_PATHS: readonly string[];   // ['participant.calworks_id']
  export function stripSensitiveFields<T>(record: T): T;     // pure, returns a copy
  export function stripSensitiveText(text: string): string;  // replaces the record span in a message, unchanged text when no record is found
  ```
  Stripping removes values at `SENSITIVE_RECORD_PATHS` and at any key whose name classifies as sensitive (`ssn`, `social`, `account_number`, `routing`, `password`, `pin`, `tax_id`, `itin`), at any depth.

- [ ] **Step 1: Write the failing tests.** Unit: `calworks_id becomes the placeholder`, `a future ssn key at depth 3 is stripped`, `ordinary fields survive byte for byte`, `the input record is not mutated`, `stripSensitiveText round-trips: the stripped message still parses with extractParticipantFromText`, `text with no record is returned unchanged`. Route: `a relay first turn sends Eve a message whose record has no calworks_id value`, `the caseworker's own words around the record are unchanged`, `a web chat first turn is unchanged`, `a follow-up turn is not rewritten`.
- [ ] **Step 2: Run** `pnpm exec vitest run --config vitest.config.node.mjs tests/agent/sensitive-record.test.ts tests/agent/eve-chat-route.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement** the helpers and apply `stripSensitiveText(text)` in the relay branch before `buildExtensionRunMessage`. Check `agent/hooks/participant.ts` stores the stripped record as session state and say so in the report; confirm Jev annotators receive the stripped record.
- [ ] **Step 4: Run** the same command plus `pnpm exec vitest run --config vitest.config.node.mjs tests/agent`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(relay): strip sensitive fields from the opening message`.

### Task 9: Event allowlist and bridge-owned Page domain

**Files:**
- Modify: `lib/relay/policy.ts`, `extension/src/bridge.ts`, `lib/relay/allowed-cdp-methods.ts` (only if a method needs justification)
- Test: `tests/relay/policy.test.ts` (extend), `tests/relay/e2e/allowlist.e2e.test.ts` (extend)

**Interfaces:**
- Consumes: the measured event list (run `ALLOWLIST_VERBOSE=1 pnpm extension:build && pnpm test:relay e2e/allowlist` and record the 23 event methods in the report).
- Produces: `ALLOWED_EVENT_METHODS` (a `ReadonlySet<string>` of measured, content-safe events) and `filterEvent` returning `null` for any event outside it; `Runtime.consoleAPICalled`, `Runtime.exceptionThrown`, `Page.javascriptDialogOpening` text, `Accessibility.nodesUpdated` and `DOM.*` mutation events are dropped unless the measured list shows the agent needs them and the redactor scrubs them (the report states each decision). After attach and after every re-attach the bridge sends `Page.enable` on the tab session itself; `Page.disable` stays refused.

- [ ] **Step 1: Write the failing tests.** Policy: `an unlisted event is dropped`, `every allowed event is in the measured list`, `console and exception events are dropped`, plus the policy regression rows from the tracker: `https://a.gov@evil.com` is out of scope, `javascript:`, `data:`, `blob:` urls are refused for `Page.navigate` and `Target.createTarget`, lowercase method names are refused, and non-string or non-object params are refused for every URL-checked method. E2E: `leaving the origin ends the run even when the agent never sent Page.enable`, `Page.disable is refused`.
- [ ] **Step 2: Run** `pnpm exec vitest run --config vitest.config.node.mjs tests/relay/policy.test.ts` and the allowlist e2e. Expected: FAIL.
- [ ] **Step 3: Implement** the event allowlist and the bridge `Page.enable`; update the allowlist e2e so an unlisted event method fails the suite.
- [ ] **Step 4: Run** `pnpm extension:build && pnpm test:relay`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(relay): event allowlist and bridge-owned Page domain`.

### Task 10: Frame tracking and scope for same-site frames

**Files:**
- Modify: `extension/src/bridge.ts`, `lib/relay/policy.ts`
- Test: `tests/relay/policy.test.ts` (extend), `tests/relay/e2e/bridge.e2e.test.ts` (extend)

**Interfaces:**
- Produces: the bridge keeps `frames: Map<frameId, url>` from `Page.frameAttached`, `Page.frameNavigated`, `Page.navigatedWithinDocument` and an initial `Page.getFrameTree`, and `contexts: Map<executionContextId, frameId>` from `Runtime.executionContextCreated`; a child session's `url` is updated from its own navigation events (not only `Target.targetInfoChanged`). A tab-session command that carries a `frameId`, or an `executionContextId` or `contextId` mapped to a frame whose URL is not in scope, is refused; an unknown context or frame is refused (fail closed); `Accessibility.getFullAXTree` results drop nodes whose `frameId` maps to an out-of-scope URL.

- [ ] **Step 1: Write the failing tests.** Policy-level: a pure `frameInScope(frames, frameId, origin, allowedFrameOrigins)` table. E2E (extend the fixtures with a same-site cross-origin iframe on another port of the same host): `a command targeting the out-of-scope same-site frame is refused`, `its fields do not appear in a snapshot`, `an unknown execution context is refused`, `a child session whose frame navigates cross-origin stops being allowed within the same event loop turn`.
- [ ] **Step 2: Run** the policy file and `pnpm extension:build && pnpm test:relay e2e/bridge`. Expected: FAIL.
- [ ] **Step 3: Implement** the maps and checks in the command path and the AX redaction step; keep same-origin frames working.
- [ ] **Step 4: Run** `pnpm extension:build && pnpm test:relay`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(extension): scope same-site frames and track child navigation`.

### Task 11: Trace attribute allowlist for relay runs

**Files:**
- Create: `lib/observability/scrubbing-exporter.ts`
- Modify: `agent/instrumentation.ts`
- Test: `tests/agent/scrubbing-exporter.test.ts`

**Interfaces:**
- Produces:
  ```ts
  export const SAFE_SPAN_ATTRIBUTES: ReadonlySet<string>;   // names, durations, model, token counts, tool name, finish reason, error flags
  export function scrubSpan(span: ReadableSpan): ReadableSpan;   // returns a copy keeping only SAFE_SPAN_ATTRIBUTES (and the span name, ids, timing, status)
  export class ScrubbingSpanExporter implements SpanExporter {
    constructor(inner: SpanExporter, opts: { enabled: () => boolean });
  }
  ```
  When `enabled()` is true every span is scrubbed before `inner.export`; if scrubbing throws for a span, that span is not exported.
- `agent/instrumentation.ts` wraps the Braintrust exporter with `enabled: () => process.env.RELAY_RUNS_ENABLED === '1'`. The root `instrumentation.ts` is unchanged (its spans carry command names and counts only; the implementer confirms this by listing the attribute keys it emits and says so in the report).

- [ ] **Step 1: Write the failing tests.** `keeps only allowlisted attributes`, `drops ai.prompt, ai.toolCall.args, ai.toolCall.result and message attributes`, `drops an attribute whose name is not on the list even when its value looks harmless`, `passes spans through untouched when disabled`, `a span whose scrub throws is not exported`, `the original span object is not mutated`. Measure first: a step in the report lists the attribute keys (names only) the Eve process emits on one local relay-run turn, and each kept key is justified.
- [ ] **Step 2: Run** `pnpm exec vitest run --config vitest.config.node.mjs tests/agent/scrubbing-exporter.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement** the exporter and the `agent/instrumentation.ts` wiring; keep `filterAISpans` and the `KEEP_SCOPES` behavior as they are.
- [ ] **Step 4: Run** the test and `pnpm exec vitest run --config vitest.config.node.mjs tests/agent`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(agent): export only allowlisted span attributes on relay runs`.

### Task 12: End-to-end proof and documentation

**Files:**
- Modify: `docs/browser-extension-todo.md`
- Create: `docs/changes/2026-10-26-add-sensitive-redaction.md`
- Test: `tests/relay/e2e/redaction.e2e.test.ts` (extend)

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Write the failing e2e test.** Drive the real `agent-browser` against the sensitive fixture through `runRelayCommand`: `a snapshot never contains a sensitive value`, `a refused command never reaches the relay (assert relay.frames shows no cdp call for it)`, `after the run the relay frames contain none of the fixture's sensitive strings, the token or the CDP URL`.
- [ ] **Step 2: Run** `pnpm extension:build && pnpm test:relay e2e/redaction`. Expected: FAIL until Tasks 1 to 6 are merged; PASS after.
- [ ] **Step 3: Docs.** Check off the seven plan 3 tracker items and the relay stream item that this plan covers; list in the tracker what is deferred (runtime Jev classification from the worker, typing sensitive values from the panel, adapter redaction of tool inputs in the client stream, the panel not rendering raw tool inputs) under plan 4; add a "Plan 3 rulings" list with the four decisions above and the cost if wrong; write the change doc in the `change-docs` format (30 to 80 lines).
- [ ] **Step 4: Run** `pnpm extension:build && pnpm test:relay` and `pnpm exec vitest run --config vitest.config.node.mjs tests/agent`. Expected: PASS.
- [ ] **Step 5: Commit** `test(relay): end-to-end redaction proof` and `docs: record plan 3 changes`.

---

## Out Of Scope

- The side panel, typing a sensitive value into the page from a panel input, and the Needs You card (plan 4). Known-value scrubbing is built and tested against values registered through the worker message; nothing registers one until plan 4.
- Calling the Jev classifier from the worker at run time (needs the authenticated route in plan 4).
- Redacting `fill` values and other ordinary inputs in the app stream to the client; the panel must not render raw tool inputs (plan 4).
- A sensitive value the caseworker types on the page themselves is protected by field masking and the command restrictions only; the design states this residual risk.
- The legacy chat route, `lib/ai/prompts/`, Kernel browser paths, Terraform, and enabling any flag.

## Self-Review Notes

- **Spec coverage:** field masking (Tasks 1, 3, 4); known-value scrubbing (3, 4); screenshots (5); records (8); `eval` and `get` refusal with skills updated (6, 7); Jev classifier regression-scored (2); traces (11); tracker items: iframes (10), child URL (10), Page domain (9), policy tests (9), event allowlist (9).
- **Largest risk to settle early:** Task 3 depends on what the AX tree exposes for `type=password`; the implementer measures it first and reports whether `DOM.describeNode` per input node is needed, which affects snapshot latency.
- **Honest limit:** Task 5's mask cannot be proven visually in CI unless the PNG can be inspected; the sequencer tests are the CI guarantee and the report must say what was actually verified.
