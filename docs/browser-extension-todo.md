# Browser Extension Todo

Living tracker for the Chrome extension work. Compiled on 2026-10-23 from the review findings and decisions made while executing plan 1 (`docs/plans/2026-10-23-browser-bridge.md`) against the design (`docs/specs/2026-10-23-browser-extension-design.md`). Check items off as they land; add new ones under the plan they belong to.

## Before Plan 1 Merges

- [x] Node project (`pnpm exec vitest run --project node`): one earlier run showed 2 failures in 1 file (40 passed, 42 s) and could not be reproduced. Six later runs passed 42 of 42 in about 2 to 4 s, including right after the relay suite and while it ran. The slow file is `tests/client/browser-commands.node.test.ts` (it launches its own Chromium; this branch did not touch it). Cause of the one-off is unidentified, probably transient load.
- [x] Browser project: run it with `CI=true pnpm exec vitest run --project browser`. Without `CI` it exits 0 with no output in this environment, on the untouched baseline too. Result: 115 tests, 112 pass, 3 fail (`form-summary-card.test.tsx` two tests, `gap-analysis-card.test.tsx` one test). The same 3 fail on a clean `browser-extension` checkout (`fa60e367`), so they are pre-existing and not caused by this branch.
- [ ] Decide who fixes the 3 pre-existing browser-project failures; the existing `test` job in `.github/workflows/test.yml` runs this project, so check CI history for whether it is already red.
- [ ] Use Node 24 for local commands: a fresh shell here defaulted to Node v20.15.1 and `pnpm` 11 failed with `ERR_UNKNOWN_BUILTIN_MODULE` (`nvm use 24`).
- [ ] The relay e2e suite leaves an `agent-browser` daemon process running after it finishes; stop daemons in the harness teardown.
- [ ] Watch the first GitHub run of the new `relay-e2e` job (headless Chromium with an extension has only been run locally).
- [ ] Choose how `feat/browser-bridge` lands: merge locally, push and open a PR into `browser-extension`, or keep. PR 324 (the plan) is still open and its commit is inside this branch.
- [ ] Confirm the decisions in "Rulings To Confirm" below.

## Owner Actions Before The Relay Is Enabled

Nothing here has been applied. The relay stays off until `RELAY_ENABLED` is `true`.

- [ ] Create Secret Manager secrets: `relay-token-secret-{dev,preview,prod}` (32+ characters) and `eve-internal-secret-{dev,preview,prod}`.
- [ ] Grant `roles/logging.configWriter` to the deploy service account, or set `relay_exclude_request_logs = false` knowingly (that leaves token-bearing request URLs in Cloud Logging).
- [ ] Set the GitHub repository variable `RELAY_ENABLED=true` only after the two items above. Turning it off later plans a destroy; on dev and prod deletion protection makes that apply fail.
- [ ] Do not set `RELAY_TOKEN_ENDPOINT_ENABLED` anywhere until the extension can authenticate to the app (plan 4b; runs are now user-bound).
- [ ] Before deploying plan 4a: migrations 0008 (Participant) and 0009 (partial unique index on `EveRun.chatId`) are applied by `scripts/start-container.sh` (`pnpm tsx lib/db/migrate`, `set -euo pipefail`) on the first deploy that carries this merge, and a failure crash-loops the new revision. BEFORE that deploy, `SELECT "chatId", count(*) FROM "EveRun" WHERE "relayOrigin" IS NOT NULL GROUP BY 1 HAVING count(*) > 1;` must return no rows (in practice none exist: `RELAY_RUNS_ENABLED` is set nowhere in deployment). No down migrations; rolling back the code is safe (old code ignores `Participant`; the index only touches relay rows). Run `scripts/seed-participants.ts` manually (the container does not), and make sure `EVE_INTERNAL_SECRET` and `RELAY_TOKEN_SECRET` are each at least 32 characters (a short `RELAY_TOKEN_SECRET` makes both token routes return 500).
- [ ] Only if running more than one relay instance: create `relay-redis-url-<env>` and first close the multi-instance gaps listed under plan 5.
- [ ] Run the manual cloud check (plan task 11) after the first apply: e2e against the deployed relay, compare round trip, snapshot and fill times with the spike (56 ms, about 600 ms, about 240 ms), and search the Eve Postgres world for relay token strings.

## Open Questions

- [ ] Which benefits sites appear on the Home screen? The list is configured in `config/relay-sites.json`, which is currently empty (plan 4a).
- [ ] Which models may caseworkers choose between? Mechanism exists: `RELAY_ALLOWED_MODELS`; the owner picks the values.
- [ ] Which field categories count as sensitive beyond SSN and bank details?
- [ ] Distribution: Chrome Enterprise force-install or an unlisted store listing, and which IT team owns it?
- [x] The participant table lives in the app's Postgres database, the one `DATABASE_URL` points to (`Participant`, migration 0008).
- [ ] IT: do managed Chrome policies set `runtime_blocked_hosts` for this extension, `DisableScreenshots`, or DLP screenshot rules? Any of these breaks `chrome.debugger` on Chrome 155 and later.
- [ ] IT: can a Chrome launch flag be set for caseworkers, and how do they start Chrome today?

## Plan 2: Agent Loop

- [x] Service-worker restart resume: the run re-attaches after a restart.
- [x] Postgres-backed Eve session continuity (replaces the in-memory `Map`).
- [x] `ask_question` as the Needs You card, and the structured progress contract (sections, counts, action log).
- [x] Verify what Eve's Postgres world stores: Eve does persist session auth attributes, so the guarantee rests on the code (Eve carries only the run reference, the agent token is fetched at call time, and tool output is redacted). A scan of a local Postgres world, including zstd frames, found neither the retired `x-eve-relay-token` header value nor the internal secret. It did not search for a minted agent token.
- [x] Eve instructions: re-open the current URL before relying on cross-site iframe content; `frame <selector>` does not work in agent-browser 0.33.2, `frame @ref` does.

Follow-ups:

- [ ] Stale `claim:` token recovery: a crash between the token claim and the final swap leaves a chat on a `claim:` token; needs a timestamped claim and stream resume at the stored index.
- [ ] A lost concurrent first-turn race leaks an Eve session (Eve has no cancel call).
- [ ] Stale-cursor replay can resurrect cleared pending questions when `setPending` landed but `advance` did not (needs a position guard or an atomic pending and cursor write); questions are also shown before they are persisted.
- [ ] An answer turn whose pending clear fails and whose stream dies before `session.waiting` leaves the answered requests pending, so the next answer is validated against stale questions (same class as the replay item above).
- [ ] A read resuming from a stale cursor stops at the earlier turn's `session.waiting`, so that turn's own output is not streamed (the token is now safe: `advance` never stores a stream token).
- [ ] `EveRun` rows are never deleted when chats are removed: `store.remove` has no caller.
- [ ] Decide whether to hold the turn claim until `session.waiting` instead of releasing it at the final swap. Today a follow-up sent while Eve is still running a turn reaches Eve (which folds it into the next turn) and its read replays from the old cursor; holding the claim closes that but widens the stranded-claim window (R7).
- [ ] Dev databases may hold `EveRun` rows with an empty continuation token (stored by follow-ups before the fix); they need a one-off cleanup.
- [ ] A relay follow-up that Eve delivers to a fresh session (the stored session's run ended) now fails the turn, but Eve has already started that fresh session without relay attributes and it runs the turn; cancel it (`POST /eve/v1/session/:id/cancel`) or decide how an ended relay run is restarted.
- [ ] `report_progress` on web chats streams a raw `reportProgress` tool card that the web UI has no renderer for.
- [ ] The built-in `agent` tool (Eve's root-only copy of the agent) gets the parent's tools and auth but runs in its own child session, so on a relay run its `browser` calls are now refused by the agent-token route's session binding. Consider disabling it on extension runs (the extension-run skill already tells the agent not to use it), or decide how a child may share the run.
- [ ] `fetchAgentToken`: in-flight dedupe, and cache eviction when the relay rejects the token.
- [ ] Restart recovery: reset only agents that attached before the extension went away; `ELEMENT_REF` recognizes only the `@eN` form; add a committed test against real agent-browser error wording.
- [ ] The stale marker sentence stays in stored caseworker text and reaches Jev corroboration context (harmless noise).

- [ ] Hub restart minors: reword the hub comment that says queued commands fail with `not attached` (a queued browser-level `attach` from the old agent could succeed) or reject queued `attach` waiters on `ext-up` from a resumed socket.
- [ ] A socket that closes between `register` and promote skips `resumed`; recovery relies on the Eve-side `not attached` rotation.
- [ ] Cross-hub gap: a hub holding a run with no subscribed agent never receives `ext-up` or `agent-reset` (single instance for now).
- [ ] `tests/relay/e2e/bridge.e2e.test.ts` teardown: use `Promise.allSettled` so a throwing `browser.stop()` still stops the relay and fixtures.

## Plan 3: Sensitive-Value Redaction

Landed on `feat/sensitive-redaction-impl`; see `docs/changes/2026-10-26-add-sensitive-redaction.md`.

- [x] Redaction in the extension worker, the sensitive-field classifier, and refusal of `eval` and value `get` (done through the argv allowlist in `lib/relay/command-policy.ts`; Jev classification at run time is under plan 4).
- [x] Strip sensitive fields from participant records server-side before the opening message is built.
- [x] Frame scope for same-site cross-origin iframes (`lib/relay/frames.ts`).
- [x] Update a child session's URL from navigation events, not only `Target.targetInfoChanged`.
- [x] Have the bridge enable the Page domain itself (and deny `Page.disable`) so leave-origin detection does not depend on the agent.
- [x] Policy regression tests: `https://a.gov@evil.com`, `javascript:`, `data:`, `blob:`, lowercase method names, non-string params.
- [x] Event-method allowlist (24 methods, dialog events rebuilt with empty text).

## Plan 3 And 4: Relay Stream Sensitivity

- [ ] Treat the whole relay stream as sensitive (partly done): the worker redacts replies and events, drops unlisted events, and strips `postData`. Still open for plan 4: the extension must not render or log raw tool inputs (browser `fill` values and `ask_question` inputs cross it), and tool inputs in the client stream are not redacted (see plan 4).

## Plan 4: Side Panel And App API

- [ ] The six screens and saved details with the 8-hour limit (the participant API is done, plan 4a).
- [x] Runtime Jev route `app/(chat)/api/relay/classify-field/route.ts` (plan 4a); `CLAUDE.md` updated.
- [ ] 4b: the worker calls classify-field (build the request with `describeFieldForJev`) and ORs the verdict with the deterministic `isSensitiveField`, so Jev can only escalate masking.
- [ ] Typing sensitive values from the panel into the page, and registering them with the worker through `addKnownValue`. Values are lost on a worker restart, so the panel re-registers them after each restart.
- [x] Server-built opening message (plan 4a): before this, the unstripped participant record transited the browser to the Next server before the route stripped it. Done: the server builds the message from the stored participant, so the record no longer transits the browser.
- [x] Adapter redaction of tool inputs in the client stream on relay runs (plan 4a). Residuals: see "Plan 4a follow-ups".
- [ ] 4b: the panel must not render or log raw tool inputs (the client stream no longer carries them on relay runs, but `ask_question` and worker-side data still must not be logged).
- [ ] A `calworks_id`-style panel flow: the field is stripped from the record, so the caseworker supplies it in the panel.
- [x] Bind runs to users before enabling the token route (plan 4a): server-created runs, `runId` to `userId`, origin from the stored run, guests refused, soft run limit (`RELAY_MAX_RUNS_PER_HOUR`), a `sub` claim with a hub peer check. (Token refresh for the agent leg is no longer needed: it fetches on demand.)
- [x] (plan 4a) The public `app/(chat)/api/relay/token/route.ts` now mints only the extension token. Done: it no longer mints or returns an `agentToken` (it used to); the agent leg gets its token from the internal agent-token route.
- [x] (plan 4a: relay-only lookup, partial unique index, other-user refusal before session creation) Chat-id squatting on the internal token route: another user who creates an `EveRun` row for the same chat id first makes the route 404 for the real owner, and if a run id leaks, that user can obtain an agent token for the run with an origin they chose (hijack). The session binding (the route mints only for the Eve session stored on the row) narrows this to the squatter's own session but does not stop it; fix with user binding, plus a partial unique index on `EveRun.chatId` where `relayOrigin` is not null, or a relay-only store lookup.
- [x] `EVE_INTERNAL_SECRET` minimum length (32; shorter is treated as unset). Owner: the deployed secret must meet it.
- [x] The agent-token route reads its body through a bounded reader (`lib/relay/read-bounded-body.ts`).
- [ ] Per-run "Allow this site?" approval for a redirect (leaving the origin currently ends the run, which breaks login redirects).
- [ ] Serialize `startRun` so concurrent calls cannot orphan a run.
- [x] Mint-side claim validation and refusing an empty secret in `mintRelayToken`; value-free log line on route misconfiguration; `Cache-Control: no-store` on error responses.
- [ ] Gate the service-worker test hooks (`bridgeHandleMessage`, `bridgeOnDebuggerDetach`) behind a build define before real extension pages exist.
- [ ] A Stop issued while the socket is reconnecting drops the `stop` frame; send it on reopen.
- [ ] Decide whether a detach for reasons other than `canceled_by_user` should end the run.

- [ ] 4b: the panel; saved details; typing sensitive values plus `addKnownValue`; the "Allow this site?" flow; `startRun` serialization; service-worker hook gating; Stop while reconnecting; the detach decision (all as listed above).
- [ ] 4b: how the extension authenticates to the app: CORS, cookies, SameSite.
- [ ] 4b: a read route for run state: `progress` and `pendingRequests` are stored on the `EveRun` row but nothing returns them, so a panel reopened mid-run cannot rebuild its state; there is also no stream-resume GET on `eve-chat`.
- [ ] 4b: a server-side Stop: spec Security 9 says Stop ends the run, but today Stop only closes the relay session and the Eve session keeps running, because no route calls Eve's `POST /eve/v1/session/:id/cancel`.
- [ ] Owner decision: the `eve-chat` route reads `request.json()` with no size bound (so does `relay/token`, behind auth), so a 10 MB body is buffered before refusal. Bounding it would change the web path too.
- [ ] 4b note: runs started from entered `details` (not a `recordId`) get no Jev annotation, silently: the Eve participant hook only recognizes a full participant record (`record_id` plus `participant.name`), so Jev features behave as if the flag were off for them.
- [ ] 4b: decide whether `action_label` text is needed in tool-card inputs (they are `{}` on relay runs; it would need a deliberately sanitized field).

### Plan 4a follow-ups

Known residuals:
- [ ] Reasoning and text deltas stream model text; a model could echo a typed value and nothing masks it.
- [ ] An SSN in an `open` URL path survives tool-input redaction (the agent's own input). Browser tool outputs are redacted only by the worker upstream; `form_summary`, `resolve_field_value` and `gap_analysis` outputs stream verbatim on relay runs (inputs become `{}`) and derive from the stripped record plus caseworker answers, so the 4b panel must treat them as applicant data.
- [ ] Orphan Eve session if two users race on the same new chat id (the pre-check is not atomic).
- [ ] The run limit is a soft cap: check-then-create, concurrent first turns can exceed it.
- [ ] A Jev prompt injection can only suppress an escalation, never un-mask a deterministically sensitive field.

Deferred minors:
- [ ] Participant search and routes: the source caps neither query length nor token count (the route caps 100 characters); `ILIKE '% t%'` scans sequentially (pg_trgm if the table grows); static and DB case folding differ for odd Unicode; the SQL path is untested without a database; no `recordId` length cap; a NUL byte in `q` gives an unhandled 500 without `no-store`; the `noStore` helper is duplicated; the list route has no summarize backstop; seed script import order.
- [ ] Run store: DB store and the unique index are untested against Postgres; the memory store's fail-closed branch is unreachable.
- [ ] `eve-chat`: the details key check and the prose guard disagree on zero-width and full-width spellings (still refused); the default model is not forced into the allowlist; DOB is non-sensitive and CIN is sensitive (owner rulings to confirm).
- [ ] Tokens and hub: no upper length bound on `sub`; extra body keys are ignored, not refused; `sub` travels in bus frames, so older-hub frames count as a mismatch; the Eve-side callers treat a short secret as unset silently (only the route logs).
- [ ] Classify route: the bounded reader does not cancel the body on the up-front `Content-Length` refusal and does not release the lock in `finally`; `describeFieldForJev` runs twice; the descriptor allowlist is duplicated; a store read failure answers `offline:chat`, distinguishable from a 404.
- [ ] Tool-input redaction: element 0 can be any short letters-and-hyphens token (restricting it to the command-policy verbs would close that); non-ref second arguments degrade to `[hidden]`; the `open` pathname is kept verbatim.
- [ ] Config route: the returned URL is the original string, not `parsed.href`; no maximum URL length; no tests for uppercase scheme, whitespace, port or `https:///`; the source-grep test is brittle.

## Plan 5: Rollout

- [ ] Test banner suppression (force-install, `--silent-debugger-extension-api`) on a managed Chrome; add `minimum_chrome_version` to the manifest.
- [ ] Multi-instance gaps, all required before raising `relay_max_instances` above 1: origin binding is per hub; a mismatched `ext-up` can close the legitimate agent; `busy` is per hub; an agent on a hub that never saw the run is never closed; no tombstone across hubs; `ext-down` and `ext-up` can reorder on Redis; pending sockets miss a remote `closeRun`.
- [ ] Bus items: Redis error listener is a silent no-op; document at-most-once async delivery.
- [ ] Relay observability: OpenTelemetry spans for connect, disconnect and round trips; log socket errors; add `origin` to the attach log line; backpressure on 32 MiB frames.
- [ ] Token exposure in the URL query: move the token to `Sec-WebSocket-Protocol` on the extension leg, a private path for the Eve to relay leg, shorter or single-use tokens.
- [ ] `promote.yml` should fail hard, not warn, when the relay image for the promoted SHA is missing while `RELAY_ENABLED` is true; add a plan-time data source for `eve-internal-secret-*`.
- [ ] Set the exact relay host in `RELAY_ALLOWED_HOSTS` instead of the `.run.app` suffix; state a number for Cloud Run concurrency (default 80, about two sockets per run).

## Cleanup Before Plans 3 And 4 Build On This

- [ ] Split `relay/hub.ts` (806 lines) and `extension/src/bridge.ts` (712 lines); suggested seams are in the final review notes: run table, extension promotion and hold, agent command path for the hub; run lifecycle, commands, events for the bridge.
- [ ] Remove duplicated logic: loopback host sets (`bridge.ts` keeps its own), `isObject`, the per-session command queue, `normalizeOrigin` reimplemented in `token.ts`, and e2e helpers defined in two test files.

## Known Limits (Accepted For Plan 1)

- Cross-site iframe fields appear in `snapshot -i` only for frames created after the debugger attached (after agent-browser `open`); `wait @ref` times out even on plain Chrome.
- Leaving the origin ends the run, so a login redirect ends it until the plan 4 approval flow.
- Non-default-port origins such as `https://host:8443` are valid exact origins.
- The relay e2e suite now passes locally.
- One relay instance only; `REDIS_URL` is refused unless `RELAY_MULTI_INSTANCE_ACKNOWLEDGED=1`.
- A run token can be replayed for its 30-minute lifetime after a run closes.
- One in-flight command can race a navigation (check-then-send); a short window of events can follow an off-origin navigation if the agent never enabled the Page domain.
- A relay run row left behind after `RELAY_RUNS_ENABLED` is turned off still continues as a relay run, while the agent-token route returns 404, so its browser calls fail closed.


### Known limits (accepted for plan 3)

Redaction and screenshots:

- Text echoed in ordinary page text and not registered (a confirmation page repeating an SSN) is not masked in snapshots or pixels (P11).
- A value set in a field and reverted before the screenshot verify step is not caught (N7b).
- Screenshots can still show canvas or bitmap text, `::marker` and `attr()` fallback content, and class or style changes that reveal existing generated content (P22).
- While values are registered, a screenshot is refused for pages with `<q>`, `counter()` or `url()` generated content, for pages over 20,000 elements per frame, for continuously churning pages, and for any page whose text shows a registered value (P21). Pages nested about 150 levels deep may also refuse (a CBOR limit in `DOM.describeNode`).
- Phone numbers and other ID-shaped values in fields are masked in screenshots (P14); 9+ digit values are masked in script results, URL query and path (P9b, P10), and host or path segments shaped like values are masked, so some order numbers and timestamps are over-masked.
- Disallowed cross-site out-of-process iframes still paint in screenshots (plan 1 ruling).

Chat prose guard (best-effort against accidental pastes, P20):

- A contiguous 9-digit value alone is not flagged without a nearby sensitive label; ZIP+4 and phone numbers are not refused.
- The card-window scan is over-inclusive and can occasionally refuse ordinary digit text.
- Filler of more than 8 tokens between a label and its value defeats the window.
- Records at deep brace nesting in unbalanced text may not be parsed (the prose guard still applies).

Same-site frames:

- Coordinate and key input into an out-of-scope frame is refused (P23), but `Origin-Agent-Cluster: ?0` with `document.domain` is page-level, `about:blank` frames inherit the parent's origin, and sandboxed in-scope frames refuse `fill` and `click`.
- Scope is per frame origin, not per ancestry.
- A screenshot is refused while any out-of-scope same-site frame is present, even an invisible one.

Traces and tooling:

- `RELAY_RUNS_ENABLED` scrubs all Eve spans (process-wide, web chats included). Terraform does not set it, so scrubbing is off until it is set. The Eve process ignores `OTEL_EXPORTER_OTLP_*` endpoint variables whatever the flag says (behavior change).
- The `agent.*`, `eve.*` and `ai.*` allowlist keys are not verified against a real Eve turn. The root exporter does not scrub `kernel.*` events (the relay path never runs there).
- A test pins agent-browser at 0.33.2; an upgrade must re-run the command-policy review.
- `agent/instructions.md` has an `evaluate` bullet that contradicts the extension-run skill; the skill overrides it.
- The classifier misses `cin` and a bare `dl`. The record redactor was frozen after 8 rounds.
- `wait @ref`, `reload` and `screenshot --full` are refused by the command policy: the first times out through the bridge and the other two are denied by it (`Page.reload`, `Page.getLayoutMetrics` are not allowlisted). `open <url>` covers re-opening a page. `frame @ref` into a cross-site iframe cannot snapshot.

## Rulings To Confirm

Decisions made while executing plan 1 without asking, with the cost if wrong.

1. Branched from the plan branch because PR 324 is open; rebase if that changes.
2. Dependencies added in one commit; `ws` pinned to 8.18.3 (one transitive consumer moved to 8.22.0).
3. The allowlist applies to session commands and browser-level `Target.createTarget`; the shim handles other browser-level methods.
4. Redis bus tests are skipped without a Redis URL, so that path is untested in CI.
5. `terraform validate` only; nothing planned or applied.
6. `Runtime.evaluate` and `Runtime.callFunctionOn` stay allowed; agent-browser needs them for every fill.
7. Network events are filtered (`*ExtraInfo` dropped, headers and post data stripped) before leaving the extension.
8. Child (iframe) sessions are checked against their own URL; `allowedFrameOrigins` defaults to none.
9. A run binds to the origin of its first attach, and an extension is replaced only after it registers.
10. Multi-instance gaps are accepted because the owner chose a single instance; `REDIS_URL` needs an explicit acknowledgement.
11. Host allowlist suffixes match on label boundaries; empty entries are refused.
12. A closed run is remembered for the token lifetime, set only on terminal closes (not on socket drops).
13. Leaving the origin is terminal in plan 1.
14. Cross-site iframe fields need a prior `open`; this is agent-browser behavior, reproduced natively.
15. Relay credentials need loopback and a shared secret; the relay URL must match `RELAY_PUBLIC_URL`; redaction runs before telemetry.
16. The token route is disabled by default, and Terraform must not set its flag.
17. The relay runs as a dedicated service account with access to one secret; request logs are excluded for it.
18. The relay is opt-in per repository via `RELAY_ENABLED`; secrets are keyed on the base environment, so previews share preview secrets.
19. The relay depends on the log exclusion, and the deploy account is not granted logging permissions in code.
20. A service-worker restart does not resume a run in plan 1 (deferred to plan 2).

### Plan 2 rulings

R1. Branch base `feat/agent-loop-impl` was cut from `browser-extension`; cost: rebase if the base moves.
R2. Node 24 is set per command; cost: a forgotten prefix gives confusing pnpm errors.
R3. Only the typed duplicate-create error counts as a run conflict; cost: other create failures surface as errors, not retries.
R4. DB tests need `TEST_DATABASE_URL` and CI skips that half; cost: store regressions go unseen in CI.
R5. Microsandbox edits from `eve:dev` are reverted; cost: none beyond re-installing on the next dev run.
R6. Cheaper model tiers were used for small tasks; cost: extra review rounds if quality slipped.
R7. Stale-claim takeover is deferred (minimum: never forward a `claim:` token); cost: a crash mid-swap strands that chat.
R8. Token fetch dedupe and eviction on relay rejection are deferred; cost: extra token requests and a rejected token reused until expiry.
R9. Chat-id squatting: plan 4a added user binding (server-created runs, a relay-only lookup, a partial unique index, refusal of another user's chat id before session creation), which closes the row-squatting path; the residual is the orphan Eve session from a race on a new chat id. The original cost was: cost: denial of the token route for the real owner, and if a run id leaks, another user who creates the row first can obtain an agent token for that run with an origin they chose (hijack). Partly mitigated by binding the token to the Eve session stored on the row (no other session, such as an orphan, can use the run), which does not stop a squatter's own session.
R10. Non-default ports stay valid origins; cost: a run can bind to an unexpected port on an allowed host.
R11. Replay resurrection and persist-before-show are deferred; cost: a cleared question can reappear after a crash.
R12. The relay stream is treated as sensitive; cost: the extension UI cannot show raw tool inputs.
R13. The marker sentence is accepted in stored text; cost: noise in Jev context.
R14. Post-restart retry only for a read-only allowlist and only on anchored transport errors; cost: some commands fail instead of retrying.
R15. Restart-recovery minors are deferred; cost: rare wrong resets or missed element refs.
R16. Hub restart minors are deferred (reword the `not attached` comment or reject queued `attach` waiters; a socket closing between register and promote skips `resumed`; a cross-hub gap for a hub with no subscribed agent; e2e teardown should use `Promise.allSettled`); cost: a rare stale attach and slower recovery.

### Plan 3 rulings

P1. Branched from `browser-extension`; cost: rebase if the base moves.
P2. Opus for the redactor, bridge, screenshot and frame tasks, cheaper tiers elsewhere; cost: extra review rounds if quality slipped.
P3a. Sensitive categories are taken as written in the plan; cost: adding or removing a category is a regex change in `lib/relay/sensitive-fields.ts`.
P3b. No runtime Jev classification from the worker; cost: the semantic second opinion on field sensitivity arrives with plan 4.
P3c. `calworks_id` is stripped from the record; cost: one extra question per run until the panel exists.
P3d. Traces keep an attribute allowlist; cost: less debuggability for extension runs while `RELAY_RUNS_ENABLED` is on.
P4. "Application number" and "Customer ID" are not sensitive, "Client ID" is, card expiry only with a card signal; cost: a leak (first two) or an unreadable field (Client ID).
P5. Jev descriptors are capped at 200 characters with digits masked to 9; cost: coarser classification context.
P6. Known values of 3 characters are scrubbed as whole tokens only, under 3 ignored; cost: a 2-character value can show.
P7. URL hosts are scrubbed like text in the model's view (routing uses the raw URL); cost: odd hosts in the agent's view.
P8. Redaction e2e acceptance required redactor round 3; cost: none after landing.
P9b. A value-shape guard (SSN-like, 9+ digit runs, Luhn 13 to 19) masks `Runtime.*` results and URL query and fragment; cost: long digit runs such as order numbers and timestamps are masked.
P10. Value-shaped path segments and userinfo are masked in the agent's view; cost: some URLs look odd to the agent.
P11. An unregistered value echoed in ordinary page text is not masked (documented); cost: a confirmation page can show a value.
P12. The redactor is frozen after round 8 (body masking is whole-body, defense in depth); cost: new redactor findings need an owner decision.
P13. A read-only `Redactor.hasKnownValue` was added for the screenshot mask; cost: none beyond one more method.
P14. Fields with a sensitive-shaped or registered value are masked in pixels; cost: phone numbers are masked in screenshots.
P15. A closed shadow root holding fields or text refuses the screenshot; cost: such pages cannot be screenshotted while values are registered.
P16. One mask queue per tab; cost: a screenshot waits for the tab's other commands.
P17. Commands name elements by `@eN` only; `get count`, `snapshot -s`, `scroll --selector`, `wait --text|--url` and `find` are removed and `press` takes an allowlist; cost: agents lose selector-based shortcuts and need more snapshots.
P18. Relay turns whose prose pairs a sensitive label with a value are refused, cap 256 KB; cost: false refusals such as "pin 1234".
P19. The chat value-shape guard is narrower than the redactor's (strict 3-2-4 SSN grouping, Luhn) so phones and ZIP+4 pass; cost: a bare 9-digit value can pass.
P20. The prose guard is best-effort against accidental pastes and frozen after round 5; cost: a determined paste can get through.
P21. A registered value in page text refuses the screenshot instead of masking it; cost: more refused screenshots.
P22. The screenshot mask is frozen after round 5; the residuals listed under Known Limits are accepted; cost: those page shapes can leak in pixels while values are registered.
P23. Coordinate and key input into out-of-scope same-site frames is refused conservatively, and the screenshot is refused while such a frame exists; cost: legitimate input near such a frame can be refused.
P24. Frame scope is frozen after three review rounds; cost: a later finding in `lib/relay/frames.ts` needs a new task.
