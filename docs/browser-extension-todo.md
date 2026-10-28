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

- [ ] Set `RELAY_EXTENSION_ORIGIN` (`chrome-extension://<id>` from `extension/EXTENSION_ID`) in deployment; unset, the extension-origin check is off. Build the released extension with `APP_ORIGIN` equal to the deployed app origin and `EXTENSION_RELEASE=1`.
- [ ] Create Secret Manager secrets: `relay-token-secret-{dev,preview,prod}` (32+ characters) and `eve-internal-secret-{dev,preview,prod}`.
- [ ] Grant `roles/logging.configWriter` to the deploy service account, or set `relay_exclude_request_logs = false` knowingly (that leaves token-bearing request URLs in Cloud Logging).
- [ ] Set the GitHub repository variable `RELAY_ENABLED=true` only after the two items above. Turning it off later plans a destroy; on dev and prod deletion protection makes that apply fail.
- [ ] Do not set `RELAY_TOKEN_ENDPOINT_ENABLED` anywhere until the panel exists (plan 4c); the auth path is done in plan 4b (runs are user-bound).
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

- [ ] The six screens and saved details with the 8-hour limit (the participant API is done, plan 4a). Plan 4c-1 built Home, both participant screens, Ready, Starting, Running (Stop only) and saved details; plan 4c-2 added the Running view, Needs You, Allow this site, Review and Confirmed.
- [x] Runtime Jev route `app/(chat)/api/relay/classify-field/route.ts` (plan 4a); `CLAUDE.md` updated.
- [x] 4b: the worker calls classify-field, sending only the `DESCRIPTOR_KEYS` string fields of each descriptor (the route sanitizes them with `describeFieldForJev` before asking Jev), and ORs the verdict with the deterministic `isSensitiveField`, so Jev can only escalate masking. A `null` answer (no verdict: Jev disabled or failing) leaves the deterministic verdict and is not cached (plan 4b; residuals in "Plan 4b follow-ups").
- [x] Typing sensitive values from the panel into the page (plan 4c-3, `docs/changes/2026-10-31-add-sensitive-value-typing.md`): the worker's `typeSensitive` registers the value and types it into the field the caseworker focused. Values are lost on a worker restart (see 'Plan 4c-3 Follow-ups').
- [x] Server-built opening message (plan 4a): before this, the unstripped participant record transited the browser to the Next server before the route stripped it. Done: the server builds the message from the stored participant, so the record no longer transits the browser.
- [x] Adapter redaction of tool inputs in the client stream on relay runs (plan 4a). Residuals: see "Plan 4a follow-ups".
- [ ] 4b: the panel must not render or log raw tool inputs (the client stream no longer carries them on relay runs, but `ask_question` and worker-side data still must not be logged).
- [ ] A `calworks_id`-style panel flow: the field is stripped from the record, so the caseworker supplies it in the panel.
- [x] Bind runs to users before enabling the token route (plan 4a): server-created runs, `runId` to `userId`, origin from the stored run, guests refused, soft run limit (`RELAY_MAX_RUNS_PER_HOUR`), a `sub` claim with a hub peer check. (Token refresh for the agent leg is no longer needed: it fetches on demand.)
- [x] (plan 4a) The public `app/(chat)/api/relay/token/route.ts` now mints only the extension token. Done: it no longer mints or returns an `agentToken` (it used to); the agent leg gets its token from the internal agent-token route.
- [x] (plan 4a: relay-only lookup, partial unique index, other-user refusal before session creation) Chat-id squatting on the internal token route: another user who creates an `EveRun` row for the same chat id first makes the route 404 for the real owner, and if a run id leaks, that user can obtain an agent token for the run with an origin they chose (hijack). The session binding (the route mints only for the Eve session stored on the row) narrows this to the squatter's own session but does not stop it; fix with user binding, plus a partial unique index on `EveRun.chatId` where `relayOrigin` is not null, or a relay-only store lookup.
- [x] `EVE_INTERNAL_SECRET` minimum length (32; shorter is treated as unset). Owner: the deployed secret must meet it.
- [x] The agent-token route reads its body through a bounded reader (`lib/relay/read-bounded-body.ts`).
- [ ] Per-run "Allow this site?" approval for a redirect. Partly done (plan 4b): the worker holds the run for 60 s (120 s cap) when the tab leaves for another https origin, and the `allowOrigin` message adds the origin (max 3). The prompt UI landed in plan 4c-2.
- [x] Serialize `startRun` so concurrent calls cannot orphan a run (plan 4b: `startRun`, `stopRun` and `resumeRun` share one queue).
- [x] Mint-side claim validation and refusing an empty secret in `mintRelayToken`; value-free log line on route misconfiguration; `Cache-Control: no-store` on error responses.
- [x] Gate the service-worker test hooks (`bridgeHandleMessage`, `bridgeOnDebuggerDetach`) behind a build define before real extension pages exist (plan 4b: `EXTENSION_TEST_HOOKS=1`; refused with `EXTENSION_RELEASE=1`).
- [x] A Stop issued while the socket is reconnecting drops the `stop` frame; send it on reopen (plan 4b: queued for 30 s or token expiry, memory only, so a worker restart loses it).
- [x] Decide whether a detach for reasons other than `canceled_by_user` should end the run (plan 4b: ends it with `debugger-detached` after 5 s without a re-attach).

- [ ] Plan 4c-1 is done (see `docs/changes/2026-10-29-add-side-panel-start-flow.md`); plan 4c-2 is done (see `docs/changes/2026-10-30-add-side-panel-running-view.md`); plan 4c-3 is done (see `docs/changes/2026-10-31-add-sensitive-value-typing.md`); plan 4c-4 remains. Plan 4c: the panel; saved details; typing sensitive values; the "Allow this site?" prompt. `startRun` serialization, hook gating, Stop while reconnecting and the detach decision landed in plan 4b (above). See "Plan 4c carry-forward".
- [x] 4b: how the extension authenticates to the app: CORS, cookies, SameSite. Settled by a spike (Chromium 145): a Lax cookie reaches the app from the worker and extension pages once `host_permissions` covers `APP_ORIGIN`; `auth.ts` is unchanged. Third-party-cookie blocking is untested.
- [x] 4b: a read route for run state (`GET /api/relay/runs/[runId]`: origin, progress, `pending`, `ended`, `endedAt`). Still open: there is no stream-resume GET on `eve-chat`.
- [x] 4b: a server-side Stop (`POST /api/relay/runs/[runId]/stop`: ends the row, cancels the Eve turn). Nothing calls it yet; the 4c panel must call it as well as the worker `stopRun`. Tokens minted before Stop stay valid up to 1800 s and the hub does not know `endedAt`.
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

### Plan 4b follow-ups

Deferred minors and residuals from the run plumbing, grouped by area.

Auth and origin check:
- [ ] Thin `resolveAppOrigin` test rows (whitespace, uppercase, `localhost.evil.com`, `[::1]`, userinfo); the manifest file mode is the umask's (0600 locally); GET-without-Origin and POST-with-Origin are asserted only in the e2e.
- [ ] A failed release build leaves an earlier `extension/dist` in place.
- [ ] The `eve-chat` 403 lacks `no-store`; the answer-turn origin case is not separately tested; `resetOriginWarningForTests` ships in the bundle.
- [x] Tests: `tsc --noEmit` reported errors in two `tests/agent` files (`relay-config`, `relay-route-origin`); fixed in the final review.

Run state and Stop:
- [ ] The UUID regex, `noStore` and `notFound` helpers are duplicated across the run routes; `endEveRun` also writes `updatedAt`; concurrency of `end` is untested against Postgres.
- [ ] A failed-claim re-read is not wrapped (a DB error gives 500, not `offline:chat`, before Eve); `cancelIfEnded` awaits a cancel up to 5 s in the race.
- [ ] Whether `/eve/v1/session/:id/cancel` is reachable through the `withEve()` rewrite on the public origin (predates this plan; owner to confirm).

`eve-chat` persistence:
- [ ] `clientGone` flips silently (an optional value-free warning); relay close-then-reconnect is untested; web reads leave the Eve body open after `session.waiting` (relay closes it).
- [ ] Cloud Run CPU allocation after a client disconnects is not established (no `cpu_idle` in Terraform); a scale-in mid-read can lose pending, cursor and heal; Eve's behavior on a closed stream body is not proven from Eve code. The SDK behavior is pinned by tests against `ai@7.0.42`.

Redactor hook:
- [ ] Empty and filled AX fields use different cache keys (one extra call); DOM results can wait up to `describeDeadlineMs`; agent-browser's internal `DOM.getAttributes` use is unverified.
- [ ] Eight permanently hung classify calls disable the hook until restart (no slot reclaim or circuit breaker); `getAttributes` over-masks a non-input whose descriptor equals a flagged field's.
- [ ] AX checkbox and radio are not asked about the hook (a label-only sensitive checkbox is not caught by Jev).

Worker Jev classifier:
- [ ] Large forms exceed the deadline at about 8 per batch and retry on the next snapshot; no backoff on a persistent outage (bounded by the 200-request budget; later fields keep the deterministic verdict); `failures()` counts rejected calls, not requests.
- [ ] The default bundle contains the classifier and app client code when `APP_ORIGIN` is empty; one test name is misleading and the e2e lacks a hung-server case.

Worker lifecycle:
- [x] A same-run-id restart after the old run already ended returned `needsRegistration: false` although values were lost. Fixed: the worker records the 20 most recent begun run ids (ids only, `chrome.storage.session`), and a `startRun` for one of them with no live run of that id needs registration.
- [ ] `registrationComplete` and `addKnownValue` are tied to a `runId` since plan 4c-1 (fixed); they are still not serialized.
- [ ] The harness leaves `test-page.html` and a test-hooks build (with a 2.5 s origin hold) in `extension/dist`, which can persist into a release build; plain and release bundles still read `typeof __ORIGIN_HOLD_MS__`.
- [ ] The detach grace relies on the agent re-attaching; if Chrome emits detaches during in-origin navigation this becomes important (unverified). A same-run-id replacement does not announce `resumed`; the endRun detach is not ordered by the lifecycle chain.
- [ ] Test gaps: `sendCommand` never asserted for refused methods, child sessions while unregistered, token-in-logs across delivery retries.
- [ ] Residuals: pending stops are lost on a worker restart within 30 s; events missed while unregistered are not replayed; `registration required` does not match Eve's rotation regex; Chrome may log WebSocket URLs containing the token on delivery retries.

Origin hold:
- [x] A command in flight across A to B to A was still sent to the new A document (a double-submit risk). Fixed: `requireSameScope` runs in the forward loop, right before the send and before a screenshot capture.
- [ ] The agent cannot dismiss an in-scope dialog during a hold; a resumed run whose tab is off-origin ends at once; a navigation announced but never committed ends the run after 60 s.
- [ ] A page on A can create a pending origin B through a never-committing request, and `allowOrigin(B)` then succeeds while the tab is on A (fails closed).
- [ ] Hub design: only `open` is bound to the token origin; after `allowOrigin` the agent works on the new origin but cannot `Page.navigate` to it. A round trip B to A without an allow lets B choose the A URL the agent resumes on.
- [ ] `P4b-f` is superseded: leaving the origin holds the run (60 s, 120 s cap) instead of ending it at once.

### Plan 4c carry-forward

Contracts the panel must honor:
- [x] (plan 4c-1) Stop: call the server stop route and the worker `stopRun`; the hub does not know `endedAt` and tokens minted before Stop stay valid up to 1800 s.
- [x] (plan 4c-1: the run session polls `getRunStatus` every 2 s and treats 7 s of continuous inactivity for its run as an end; an extra read of the server first) Call the server Stop for worker-initiated ends too (only while a panel is open; see I1 under 'Must land before') (`left-origin`, `token-expired`, `debugger-canceled`, `debugger-detached`, tab closed). Those tell only the hub: the server row stays live, the Eve turn keeps running and `relay/agent-token` keeps minting. The bridge has no push, so poll `getRunStatus` to notice an end.
- [x] (plan 4c-1: refresh at 5 min left, retry every 60 s, then `registrationComplete(runId, 0)`) Token refresh (a same-run-id `startRun`): it is refused when the tab is on an origin added by `allowOrigin` (only the token origin is checked); it drops the extra origins, detaches the debugger, sends no `resumed`, clears known values and requires re-registration. A refused refresh means the run dies at `exp` (see the 4c-2 carry-forward). Refresh BEFORE `exp`: the worker's own `token-expired` sends `stop`, and the hub then closes the run for good.
- [ ] Jev disabled is a supported configuration: `classify-field` answers `null` for unflagged fields and masking is deterministic only. `jevFailures` grows with every `null`, so it is not an outage signal.
- [ ] After a worker restart, send `runId` with `registrationComplete` and call `registrationComplete` (count may be 0) promptly; the run stalls fail-closed until then. Partly done (plan 4c-1): the panel sends `runId` and calls `registrationComplete(runId, 0)` after every `startRun` and refresh. Open: a worker restart without a panel refresh is not detected, and the count is always 0 (the worker never verifies it; values typed before a restart are lost, see 'Plan 4c-3 Follow-ups').
- [x] Hide pending questions when the run state says `ended` (done in plan 4c-2: the route returns `[]` once ended and the panel shows no cards).
- [x] The allow-origin prompt shows the tab's actual URL (origin plus path) as well as `pendingOrigin`, and is never auto-approved (plan 4c-2). A prompt-injected agent can still trigger holds by clicking links; the card shows where the tab is.
- [ ] During a hold the Eve tool returns the out-of-scope refusal; the agent should be told to wait and retry.
- [ ] Show `jevFailures` as a count, not as an outage signal.
- [ ] `RELAY_EXTENSION_ORIGIN` must be set at deploy for the origin check; `APP_ORIGIN` must exactly match the deployed app origin for cookie auth.
- [ ] Owner follow-ups: set `RELAY_EXTENSION_ORIGIN`; third-party-cookie blocking is untested; Cloud Run CPU after a disconnect; scale-in mid-read; whether the Eve cancel path is reachable through the public rewrite.

Code items for 4c or a follow-up:
- [x] (plan 4c-1) The bridge checks `runId` on `registrationComplete` and `addKnownValue`.
- [ ] Owner or 4c decision: a server-side backstop for Stop, re-checking `endedAt` on the agent-token cache path (`lib/relay/agent-token.ts`) so an ended run stops getting tokens from the cache.
- [ ] The hub tombstone (`closedRunTtlMs`) equals the token TTL measured from the close, so a token minted later from a still-live row can re-register the run after the tombstone expires.

## Plan 4c-1 follow-ups

Must land before `RELAY_RUNS_ENABLED` is set anywhere:
- [ ] F1: the agent's first command before the extension registers is dispatched while the hub's extension state is `unknown`, and times out after 30 s with `extension did not answer`, which is not retried. Options: add `extension did not answer` to `CONNECTION_LOST` in `lib/relay/eve-relay-browser.ts` so the first (retry-safe) snapshot is rotated and retried, and/or queue commands in `unknown` in `relay/hub.ts` with a per-waiter timer (single instance only; a multi-hub setup needs an ext-query). Treating `unknown` like `away` is not an option.
- [ ] F2: orphan run when the panel's 15 s first-chunk timeout fires before `eve-chat` answers: the route may still create the session afterwards, and the panel's stop got a 404. Fix server-side (after `store.create` in the relay first turn, end the row and cancel the Eve turn if the request signal aborted; Next 16 abort behavior is unverified) and/or have the panel retry stop once after 15 to 20 s.
- [ ] I1: a run the worker ends while the panel is closed is never stopped on the server. Worker-initiated end detection runs only inside an open panel's session, and `rebuildRun` returns null as soon as the worker is inactive: the server row stays live, the Eve turn keeps running, `relay/agent-token` keeps minting, and the run counts toward the hourly limit. Options: on panel open, stop every recent with no `endedReason` whose server row is live and that the worker does not hold; or have the worker call the stop route itself.
- [ ] I2: closing the panel during Starting orphans the run. The panel page unloads after the `eve-chat` POST created the row but before `worker.startRun`, so cleanup never runs; F2's server-side abort fix does not cover it. Options: write the in-flight run id (id only) to `chrome.storage.session` before the POST and stop it on the next open if the worker does not hold it; or a server or hub reaper for relay runs whose extension never registered. Relates to 'The Starting screen has no Cancel'.
- [ ] I4: every panel open mid-run refreshes the token at once (the estimated expiry equals the refresh threshold), and the same-run-id `startRun` detaches the debugger under any in-flight agent command, clears known values and drops allowed origins; a second window does the same. Cheap fix: have `getRunStatus` return the token's `exp` (`claims.exp`, not secret) so a rebuild uses the real expiry.
- [ ] The participant name in the GET query string (`/api/participants?q=`): server and access logs can capture a participant name. Owner or API decision before the flag.
- [ ] The bridge reports a `restarting` status in `getRunStatus` during a same-run-id refresh, so the panel does not have to wait out the 7 s inactive rule. A restart gap over about 8 s (a hung detach or storage call after the 5 s screenshot wait) would stop a healthy run.

Other follow-ups:
- [ ] The Starting screen has no Cancel.
- [ ] The session's `workerGone` branch for a row that already ended skips the scoped worker stop.
- [ ] Dead constant `GRACE_MS` in `extension/panel/lib/run-session.ts`; the bridge-constant test reads `bridge.ts` with a regex (export `STOP_WAIT_MS` instead); no render tests for `running.tsx` copy.
- [ ] Single-run worker: a start in a second panel window replaces the first window's worker run (the first window's session then notices the replacement and stops its server run).
- [x] Every panel open mid-run refreshes the token at once, which detaches the debugger and drops allowed origins. Known values: solved in plan 4c-3, the worker carries them across a same-run refresh. Open: re-allow or re-prompt origins.
- [ ] `siteName` is the hostname for "Use this tab" and is cut to 100 UTF-16 units, which can split a surrogate pair.
- [ ] The recents read-modify-write is serialized by a mutex within one window, not across windows.
- [x] `ensureSweepAlarm` now checks `alarms.get` first and creates the alarm only when none exists, and the worker also sweeps once at every start. Still open: it uses the global `chrome`.
- [ ] Cross-window start: after a successful worker start, `worker_unreachable` can leave a worker run against a stopped server run.
- [ ] If the server stop fails after a worker-initiated end it is not retried; Stop can take 10 s or more plus the refresh before the worker stop; the elapsed timer restarts on a rebuilt run; focus drops when Stop is disabled.

Minors by area:
- Build: the token extractor does not fail when the `.dark` block is missing and ignores braces in comments or strings; the `@/lib/utils` alias also rewrites `@/lib/utils/foo` (fails at resolve); `panel.js` is about 746 KB unminified.
- Shell and copy: lookups for error codes use plain objects, so inherited keys (`constructor`) match (use `Object.hasOwn`); the reducer's `probeSucceeded` and `probeFailed` are unguarded outside `loading`; no tests for `ws:` or credentialed relay strings or a forged code object; the e2e console filter matches the substring `status of 401`; no focus management on the placeholder or a non-retryable error; unmapped worker strings fall to `worker_error`; a local `codeForStatus` duplicates `app-api`'s and disagrees on 404.
- Recents: entries with a future `startedAt` never expire; stored duplicate run ids are not deduped on read; a double click can open two site tabs; `goHome` may be unused; `tabs.query` with `lastFocusedWindow` is unverified when the panel is a tab; real `chrome.tabs` flows are not e2e.
- Search: retry goes through the debounce; stale rows show for up to 250 ms; "Sign in again" drops the chosen target; the stale-response e2e is weak; `key={recordId}` can warn on duplicates; no rendered test of the StrictMode effect order.
- Saved details: `startSavedDetailsSweep` adds a listener per call (called once); saving all-empty values clears the old record by contract; stale header comments in `service-worker.test.ts` and `saved-details.ts`.
- Ready: dead `starting` and disabled code in `ReadyScreen`; no screen-level test (invalid, double press); a stale `invalid` alert after a model change; a single model that is not the default is unhandled; the panel assumes `config.defaultModel` equals the server's first allowed model.
- Starter: a local `newId` that throws rejects instead of returning `ok: false`; the e2e `stopWorkerRun` is not in `finally`.
- Session: `waitForTimeout` remains at `panel-shell.e2e.test.ts` lines near 293 to 296; the `expiresAt` value is Unix seconds (do not compare with milliseconds).
- Final review: site buttons fail with 'The tab left the site' when the site URL redirects to another origin (www or SSO).
- Final review: the 100-UTF-16-unit search cut can leave a lone surrogate, so `encodeURIComponent` throws and Try again repeats it.
- Final review: Back from Ready to Details loses typed values (`goBack` drops the participant; `DetailsScreen` starts empty).
- Final review: inline arrow handlers are not `handle`-prefixed (`participant.tsx`, `app.tsx`).
- Final review: the panel never applies `.dark` (light only) and the extension has no manifest icons.
- Final review: `bad_request:database` (400) shows 'check what you entered'.
- Final review: a 401 on Start shows the signed-out copy on Ready with no sign-in action.
- Final review: with `RELAY_RUNS_ENABLED` unset the probe gets 404 on `relay/config` and shows the non-retryable `not_found` screen with no button, a dead end (add a retry or a clearer 'not available yet' message).
- Not covered by e2e: starting a run with entered details through the real flow; the relay-e2e CI job time against its 20 minute limit.

## Plan 4c-2 carry-forward

- [x] Running view with sections, progress and the action log (plan 4c-2).
- [x] Needs You cards from `getRunState.pending`; ordinary answers go through `eve-chat` `inputResponses`; hidden once the run is `ended` (plan 4c-2). Sensitive questions only offer `provided` and `skip`.
- [ ] Token refresh is refused while the tab is on an origin added by `allowOrigin` (only the token origin is checked), so the run then dies at `exp`; fix the refresh or re-allow the origin before the run can outlive its token.
- [x] Typing sensitive values into the page (plan 4c-3: `typeSensitive`; the worker carries values across a refresh).
- [x] The allow-this-site prompt shows the tab's actual URL as well as `pendingOrigin` and is never auto-approved (plan 4c-2).
- [x] Review and Confirmed screens (plan 4c-2; 'I submitted it' is manual).
- [ ] "Show me the unfilled field" (plan 4c-4), automatic submit detection and the confirmation number (plan 4c-4).
- [ ] The Eve tool waits and retries during an origin hold instead of returning the out-of-scope refusal. Panel consequence: if the agent ends its turn on the refusal, the hold clearing shows Review ("The agent has stopped") and nothing in the panel can resume it, so Allow achieves nothing.

## Plan 4c-2 Follow-ups

- [ ] The panel-shell e2e now takes about 177 s (plan 4c-3); check it against the relay-e2e CI job's 20 minute limit, and consider splitting the Needs You scenario (about 55 s of its 90 s budget).
- [ ] A stale `review` phase can show after a replayed turn: a follow-up arriving between turn A's final swap and its `session.waiting` replays A's waiting and leaves `turnIdle` true while B runs, and "I submitted it" then cancels B. Root fix: hold the claim until `session.waiting` (see the claim item above). The panel offers no follow-up input while `working`. The variant a panel answer can reach instead ends on a re-asked, already-answered Needs You card; a cheap guard is to show `needs_you` only when `turnIdle` too, with idle written after `advance` (optional: the review judged the current ordering acceptable).
- [ ] The Confirmed stop-failed copy mentions a Stop control that is not on that screen.
- [ ] `turnIdle` is written before the cursor `advance`, so `review` can show a moment before the cursor moves; write it after, in a `finally`. The rollback branch where the row was not idle has no test.
- [x] A free-text sensitive question had no panel path (dead-end card); plan 4c-3 gives it the typing card and answers `provided`. Open: a bare SSN or short ID in a free-text answer can pass the server's prose guard; the card only warns.
- [ ] A server refusal clears every typed text field in the batch (the server does not say which). Whether Eve splits parallel `ask_question` calls across separate `input.requested` events is unverified (decides how often `missing_answer` occurs).
- [ ] Allow-site: a second press after an unanswered worker call can show 'hold ended' though the first went through; `tabNotFound` and `token_invalid` refusals show the generic hold-ended copy; the double announcement (focus plus `role=alert`) and the note persisting after the tab returns are unaddressed.
- [ ] A user Stop in flight followed by "I submitted it" returns the same promise, so the run ends as `user` yet Confirmed shows.
- [ ] `RunState.phase` is typed `RunPhase | string`; the session emits every 3 s even when nothing changed (memoise in the UI).
- [ ] Missing tests: rendered hold card, `openedMidRun` render, a B-to-C allow e2e, progress sections checked only by count text in the e2e.

## Plan 4c-3 Follow-ups

- [ ] Jev-only masking verdicts are not durable: a verdict can be evicted (LRU, 256) or fail after a worker restart, and screenshot masking never consults Jev. So "field masking still applies" after a restart is false for fields only Jev flags while Jev is down. Option: pin true verdicts for typed fields for the run.
- [ ] `masksFilledField` runs under the shared tab hold, so a Jev call (up to about 3.5 s) can delay a screenshot.
- [ ] `typingNow` is module-level, not per run. No test that it is released when typing throws; the serialization test does not assert that nothing was registered; no e2e for a disabled-and-focused, date, hidden or textarea field.
- [ ] The same value can be filled twice after toggling Skip (the page field may append or replace). The browser's password-save prompt is unverified (an `autoComplete="new-password"` option). The source-regex test for value hygiene is weak.
- [ ] `registrationComplete` never verifies its count against the worker's value count. A worker restart loses the values and the panel re-sends 0; values typed before the restart are not scrubbed.
- [ ] Fields with key-driven input masks get no key events. A page `beforeinput` handler can move focus so the text lands elsewhere (still scrubbed when 3+ characters).
- [ ] Worker console output is only partly captured in the e2e; no test of a successful screenshot after a refresh.
- [ ] Copy: `value_not_accepted` could say the value was empty or too long; the copy test checks only length.

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
