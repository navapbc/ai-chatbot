# Add the agent loop for extension runs

## Summary

Moved Eve run continuity into Postgres, added structured `ask_question` and progress events for extension runs, stopped Eve from holding relay credentials, and made the extension recover a run after a service-worker restart.

## Changes made

- Added the `EveRun` table (`lib/db/schema.ts`, migration `0007_zippy_magik.sql`), its queries in `lib/db/queries.ts`, and `lib/ai/eve/run-store.ts` and `db-run-store.ts`. These replace the in-memory `session-continuity.ts`. Only a typed duplicate-create error counts as a conflict.
- Changed `app/(chat)/api/eve-chat/route.ts` to resume sessions from Postgres, create relay runs, and refuse to forward a `claim:` token.
- Added `app/(chat)/api/relay/agent-token/route.ts` and `lib/relay/agent-token.ts`. The agent leg now fetches its relay token on demand with the internal secret; Eve carries a run reference (`x-eve-relay-run`) and the relay URL, not the token. `agent/channels/eve.ts` and `lib/ai/eve/loopback-attributes.ts` no longer read `x-eve-relay-token`.
- Added `lib/ai/eve/input-requests.ts` and stream adapter changes: `input.requested` becomes one structured Needs You card with validated answers. Added `lib/relay/progress.ts` and `agent/tools/report_progress.ts` for the structured progress contract (sections, counts, action log).
- Added the `agent/skills/extension-run` skill and an `agent/instructions.md` line: extension-run precedence, re-opening the current URL before relying on cross-site iframe content, and `frame @ref` (not `frame <selector>`). `form_summary`, `gap_analysis` and the form review subagent tool gained progress guards.
- Changed `lib/relay/eve-relay-browser.ts`, `extension/src/bridge.ts` and `relay/hub.ts` so a run is re-attached after a service-worker restart. A command is retried once, only for a read-only allowlist and only on an anchored transport error.
- Added `scripts/check-workflow-secrets.ts`: scans every text, json, jsonb and bytea column of the Eve `workflow` schema, decompresses zstd frames, and prints per-column row counts only.
- Added tests under `tests/agent/` and `tests/relay/`, and updated `CLAUDE.md`, `agent/README.md` and `scripts/start-container.sh`.

## What it improves or fixes

Before: session continuity lived in a per-instance `Map` and was lost on restart; a question from the agent was plain text; the relay token was sent to Eve and could land in its stored state; and after a service-worker restart every command failed with "not attached". After: continuity survives restarts, questions and progress arrive as typed events, Eve receives only a run reference, and the extension resumes the run.

Manual check against a throwaway local Postgres with Eve running its Postgres world (3 model turns, legacy `x-eve-relay-token` header and internal secret sent on purpose): the scan covered 46 columns, 699 values and 41 zstd frames, found neither the token nor the internal secret in any row, and found a control string from the first user message in 40 rows across the event, run and step tables. In the same session a turn asking two parallel `ask_question` calls produced one `input.requested` event holding both requests.

Not done, and tracked in [the todo tracker](../browser-extension-todo.md): binding runs to users, stale `claim:` token recovery, and the restart-recovery minors.

## How to run / verify

```bash
export PATH=/Users/kaylynvannorstrand/.nvm/versions/node/v24.20.0/bin:$PATH
pnpm exec vitest run --config vitest.config.node.mjs tests/agent tests/relay
```

The `EveRun` store tests need `TEST_DATABASE_URL` and are skipped without it. To repeat the credential check, start Eve with `WORKFLOW_POSTGRES_URL` set, run a few turns, then:

```bash
pnpm tsx scripts/check-workflow-secrets.ts <string>...
```

Exit code 1 means a string was found; output labels strings `needle#1`, `needle#2` and never prints them.
