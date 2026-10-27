# Add server-side run API for the extension

## Summary

Added the server half of the Chrome extension side panel (plan 4a): a Neon-backed participant API, a server-built opening message for relay runs, user-bound relay tokens, a field-classification route, a config route, and redaction of tool inputs in the client stream. The extension panel that calls these routes is plan 4b. The web chat path and the legacy `chat` route are unchanged.

## Changes made

- Added `app/(chat)/api/participants` (search) and `participants/[recordId]` (summary only: record id, name, county), backed by `lib/data/participant-source.ts` and a new `Participant` table. `lib/data/participants.ts` still holds the seed records.
- Added `lib/relay/route-guard.ts`: every relay route returns a bodyless 404 when `RELAY_RUNS_ENABLED` is not `1`, 401 without a session and 403 for guest accounts.
- `eve-chat`: a relay first turn now takes `relay: { origin, participant, model? }` and no client message. `lib/relay/run-input.ts` validates it, checks the details for sensitive keys and builds the opening message on the server, so the unstripped record no longer transits the browser. Added a model allowlist (`RELAY_ALLOWED_MODELS`), a per-user run limit (`RELAY_MAX_RUNS_PER_HOUR`, default 20) and a refusal when another user already holds the chat id.
- Run store: relay-only `getByChatId`, `countRecentRelayRuns`, and a partial unique index on `EveRun.chatId` where `relayOrigin` is not null.
- Tokens: `lib/relay/token.ts` requires a `sub` claim; `relay/hub.ts` refuses an extension and agent pair with different `sub` values. The public `relay/token` route mints only the extension token, reads the origin from the stored run and sends `no-store` on errors. The internal agent-token route reads its body through `lib/relay/read-bounded-body.ts`.
- `lib/relay/internal-secret.ts`: one length rule, `EVE_INTERNAL_SECRET` shorter than 32 characters is treated as unset by every consumer.
- Added `relay/classify-field` (Jev sensitive-field judgment, fails toward sensitive) and `relay/config` (sites from `config/relay-sites.json`, allowed models, default model).
- `lib/relay/tool-input.ts` and the stream adapter: on relay runs, tool inputs in the client stream are hidden (`browser` argv keeps only the verb and snapshot refs; other tools send `{}`).
- Added `scripts/seed-participants.ts` and migrations 0008 and 0009; new tests under `tests/agent` and `tests/relay`.

## What it improves or fixes

Before, the extension would have sent the whole participant record and the opening message from the browser, the public token route also minted an agent token, any user could create an `EveRun` row for another user's chat id, and `fill` values appeared in the client stream. Now the server owns the record and the message, tokens are bound to the signed-in user, relay run rows are unique per chat id, and the client stream carries no tool inputs.

Known limits:
- The run limit is check-then-create, so concurrent first turns from one user can exceed it by the number in flight.
- An orphan Eve session can still be created if two users race on the same new chat id.
- Reasoning and text deltas are not masked; a model could echo a typed value.
- An SSN embedded in an `open` URL path survives tool-input redaction.
- A Jev prompt injection can only suppress an escalation: the caller must OR the Jev verdict with `isSensitiveField`, so a deterministically sensitive field stays masked.

## How to run / verify

Before deploying:
- Migrations 0008 and 0009 are generated, not applied. 0009 fails if production already holds duplicate relay rows for one chat id; check that first.
- The deployed `EVE_INTERNAL_SECRET` must be at least 32 characters, or relay stays refused.
- Optional env vars: `RELAY_ALLOWED_MODELS` (comma-separated; unknown values fall back to `claude-opus-5`), `RELAY_MAX_RUNS_PER_HOUR`.
- The seed script was not run: `dotenv -e .env.local -- tsx scripts/seed-participants.ts`.

```bash
pnpm exec vitest run --config vitest.config.node.mjs tests/agent tests/relay --exclude 'tests/relay/e2e/**'
```
