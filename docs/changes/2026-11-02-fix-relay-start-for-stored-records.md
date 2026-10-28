# Fix Relay Start For Stored Records

## Summary

A relay start was refused for every participant read from Postgres, because the
sensitive-text guard misread a date in the stored record's key order. Fixed the
guard, log the (fixed) reason when a relay start is refused, and added a guide
and script for running the extension locally.

## Changes made

- `lib/data/sensitive-record.ts`: in `hasSensitiveValueInProse`, a sensitive key
  whose value is withheld (or empty) no longer reads ahead into the next JSON
  member. The new `NEXT_JSON_MEMBER` check stops the scan at `, "key":`. Prose
  such as `ssn: null 123-45-6789` has no quoted key and is still refused.
- `app/(chat)/api/eve-chat/route.ts`: `prepareRelayStart` logs
  `[eve-chat] relay start refused: <reason>` (one of the route's fixed codes,
  never a value from the request) and a separate line when `RELAY_PUBLIC_URL`
  is unset or not a relay URL.
- `tests/agent/sensitive-record.test.ts`: every seeded participant in Postgres
  `jsonb` key order must not be refused; prose and a second sensitive key after
  a withheld one must still be refused.
- `docs/local-extension.md` and `scripts/dev-relay.sh`: the local setup (env
  names, migrations and seed, a signed-in user in the same database, process
  order, loading the extension) and a script that starts the relay after
  checking the required variable names (never printing values).
- `CLAUDE.md`: one pointer to the local guide.

## What it improves or fixes

Before: `POST /api/eve-chat` answered 400 for every participant stored in
Postgres and logged nothing. `jsonb` stores keys shortest first, so
`calworks_id` (withheld by the stripper) was followed by `date_of_birth`, and
the guard took the date for a leaked value (`sensitive_value_in_message`). The
tests only used the seed file's key order, so none caught it.
After: the stored record starts a run, and a refusal says why in the server log.

## How to run / verify

```bash
pnpm exec vitest run --config vitest.config.node.mjs tests/agent/sensitive-record.test.ts
```

To start a run locally follow `docs/local-extension.md`.
