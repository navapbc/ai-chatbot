# Running The Extension Locally

This is how to run the browser extension against a local app, Eve agent and
relay. It is a development setup, not a deployment guide. Nothing here sets a
flag in a deployed environment.

## What You Need

Four processes and one browser extension:

| Piece | Command | Port |
|---|---|---|
| Eve agent | `pnpm eve:dev` | 2000 |
| Next app | `pnpm dev` | 3000 |
| Relay | `scripts/dev-relay.sh` | 8080 |
| Extension | `APP_ORIGIN=http://localhost:3000 pnpm extension:build`, then load `extension/dist` | |

Use Node 24. `pnpm` crashes on Node 20.

## 1. Environment

Put these in `.env.local`. Any random string of 32 or more characters works
for the two secrets. Generate one with `openssl rand -hex 32`.

| Name | Value |
|---|---|
| `DATABASE_URL`, `AUTH_SECRET` | Already needed by the app. |
| `RELAY_RUNS_ENABLED` | `1`, exactly. Anything else leaves relay runs off. |
| `RELAY_TOKEN_ENDPOINT_ENABLED` | `1`, exactly. |
| `RELAY_TOKEN_SECRET` | A random string of 32 or more characters. The app and the relay both read it from `.env.local`, so they share it. |
| `EVE_INTERNAL_SECRET` | A random string of 32 or more characters. The app and Eve both read it, so restart Eve after changing it. |
| `RELAY_PUBLIC_URL` | The relay's URL, for example `http://localhost:8080`. No path, query or credentials. `http` and `ws` are accepted for localhost only. |
| `RELAY_EXTENSION_ORIGIN` | Optional. `chrome-extension://` followed by the contents of `extension/EXTENSION_ID`. Unset turns the extension-origin check off. |

`scripts/dev-relay.sh` checks that the required names are set and prints the
names that are missing. It never prints a value.

## 2. Database

The app does not run migrations in `pnpm dev`. Apply them once, and seed the
participants:

```bash
pnpm db:migrate
pnpm exec dotenv -e .env.local -- tsx scripts/seed-participants.ts
```

Check that `DATABASE_URL` is a development database first. Migrations only add
tables and columns.

## 3. A Signed-In User In This Database

A relay run needs a real (non-guest) user whose row is in this database. A login
cookie from another database passes `AUTH_SECRET` but fails the foreign key on
`EveRun.userId` with `User ... is not present`. If you see that error, sign out
of `http://localhost:3000`, register or sign in again, and use the same Chrome
profile that has the extension.

## 4. Start The Processes

In this order, one terminal each:

```bash
pnpm eve:dev
```

```bash
pnpm dev
```

```bash
scripts/dev-relay.sh
```

Stop any older Eve or relay process first. Eve reads `.env.local` only when it
starts, and `pnpm dev` attaches to an Eve that is already running (its log says
`A dev server is already running for this eve agent`), so an Eve started before
you added a variable does not see it.

The relay refuses any host it is not told to allow. The script sets
`RELAY_ALLOWED_HOSTS=localhost,127.0.0.1`. You should see
`{"event":"listening","port":8080}`.

## 5. Load The Extension

1. Build with the app origin: `APP_ORIGIN=http://localhost:3000 pnpm extension:build`.
   Outside a release build the origin may be `https://...` or
   `http://localhost:<port>`.
2. Open `chrome://extensions`, turn on Developer mode, choose Load unpacked and
   select `extension/dist`. Pin it and click the icon to open the side panel.
3. After a rebuild, click Reload on the extension.

The panel calls the app with your session cookie, so sign in on
`APP_ORIGIN` in the same profile.

## 6. Start A Run

1. Open an `https` page with a form in a tab. The relay only accepts `https`
   targets, so a `http://localhost` page cannot be the target. A site's home
   page is not a form; the agent will say so.
2. In the panel, search for a participant, choose Use this tab and Start.
   `config/relay-sites.json` is empty, so there are no site buttons yet.

## When It Does Not Work

Read the `pnpm dev` terminal first. The messages and what they mean:

| What you see | Cause |
|---|---|
| Panel says "Cannot reach the app" | Nothing is listening on `APP_ORIGIN`, or the extension was built with a different port. |
| `POST /api/participants` 500, `getEveRun` or `Failed to search participants` | Migrations were not applied to this database. |
| `[eve-chat] relay start refused: <reason>` | The first turn was refused. `model_not_allowed`, `unknown_participant`, `invalid_origin` (the tab is not `https`) and `sensitive_value_in_message` (the stored record tripped the sensitive-text guard) are the reasons you can hit. A start with `RELAY_PUBLIC_URL` unset or invalid also logs a line saying so. |
| `EveRun_userId_User_id_fk` violation | The signed-in user is not in this database. See step 3. |
| "The browser tab isn't connected" from the agent | The relay is not running or not reachable at `RELAY_PUBLIC_URL`. |
| The agent says the site is not an application form | The tab is a home page, not a form. |

The DEP0169 and `url.parse()` warnings in the terminal are noise.
