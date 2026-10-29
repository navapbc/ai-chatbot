# Pilot Rollout Runbook

How to turn on extension relay runs for a single-instance pilot: one relay
instance per environment, the Labs ASP Browser Bridge extension force-installed
on managed Chrome, and relay runs enabled in one environment at a time. Follow
the sections in order. Each step says who does it and what to check.

Who:

- **Owner**: the product/engineering owner who decides and holds keys.
- **Platform**: whoever has GCP project and GitHub repository admin rights.
- **IT**: the team that manages the caseworkers' Chrome.
- **Claude-reviewable**: a repository change (a PR) that an engineer or Claude
  can write and review; no cloud access needed.

Rules for the whole runbook:

- No secret value goes into this document, a ticket, a chat, a PR, a workflow
  input or a log. Secrets are generated and stored in one pipe (step 2.1) and
  never printed.
- Placeholders: `<project-id>`, `<project-number>`, `<region>` (the workflows
  use `us-central1`), `<env>` (`dev` or `prod`), `<domain_name>` (the Terraform
  variable), `<app-origin>`, `<relay-host>`, `<update-host>`. Replace them
  locally; do not commit the filled values to this file.
- Nothing in the repository turns a flag on. Every new Terraform variable
  defaults to today's behavior (relay off, run flags unset).

## Known Limits

These are accepted for the pilot. Tell pilot users about the ones they will
notice.

- **One relay instance.** `relay_max_instances` must be 1 (validation in
  `terraform/variables.tf`). Redis and the multi-instance gaps are out of
  scope.
- **Run state and tombstones live in the relay's memory.** A relay restart,
  redeploy or Cloud Run recycle ends every live run. The tombstone that refuses
  a re-dial with a closed run's token is lost on restart too.
- **The one-hour socket limit.** The relay's request timeout is Cloud Run's
  maximum, `3600s` (`terraform/relay.tf`). A WebSocket is a request, so no run
  connection lasts longer than one hour. The panel shows this as a normal end.
- **The agent token still travels in a URL query** on the agent leg
  (`/cdp?token=`). agent-browser takes a plain `ws://` URL and cannot send a
  header. The request-log exclusion (section 3) is the only mitigation. The
  extension leg now sends its token as `Sec-WebSocket-Protocol: relay.v1,
  <token>`. During the transition the relay still accepts the extension token
  from the query on `/extension` (`relay/server.ts`, "transition only"). That
  fallback is to be removed once no old extension builds remain.
- **Chrome's debugging bar** ("Labs ASP Browser Bridge started debugging this
  browser") appears while the extension is attached to a tab. Hiding it is
  untested (section 7). The Ready screen tells the caseworker the bar is
  expected (`DEBUGGER_BAR_COPY` in `extension/panel/lib/copy.ts`).
- **Submit detection** sees only form navigations. XHR and single-page-app
  submits are not detected. A multi-step form's Next, Save or Back POST raises
  the "Did you submit the form?" prompt.
- **An echoed SSN on a confirmation page** can appear unmasked in the agent's
  own accessibility snapshot. The shape guard covers `Runtime.*` results and
  URLs only. This is the spec's accepted residual (`docs/browser-extension-todo.md`,
  "Plan 4c-4 Follow-ups").
- **Overlay painting into agent screenshots, and the infobar while the submit
  watch holds its attach, are unverified.**
- **No OpenTelemetry spans in the relay.** Observability is structured logs
  plus one optional log-based metric (section 4).

## 1. Decisions To Confirm

Owner, before anything else. Record each answer in the pilot ticket (values
only where they are not secrets).

| Decision | Where it lands | What to check |
| --- | --- | --- |
| Pilot environment (`dev` first is recommended) | Terraform variables, section 4 | The extension is built for exactly one app origin, so a dev build cannot be used against prod. |
| Benefits sites on the Home screen | `config/relay-sites.json` (today `[]`) | Each entry is `{ "id", "name", "url" }`: `id` matches `^[a-z0-9-]{1,64}$`, `name` is 1 to 100 characters, `url` is https with no user or password (`lib/relay/config.ts`; invalid entries are dropped). The file is compiled into the app, so a change needs a PR and a deploy (Claude-reviewable). With `[]`, Home offers only "Use this tab". |
| Models caseworkers may choose | `relay_allowed_models` (Terraform), sets `RELAY_ALLOWED_MODELS` | Ids must be keys of `MODEL_MAP` in `lib/ai/eve/model-map.ts`. Unknown ids are dropped silently; with none left the default `claude-opus-5` applies. |
| Minimum Chrome version | `minimum_chrome_version` in `extension/manifest.json` (today `"120"`) | Ask IT which Chrome channel and version the pilot machines run. Raise the value only if the pilot relies on newer behavior. Older Chrome refuses to install the extension. |
| Extension hosting | Self-hosted CRX plus `updates.xml` (decision P5-g) or a Chrome Web Store listing | Self-hosted keeps the id `ppcenadbeddfnlgklnccjkgjegmbkkkl` (`extension/EXTENSION_ID`). A store listing may assign another id; see section 6.4. |
| Per-user run cap | `RELAY_MAX_RUNS_PER_HOUR` on the app (default 20) | No Terraform variable sets it; the default applies unless a PR adds one. |
| Participants for the pilot | `Participant` table, seeded by `scripts/seed-participants.ts` from `lib/data/participants.ts` | The seed records are fixtures. Decide whether the pilot uses them or real records, and who loads real ones. |

## 2. Secrets

Platform, in Secret Manager, for **both dev and prod** before
`RELAY_ENABLED` is set (section 4 explains why prod too).

Names (from `terraform/relay.tf`, suffixed with the base environment):

- `relay-token-secret-<env>`: signs run tokens; read by the relay and the app.
  At least 32 characters, or both token routes return 500 and the relay
  refuses to start.
- `eve-internal-secret-<env>`: `EVE_INTERNAL_SECRET` on the app and the Eve
  process. Shorter than 32 characters is treated as unset
  (`lib/relay/internal-secret.ts`).

The `-preview` pair is needed only if a preview relay is ever deployed some
other way: `deploy.yml` never gives a preview environment a relay.

### 2.1 Create

Generate and store in one pipe, so the value is never shown, copied or saved.
`tr -d '\n'` strips the trailing newline; neither the relay nor the app trims
the value.

```sh
openssl rand -hex 32 | tr -d '\n' | \
  gcloud secrets create relay-token-secret-<env> \
    --project=<project-id> --replication-policy=automatic --data-file=-

openssl rand -hex 32 | tr -d '\n' | \
  gcloud secrets create eve-internal-secret-<env> \
    --project=<project-id> --replication-policy=automatic --data-file=-
```

`openssl rand -hex 32` gives 64 hex characters. Never paste a value into a
ticket, chat, PR or workflow input.

### 2.2 Check

Platform. Presence and version only, never the value:

```sh
gcloud secrets versions list relay-token-secret-<env> --project=<project-id>
gcloud secrets versions list eve-internal-secret-<env> --project=<project-id>
```

Each must list one `enabled` version. Terraform reads both secrets at plan time
(`data "google_secret_manager_secret"` for `relay_token` and `eve_internal` in
`terraform/relay.tf`), so a missing secret fails the plan with a clear error
rather than a later revision.

## 3. Logging Exclusion Permission And Sinks

Platform, before `RELAY_ENABLED` is set.

Cloud Run's platform request logs (`run.googleapis.com/requests`) record the
full request URL, and the agent leg's URL carries its run token (and the
extension leg's, from an old extension build using the query fallback). The
relay deploy creates `google_logging_project_exclusion.relay_request_logs`
(`relay-request-logs-<env>`), which drops those request logs for the relay
service. The relay's own stdout logs never carry a URL, query, header or
token.

### 3.1 Grant or decide

The deploy identity, `github-actions-deploy`, needs `logging.exclusions.create`
for the exclusion and, if the metric is turned on (section 4),
`logging.logMetrics.create`. `roles/logging.configWriter` has both.
`terraform/iam.tf` does not grant it. The account also holds `roles/editor`
(`terraform/github-actions/github-actions-auth.tf`); whether that already
covers both permissions is **unconfirmed**: check, do not assume.

Pick one, and record the choice:

- Grant `roles/logging.configWriter` to `github-actions-deploy` (recommended).
- Or set `relay_exclude_request_logs = false` knowingly. Token-bearing request
  URLs then stay in Cloud Logging for anyone with log read access, for the
  token's lifetime (30 minutes).

The relay service depends on the exclusion, so a missing permission fails the
apply before the relay exists, never with a live relay logging tokens.

### 3.2 Check the sinks

Platform. The exclusion applies to the `_Default` sink only. Any other sink
that matches relay request logs still receives token URLs.

```sh
gcloud logging sinks list --project=<project-id>
gcloud logging sinks list --organization=<org-id>   # org-level aggregated sinks
gcloud logging sinks list --folder=<folder-id>      # if the project sits in a folder
```

For each sink other than `_Default` and `_Required`, read its filter. If it
could match `resource.type="cloud_run_revision"` for
`labs-asp-relay-<env>`, add an exclusion to that sink or narrow its filter
before relay runs are enabled. Also check the relay log bucket's retention
and IAM match the app's logs: the relay's `accepted` line now carries the
run's `origin`, which reveals which benefits site a caseworker is working on.
Do not add the user's `sub` to relay logs.

**Unverified:** whether Cloud Run's automatic request traces (Cloud Trace)
store the request URL. Check one relay trace after the first run (section 8).

## 4. Repository And Terraform Variables

### 4.1 How the switches work today

- `RELAY_ENABLED` is a **repository-level** GitHub Actions variable. It must
  never be set at the environment level: the `setup` job in
  `.github/workflows/deploy.yml` has no `environment:`, but the `promote` job
  in `.github/workflows/promote.yml` does, so an environment-level value would
  make the two workflows disagree and a develop push would plan a destroy of
  the relay.
- `deploy.yml` turns the relay on only for `dev` and `prod`. Setting
  `RELAY_ENABLED=true` therefore deploys the dev relay on the next push to
  `develop` **and the prod relay on the next push to `main` or the next prod
  promote**. It cannot be scoped to one environment. That is why section 2
  creates the prod secrets too: without them every prod plan fails at the
  secret data source, which blocks all prod deploys, not just the relay. If
  prod must not get a relay yet, change the allowlist in `deploy.yml`'s
  "Determine environment" step first (Claude-reviewable).
- **"One environment" applies to the run flags**, not to the relay service:
  relay runs are turned on per environment with the Terraform variables below.
- The pilot Terraform variables (in `terraform/variables.tf`, all defaulting to
  off):

  | Variable | Default | Sets |
  | --- | --- | --- |
  | `relay_min_instances` | `0` | relay `min_instance_count`; the pilot sets `1` (0 or 1 only) |
  | `relay_runs_enabled` | `false` | `RELAY_RUNS_ENABLED=1` on the app |
  | `relay_token_endpoint_enabled` | `false` | `RELAY_TOKEN_ENDPOINT_ENABLED=1` on the app |
  | `relay_extension_origin` | `""` | `RELAY_EXTENSION_ORIGIN` on the app |
  | `relay_allowed_models` | `""` | `RELAY_ALLOWED_MODELS` on the app |
  | `relay_allowed_hosts` | `""` | `RELAY_ALLOWED_HOSTS` on the relay |
  | `relay_allowed_host_suffixes` | `".run.app"` | `RELAY_ALLOWED_HOST_SUFFIXES` on the relay |
  | `relay_command_duration_metric_enabled` | `false` | the `relay-command-duration-<env>` log-based metric |
  | `relay_exclude_request_logs` | `true` | the request-log exclusion (section 3) |

  The app-side ones take effect only where the relay is deployed.
- **Neither workflow passes any of these yet.** Setting them needs a PR
  (Claude-reviewable) that wires each one into **both**
  `deploy.yml`'s "Terraform Plan" step (as `-var=`) and `promote.yml`'s
  "Terraform Plan" step (as a `TF_VAR_` environment entry, because a tag that
  predates a variable would reject `-var=`). Wiring only one lets the other
  workflow plan the value back to its default on its next run (the same hazard
  `jev_features` documents). Both Terraform jobs have `environment:`, so the
  values can come from GitHub environment-level variables (`dev`, `prod`);
  their names are to be chosen in that PR (none exist yet).

### 4.2 Before setting `RELAY_ENABLED`

Platform. Check its current value (Settings, Secrets and variables, Actions,
Variables; or `gh variable list`, read-only). If it is already `'true'` and a
preview environment already has a relay from before the dev/prod allowlist,
that preview's next push plans a destroy of its relay (deletion protection is
off for previews). That is expected; merge knowing it.

Prerequisites: sections 2 and 3 done for dev and prod; migration check
(section 5) done; the variable-wiring PR from 4.1 merged.

### 4.3 First apply: the relay, runs off

Platform.

1. Set the environment variables for dev through the wiring from 4.1:
   `relay_min_instances = 1` and, optionally,
   `relay_command_duration_metric_enabled = true` (needs
   `logging.logMetrics.create`). Leave the run flags off.
2. Set the repository variable `RELAY_ENABLED=true`.
3. Push to `develop` (or merge the next PR into it). The "Deploy Infrastructure"
   workflow builds `labs-asp-relay:<short sha>` and applies.
4. Check: the apply succeeded; the service `labs-asp-relay-dev` exists with
   min instances 1; the exclusion `relay-request-logs-dev` exists; then run
   the smoke checks in 8.1 and 8.2.
5. Read the relay host:

   ```sh
   gcloud run services describe labs-asp-relay-dev --region=<region> \
     --project=<project-id> --format='value(status.url)'
   ```

   This is the service `uri` that the app gets as `RELAY_PUBLIC_URL` (and the
   Terraform output `relay_service_url`). Strip `https://` to get
   `<relay-host>`. Cloud Run can serve a service on two URL forms
   (`<service>-<project-number>.<region>.run.app` and an older hashed form);
   **confirm** which one `status.url` returns in this project, since that is
   the one the app and extension use.

### 4.4 Second apply: the exact host (two applies to avoid a 421 lockout)

A wrong host makes every relay request answer 421, including `/healthz`.

1. Platform: set `relay_allowed_hosts = "<relay-host>"` and keep
   `relay_allowed_host_suffixes` at `".run.app"`. Apply (push to `develop`).
   The validation rejects a scheme, port, path or space.
2. Check: `/healthz` answers 200 on `<relay-host>` (8.1).
3. Platform: set `relay_allowed_host_suffixes = ""`. Apply.
4. Check: `/healthz` still answers 200 on `<relay-host>`. If it answers 421,
   set the suffix back to `".run.app"` and apply to recover.

The `lifecycle` precondition in `terraform/relay.tf` refuses a plan with both
lists empty.

### 4.5 Promotion while the relay is enabled

Platform and Owner. `promote.yml` now **fails** when the relay image for the
promoted commit is missing and `RELAY_ENABLED` is `'true'` (the chatbot image
check still only warns). Consequences:

- Promote only versions whose commit was deployed through `develop` or `main`
  with the relay on, so `labs-asp-relay:<short sha>` exists.
- You cannot roll back by promoting a tag built before `RELAY_ENABLED=true`, or
  a commit built only on a feature branch (preview builds never build the
  relay image).
- Do not unset `RELAY_ENABLED` to get past the failure: that plans a destroy of
  the relay.

### 4.6 Platform notes

- `terraform fmt -check` exits 3 on 8 pre-existing unformatted files
  (`cloud_sql.tf`, `iam.tf`, `storage.tf`, and files under `github-actions/`
  and `shared-preview-vpc/`). That is not caused by the relay changes.
- The variable validations and the `relay_extension_origin` precondition have
  not been exercised with non-default values; the first plan with them is the
  first test. Read that plan before applying.

## 5. Migrations

Platform, before the first deploy that carries migrations 0008 to 0011
(`lib/db/migrations/`).

- `scripts/start-container.sh` applies migrations at container start
  (`pnpm tsx lib/db/migrate`, `set -euo pipefail`). A failed migration
  crash-loops the new revision. There are no down migrations; rolling the code
  back is safe (old code ignores `Participant`; the index only touches relay
  rows).
- 0008 adds `Participant`; 0009 adds the partial unique index on
  `EveRun.chatId` for relay rows; 0010 adds `EveRun.endedAt`; 0011 adds
  `EveRun.turnIdle`.
- **Before** that deploy, run this against the target database and expect no
  rows:

  ```sql
  SELECT "chatId", count(*) FROM "EveRun" WHERE "relayOrigin" IS NOT NULL GROUP BY 1 HAVING count(*) > 1;
  ```

  If it returns rows, stop: 0009 will fail. In practice none exist, since
  `RELAY_RUNS_ENABLED` has been set nowhere.
- After the migrations, seed participants manually; the container does not:

  ```sh
  pnpm exec tsx scripts/seed-participants.ts
  ```

  The script needs `DATABASE_URL` for the target database in its environment.
  How to reach the Cloud SQL instance from your machine (proxy, bastion) is
  **confirm** with Platform; never paste the connection string into a ticket.

## 6. Build, Package And Host The Extension

The extension is built for exactly one app origin (`APP_ORIGIN`, baked in at
build time, and `host_permissions` covers only `<app-origin>/*`), so dev and
prod need separate packages. The app origin is the deployed one: with the
custom domain, `https://dev.<domain_name>` for dev and `https://<domain_name>`
for prod (`terraform/cloud_run.tf`). Confirm against the browser address bar.

### 6.1 Release build check in CI (optional)

Platform. Run the "Extension Release" workflow
(`.github/workflows/extension-release.yml`, `workflow_dispatch`) with inputs
`environment` (`dev` or `prod`, names the artifact only), `app_origin`,
`version`, and optionally both `update_url` and `package_url`. It uploads
`extension.zip` (and `updates.xml` when both URLs are given) as an artifact
for inspection only. It holds no key and never produces a CRX.
`actions/upload-artifact@v4` is new to this repository's workflows; if the
organization restricts actions, it may need to be allowed.

### 6.2 Build the signed package locally

Owner, on a machine that holds the extension's private key. The key (`.pem`)
is never in the repository, CI or a workflow.

```sh
EXTENSION_PEM_PATH=/path/outside/the/repo/extension.pem \
pnpm exec tsx scripts/package-extension.ts \
  --app-origin <app-origin> \
  --out ./release \
  --version <x.y.z> \
  --update-url https://<update-host>/<path>/updates.xml \
  --package-url https://<update-host>/<path>/extension.crx
```

- The key must be an unencrypted PEM, in a file outside the repository, that
  matches the manifest `key`; anything else is refused with a fixed message.
- `--update-url` and `--package-url` go together, and `--update-url` must end
  with `updates.xml`. With them, the packaged manifest carries `update_url`.
- Outputs in `./release`: `extension.zip`, `updates.xml`, `extension.crx`.
- Check the last line: `packaged extension <id> version <x.y.z> for <app-origin>`.
  `<id>` must equal `extension/EXTENSION_ID`.
- Raise `--version` for every release; Chrome updates only when the version in
  `updates.xml` increases.
- **The CRX3 output is untested in real Chrome** (it is verified only by a
  test-written parser and a throwaway key). If Chrome refuses it, fall back to
  `chrome --pack-extension=<unpacked dir> --pack-extension-key=<pem>` on the
  release build output, and report it.

### 6.3 Host

Owner or Platform. Host `extension.crx` and `updates.xml` at the two URLs
used above, over https, reachable by Chrome's updater on the caseworkers'
machines (a bucket behind sign-in is not). Neither file holds a secret: the app
origin is the only value built in, and there are no tokens.

### 6.4 Extension id check

Owner. All of these must name the same id:

| Where | How to read it |
| --- | --- |
| `extension/EXTENSION_ID` | `cat extension/EXTENSION_ID` (today `ppcenadbeddfnlgklnccjkgjegmbkkkl`) |
| The package | the `packaged extension <id>` line from 6.2 |
| A pilot machine | `chrome://extensions` with Developer mode on shows the id |
| `RELAY_EXTENSION_ID` on the relay | set by Terraform from `extension/EXTENSION_ID` (`terraform/relay.tf`) |
| `RELAY_EXTENSION_ORIGIN` on the app | `relay_extension_origin = "chrome-extension://<id>"`; the precondition in `terraform/relay.tf` refuses any other value |

If a Chrome Web Store listing assigns another id (P5-g), re-derive it and
change `extension/EXTENSION_ID`, the manifest `key`, `relay_extension_origin`
and the policy together (Claude-reviewable for the files). A store upload may
also need the manifest `key` removed first and rejects `update_url` (build
without the two URL flags); both are unverified.

### 6.5 Policy

IT, with the Owner. Fill `deploy/chrome-policy/extension-policy.template.json`
(placeholders `__EXTENSION_ID__`, `__UPDATE_URL__`) as described in
`deploy/chrome-policy/README.md`, and apply it with IT's tooling (Google Admin
console, Group Policy/ADMX or MDM). `__UPDATE_URL__` is the `--update-url`
value. In the README's `sed` example, `&` in a URL must be escaped as `\&`.

The template deliberately sets no `runtime_blocked_hosts`, no
`blocked_permissions` and no `runtime_allowed_hosts`. IT must make sure no
other policy applies `runtime_blocked_hosts`, `DisableScreenshots` or a
screenshot DLP rule to this extension (section 7).

## 7. Managed-Chrome Tests

IT runs these on one pilot machine with the Owner watching, before any
caseworker uses the extension. None has been tested in this project, and
nothing has been exercised on a managed Chrome 155 or later. Record the Chrome
version (`chrome://version`) with every result.

1. **Force-install.** `chrome://policy` shows `ExtensionInstallForcelist` and
   `ExtensionSettings` with status OK. `chrome://extensions` shows the
   extension as installed by the administrator, with the id from 6.4. Then
   publish a higher version in `updates.xml` and confirm Chrome updates
   (`chrome://extensions`, Update).
2. **Version floor.** The machine runs at least `minimum_chrome_version`
   (`120`).
3. **Debugging bar, force-installed.** Start a run. Does the bar "Labs ASP
   Browser Bridge started debugging this browser" appear? The Ready screen
   shows "While the agent works, Chrome may show a bar saying the browser is
   being debugged; that is expected." either way. Dismissing the bar ends the
   run (`debugger-canceled`); tell caseworkers not to.
4. **Banner suppression flag.** Only if IT can control how Chrome starts:
   launch with `--silent-debugger-extension-api`, confirm it appears on
   `chrome://version` (Command Line), and repeat item 3. Chrome documents the
   flag only for extension background pages, so it may not hide the bar for
   tabs. Do not tell caseworkers the bar is hidden until this passes.
5. **Host access policy on Chrome 155+.** Start a run on each pilot site. The
   attach must succeed. If a `runtime_blocked_hosts` policy reaches the
   extension, Chrome rejects the attach with "Host access is restricted by
   policy." and the panel shows instead: "Chrome is blocking part of what this
   extension needs here because of your organization's policy. Ask IT to allow
   the extension." The agent is told "This browser is restricted by your
   organization's policy for this site." and ends its turn.
6. **Screenshot policy on Chrome 155+.** Same, for `DisableScreenshots` or a
   screenshot DLP rule ("Screenshot capture is restricted by policy."). Any
   screenshot the agent takes during item 5's run must work.
7. **`DeveloperToolsAvailability`.** Record the value IT sets (or that it is
   unset) and run item 5 under exactly that value; its effect on
   `chrome.debugger` is untested.
8. **Optional negative test.** On a test profile only, apply
   `runtime_blocked_hosts` for the extension and confirm item 5's policy copy
   appears. Known gap: a policy error raised inside the attach's inner
   `Page.enable` or `Page.getFrameTree` steps shows the generic error instead
   of the policy copy, and the screenshot-time mapping has been code-reviewed
   only.

The `--disable-features=ExtensionDebuggerStrictPolicyRestrictions` launch
flag is reported removed in Chrome 160; it is not a plan.

## 8. Smoke Checklist

Platform runs 8.1 to 8.3 after the first apply (4.3) and again after the
exact-host applies (4.4). The Owner runs 8.4 onward after the run flags are on
(section 9) in the pilot environment.

### 8.1 Health

```sh
curl -s -o /dev/null -w '%{http_code}\n' https://<relay-host>/healthz
```

Expect `200`. The Cloud Run startup probe is TCP, so a deployed revision can
be up while `/healthz` answers 421: always check it.

### 8.2 Wrong Host and refused upgrades

- **Wrong Host answers 421.** After the second exact-host apply, request
  `/healthz` on the relay's other Cloud Run URL form, if one exists; expect
  `421`. Whether Google's front end passes an arbitrary `Host` header to the
  container is unverified, so `curl -H 'Host: ...'` may not reach the relay;
  the 421 behavior itself is covered by `tests/relay/server.test.ts`. Mark the
  result **confirm** if no second URL form exists.
- **No or bad token.** A WebSocket upgrade to `/extension` or `/cdp` with no
  token or a made-up one must be refused. The relay logs `upgrade-rejected`
  with `reason` `no-token` or `token` (severity WARNING).

### 8.3 Request logs and sinks

Logs Explorer, project `<project-id>`, last hour:

```
resource.type="cloud_run_revision"
resource.labels.service_name="labs-asp-relay-<env>"
logName:"run.googleapis.com%2Frequests"
```

Expect no entries. A WebSocket's request log is written when the socket
closes (up to an hour later), so repeat this after the run in 8.4 has ended.
An empty result proves only the `_Default` sink; repeat the sink review from
3.2.

Also search all project logs for the relay token's fixed prefix,
`eyJydW5JZCI6` (base64url of `{"runId":"`), and for `/cdp?token=`. Expect no
entries. If something matches, do not copy it anywhere; restrict access and
report it.

### 8.4 One run end to end

Owner, on a pilot machine with the force-installed extension, signed in to
`<app-origin>`:

- Home shows the configured sites; Ready shows the debugging-bar sentence.
- Start a run; the agent snapshots and fills a field; an `ask_question` card
  appears and can be answered; "Fill it in" types a sensitive value into a
  focused field; Stop ends the run (server, then worker) and the Recents entry
  shows it ended.
- Relay logs (Logs Explorer, `labs-asp-relay-<env>`, stdout) show `accepted`
  for both roles with the run's `origin`, `command` lines, and `run-closed`.
  No line carries a URL, query, header, token or page content.
- Another signed-in user, or a signed-out browser, asking
  `GET /api/relay/runs/<run id>` gets 404.

### 8.5 Timings against the spike

Compare with the spike: about 56 ms round trip, about 600 ms snapshot, about
240 ms fill. The relay-side round trip per CDP command is the `durationMs` of
`jsonPayload.event="command"` lines (the
`relay-command-duration-<env>` metric extracts `EXTRACT(jsonPayload.durationMs)`
when enabled). Snapshot and fill times are visible from the agent side (tool
call durations). Record the medians; investigate anything several times slower.

### 8.6 Eve Postgres world

Platform. The Eve workflow world lives in the same database as the app
(`WORKFLOW_POSTGRES_URL` reads the `database-url-dev` or
`database-url-production` secret, `terraform/cloud_run.tf`). Search its text and JSON columns for
`eyJydW5JZCI6` and `token=`, counting matches only so a hit never prints a
token. List the workflow schemas first (`\dn`; they are created by
`scripts/bootstrap-workflow-db.ts` and `@workflow/world-postgres`); table
names are not listed here because they belong to that package. Expect zero.
Plain `eyJ` also matches other JSON in base64 (for example session cookies),
so use the specific prefix.

### 8.7 Mid-run relay restart

Owner and Platform together, on dev. Start a run, then make the relay start a
new revision while the run is live (the simplest Terraform-neutral way: merge
any commit to `develop`, which builds a new relay image tag and deploys it).
Expected result, by design: **the run ends cleanly**. The panel shows the run
as ended, the server row is ended (by the panel's session, or by the orphan
sweep on the next panel open), and a new run starts normally afterwards. The
run is not expected to recover: run state is in memory. Anything else (a
panel stuck on Running, an agent still minting tokens for the old run) is a
finding.

Also note when a run passes one hour: the socket closes at Cloud Run's
`3600s` limit and the run ends.

## 9. Turning Relay Runs On, And Rollback

### 9.1 Flags on, last

Platform, after sections 1 to 8.3 pass and section 7 passes on a pilot
machine. One apply, pilot environment only:

- `relay_runs_enabled = true` (sets `RELAY_RUNS_ENABLED=1`)
- `relay_token_endpoint_enabled = true` (sets `RELAY_TOKEN_ENDPOINT_ENABLED=1`)
- `relay_extension_origin = "chrome-extension://<id>"` (6.4)
- `relay_allowed_models` as decided in section 1

`RELAY_RUNS_ENABLED=1` also turns on trace scrubbing
(`lib/observability/scrubbing-exporter.ts`): it applies to every span the Eve
process exports, web chats included.
Then run 8.4 to 8.7.

### 9.2 Rollback

1. **Flags off first** (Platform): set `relay_runs_enabled = false` and
   `relay_token_endpoint_enabled = false` and apply. The relay routes answer
   404. A relay run row left from before continues as a relay run, but the
   agent-token route answers 404, so its browser calls fail closed. The panel's
   config probe then answers 404 and shows the `not_found` screen with no
   button (a known dead end).
2. **The relay itself can stay**: with the flags off it serves nothing useful.
3. **Removing the relay** (Platform, deliberately): unsetting `RELAY_ENABLED`
   plans a destroy of the relay in dev and prod. Deletion protection on the
   dev and prod relay services makes that apply fail. To remove it, first set
   `deletion_protection` false on the relay service (Claude-reviewable change
   in `terraform/relay.tf`, applied), or `terraform state rm` it, then unset
   `RELAY_ENABLED`. Unsetting it also changes what promote does (4.5).
4. **Code rollback**: see 4.5; only versions built through `develop` or `main`
   with the relay on can be promoted while `RELAY_ENABLED` is `'true'`.

## 10. Not Covered, And Open Questions

Deliberately out of scope for this pilot:

- More than one relay instance, `REDIS_URL`, `RELAY_MULTI_INSTANCE_ACKNOWLEDGED`
  and every multi-instance gap listed under "Plan 5: Rollout" in
  `docs/browser-extension-todo.md`.
- OpenTelemetry spans in the relay (logs and one metric only, decision P5-h).
- Submit-detection tuning on real portals (XHR and single-page submits,
  multi-step Next prompts).
- Removing the `/extension` query-token fallback (after old builds are gone),
  and a shorter or single-use agent token.
- A Chrome Web Store listing, unless chosen in section 1.

Open questions for IT (also in `docs/browser-extension-todo.md`, "Open
Questions"):

- Which Chrome channel and version do the pilot machines run, and when do they
  reach Chrome 155?
- Can a launch flag (`--silent-debugger-extension-api`) be set on the managed
  fleet, and how do caseworkers start Chrome today?
- Do managed policies set `runtime_blocked_hosts` for this extension,
  `DisableScreenshots`, or screenshot DLP rules?
- Which value of `DeveloperToolsAvailability` do the machines use?
- Which team owns distribution (force-install or a store listing)?

Open questions for the Owner: the site list (`config/relay-sites.json`), the
allowed models, extra sensitive-field categories beyond SSN and bank details,
and whether the pilot uses the fixture participants.
