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
  redeploy or Cloud Run recycle drops every live run's sockets and loses the
  relay's state for them. The tombstone that refuses a re-dial with a closed
  run's token is lost on restart too, so a closed run's token can dial a
  restarted relay until it expires.
- **The one-hour socket limit.** The relay's request timeout is Cloud Run's
  maximum, `3600s` (`terraform/relay.tf`). A WebSocket is a request, so no
  single connection lasts longer than one hour.
- **What a dropped socket does to a run is not verified end to end.** From the
  code: the extension worker reconnects with the same token (backoff from 1 s
  to 30 s) unless the relay closed with 1000 or 1008 (`FINAL_CLOSE_CODES` in
  `extension/src/bridge.ts`); the Eve `browser` tool treats a lost connection
  by rotating its agent-browser session and retrying a retry-safe command
  (`lib/relay/eve-relay-browser.ts`); an in-flight command fails. The panel
  shows no reconnecting state: it stays on Running while the worker still
  holds the run. Whether a run recovers or ends after a restart or the
  one-hour close is checked in 8.7 (**confirm**). Plan decision P5-c assumed
  these end live runs.
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
  plus one optional log-based metric (section 4). Relay log lines carry a
  `severity` (`WARNING` for refusals such as `upgrade-rejected`), and the
  `accepted` line carries the run's `origin`.
- **Outbound backpressure cap.** The relay accepts inbound messages up to
  32 MiB and closes a peer whose outbound buffer passes 64 MiB
  (`MAX_BUFFERED_BYTES` in `relay/hub.ts`, logged as `backpressure-closed`),
  which ends that run. The relay has 512Mi of memory.

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

Run section 5's migration check before any PR from this section merges or
gets a preview deploy.

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
  secret data source, which blocks all prod deploys, not just the relay.
  **Setting `RELAY_ENABLED` requires the prod secrets (and the section 3
  permission) first.** There is no supported way to keep prod without a relay
  while it is `'true'`: narrowing the allowlist in `deploy.yml` alone is not
  enough, because `promote.yml` keys only on `vars.RELAY_ENABLED` (its "Verify
  images exist" step and `TF_VAR_relay_image_url`), so prod promotes would
  still plan the relay.
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
- **How to set them: GitHub environment-level variables.** Set these on the
  `dev` or `prod` environment (Settings, Environments, `<env>`, Environment
  variables), **not** at repository level, where a value would reach every
  environment, previews included (harmless where no relay is deployed, but
  not what you mean):

  | GitHub environment variable | Terraform variable | Accepted values |
  | --- | --- | --- |
  | `RELAY_MIN_INSTANCES` | `relay_min_instances` | `0` or `1` |
  | `RELAY_RUNS_ENABLED` | `relay_runs_enabled` | `true` or `false` |
  | `RELAY_TOKEN_ENDPOINT_ENABLED` | `relay_token_endpoint_enabled` | `true` or `false` |
  | `RELAY_EXTENSION_ORIGIN` | `relay_extension_origin` | `chrome-extension://<32 characters a-p>` |
  | `RELAY_ALLOWED_MODELS` | `relay_allowed_models` | comma-separated ids of `A-Z a-z 0-9 . _ / : -` |
  | `RELAY_ALLOWED_HOSTS` | `relay_allowed_hosts` | comma-separated bare host names |
  | `RELAY_ALLOWED_HOST_SUFFIXES` | `relay_allowed_host_suffixes` | comma-separated suffixes (leading dot allowed), or `none` |
  | `RELAY_COMMAND_DURATION_METRIC_ENABLED` | `relay_command_duration_metric_enabled` | `true` or `false` |

  `relay_exclude_request_logs` is not wired; it keeps its default `true`.
- **Both workflows pass them the same way.** The `terraform` job in
  `deploy.yml` and the `promote` job in `promote.yml` each have a "Relay
  Terraform variables" step, just before "Terraform Plan", that maps the eight
  `vars.*` above into its environment and runs
  `scripts/relay-terraform-vars.sh >> "$GITHUB_ENV"`. The script prints a
  `TF_VAR_<name>=<value>` line for each variable that is set, and the plan
  step reads them as environment. One script for both means a promote cannot
  plan a value back to its default (the hazard `jev_features` documents).
  `promote.yml` takes the script from the commit the workflow runs from (a
  sparse checkout of `github.sha` into `workflow-ref/`), not from the
  promoted tag, so it works for tags that predate it; Terraform at an older
  tag ignores any `TF_VAR_` it does not declare.
- **An unset (or blank) variable is not passed**, so Terraform's default
  applies (the table above in this section). That is why
  `relay_allowed_host_suffixes` keeps `.run.app` until you deliberately set
  `RELAY_ALLOWED_HOST_SUFFIXES=none`: deleting the variable restores
  `.run.app`, it does not empty the list. `none` is the only way to pass an
  empty suffix list, and the script refuses it unless `RELAY_ALLOWED_HOSTS` is
  set in the same environment (section 4.4).
- **An invalid value fails the step** before any plan, with a fixed message
  that names the variable and never prints the value; check the table above.
  Host names and suffixes are lowercased; spaces around commas are dropped.
- **Older tags.** Promoting a tag whose `terraform/variables.tf` predates a
  pilot variable plans that tag's own behavior for it (for example its fixed
  `.run.app` suffix or minimum 0 instances), whatever the environment variable
  says, until the next `deploy.yml` run or a newer promote. Only promote tags
  built after this wiring once the pilot variables are set.

### 4.2 Before setting `RELAY_ENABLED`

Platform. Check its current value (Settings, Secrets and variables, Actions,
Variables; or `gh variable list`, read-only). If it is already `'true'` and a
preview environment already has a relay from before the dev/prod allowlist,
that preview's next push plans a destroy of its relay. Deletion protection is
off only for `preview-*` environments (`terraform/relay.tf`); the plain
`preview` environment (feature-branch pushes, the `workflow_dispatch` default)
keeps it on, so there the destroy FAILS and blocks every `preview` deploy
until the relay is dealt with. Check the variable value first.

Prerequisites: sections 2 and 3 done for dev and prod; the migration check
(section 5) done for the dev and prod databases before the wiring PR merges
or previews; the variable-wiring PR from 4.1 merged.

### 4.3 First apply: the relay, runs off

Platform. Written for `<env>`; do dev first. Prod gets the relay on its next
deploy too (4.1), so repeat 4.3 step 4 and section 4.4 for prod after that
deploy, and run 8.1 to 8.3 there as well (section 8).

How applies happen: dev applies on a push to `develop`, prod on a push to
`main` or a prod promote. If you use `workflow_dispatch` on "Deploy
Infrastructure" instead, run it from `develop` (for dev) or `main` (for prod):
dispatching with `environment` `dev` from any other branch deploys that
branch's code to dev.

1. Set the GitHub environment variables for `<env>` (4.1):
   `RELAY_MIN_INSTANCES=1` and, optionally,
   `RELAY_COMMAND_DURATION_METRIC_ENABLED=true` (needs
   `logging.logMetrics.create`). Leave `RELAY_RUNS_ENABLED` and
   `RELAY_TOKEN_ENDPOINT_ENABLED` unset.
2. Set the repository variable `RELAY_ENABLED=true`.
3. Push to `develop` (or merge the next PR into it) for dev. The "Deploy
   Infrastructure" workflow builds `labs-asp-relay:<short sha>` and applies.
4. Check: the apply succeeded; the service `labs-asp-relay-<env>` exists with
   min instances 1; the exclusion `relay-request-logs-<env>` exists; then run
   the smoke checks in 8.1 to 8.3.
5. Read the relay host **from the value the app uses**: the app's
   `RELAY_PUBLIC_URL`, which Terraform sets from the relay service's `uri`
   (`google_cloud_run_v2_service.relay[*].uri`, also the Terraform output
   `relay_service_url`). It is a plain, non-secret value. Either:

   - the Terraform output (read-only; from `terraform/` after
     `terraform init -backend-config="prefix=terraform/state/<env>"`):
     `terraform output -raw relay_service_url`; or
   - the app revision's environment, printing only that one variable (the
     app's environment may hold other values that should not be printed):

     ```sh
     gcloud run services describe ai-chatbot-<env> --region=<region> \
       --project=<project-id> --format=json \
       | jq -r '.spec.template.spec.containers[0].env[] | select(.name=="RELAY_PUBLIC_URL") | .value'
     ```

     (**confirm** the JSON path against your gcloud version.)

   Strip `https://` to get `<relay-host>`. Do not take it from
   `gcloud run services describe labs-asp-relay-<env> --format='value(status.url)'`:
   Cloud Run can serve a service on two URL forms
   (`<service>-<project-number>.<region>.run.app` and an older hashed form),
   and only the host in `RELAY_PUBLIC_URL` is the one the app hands to the
   extension and the agent. Listing both forms in `relay_allowed_hosts`
   (comma-separated) is allowed.

### 4.4 Second apply: the exact host (two applies to avoid a 421 lockout)

A wrong host makes every relay request answer 421, including `/healthz`.
While the `.run.app` suffix is still set, the relay accepts a host that is in
`RELAY_ALLOWED_HOSTS` **or** ends with the suffix (`hostOk` in
`relay/server.ts`), so any check made with the suffix still set passes even
with a typo or the wrong URL form: **it proves nothing**. The safety check is
a comparison of values, done before dropping the suffix. This switch is safe
to do now only because the run flags are still off: a lockout breaks no
caseworker's run, and the suffix can be restored with another apply.

1. Platform: set the environment variable `RELAY_ALLOWED_HOSTS=<relay-host>`
   (from 4.3 step 5) and leave `RELAY_ALLOWED_HOST_SUFFIXES` unset, so
   `relay_allowed_host_suffixes` stays `".run.app"`. Apply (for dev, push to
   `develop`). The workflow step and the Terraform validation reject a
   scheme, port, path or space.
2. Check, before going on: the relay revision's `RELAY_ALLOWED_HOSTS` contains
   exactly the host part of the app's `RELAY_PUBLIC_URL`. Read both values
   one variable at a time (same `jq` filter as 4.3 step 5, with
   `labs-asp-relay-<env>` and `RELAY_ALLOWED_HOSTS` for the relay) and compare
   them character for character. If they differ, fix `relay_allowed_hosts`
   and apply again; do not drop the suffix.
3. Platform: set the environment variable `RELAY_ALLOWED_HOST_SUFFIXES=none`
   (passes `relay_allowed_host_suffixes = ""`; deleting the variable would
   not, see 4.1). Apply.
4. Check: `/healthz` answers 200 on `<relay-host>` (8.1), and a wrong Host
   answers 421 (8.2). If `/healthz` answers 421, delete
   `RELAY_ALLOWED_HOST_SUFFIXES` (back to `".run.app"`) and apply to recover.

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

Platform. **Do this before anything in section 4 merges or deploys.** The
migrations ship with the code, not with `RELAY_ENABLED`: the first deploy of
any commit that carries them applies them, whatever the relay variables say.
That can be a preview deploy of a PR from this work (every preview uses the
dev Cloud SQL instance, `terraform/main.tf`), the merge of
the section 4.1 wiring PR, or any merge to `develop` or `main` after this
branch lands.

Which migrations are pending: when this runbook was written, the local copy
of `origin/develop` (`6cca9703`) ended at `0006` and `origin/main`
(`5bb797f7`) had none of `0007` to `0011`; `origin/browser-extension` had
them. Those refs may be stale: run `git fetch` and
`git ls-tree --name-only origin/develop lib/db/migrations/` (and the same for
`origin/main`) to see what is actually deployed before relying on this.

- `scripts/start-container.sh` applies migrations at container start
  (`pnpm tsx lib/db/migrate`, `set -euo pipefail`). A failed migration
  crash-loops the new revision. There are no down migrations; rolling the code
  back is safe (old code ignores `Participant`; the index only touches relay
  rows).
- 0007 creates `EveRun` (with `relayOrigin`); 0008 adds `Participant`; 0009 adds the partial unique index on
  `EveRun.chatId` for relay rows; 0010 adds `EveRun.endedAt`; 0011 adds
  `EveRun.turnIdle`.
- **Before** the first merge or preview deploy that carries them, run this
  against **each** database the code will reach and expect no rows: the dev
  database (dev and every preview), and the prod database (before the merge
  to `main` or the prod promote that carries them):

  ```sql
  SELECT "chatId", count(*) FROM "EveRun" WHERE "relayOrigin" IS NOT NULL GROUP BY 1 HAVING count(*) > 1;
  ```

  If it returns rows, stop: 0009 will fail. If it fails because `EveRun` does
  not exist, that database predates 0007, no relay rows can exist, and 0007
  and 0009 apply together. In practice no duplicates exist, since
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
- Check the last line: `packaged extension <id> version <x.y.z> for <app-origin>: <files written>`.
  `<id>` must equal `extension/EXTENSION_ID`, and the list of files written
  must include `extension.crx`. An unset or empty `EXTENSION_PEM_PATH` is not
  an error: the script then writes only `extension.zip` and `updates.xml`.
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
6. **Screenshot policy on Chrome 155+.** For `DisableScreenshots` or a
   screenshot DLP rule ("Screenshot capture is restricted by policy."), any
   screenshot the agent takes during item 5's run must work. If one is
   refused, the expected outcome is that the AGENT tells the caseworker IT must
   allow the extension (it is given the fixed message and ends its turn); the
   panel shows NO notice, because the panel notice is set only for a refused
   attach. IT should look for `DisableScreenshots`, a screenshot DLP rule or
   `runtime_blocked_hosts` covering the site in `chrome://policy`.
7. **`DeveloperToolsAvailability`.** Record the value IT sets (or that it is
   unset) and run item 5 under exactly that value; its effect on
   `chrome.debugger` is untested.
8. **Optional negative test.** On a test profile only, apply
   `runtime_blocked_hosts` for the extension and confirm item 5's policy copy
   appears (a refused attach). Known gaps: a refused CDP command or screenshot
   sets no panel notice (the agent's text only), and a policy error raised inside the attach's inner
   `Page.enable` or `Page.getFrameTree` steps shows the generic error instead
   of the policy copy, and the screenshot-time mapping has been code-reviewed
   only.

The `--disable-features=ExtensionDebuggerStrictPolicyRestrictions` launch
flag is reported removed in Chrome 160; it is not a plan.

## 8. Smoke Checklist

Platform runs 8.1 to 8.3 in **every environment that has a relay**, dev and
prod: after its first apply (4.3) and again after its exact-host applies
(4.4). Prod gets its relay on its first deploy after `RELAY_ENABLED` is set
(4.1), so schedule the prod checks for right after that deploy, even if the
pilot runs on dev. The Owner runs 8.4 onward after the run flags are on
(section 9), in the pilot environment.

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
`eyJydW5JZCI6` (the base64url of `{"runId":`, 12 characters, which every
relay token starts with), and for `/cdp?token=`. Do not open
matching entries in Logs Explorer: a match would display the token. Count
timestamps only:

```sh
gcloud logging read '"eyJydW5JZCI6"' --project=<project-id> --freshness=1d \
  --format='value(timestamp)' | wc -l
gcloud logging read '"/cdp?token="' --project=<project-id> --freshness=1d \
  --format='value(timestamp)' | wc -l
```

Expect `0` for both. If either is not zero, do not copy the entries anywhere;
restrict log access and report it with the timestamps only. (**confirm** that
a quoted string in the filter is the global text search in your gcloud
version.)

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
What to expect is **not verified** (see Known Limits). From the code, the
worker reconnects with the same token and the agent's `browser` tool rotates
and retries, so the run may continue on the new revision with its in-flight
command failed; plan decision P5-c assumed it ends. Record which happens:

- **It continues:** the panel stays on Running, the next agent command
  succeeds, and Stop still ends the run normally (server row ended, Recents
  marked).
- **It ends:** the panel shows the run as ended, the server row is ended (by
  the panel's session, or by the orphan sweep on the next panel open), and a
  new run starts normally afterwards.

Either is acceptable. A panel stuck on Running while the agent keeps failing,
or the agent still minting tokens for an ended run, is a finding.

Also run one session past one hour and record the same for the socket close
at Cloud Run's `3600s` limit.

## 9. Turning Relay Runs On, And Rollback

### 9.1 Flags on, last

Platform, after sections 1 to 8.3 pass and section 7 passes on a pilot
machine. One apply, pilot environment only:

- `relay_runs_enabled = true` (sets `RELAY_RUNS_ENABLED=1`)
- `relay_token_endpoint_enabled = true` (sets `RELAY_TOKEN_ENDPOINT_ENABLED=1`)
- `relay_extension_origin = "chrome-extension://<id>"` (6.4)
- `relay_allowed_models` as decided in section 1

Through the GitHub environment variables of that environment (4.1):
`RELAY_RUNS_ENABLED=true`, `RELAY_TOKEN_ENDPOINT_ENABLED=true`,
`RELAY_EXTENSION_ORIGIN` and `RELAY_ALLOWED_MODELS`.

`RELAY_RUNS_ENABLED=1` on the app also turns on trace scrubbing
(`lib/observability/scrubbing-exporter.ts`): it applies to every span the Eve
process exports, web chats included.
Then run 8.4 to 8.7.

### 9.2 Rollback

1. **Flags off first** (Platform): set `relay_runs_enabled = false` and
   `relay_token_endpoint_enabled = false` (set the environment variables
   `RELAY_RUNS_ENABLED` and `RELAY_TOKEN_ENDPOINT_ENABLED` to `false`, or
   delete them) and apply. The relay routes answer
   404. A relay run row left from before continues as a relay run, but the
   agent-token route answers 404, so its browser calls fail closed. The panel's
   config probe then answers 404 and shows the `not_found` screen with no
   button (a known dead end).
2. **The relay itself can stay**: with the flags off it serves nothing useful.
3. **Removing the relay** (Platform, deliberately): unsetting `RELAY_ENABLED`
   plans a destroy of the relay in dev and prod. Deletion protection on the
   dev and prod relay services makes that apply fail. To remove it, first set
   `deletion_protection` false on the relay service (Claude-reviewable change
   in `terraform/relay.tf`, applied), then unset `RELAY_ENABLED` so the next
   apply destroys the relay service together with its exclusion, account and
   invoker grant, in the order `depends_on` in `terraform/relay.tf` sets (the
   service goes first, the request-log exclusion after it). Do not use
   `terraform state rm` on the service: the relay would keep running,
   unmanaged, and the next apply would delete the exclusion, the service
   account and the public invoker grant while it is still live, so its
   token-bearing request URLs would reach the logs again. Unsetting
   `RELAY_ENABLED` also changes what promote does (4.5).
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
