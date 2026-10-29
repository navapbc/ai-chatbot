# Wire Relay Terraform Variables

## Summary

The relay pilot Terraform variables added in PR 347 can now be set per environment. `deploy.yml` and `promote.yml` both run `scripts/relay-terraform-vars.sh`, which turns eight GitHub environment-level variables into `TF_VAR_` entries before the plan. Nothing was applied or deployed, and no variable is set anywhere, so every plan is unchanged until someone sets one.

## Changes made

- `scripts/relay-terraform-vars.sh` (new): reads `RELAY_RUNS_ENABLED`, `RELAY_TOKEN_ENDPOINT_ENABLED`, `RELAY_EXTENSION_ORIGIN`, `RELAY_ALLOWED_MODELS`, `RELAY_ALLOWED_HOSTS`, `RELAY_ALLOWED_HOST_SUFFIXES`, `RELAY_MIN_INSTANCES` and `RELAY_COMMAND_DURATION_METRIC_ENABLED`. It prints `TF_VAR_<terraform name>=<value>` only for a variable that is set (non-blank), after validating it: bools `true`/`false`, minimum instances `0`/`1`, the origin `chrome-extension://<32 a-p>`, host and suffix lists of bare names (trimmed, lowercased, no empty entry), and model lists of `A-Za-z0-9._/:-`. A value with a control character is refused, so a line appended to `$GITHUB_ENV` cannot add a variable. On any failure it prints one fixed line naming the variable to stderr, nothing to stdout, and exits 1. `RELAY_ALLOWED_HOST_SUFFIXES=none` prints an empty `TF_VAR_relay_allowed_host_suffixes`, and only when `RELAY_ALLOWED_HOSTS` is set.
- `.github/workflows/deploy.yml`: a "Relay Terraform variables" step before "Terraform Plan" in the `terraform` job maps the eight `vars.*` and runs `bash scripts/relay-terraform-vars.sh >> "$GITHUB_ENV"`. The existing `-var=` flags are unchanged.
- `.github/workflows/promote.yml`: a sparse checkout of `github.sha` into `workflow-ref/` (after the tag checkout) and the same step, running `workflow-ref/scripts/relay-terraform-vars.sh`.
- `terraform/variables.tf`: comments only (the pilot block no longer says the workflows do not pass these; the host switch names `none`).
- Docs: `docs/rollout-pilot.md` sections 4.1, 4.3, 4.4, 9.1 and 9.2; `docs/browser-extension-todo.md`; `CLAUDE.md`.
- Tests: `tests/agent/relay-terraform-vars.test.ts` (new) runs the script with a controlled environment; `tests/agent/relay-deploy-config.test.ts` checks that both workflows map every variable and run the script before their plan, that promote takes it from `github.sha` after the tag checkout, and that every emitted variable is declared in `terraform/variables.tf`.

## What it improves or fixes

- Before: the pilot variables existed in Terraform but neither workflow passed them, so they could not be set, and wiring one workflow would have let the other plan them back to defaults. After: one script feeds both, so they cannot disagree.
- An unset GitHub variable expands to `''`. Passing it straight through would have blanked `relay_allowed_host_suffixes` (skipping the exact-host comparison step) and failed the bool and number variables. The script passes only set variables, so unset keeps the default.
- With unset meaning default, the suffix list could not be emptied; `none` is the explicit way, guarded by requiring the host list. An empty `TF_VAR_` was checked to yield `""` (not the default) with Terraform 1.9.0 in a scratch module.
- Promote works for tags older than the script: the script comes from the workflow commit, and Terraform at an older tag ignores undeclared `TF_VAR_` entries. Such a tag still plans its own behavior for variables it predates (documented in the runbook).

## How to run / verify

```bash
export PATH=/Users/kaylynvannorstrand/.nvm/versions/node/v24.20.0/bin:$PATH
bash -n scripts/relay-terraform-vars.sh
pnpm exec vitest run --config vitest.config.node.mjs tests/agent tests/relay --exclude 'tests/relay/e2e/**'
env -i PATH=/usr/bin:/bin RELAY_MIN_INSTANCES=1 bash scripts/relay-terraform-vars.sh
```

The script has not run inside a GitHub Actions workflow yet; the first deploy or promote after a variable is set is the first real test. Read that plan before applying. Setting the variables is in [the runbook](../rollout-pilot.md), section 4.
