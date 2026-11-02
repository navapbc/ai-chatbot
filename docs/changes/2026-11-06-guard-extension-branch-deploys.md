# Guard Extension Branch Deploys

## Summary

The extension branch (`browser-extension`) can no longer deploy, promote or clean up a preview through the GitHub workflows. A new script, `scripts/guard-deploy-ref.sh`, runs as the first step of `deploy.yml`, `promote.yml` and `cleanup-preview.yml` and fails the run before any build, Google Cloud or Terraform step. Nothing was deployed or applied.

## Changes made

- `scripts/guard-deploy-ref.sh` (new): reads only `GITHUB_EVENT_NAME`, `GUARD_REF_NAME` (`github.ref_name`), `GUARD_HEAD_REF` (`github.head_ref`) and `GUARD_ENVIRONMENT` (`inputs.environment`). It refuses a `workflow_dispatch` from `browser-extension`, a `pull_request` whose source branch is `browser-extension`, and a dispatch of `dev` from any branch but `develop` or of `prod` from any branch but `main`. `preview` stays allowed from other branches; push and unknown events pass. A refusal prints one fixed `::error::` line (no input value) to stderr and exits 1; success prints nothing. The branch name is the `EXTENSION_BRANCH` constant.
- `.github/workflows/deploy.yml`: `setup` starts with a sparse, credential-less checkout of the script and the guard step; the values reach it through `env:` only.
- `.github/workflows/promote.yml`: `promote` starts with a sparse checkout of `github.sha` into `guard-ref/` and the guard step, before the tag checkout.
- `.github/workflows/cleanup-preview.yml`: the guard runs after the existing checkout and before Google Cloud authentication, so closing a pull request from `browser-extension` runs no `terraform destroy` or `gsutil`.
- Docs: `docs/rollout-pilot.md` section 4.7 (and a pointer in 4.3); `CLAUDE.md` Deployment paragraph.
- Tests: `tests/agent/deploy-ref-guard.test.ts` runs the script with a controlled environment for every rule and checks that no input value is echoed, and checks the step order and `env:` mappings in all three workflows.

## What it improves or fixes

- `dev`, `preview` and every `preview-pr-N` use the dev Cloud SQL instance. Before, a dispatch of "Deploy Infrastructure" from `browser-extension` (any environment), or a pull request from it into `develop`/`main`, would have applied this branch's migrations to the database the develop site uses. Those runs now fail at their first step.
- A dispatch of `dev` or `prod` from a feature branch based on this branch is refused too, so only `develop` deploys dev and only `main` deploys prod.
- The guard lives only in this branch's workflow files (GitHub runs the file from the dispatched ref or the pull request's merge commit). `develop` and `main` do not carry it, so their deploys and promotes are unchanged.
- Not covered: the guard matches the branch name only. A branch cut from `browser-extension` can still be dispatched to `preview` or opened as a pull request into `develop`/`main`, and both use the dev Cloud SQL instance.
- Not covered: a merge of `browser-extension` into `develop` or `main` deploys on the following push and cannot be blocked by a workflow. The protection for that is a branch ruleset or required review, which is the owner's setting.

## How to run / verify

```bash
export PATH=/Users/kaylynvannorstrand/.nvm/versions/node/v24.20.0/bin:$PATH
bash -n scripts/guard-deploy-ref.sh
pnpm exec vitest run --config vitest.config.node.mjs tests/agent tests/relay --exclude 'tests/relay/e2e/**'
env -i PATH=/usr/bin:/bin GITHUB_EVENT_NAME=workflow_dispatch GUARD_REF_NAME=browser-extension GUARD_ENVIRONMENT=preview bash scripts/guard-deploy-ref.sh; echo "exit $?"
```

The guard has not run inside GitHub Actions yet. The first dispatch from this branch (expected to fail at "Guard deploy ref") is the first real check.
