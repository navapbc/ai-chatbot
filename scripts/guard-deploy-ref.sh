#!/usr/bin/env bash
# Refuses a deploy or promote of the extension branch to the shared
# environments, and a dispatch of dev or prod from the wrong branch.
#
# Run as the first step of deploy.yml (setup), promote.yml (promote) and
# cleanup-preview.yml (cleanup). Reads only these variables, mapped with `env:`
# in the workflow (never interpolated into the script):
#   GITHUB_EVENT_NAME  the triggering event
#   GUARD_REF_NAME     github.ref_name (the branch a dispatch runs from)
#   GUARD_HEAD_REF     github.head_ref (a pull request's source branch)
#   GUARD_ENVIRONMENT  inputs.environment on a dispatch, empty otherwise
#
# Rules:
#   - workflow_dispatch from the extension branch: refused.
#   - pull_request from the extension branch: refused.
#   - workflow_dispatch of dev: only from develop; of prod: only from main;
#     any other environment (preview, or none): from any other branch.
#   - push and any other event: allowed (deploy.yml pushes only on
#     develop and main).
# On refusal prints one fixed line (no input value) to stderr and exits 1;
# otherwise prints nothing and exits 0. See docs/rollout-pilot.md section 4.7.

set -euo pipefail

EXTENSION_BRANCH=browser-extension

event="${GITHUB_EVENT_NAME:-}"
ref_name="${GUARD_REF_NAME:-}"
head_ref="${GUARD_HEAD_REF:-}"
environment="${GUARD_ENVIRONMENT:-}"

refuse() {
  echo "::error::guard-deploy-ref: $1 (see docs/rollout-pilot.md section 4.7)" >&2
  exit 1
}

case "$event" in
  workflow_dispatch)
    if [[ "$ref_name" == "$EXTENSION_BRANCH" ]]; then
      refuse "the extension branch is never deployed to the shared environments"
    fi
    if [[ "$environment" == "dev" && "$ref_name" != "develop" ]]; then
      refuse "dev can only be dispatched from develop"
    fi
    if [[ "$environment" == "prod" && "$ref_name" != "main" ]]; then
      refuse "prod can only be dispatched from main"
    fi
    ;;
  pull_request)
    if [[ "$head_ref" == "$EXTENSION_BRANCH" ]]; then
      refuse "the extension branch is never deployed to the shared environments"
    fi
    ;;
  *)
    ;;
esac

exit 0
