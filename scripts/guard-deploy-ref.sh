#!/usr/bin/env bash
# Refuses a deploy or promote of the extension branch to the shared
# environments, and a dispatch of dev or prod from the wrong branch.
#
# Run as the first step of deploy.yml (setup), promote.yml (promote) and
# cleanup-preview.yml (cleanup). Reads only these variables, mapped with `env:`
# in the workflow (never interpolated into the script):
#   GUARD_EVENT_NAME   github.event_name
#   GUARD_REF          github.ref, the full ref a dispatch runs from
#                      (refs/heads/<branch> or refs/tags/<tag>)
#   GUARD_HEAD_REF     github.head_ref, a pull request's source branch (short)
#   GUARD_ENVIRONMENT  inputs.environment on a dispatch ('preview' for cleanup)
#
# Rules:
#   - workflow_dispatch:
#       - from any ref that is not a branch (a tag): refused;
#       - from the extension branch: refused;
#       - an environment other than exactly dev, preview or prod: refused;
#       - dev only from develop, prod only from main; preview from any other
#         branch.
#   - pull_request from the extension branch: refused.
#   - push and any other event: allowed (deploy.yml pushes only on
#     develop and main).
# On refusal prints one fixed line (no input value) to stderr and exits 1;
# otherwise prints nothing and exits 0. See docs/rollout-pilot.md section 4.7.

set -euo pipefail

EXTENSION_BRANCH=browser-extension

event="${GUARD_EVENT_NAME:-}"
ref="${GUARD_REF:-}"
head_ref="${GUARD_HEAD_REF:-}"
environment="${GUARD_ENVIRONMENT:-}"

refuse() {
  echo "::error::guard-deploy-ref: $1 (see docs/rollout-pilot.md section 4.7)" >&2
  exit 1
}

case "$event" in
  workflow_dispatch)
    if [[ "$ref" != refs/heads/* ]]; then
      refuse "a deploy can only be dispatched from a branch"
    fi
    if [[ "$ref" == "refs/heads/$EXTENSION_BRANCH" ]]; then
      refuse "the extension branch is never deployed to the shared environments"
    fi
    case "$environment" in
      dev | preview | prod) ;;
      *) refuse "the environment must be dev, preview or prod" ;;
    esac
    if [[ "$environment" == "dev" && "$ref" != "refs/heads/develop" ]]; then
      refuse "dev can only be dispatched from develop"
    fi
    if [[ "$environment" == "prod" && "$ref" != "refs/heads/main" ]]; then
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
