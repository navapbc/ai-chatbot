#!/usr/bin/env bash
# Turns the relay pilot settings (GitHub ENVIRONMENT-level variables, mapped
# into this step's env by .github/workflows/deploy.yml and promote.yml) into
# TF_VAR_<name>=<value> lines for "$GITHUB_ENV". Both workflows run this one
# script so they cannot drift: wiring a variable into only one of them would
# let the other plan it back to its Terraform default.
#
# An unset or blank variable prints nothing, so Terraform's default applies
# (an unset GitHub variable expands to ''). The one way to pass an empty value
# is RELAY_ALLOWED_HOST_SUFFIXES=none, which prints an empty
# TF_VAR_relay_allowed_host_suffixes and is accepted only together with
# RELAY_ALLOWED_HOSTS (second apply of the exact-host switch, see
# docs/rollout-pilot.md section 4.4).
#
# Every value is checked before anything is printed. On any failure the script
# prints one fixed line naming the variable to stderr, never the value, prints
# nothing to stdout and exits 1. Values are single-line by construction (no
# control characters), so appending to "$GITHUB_ENV" cannot add a variable.
#
# Bash 3.2 compatible (macOS runs the tests): no ${v,,}, mapfile or
# associative arrays.
set -euo pipefail
export LC_ALL=C

out=''

fail() {
  printf 'relay-terraform-vars: %s\n' "$1" >&2
  exit 1
}

invalid() {
  fail "$1 is not valid (see docs/rollout-pilot.md section 4.1)"
}

# Every helper below returns through the global R, never through $(...): a
# failure (exit 1) inside a command substitution would end only the subshell.

# R = the value of variable $1, trimmed of surrounding spaces; refuses control
# characters (newlines included) before anything else.
read_var() {
  R="${!1-}"
  if [[ "$R" == *[[:cntrl:]]* ]]; then
    invalid "$1"
  fi
  R="${R#"${R%%[! ]*}"}"
  R="${R%"${R##*[! ]}"}"
}

emit() {
  out="${out}TF_VAR_$1=$2"$'\n'
}

bool_var() {
  read_var "$1"
  [[ -z "$R" ]] && return 0
  [[ "$R" == "true" || "$R" == "false" ]] || invalid "$1"
  emit "$2" "$R"
}

# R = the comma-separated list $2 of variable $1, every entry trimmed,
# lowercased when $4 is "lower" and matched against $3; no empty entry.
list() {
  local name="$1" rest="$2," pattern="$3" lower="$4" entry joined=''
  while [[ -n "$rest" ]]; do
    entry="${rest%%,*}"
    rest="${rest#*,}"
    entry="${entry#"${entry%%[! ]*}"}"
    entry="${entry%"${entry##*[! ]}"}"
    if [[ "$lower" == "lower" ]]; then
      entry="$(printf '%s' "$entry" | tr 'A-Z' 'a-z')"
    fi
    [[ -n "$entry" && "$entry" =~ $pattern ]] || invalid "$name"
    joined="${joined:+${joined},}${entry}"
  done
  R="$joined"
}

HOST_PATTERN='^[a-z0-9]([a-z0-9.-]*[a-z0-9])?$'
SUFFIX_PATTERN='^\.?[a-z0-9]([a-z0-9.-]*[a-z0-9])?$'
MODEL_PATTERN='^[A-Za-z0-9._/:-]+$'
ORIGIN_PATTERN='^chrome-extension://[a-p]{32}$'
R=''

bool_var RELAY_RUNS_ENABLED relay_runs_enabled
bool_var RELAY_TOKEN_ENDPOINT_ENABLED relay_token_endpoint_enabled
bool_var RELAY_COMMAND_DURATION_METRIC_ENABLED relay_command_duration_metric_enabled

read_var RELAY_MIN_INSTANCES
if [[ -n "$R" ]]; then
  [[ "$R" == "0" || "$R" == "1" ]] || invalid RELAY_MIN_INSTANCES
  emit relay_min_instances "$R"
fi

read_var RELAY_EXTENSION_ORIGIN
if [[ -n "$R" ]]; then
  [[ "$R" =~ $ORIGIN_PATTERN ]] || invalid RELAY_EXTENSION_ORIGIN
  emit relay_extension_origin "$R"
fi

read_var RELAY_ALLOWED_MODELS
if [[ -n "$R" ]]; then
  list RELAY_ALLOWED_MODELS "$R" "$MODEL_PATTERN" keep
  emit relay_allowed_models "$R"
fi

read_var RELAY_ALLOWED_HOSTS
hosts_set=''
if [[ -n "$R" ]]; then
  list RELAY_ALLOWED_HOSTS "$R" "$HOST_PATTERN" lower
  emit relay_allowed_hosts "$R"
  hosts_set=1
fi

read_var RELAY_ALLOWED_HOST_SUFFIXES
# Any casing of none is the sentinel: otherwise "None" would be lowercased by
# list() into a literal suffix and drop .run.app without the host list.
if [[ "$(printf '%s' "$R" | tr 'A-Z' 'a-z')" == "none" ]]; then
  [[ -n "$hosts_set" ]] || fail "RELAY_ALLOWED_HOST_SUFFIXES=none needs RELAY_ALLOWED_HOSTS"
  emit relay_allowed_host_suffixes ""
elif [[ -n "$R" ]]; then
  list RELAY_ALLOWED_HOST_SUFFIXES "$R" "$SUFFIX_PATTERN" lower
  emit relay_allowed_host_suffixes "$R"
fi

printf '%s' "$out"
