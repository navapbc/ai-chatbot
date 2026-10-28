#!/usr/bin/env bash
# Starts the relay for local extension development.
#
# Usage: scripts/dev-relay.sh
#
# Loads .env.local, checks that the variables a local extension run needs are
# set (it prints only their names, never their values), then runs
# relay/main.ts on PORT (default 8080), allowing the localhost hosts.
# See docs/local-extension.md.
set -euo pipefail

cd "$(dirname "$0")/.."

ENV_FILE=".env.local"
if [ ! -f "$ENV_FILE" ]; then
  echo "dev-relay: $ENV_FILE not found in $(pwd)" >&2
  exit 1
fi

# Presence check only: a name counts as set when it has a non-empty value.
is_set() {
  grep -Eq "^[[:space:]]*$1=[^[:space:]]" "$ENV_FILE"
}

missing=()
for name in \
  DATABASE_URL \
  AUTH_SECRET \
  RELAY_RUNS_ENABLED \
  RELAY_TOKEN_ENDPOINT_ENABLED \
  RELAY_TOKEN_SECRET \
  EVE_INTERNAL_SECRET \
  RELAY_PUBLIC_URL; do
  is_set "$name" || missing+=("$name")
done

if [ "${#missing[@]}" -gt 0 ]; then
  echo "dev-relay: these variables are missing or empty in $ENV_FILE:" >&2
  printf '  %s\n' "${missing[@]}" >&2
  echo "See docs/local-extension.md for what each one is." >&2
  exit 1
fi

# The two flags must be exactly 1.
for name in RELAY_RUNS_ENABLED RELAY_TOKEN_ENDPOINT_ENABLED; do
  if ! grep -Eq "^[[:space:]]*$name=1[[:space:]]*$" "$ENV_FILE"; then
    echo "dev-relay: $name must be exactly 1 in $ENV_FILE" >&2
    exit 1
  fi
done

# The relay and the app must share RELAY_TOKEN_SECRET (both read .env.local).
# The relay checks its length itself and exits with a generic message.
export PORT="${PORT:-8080}"
export RELAY_ALLOWED_HOSTS="${RELAY_ALLOWED_HOSTS:-localhost,127.0.0.1}"

echo "dev-relay: starting on port $PORT (allowed hosts: $RELAY_ALLOWED_HOSTS)"
exec pnpm exec dotenv -e "$ENV_FILE" -- tsx relay/main.ts
