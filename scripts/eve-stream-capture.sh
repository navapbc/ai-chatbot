#!/usr/bin/env bash
# Captures the raw NDJSON event stream from an Eve session for analysis.
# Set BASE to match the dev server from Task 1 (port 2000 for `eve dev`,
# 3000 if Eve is mounted into Next), or a Vercel preview URL in Task 5.
# Usage: BASE=http://127.0.0.1:2000 ./scripts/eve-stream-capture.sh
set -euo pipefail
BASE="${BASE:-http://127.0.0.1:2000}"
OUT="${OUT:-eve-stream-capture.ndjson}"

resp=$(curl -sD - -o /dev/null -X POST "$BASE/eve/v1/session" \
  -H 'content-type: application/json' \
  -d '{"message":"Read the reference field-patterns.md and tell me what it covers."}')
sid=$(printf '%s' "$resp" | tr -d '\r' | awk -F': ' 'tolower($1)=="x-eve-session-id"{print $2}')
echo "session: $sid"
curl -N "$BASE/eve/v1/session/$sid/stream" | tee "$OUT"
