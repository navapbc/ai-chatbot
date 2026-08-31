#!/usr/bin/env bash
#
# Container entrypoint: schema steps, then the Eve runtime and Next.js together.
#
# Why two processes instead of letting withEve() spawn Eve for us: eve/next's
# `startEveProductionServer` only runs when Next re-evaluates `rewrites()` at
# server start. Under Next 16 the rewrite is baked into
# `.next/routes-manifest.json` at build time and `rewrites()` is NOT called
# again by `next start`, so nothing ever spawns the Eve process and every
# /eve/v1/** request 502s with ECONNREFUSED on 127.0.0.1:4274. Verified locally:
# `next build && next start` alone reproduces exactly that. The baked rewrite is
# correct, so we simply start the process it points at.
#
# EVE_PORT must stay in sync with eve/next's default local production port
# (4274, overridable at BUILD time via EVE_NEXT_PRODUCTION_PORT). It is the
# destination compiled into the routes manifest.

set -euo pipefail

EVE_PORT="${EVE_NEXT_PRODUCTION_PORT:-4274}"
EVE_ENTRY=".output/server/index.mjs"

if [[ ! -f "${EVE_ENTRY}" ]]; then
  echo "[start] FATAL: ${EVE_ENTRY} missing — 'pnpm eve build' did not run in the image build." >&2
  exit 1
fi

echo "[start] applying app migrations"
pnpm tsx lib/db/migrate

echo "[start] applying Eve workflow schema"
pnpm tsx scripts/bootstrap-workflow-db.ts

# Eve listens on loopback only: it is reached through Next's /eve/v1/** rewrite,
# never directly from outside the container. PORT is set per-process so it does
# not disturb the PORT Cloud Run hands to Next.
echo "[start] starting Eve runtime on 127.0.0.1:${EVE_PORT}"
HOST=127.0.0.1 NITRO_HOST=127.0.0.1 NITRO_PORT="${EVE_PORT}" PORT="${EVE_PORT}" \
  node "${EVE_ENTRY}" &
EVE_PID=$!

# Invoked directly rather than through `pnpm start`: pnpm would be the process
# we hold a PID for, while the actual server runs as its grandchild, so the
# shutdown path below would signal the wrapper and leave Next serving. Verified:
# with `pnpm start`, killing Eve logged the shutdown but left :3000 listening.
echo "[start] starting Next.js"
./node_modules/.bin/next start &
NEXT_PID=$!

# Either process dying makes the container unhealthy: a live Next with a dead
# Eve serves 502s on every agent request, which Cloud Run's HTTP health check
# would not catch. Exit instead and let Cloud Run replace the instance.
terminate() {
  trap - TERM INT
  kill "${EVE_PID}" "${NEXT_PID}" 2>/dev/null || true
  # Escalate rather than `wait` indefinitely: a wedged child would otherwise
  # hold the container open with a half-dead service behind it.
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    kill -0 "${EVE_PID}" 2>/dev/null || kill -0 "${NEXT_PID}" 2>/dev/null || return 0
    sleep 1
  done
  kill -9 "${EVE_PID}" "${NEXT_PID}" 2>/dev/null || true
}
trap terminate TERM INT

# Polled rather than `wait -n $PID...`: that form needs bash >= 5.1. The image
# (node:24-slim) has it, but macOS ships bash 3.2, so `wait -n` cannot be
# exercised on a developer machine — it fails with "invalid option" and the
# supervisor silently stops supervising. A poll behaves identically on both, so
# this logic is actually testable before it ships.
while kill -0 "${EVE_PID}" 2>/dev/null && kill -0 "${NEXT_PID}" 2>/dev/null; do
  sleep 2
done

if kill -0 "${EVE_PID}" 2>/dev/null; then
  echo "[start] Next.js exited; shutting the container down" >&2
else
  echo "[start] Eve runtime exited; shutting the container down" >&2
fi

terminate
exit 1
