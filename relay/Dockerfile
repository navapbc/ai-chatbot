# Browser relay (relay/main.ts) for Cloud Run.
#
# Build from the repo root so lib/relay is in the context:
#   docker build -f relay/Dockerfile -t relay:local .
#
# The build stage bundles relay/main.ts and its dependencies (ws, redis) into a
# single ESM file; the runtime stage holds only that file and node. No source,
# no node_modules, no agent-browser binary.

FROM node:24-slim AS builder

# Pinned to match packageManager in package.json (same as the root Dockerfile).
RUN npm install -g pnpm@11.18.0

WORKDIR /app

# Manifests first so the install layer is cached across source-only changes.
# pnpm-workspace.yaml carries the overrides the lockfile was resolved with;
# without it --frozen-lockfile fails.
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile --ignore-scripts

COPY relay ./relay
COPY lib/relay ./lib/relay

# ws and redis are CommonJS and call require() for node built-ins, which an
# ESM bundle does not provide ("Dynamic require of ... is not supported"); the
# banner gives the bundle a real require. bufferutil and utf-8-validate are
# optional native accelerators that ws loads inside a try/catch, so they stay
# external and ws falls back to its pure-JS path.
RUN pnpm exec esbuild relay/main.ts \
    --bundle \
    --platform=node \
    --format=esm \
    --target=node24 \
    --external:bufferutil \
    --external:utf-8-validate \
    "--banner:js=import { createRequire } from 'node:module'; const require = createRequire(import.meta.url);" \
    --outfile=dist/relay.mjs

FROM node:24-slim AS runtime

ENV NODE_ENV=production

WORKDIR /app
COPY --from=builder --chown=node:node /app/dist/relay.mjs ./relay.mjs

# The slim image ships a non-root `node` user.
USER node

# Cloud Run injects PORT; 8080 is the relay's own default.
EXPOSE 8080

# Docker-only (Cloud Run ignores HEALTHCHECK and uses its own TCP startup
# probe). Uses node's fetch because the slim image has no curl.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["node", "-e", "fetch('http://127.0.0.1:'+(process.env.PORT||8080)+'/healthz').then(r=>process.exit(r.ok?0:1),()=>process.exit(1))"]

CMD ["node", "relay.mjs"]
