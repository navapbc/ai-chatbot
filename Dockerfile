# Dockerfile for AI Chatbot (Next.js client)
FROM node:24-slim AS base

# Install basic system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    ca-certificates \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Install pnpm globally (pinned to match packageManager in package.json)
RUN npm install -g pnpm@11.18.0

# Stage 1: Build stage
FROM base AS builder

# Set working directory
WORKDIR /app

# Copy app source (kept at /app/client to match the pre-migration image layout)
COPY . ./client/

# Go to client directory and install dependencies
WORKDIR /app/client
RUN rm -rf node_modules && pnpm install --frozen-lockfile --ignore-scripts

# Build args
ARG NEXT_PUBLIC_POSTHOG_KEY
ARG NEXT_PUBLIC_POSTHOG_HOST=https://us.i.posthog.com
ARG USE_GUEST_LOGIN=false
ARG KERNEL_API_KEY
ARG ENVIRONMENT=dev

# Set environment variables for build time
ENV NEXT_PUBLIC_POSTHOG_KEY=${NEXT_PUBLIC_POSTHOG_KEY}
ENV NEXT_PUBLIC_POSTHOG_HOST=${NEXT_PUBLIC_POSTHOG_HOST}
ENV USE_GUEST_LOGIN=${USE_GUEST_LOGIN}
ENV NEXT_PUBLIC_USE_GUEST_LOGIN=${USE_GUEST_LOGIN}
ENV KERNEL_API_KEY=${KERNEL_API_KEY}
ENV ENVIRONMENT=${ENVIRONMENT}
ENV NEXT_PUBLIC_ENVIRONMENT=${ENVIRONMENT}

# Build Next.js client only (migrations run at container startup)
RUN pnpm next build

# Stage 2: Runtime stage
FROM base AS runtime

# Copy built application
WORKDIR /app
COPY --from=builder /app/client ./client

# agent-browser ships prebuilt binaries for all 7 platforms (~87MB) in the npm
# tarball. The package's postinstall only downloads a binary that is already
# present, so `--ignore-scripts` costs us nothing — we just need the execute bit
# and a stable path. Drop the six binaries this image can't run (~74MB).
#
# Keep `bin/agent-browser.js`: pnpm's `node_modules/.bin/agent-browser` shim
# execs it by name, and `pnpm start` prepends `node_modules/.bin` to PATH — so
# deleting it makes every invocation fail with MODULE_NOT_FOUND even though
# /usr/local/bin has a working symlink. Replace it with a shim that execs the
# native binary, so both resolution paths reach the same place.
#
# The emptiness check below tests the binary path, not its dirname: `dirname ""`
# returns `.`, which would aim the prune at the working directory.
RUN set -eu; \
    AB_BIN="$(find /app/client/node_modules/.pnpm \
        -path '*/node_modules/agent-browser/bin/agent-browser-linux-x64' \
        -print -quit)"; \
    test -n "$AB_BIN" || { echo 'agent-browser linux-x64 binary not found'; exit 1; }; \
    AB_DIR="$(dirname "$AB_BIN")"; \
    find "$AB_DIR" -type f ! -name 'agent-browser-linux-x64' -delete; \
    chmod +x "$AB_BIN"; \
    printf '#!/usr/bin/env node\nrequire("node:child_process").spawnSync(%s,process.argv.slice(2),{stdio:"inherit"});\n' \
        "\"$AB_BIN\"" > "$AB_DIR/agent-browser.js"; \
    chmod +x "$AB_DIR/agent-browser.js"; \
    ln -sf "$AB_BIN" /usr/local/bin/agent-browser; \
    agent-browser --version

# Create a non-root user for better security
RUN groupadd -r nextjs && useradd -r -g nextjs -d /app -s /bin/bash nextjs

# Change ownership of the app directory to the nextjs user
RUN chown -R nextjs:nextjs /app

# Switch to non-root user
USER nextjs

# agent-browser's daemon creates its control socket under $HOME/.agent-browser.
# HOME defaults to /app, which is not writable when the container runs with a
# read-only root filesystem, and the daemon exits with "Failed to create socket
# directory". /tmp is writable in every environment we deploy to.
ENV HOME=/tmp
ENV AGENT_BROWSER_PROVIDER=kernel

# Invoke the native binary directly. `pnpm start` puts node_modules/.bin ahead
# of /usr/local/bin on PATH, so a bare `agent-browser` would go through pnpm's
# Node shim and pay a process spawn on every command.
ENV AGENT_BROWSER_BIN=/usr/local/bin/agent-browser

# Set working directory to client
WORKDIR /app/client

# Expose Next.js port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:3000/health || curl -f http://localhost:3000 || exit 1

# Start Next.js server (run migrations first, then start)
CMD ["sh", "-c", "pnpm tsx lib/db/migrate && pnpm start"]
