# Multi-stage Dockerfile for Next.js Client + Mastra + Playwright MCP setup
FROM node:20-slim AS base

# Install system dependencies needed for Playwright and git
RUN apt-get update && apt-get install -y \
    git \
    wget \
    gnupg \
    ca-certificates \
    procps \
    curl \
    libxss1 \
    libgconf-2-4 \
    libxrandr2 \
    libasound2 \
    libpangocairo-1.0-0 \
    libatk1.0-0 \
    libc6-dev \
    libdrm2 \
    libgtk-3-0 \
    libgdk-pixbuf2.0-0 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxi6 \
    libxtst6 \
    libnss3 \
    libcups2 \
    libxrandr2 \
    libasound2 \
    libpangocairo-1.0-0 \
    libatk1.0-0 \
    libdrm-common \
    libdrm2 \
    libxss1 \
    libgconf-2-4 \
    && rm -rf /var/lib/apt/lists/*

# Install pnpm globally
RUN npm install -g pnpm

# Set working directory
WORKDIR /app

# Stage 1: Build stage
FROM base AS builder

# Copy parent package files for Mastra backend build
COPY package.json pnpm-lock.yaml ./parent/
COPY src ./parent/src/
COPY browser-streaming-server.js ./parent/

# Copy client package files
COPY client/package.json client/pnpm-lock.yaml ./

# Install parent dependencies first (for Mastra build)
WORKDIR /app/parent
RUN pnpm install --frozen-lockfile

# Build Mastra backend
RUN pnpm run build

# Install client dependencies
WORKDIR /app
RUN pnpm install --frozen-lockfile --ignore-scripts

# Set environment variable for Mastra build path (Docker-specific)
ENV MASTRA_DIR=/app/parent/src/mastra

# Disable Mastra's internal browser streaming service (we use standalone server)
ENV DISABLE_MASTRA_BROWSER_STREAMING=true

# Install Playwright MCP server for build-time use
RUN npm install -g @playwright/mcp@latest

# Set environment variable for browser path (standard Playwright location)
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

# Install Playwright browsers for build-time use
RUN /usr/local/lib/node_modules/@playwright/mcp/node_modules/.bin/playwright install --with-deps chromium

# Copy client source code
COPY client/ .

# Start Playwright MCP server in background and build client (using same config as main server)
RUN /usr/local/lib/node_modules/@playwright/mcp/cli.js --port 8931 --isolated --browser chromium --no-sandbox \
      --user-agent "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36" \
      --viewport-size "1920,1080" &\
    PLAYWRIGHT_PID=$! &&\
    echo "Waiting for Playwright MCP server to be ready..." &&\
    sleep 5 &&\
    echo "Playwright MCP server started for build" &&\
    pnpm run build &&\
    kill $PLAYWRIGHT_PID || true

# Stage 2: Runtime stage with Playwright
FROM base AS runtime

# Install Playwright MCP server (includes playwright as dependency) 
RUN npm install -g @playwright/mcp@latest

# Set environment variable for browser path (standard Playwright location)
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

# Install Playwright browsers using the MCP server's bundled Playwright version
RUN /usr/local/lib/node_modules/@playwright/mcp/node_modules/.bin/playwright install --with-deps chromium

# Verify browsers are installed and show their location
RUN /usr/local/lib/node_modules/@playwright/mcp/node_modules/.bin/playwright install --dry-run && \
    echo "Browser installation verified using MCP server's Playwright version. Browsers located at: $PLAYWRIGHT_BROWSERS_PATH" && \
    ls -la /ms-playwright/ || ls -la ~/.cache/ms-playwright/ || echo "Browser path not found, but installation completed"

# Copy built application, node_modules, and browsers from builder stage
COPY --from=builder /app .
COPY --from=builder /app/parent/browser-streaming-server.js /app/

# Create a non-root user for better security
RUN groupadd -r nextjs && useradd -r -g nextjs -d /app -s /bin/bash nextjs

# Copy and setup startup script
COPY client/start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Change ownership of the app directory and browser directory to the nextjs user
RUN chown -R nextjs:nextjs /app && \
    chown -R nextjs:nextjs /ms-playwright && \
    mkdir -p /app/artifacts && \
    chown -R nextjs:nextjs /app/artifacts

# Switch to non-root user
USER nextjs

# Expose ports
EXPOSE 3000 8931 8933

# Health check for Next.js
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:3000/api/health || curl -f http://localhost:3000 || exit 1

# Start all services
CMD ["/app/start.sh"]
