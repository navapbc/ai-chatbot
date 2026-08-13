#!/bin/bash

# VM startup script for Container-Optimized OS
# Runs the browser-streaming container

set -euo pipefail

# Log function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [STARTUP] $1" | systemd-cat -t vm-startup
}

log "Starting services initialization..."

# Configure Docker for Artifact Registry
log "Configuring Docker for Artifact Registry..."
export DOCKER_CONFIG=/tmp/.docker
mkdir -p "$DOCKER_CONFIG"

if command -v docker-credential-gcr >/dev/null 2>&1; then
    log "Using docker-credential-gcr..."
    docker-credential-gcr configure-docker --registries=us-central1-docker.pkg.dev
else
    log "Using gcloud auth configure-docker..."
    gcloud auth configure-docker us-central1-docker.pkg.dev --quiet
fi

# Pull images
log "Pulling browser-streaming image: ${browser_image}"
DOCKER_CONFIG="$DOCKER_CONFIG" docker pull "${browser_image}" || {
    log "Failed to pull browser image"
    exit 1
}

# Create Docker network
log "Creating Docker network..."
docker network create browser-network 2>/dev/null || log "Network already exists"

# Stop and remove existing containers
log "Cleaning up existing containers..."
docker stop browser-streaming 2>/dev/null || true
docker rm browser-streaming 2>/dev/null || true

# Create artifacts directory
mkdir -p /tmp/artifacts
chmod 755 /tmp/artifacts

# Write Vertex AI credentials to file for Docker container mounting
# Clean up if previous run left a directory instead of file (Docker creates dirs for missing mount paths)
log "Writing Vertex AI credentials file..."
rm -rf /tmp/vertex-ai-credentials.json
cat > /tmp/vertex-ai-credentials.json << 'EOFCREDS'
${vertex_ai_credentials}
EOFCREDS
chmod 644 /tmp/vertex-ai-credentials.json

# Start browser-streaming container
log "Starting browser-streaming container..."
docker run -d \
    --name browser-streaming \
    --restart unless-stopped \
    --network browser-network \
    -p 8931:8931 \
    -p 8933:8933 \
    -v /tmp/artifacts:/app/artifacts \
    -e ENVIRONMENT="${environment}" \
    -e GCP_PROJECT_ID="${project_id}" \
    "${browser_image}"

# Wait for browser service to be running
log "Waiting for browser-streaming container to start..."
sleep 10
if ! docker ps | grep -q browser-streaming; then
    log "Browser-streaming container failed to start"
    docker logs browser-streaming
    exit 1
fi
log "Browser-streaming container is running"


# Set up log forwarding
log "Setting up log forwarding..."
docker logs -f browser-streaming &

log "All services started successfully!"

# Signal readiness
ZONE=$(curl -H "Metadata-Flavor: Google" -s http://metadata.google.internal/computeMetadata/v1/instance/zone | cut -d/ -f4)
INSTANCE_NAME=$(curl -H "Metadata-Flavor: Google" -s http://metadata.google.internal/computeMetadata/v1/instance/name)

gcloud compute instances add-metadata "$${INSTANCE_NAME}" \
  --metadata services-ready=true \
  --zone="$${ZONE}" || {
  log "Warning: Could not set readiness metadata"
}

log "Services ready!"
