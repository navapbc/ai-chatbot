# Browser relay - bridges agent-browser (in the app container) to the bridge
# extension in a caseworker's browser. Image: relay/Dockerfile.
#
# Skipped entirely when relay_image_url is empty, and the app container's relay
# env vars in cloud_run.tf follow the same switch.

locals {
  relay_enabled = var.relay_image_url != ""

  # Suffixed with base_environment, not environment: every preview-pr-N shares
  # the "-preview" secrets, the same way database-url-preview is shared.
  # Per-PR secret names could not be created ahead of a PR's first deploy, and a
  # missing secret fails the revision. Consequence: all preview environments
  # share one signing key, so a run token minted by one preview's app is also
  # accepted by every other preview's relay (dev and prod keys stay separate).
  # The owner creates these (Terraform never holds the values):
  # relay-token-secret-{dev,preview,prod} (>= 32 chars) and
  # eve-internal-secret-{dev,preview,prod}.
  relay_token_secret_name  = "relay-token-secret-${local.base_environment}"
  eve_internal_secret_name = "eve-internal-secret-${local.base_environment}"

  # Same shape as cloud-run-${var.environment}. Account ids are 6-30 chars;
  # "relay-preview-pr-N" stays under 30 until N has 13 digits, and substr keeps
  # any longer id deterministic rather than failing the plan.
  relay_account_id = substr("relay-${var.environment}", 0, 30)
}

# Dedicated identity (R26): no project-level roles at all. Its only grant is
# read access to this environment's relay-token-secret, below, so the relay
# does not inherit the shared Cloud Run account's project-wide secret access.
resource "google_service_account" "relay" {
  count = local.relay_enabled ? 1 : 0

  account_id   = local.relay_account_id
  display_name = "Browser relay (${var.environment})"
  description  = "Runs the browser relay Cloud Run service in ${var.environment}; reads only its token secret"
}

# The secret (and its value) is created by the owner, not Terraform; reading it
# here makes the plan fail early, with a clear error, if it does not exist yet.
data "google_secret_manager_secret" "relay_token" {
  count = local.relay_enabled ? 1 : 0

  project   = local.project_id
  secret_id = local.relay_token_secret_name
}

resource "google_secret_manager_secret_iam_member" "relay_token_access" {
  count = local.relay_enabled ? 1 : 0

  project   = data.google_secret_manager_secret.relay_token[0].project
  secret_id = data.google_secret_manager_secret.relay_token[0].secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.relay[0].email}"
}

resource "google_cloud_run_v2_service" "relay" {
  count = local.relay_enabled ? 1 : 0

  name     = local.env_config.relay_service_name
  location = local.region

  # Disable deletion protection for preview environments to allow easy cleanup
  deletion_protection = startswith(var.environment, "preview-") ? false : true

  template {
    # Dedicated account; reads only relay-token-secret-<base_environment>.
    service_account = google_service_account.relay[0].email

    # No vpc_access: the relay talks only to its two WebSocket peers and needs
    # neither the database nor the NAT's static IP.

    containers {
      image = var.relay_image_url

      resources {
        limits = {
          cpu    = "1"
          memory = "512Mi"
        }
      }

      # PORT is injected by Cloud Run; relay/main.ts reads it.

      env {
        name = "RELAY_TOKEN_SECRET"
        value_source {
          secret_key_ref {
            secret  = local.relay_token_secret_name
            version = "latest"
          }
        }
      }

      env {
        name  = "RELAY_ALLOWED_HOST_SUFFIXES"
        value = ".run.app"
      }

      env {
        name  = "RELAY_EXTENSION_ID"
        value = trimspace(file("${path.module}/../extension/EXTENSION_ID"))
      }

      # Deliberately no REDIS_URL / RELAY_MULTI_INSTANCE_ACKNOWLEDGED: the
      # supported deployment is one instance with the in-process bus (see
      # relay_max_instances in variables.tf).

      ports {
        container_port = 8080
      }
    }

    # Request concurrency stays at Cloud Run's default (80 per instance). Each
    # run holds about two WebSockets (extension and agent), so one instance
    # carries roughly 40 concurrent runs.
    scaling {
      min_instance_count = 0
      max_instance_count = var.relay_max_instances
    }

    # Each run is two long-lived WebSockets (extension and agent) that must
    # meet in one process; with a single instance there is nothing to pin.
    session_affinity = false

    # Cloud Run's maximum. A WebSocket is a request, so this caps how long one
    # run's connection lasts before the client must reconnect.
    timeout = "3600s"
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }

  labels = merge(local.common_labels, {
    environment = var.environment
    component   = "browser-relay"
  })

  depends_on = [
    google_project_service.required_apis,
    # The revision resolves RELAY_TOKEN_SECRET as this account at deploy time.
    google_secret_manager_secret_iam_member.relay_token_access,
  ]
}

# Public invoker: the extension connects from a caseworker's browser and
# cannot attach a Google identity token, so Cloud Run IAM cannot be the gate.
# Every WebSocket must instead present a run token signed with
# RELAY_TOKEN_SECRET; the relay refuses the upgrade otherwise.
resource "google_cloud_run_v2_service_iam_member" "relay_public_access" {
  count = local.relay_enabled ? 1 : 0

  name     = google_cloud_run_v2_service.relay[0].name
  location = google_cloud_run_v2_service.relay[0].location
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# Request-log exclusion (R27). Both WebSocket legs authenticate with a run token
# in the URL query, and Cloud Run's platform request logs record the full URL,
# so anyone with log read access on the project could lift a live token for the
# token's lifetime (up to 30 minutes). This drops only the relay service's
# platform request logs (run.googleapis.com/requests); the relay's own
# application logs on stdout are unaffected and never contain tokens.
#
# This is a project-level logging resource (one per environment, named
# uniquely) that the owner applies; set relay_exclude_request_logs = false to
# keep request logs, accepting the exposure above.
resource "google_logging_project_exclusion" "relay_request_logs" {
  count = local.relay_enabled && var.relay_exclude_request_logs ? 1 : 0

  project     = local.project_id
  name        = "relay-request-logs-${var.environment}"
  description = "Drops Cloud Run request logs for ${local.env_config.relay_service_name}: the URL carries a relay run token"
  filter      = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${local.env_config.relay_service_name}\" AND logName:\"run.googleapis.com%2Frequests\""
}
