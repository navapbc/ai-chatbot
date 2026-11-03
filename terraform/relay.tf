# Browser relay - bridges agent-browser (in the app container) to the bridge
# extension in a caseworker's browser. Image: relay/Dockerfile.
#
# Skipped entirely when relay_image_url is empty, and the app container's relay
# env vars in cloud_run.tf follow the same switch.
#
# Owner checklist before the first apply with the relay enabled:
# - Secret Manager secrets relay-token-secret-<base_environment> (>= 32 chars)
#   and eve-internal-secret-<base_environment>, each with a version.
# - roles/logging.configWriter for the deploy service account
#   (github-actions-deploy), which needs logging.exclusions.create for the
#   request-log exclusion below; its current roles (iam.tf) do not include it.
#   Or set relay_exclude_request_logs = false knowingly, which leaves
#   token-bearing request URLs in Cloud Logging. The relay service depends on
#   the exclusion, so a missing permission fails the apply before the relay
#   exists, never with a live relay logging tokens.

locals {
  relay_enabled = var.relay_image_url != ""

  # Suffixed with base_environment, not environment: every preview-pr-N shares
  # the "-preview" secrets, the same way database-url-preview is shared.
  # Per-PR secret names could not be created ahead of a PR's first deploy, and a
  # missing secret fails the revision. Consequence: all preview environments
  # share one signing key, so a run token minted by one preview's app is also
  # accepted by every other preview's relay (dev and prod keys stay separate).
  # The owner creates these (Terraform never holds the values):
  # relay-token-secret-{dev,prod} (>= 32 chars) and
  # eve-internal-secret-{dev,prod}. deploy.yml deploys the relay to dev and
  # prod only; the -preview pair is needed only if a preview relay is ever
  # deployed some other way.
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

# Same early check for the app's EVE_INTERNAL_SECRET (cloud_run.tf): without
# it a missing secret fails the app's Cloud Run revision at apply time, after
# the relay already exists. The app account reads it through its project-level
# secretAccessor role, so no per-secret grant is needed.
data "google_secret_manager_secret" "eve_internal" {
  count = local.relay_enabled ? 1 : 0

  project   = local.project_id
  secret_id = local.eve_internal_secret_name
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

      # Host allowlist (relay/server.ts): a request passes when its Host is in
      # RELAY_ALLOWED_HOSTS or ends with a RELAY_ALLOWED_HOST_SUFFIXES entry.
      # Defaults keep today's ".run.app" suffix only; see relay_allowed_hosts
      # in variables.tf for the two-apply switch to the exact host.
      dynamic "env" {
        for_each = var.relay_allowed_host_suffixes != "" ? [1] : []
        content {
          name  = "RELAY_ALLOWED_HOST_SUFFIXES"
          value = var.relay_allowed_host_suffixes
        }
      }

      dynamic "env" {
        for_each = var.relay_allowed_hosts != "" ? [1] : []
        content {
          name  = "RELAY_ALLOWED_HOSTS"
          value = var.relay_allowed_hosts
        }
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
    # Minimum 0 by default (scale to zero, as before); the pilot sets 1 so a
    # cold start does not drop live runs.
    scaling {
      min_instance_count = var.relay_min_instances
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

  lifecycle {
    # With neither list set the relay answers 421 to every request.
    precondition {
      condition     = var.relay_allowed_hosts != "" || var.relay_allowed_host_suffixes != ""
      error_message = "Set relay_allowed_hosts or relay_allowed_host_suffixes: with neither, the relay refuses every request (421)."
    }

    # The app's RELAY_EXTENSION_ORIGIN and the relay's RELAY_EXTENSION_ID must
    # name the same extension, or every panel request is refused.
    precondition {
      condition     = var.relay_extension_origin == "" || var.relay_extension_origin == "chrome-extension://${trimspace(file("${path.module}/../extension/EXTENSION_ID"))}"
      error_message = "relay_extension_origin must be chrome-extension://<the id in extension/EXTENSION_ID>. If the extension id changed (P5-g, e.g. a Chrome Web Store listing assigned another id), re-derive it, update extension/EXTENSION_ID (which sets RELAY_EXTENSION_ID) and relay_extension_origin together."
    }
  }

  depends_on = [
    google_project_service.required_apis,
    # The revision resolves RELAY_TOKEN_SECRET as this account at deploy time.
    google_secret_manager_secret_iam_member.relay_token_access,
    # Created before the relay serves anything (a failed exclusion stops the
    # apply before token URLs can reach request logs) and destroyed only after
    # the relay is gone. Zero instances when relay_exclude_request_logs = false.
    google_logging_project_exclusion.relay_request_logs,
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
# Exclusions apply to the _Default sink only: any user-defined log sink in the
# project would still receive relay request URLs containing tokens.
#
# This is a project-level logging resource (one per environment, named
# uniquely) that the owner applies; the deploy identity needs
# roles/logging.configWriter (see the checklist at the top). Set
# relay_exclude_request_logs = false to keep request logs, accepting the
# exposure above.
resource "google_logging_project_exclusion" "relay_request_logs" {
  count = local.relay_enabled && var.relay_exclude_request_logs ? 1 : 0

  project     = local.project_id
  name        = "relay-request-logs-${var.environment}"
  description = "Drops Cloud Run request logs for ${local.env_config.relay_service_name}: the URL carries a relay run token"
  filter      = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${local.env_config.relay_service_name}\" AND logName:\"run.googleapis.com%2Frequests\""
}

# Command round-trip time as seen by the relay (P5-h), from the hub's flat
# `{event: "command", durationMs}` log line. Off by default. Exact match on
# event (`=`, not `:`) so command-failed and command-denied lines are not
# counted. No label extractors: the line also carries runId, which must not
# become a metric label. Creating it needs logging.logMetrics.create on the
# deploy identity (roles/logging.configWriter, as for the exclusion above).
resource "google_logging_metric" "relay_command_duration" {
  count = local.relay_enabled && var.relay_command_duration_metric_enabled ? 1 : 0

  project     = local.project_id
  name        = "relay-command-duration-${var.environment}"
  description = "Relay command round trip (ms) for ${local.env_config.relay_service_name}"
  filter      = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${local.env_config.relay_service_name}\" AND jsonPayload.event=\"command\""

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "DISTRIBUTION"
    unit        = "ms"
  }

  value_extractor = "EXTRACT(jsonPayload.durationMs)"

  # 1 ms to about 17 min in powers of two.
  bucket_options {
    exponential_buckets {
      num_finite_buckets = 20
      growth_factor      = 2
      scale              = 1
    }
  }
}
